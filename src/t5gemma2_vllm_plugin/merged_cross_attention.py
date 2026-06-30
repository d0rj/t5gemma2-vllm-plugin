# SPDX-License-Identifier: Apache-2.0
"""Merged self+cross attention for T5Gemma-style decoders.

This module re-implements the HF-exact attention used by the vllm-factory
reference decoder: one softmax over the concatenation of causal decoder keys
and bidirectional encoder keys.  The vLLM paged self-attention block table is
not a reliable source for reconstructing the full decoder history in this
encoder-decoder path, so the layer keeps its own compact per-request decoder KV
history while still updating vLLM's KV cache for the engine.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import AttentionType

from .kernels.flash_t5gemma2_attention import flash_t5gemma2_attention


def _split_and_pad_cross_kv(
    cross_key: torch.Tensor,
    cross_value: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated encoder KV into per-request tensors and pad.

    Args:
        cross_key: ``[total_cross_tokens, num_kv_heads, head_dim]``.
        cross_value: same shape as ``cross_key``.
        seq_lens: ``[num_seqs]`` encoder lengths per request.
        max_seq_len: maximum encoder length in the batch.

    Returns:
        Padded key and value tensors of shape
        ``[num_seqs, max_seq_len, num_kv_heads, head_dim]``.
    """
    num_seqs = seq_lens.shape[0]
    num_kv_heads = cross_key.shape[1]
    head_dim = cross_key.shape[2]
    device = cross_key.device
    dtype = cross_key.dtype

    key_out = torch.zeros(
        num_seqs, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    value_out = torch.zeros_like(key_out)

    offset = 0
    for i, length in enumerate(seq_lens.long().tolist()):
        if length > 0:
            key_out[i, :length] = cross_key[offset : offset + length]
            value_out[i, :length] = cross_value[offset : offset + length]
            offset += length

    return key_out, value_out


def _split_flat_tokens(
    tensor: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> list[torch.Tensor]:
    starts = query_start_loc.long().tolist()
    return [tensor[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


class MergedCrossAttention(nn.Module):
    """One softmax over causal decoder KV and encoder KV.

    This is the attention used by T5Gemma2 decoders.  It is intentionally
    different from sequential cross-attention: the query attends to both the
    decoder history and the encoder output in a single normalization, which
    matches the upstream ``transformers`` reference implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        *,
        num_kv_heads: int,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        prefix: str,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError(
                "MergedCrossAttention does not support quantized KV"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.sliding_window = sliding_window or 0

        # Decoder self-attention layer.  Its KV cache is updated and then read
        # back so that we can concatenate self and cross keys for the merged
        # attention kernel.
        self.self_attn = Attention(
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.self_cache",
            attn_type=AttentionType.DECODER,
        )
        # Cross-attention layer is only used to obtain vLLM's encoder-decoder
        # scheduling metadata (sequence lengths, etc.).  Its forward is never
        # called because the encoder KV is passed explicitly.
        self.cross_attn = CrossAttention(
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            logits_soft_cap=logits_soft_cap,
            prefix=f"{prefix}.cross_cache",
        )
        self._self_key_cache: list[torch.Tensor] = []
        self._self_value_cache: list[torch.Tensor] = []

    @staticmethod
    def _window_size_tuple(window: Any) -> tuple[int, int]:
        if window is None:
            return (-1, -1)
        if isinstance(window, (list, tuple)):
            return (int(window[0]), int(window[1]))
        return (int(window), 0)

    def forward(
        self,
        query: torch.Tensor,
        self_key: torch.Tensor,
        self_value: torch.Tensor,
        cross_key: torch.Tensor | None,
        cross_value: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = query.view(-1, self.num_heads, self.head_size)
        self_key = self_key.view(-1, self.num_kv_heads, self.head_size)
        self_value = self_value.view(-1, self.num_kv_heads, self.head_size)

        self_meta, _, _, _ = get_attention_context(self.self_attn.layer_name)
        cross_meta, _, _, _ = get_attention_context(self.cross_attn.layer_name)

        if self_meta is None:
            # Profiling run: vLLM only needs the output shape.
            return query.new_zeros(query.shape[0], query.shape[1] * query.shape[2])

        num_tokens = self_meta.num_actual_tokens
        query = query[:num_tokens]
        self_key = self_key[:num_tokens]
        self_value = self_value[:num_tokens]

        # Keep vLLM's engine-side KV cache current, but do not read the decoder
        # history back from the paged block table.  For this encoder-decoder
        # model, the table can expose only the current block during decode.
        unified_kv_cache_update(self_key, self_value, self.self_attn.layer_name)

        query_start_loc = self_meta.query_start_loc
        query_chunks = _split_flat_tokens(query, query_start_loc)
        key_chunks = _split_flat_tokens(self_key, query_start_loc)
        value_chunks = _split_flat_tokens(self_value, query_start_loc)
        if positions is not None:
            positions = positions[:num_tokens].to(device=query.device, dtype=torch.long)
            position_chunks = _split_flat_tokens(positions, query_start_loc)
        else:
            position_chunks = []
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(query_chunks)

        if len(self._self_key_cache) < num_seqs:
            missing = num_seqs - len(self._self_key_cache)
            empty = query.new_empty(0, self.num_kv_heads, self.head_size)
            self._self_key_cache.extend([empty] * missing)
            self._self_value_cache.extend([empty] * missing)

        self_histories: list[torch.Tensor] = []
        self_value_histories: list[torch.Tensor] = []
        q_start_pos_values: list[int] = []
        for i in range(num_seqs):
            q_len = int(query_lens[i].item())
            if position_chunks:
                pos_chunk = position_chunks[i]
                start_pos = int(pos_chunk[0].item()) if pos_chunk.numel() else 0
            else:
                seq_len = int(self_meta.seq_lens[i].item())
                start_pos = seq_len - q_len
            key_chunk = key_chunks[i]
            value_chunk = value_chunks[i]
            cached_key = self._self_key_cache[i]
            cached_value = self._self_value_cache[i]

            if start_pos <= 0 or cached_key.shape[0] != start_pos:
                new_key = key_chunk.detach().clone()
                new_value = value_chunk.detach().clone()
            else:
                new_key = torch.cat([cached_key, key_chunk.detach().clone()], dim=0)
                new_value = torch.cat(
                    [cached_value, value_chunk.detach().clone()],
                    dim=0,
                )

            self._self_key_cache[i] = new_key
            self._self_value_cache[i] = new_value
            q_start_pos_values.append(max(new_key.shape[0] - q_len, 0))
            self_histories.append(new_key)
            self_value_histories.append(new_value)

        max_query_len = int(query_lens.max().item()) if num_seqs else 0
        max_self_len = max((history.shape[0] for history in self_histories), default=0)
        q_start_pos = torch.tensor(
            q_start_pos_values,
            dtype=torch.int32,
            device=query.device,
        )
        q_padded = query.new_zeros(
            num_seqs, max_query_len, self.num_heads, self.head_size
        )
        self_kv = query.new_zeros(
            num_seqs, max_self_len, self.num_kv_heads, self.head_size
        )
        self_vv = torch.zeros_like(self_kv)
        for i in range(num_seqs):
            q_len = int(query_lens[i].item())
            self_len = self_histories[i].shape[0]
            q_padded[i, :q_len] = query_chunks[i]
            self_kv[i, :self_len] = self_histories[i]
            self_vv[i, :self_len] = self_value_histories[i]

        if cross_key is not None and cross_value is not None and cross_meta is not None:
            cross_key = cross_key.view(-1, self.num_kv_heads, self.head_size)
            cross_value = cross_value.view(-1, self.num_kv_heads, self.head_size)
            max_cross_len = int(cross_meta.max_seq_len)
            cross_kv, cross_vv = _split_and_pad_cross_kv(
                cross_key,
                cross_value,
                cross_meta.seq_lens,
                max_cross_len,
            )
            key = torch.cat([self_kv, cross_kv], dim=1)
            value = torch.cat([self_vv, cross_vv], dim=1)
            self_len = max_self_len
            # Mask: 1 for actual self tokens and actual cross tokens, 0 elsewhere.
            key_mask = torch.zeros(
                num_seqs, key.shape[1], dtype=torch.int32, device=key.device
            )
            for i, history in enumerate(self_histories):
                length = history.shape[0]
                key_mask[i, :length] = 1
            for i, length in enumerate(cross_meta.seq_lens.long().tolist()):
                key_mask[i, self_len : self_len + length] = 1
        else:
            key = self_kv
            value = self_vv
            self_len = max_self_len
            key_mask = torch.zeros(
                num_seqs, key.shape[1], dtype=torch.int32, device=key.device
            )
            for i, history in enumerate(self_histories):
                length = history.shape[0]
                key_mask[i, :length] = 1

        # The Triton kernel expects [B, H, S, D].
        q4d = q_padded.transpose(1, 2).contiguous()
        k4d = key.transpose(1, 2)
        v4d = value.transpose(1, 2)

        attn_output = flash_t5gemma2_attention(
            q4d,
            k4d,
            v4d,
            key_mask=key_mask,
            q_start_pos=q_start_pos,
            softcap=self.logits_soft_cap,
            sliding_window=self.sliding_window,
            is_causal=True,
            self_len=self_len,
            sm_scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        output = query.new_empty(num_tokens, self.num_heads * self.head_size)
        offset = 0
        for i in range(num_seqs):
            q_len = int(query_lens[i].item())
            output[offset : offset + q_len] = attn_output[i, :q_len].reshape(q_len, -1)
            offset += q_len
        return output


__all__ = ["MergedCrossAttention"]
