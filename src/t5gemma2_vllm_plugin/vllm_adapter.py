"""Native vLLM serving adapter for T5Gemma 2 + DFlash."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

from .config import T5Gemma2DecoderConfig, get_t5gemma2_text_config
from .merged_cross_attention import MergedCrossAttention
from .t5gemma2_encoder import (
    T5Gemma2Encoder,
    T5Gemma2MLP,
    T5Gemma2RotaryEmbedding,
    T5Gemma2TextScaledWordEmbedding,
    apply_rotary_pos_emb,
)
from .t5gemma2_model import T5Gemma2Decoder


def _tp_heads(total: int) -> int:
    world = get_tensor_model_parallel_world_size()
    return total // world if total >= world else 1


class T5Gemma2VllmMergedAttention(nn.Module):
    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
        *,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_heads = _tp_heads(config.num_attention_heads)
        self.num_kv_heads = _tp_heads(config.num_key_value_heads)
        quant = vllm_config.quant_config
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.attn = MergedCrossAttention(
            self.num_heads,
            self.head_dim,
            float(config.query_pre_attn_scalar) ** -0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant,
            logits_soft_cap=config.attn_logit_softcapping,
            sliding_window=window,
            prefix=f"{prefix}.merged_attn",
        )

    @staticmethod
    def _apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [T,H,D] -> [1,H,T,D], matching the parity implementation.
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        q4, k4 = apply_rotary_pos_emb(q4, k4, *position_embeddings)
        return q4.squeeze(0).transpose(0, 1), k4.squeeze(0).transpose(0, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        self_k, _ = self.k_proj(hidden_states)
        self_v, _ = self.v_proj(hidden_states)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        self_k = self.k_norm(self_k.view(-1, self.num_kv_heads, self.head_dim))
        self_v = self_v.view(-1, self.num_kv_heads, self.head_dim)
        q, self_k = self._apply_rope(q, self_k, position_embeddings)

        cross_k = cross_v = None
        if encoder_hidden_states is not None:
            cross_k, _ = self.k_proj(encoder_hidden_states)
            cross_v, _ = self.v_proj(encoder_hidden_states)
            cross_k = self.k_norm(cross_k.view(-1, self.num_kv_heads, self.head_dim))
            cross_v = cross_v.view(-1, self.num_kv_heads, self.head_dim)

        output = self.attn(q, self_k, self_v, cross_k, cross_v, positions=positions)
        output, _ = self.o_proj(output)
        return output


class T5Gemma2VllmDecoderLayer(nn.Module):
    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
        *,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = T5Gemma2VllmMergedAttention(
            config,
            layer_idx=layer_idx,
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.pre_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = T5Gemma2MLP(
            config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            encoder_hidden_states,
            position_embeddings,
            positions,
        )
        hidden_states = residual + self.post_self_attn_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_feedforward_layernorm(hidden_states)


class T5Gemma2VllmDecoder(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        shared_embed_tokens: T5Gemma2TextScaledWordEmbedding | None = None,
    ) -> None:
        super().__init__()
        outer = vllm_config.model_config.hf_config
        config = get_t5gemma2_text_config(outer, is_encoder=False)
        self.config = config
        if shared_embed_tokens is not None:
            self.embed_tokens = shared_embed_tokens
        else:
            self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                embed_scale=float(torch.tensor(config.hidden_size**0.5, dtype=torch.bfloat16)),
                eoi_token_index=getattr(outer, "eoi_token_index", None),
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: T5Gemma2VllmDecoderLayer(
                config,
                layer_idx=int(prefix.rsplit(".", 1)[-1]),
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        position_ids = positions.unsqueeze(0)
        position_embeddings = {
            kind: self.rotary_emb(hidden_states, position_ids, layer_type=kind)
            for kind in set(self.config.layer_types)
        }
        aux = self._maybe_add_hidden_state([], 0, hidden_states, None)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                position_embeddings[self.config.layer_types[idx]],
                positions,
            )
            self._maybe_add_hidden_state(aux, idx + 1, hidden_states, None)
        hidden_states = self.norm(hidden_states)
        return (hidden_states, aux) if aux else hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return T5Gemma2Decoder.load_weights(self, weights)


class T5Gemma2VllmForConditionalGeneration(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsEagle, SupportsEagle3
):
    """Serving model whose emitted auxiliary states contain decoder tokens only."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.encoder = T5Gemma2Encoder(vllm_config, prefix=maybe_prefix(prefix, "model.encoder"))
        # T5Gemma2 ties encoder/decoder input embeddings; share the object so
        # weight loading and updates affect both sides consistently.
        self.model = T5Gemma2VllmDecoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model.decoder"),
            shared_embed_tokens=self.encoder.embed_tokens,
        )
        decoder = get_t5gemma2_text_config(self.config, is_encoder=False)
        self.logits_processor = LogitsProcessor(
            decoder.vocab_size,
            soft_cap=decoder.final_logit_softcapping,
        )
        self._encoder_outputs_cache: torch.Tensor | None = None

    def get_language_model(self) -> nn.Module:
        # Eagle/DFlash expects get_language_model().model to be an
        # EagleModelMixin. Returning the wrapper also keeps multimodal token
        # embedding functional through this class' embed_input_ids().
        return self

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        encoder_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))
        if encoder_ids is None:
            return []
        items = encoder_ids if isinstance(encoder_ids, list) else encoder_ids.unbind(0)
        outputs = []
        for ids in items:
            ids = ids.flatten()
            positions = torch.arange(ids.numel(), device=ids.device).unsqueeze(0)
            output = self.encoder(
                input_ids=ids.unsqueeze(0),
                attention_mask=torch.ones_like(ids).unsqueeze(0),
                position_ids=positions,
            )
            outputs.append(output.squeeze(0))
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        del intermediate_tensors
        if isinstance(encoder_outputs, list):
            encoder_outputs = torch.cat(encoder_outputs, dim=0)
        if encoder_outputs is not None:
            self._encoder_outputs_cache = encoder_outputs
        else:
            encoder_outputs = self._encoder_outputs_cache
        return self.model(input_ids, positions, inputs_embeds, encoder_outputs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.encoder.embed_tokens, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        encoder_weights = []
        decoder_weights = []
        for name, weight in weights:
            name = name.removeprefix("model.")
            if name.startswith("encoder."):
                encoder_weights.append((name.removeprefix("encoder."), weight))
            elif name.startswith("decoder."):
                decoder_weights.append((name.removeprefix("decoder."), weight))
        self.encoder.load_weights(encoder_weights)
        loaded_params = self.model.load_weights(decoder_weights)

        # T5Gemma2 ties the input/output embeddings between encoder and decoder.
        # The checkpoint only stores encoder.embed_tokens.*, so copy them to the
        # decoder when the decoder did not load its own embedding weights.
        if "embed_tokens.weight" not in loaded_params:
            self.model.embed_tokens.weight.data.copy_(
                self.encoder.embed_tokens.weight.data
            )
        if hasattr(self.model.embed_tokens, "eoi_embedding") and \
           hasattr(self.encoder.embed_tokens, "eoi_embedding") and \
           "embed_tokens.eoi_embedding" not in loaded_params:
            self.model.embed_tokens.eoi_embedding.data.copy_(
                self.encoder.embed_tokens.eoi_embedding.data
            )


__all__ = ["T5Gemma2VllmForConditionalGeneration"]
