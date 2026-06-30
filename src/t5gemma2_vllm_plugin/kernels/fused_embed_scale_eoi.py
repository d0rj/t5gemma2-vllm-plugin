# SPDX-License-Identifier: Apache-2.0
"""Fused embedding scale + EOI replacement.

Replaces two passes over (B, S, D):
  1. embeddings *= embed_scale
  2. embeddings = where(input_ids == eoi_idx, eoi_embedding, embeddings)
with a single kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_embed_scale_eoi_kernel(
    Embeddings_ptr,
    InputIds_ptr,
    EOI_Embedding_ptr,
    Out_ptr,
    embed_scale,
    eoi_token_index,
    batch_seq,
    hidden_dim,
    stride_emb_row,
    stride_emb_d,
    stride_out_row,
    stride_out_d,
    HAS_EOI: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_seq:
        return

    if HAS_EOI:
        token_id = tl.load(InputIds_ptr + row_idx)
        is_eoi = token_id == eoi_token_index
    else:
        is_eoi = False

    for block_start in range(0, hidden_dim, BLOCK_D):
        cols = block_start + tl.arange(0, BLOCK_D)
        mask = cols < hidden_dim

        emb = tl.load(
            Embeddings_ptr + row_idx * stride_emb_row + cols * stride_emb_d,
            mask=mask,
            other=0.0,
        )

        if HAS_EOI:
            eoi_val = tl.load(EOI_Embedding_ptr + cols, mask=mask, other=0.0)
            out = tl.where(is_eoi, eoi_val, emb * embed_scale)
        else:
            out = emb * embed_scale

        tl.store(
            Out_ptr + row_idx * stride_out_row + cols * stride_out_d,
            out,
            mask=mask,
        )


def fused_embed_scale_eoi(
    embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    embed_scale: float,
    eoi_token_index: int | None,
    eoi_embedding: torch.Tensor | None,
) -> torch.Tensor:
    """Scale embeddings and replace EOI tokens in one pass.

    Args:
        embeddings: [*, D] raw embedding lookup output.
        input_ids: [*] token IDs (same leading dims as embeddings).
        embed_scale: multiplicative scale factor.
        eoi_token_index: token ID for EOI, or None to skip replacement.
        eoi_embedding: [D] replacement vector for EOI tokens.

    Returns:
        Scaled (and optionally EOI-replaced) embeddings, same shape.
    """
    orig_shape = embeddings.shape
    hidden_dim = orig_shape[-1]
    flat_emb = embeddings.reshape(-1, hidden_dim).contiguous()
    flat_ids = input_ids.reshape(-1).contiguous()
    out = torch.empty_like(flat_emb)
    batch_seq = flat_emb.shape[0]

    HAS_EOI = eoi_token_index is not None and eoi_embedding is not None
    if not HAS_EOI:
        eoi_embedding = torch.empty(0, device=embeddings.device, dtype=embeddings.dtype)
        eoi_token_index = -1
    else:
        eoi_embedding = eoi_embedding.to(dtype=embeddings.dtype, device=embeddings.device).contiguous()

    BLOCK_D = min(triton.next_power_of_2(hidden_dim), 1024)

    _fused_embed_scale_eoi_kernel[(batch_seq,)](
        flat_emb,
        flat_ids,
        eoi_embedding,
        out,
        embed_scale,
        eoi_token_index,
        batch_seq,
        hidden_dim,
        flat_emb.stride(0),
        flat_emb.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_EOI=HAS_EOI,
        BLOCK_D=BLOCK_D,
    )

    return out.view(orig_shape)


__all__ = ["fused_embed_scale_eoi"]
