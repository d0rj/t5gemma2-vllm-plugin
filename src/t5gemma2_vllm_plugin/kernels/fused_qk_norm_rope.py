# SPDX-License-Identifier: Apache-2.0
"""Fused GemmaRMSNorm + RoPE for Q and K head vectors.

Replaces three separate memory passes (q_norm, k_norm, fused_rope)
with a single kernel that, for each head vector:
  1. Computes RMS = sqrt(sum(x^2)/D + eps)
  2. Applies GemmaRMSNorm: x = x * (1 + w) / RMS
  3. Applies RoPE rotation: x_out = x * cos + rotate_half(x) * sin
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_rope_kernel(
    X_ptr,
    Out_ptr,
    W_ptr,
    Cos_ptr,
    Sin_ptr,
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    eps,
    stride_x_bsh,
    stride_x_d,
    stride_cos_b,
    stride_cos_s,
    stride_cos_d,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    total = batch_size * seq_len * num_heads
    if pid >= total:
        return

    bsh_idx = pid
    b_idx = bsh_idx // (seq_len * num_heads)
    rem = bsh_idx % (seq_len * num_heads)
    s_idx = rem // num_heads
    _ = rem % num_heads  # h_idx not needed but keeps index logic readable

    x_base = X_ptr + bsh_idx * stride_x_bsh
    out_base = Out_ptr + bsh_idx * stride_x_bsh

    half_d: tl.constexpr = head_dim // 2

    rms_acc = 0.0
    for block_start in range(0, head_dim, BLOCK_D):
        cols = block_start + tl.arange(0, BLOCK_D)
        mask = cols < head_dim
        x = tl.load(x_base + cols * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        rms_acc += tl.sum(x * x, axis=0)

    rms = tl.sqrt(rms_acc / head_dim + eps)

    for block_start in range(0, head_dim, BLOCK_D):
        cols = block_start + tl.arange(0, BLOCK_D)
        mask = cols < head_dim

        x = tl.load(x_base + cols * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        normed = x / rms * (1.0 + w)

        cos_ptrs = Cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s + cols * stride_cos_d
        sin_ptrs = Sin_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s + cols * stride_cos_d
        cos_val = tl.load(cos_ptrs, mask=mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)

        is_first_half = cols < half_d
        pair_cols = tl.where(is_first_half, cols + half_d, cols - half_d)
        pair_mask = pair_cols < head_dim

        x_pair = tl.load(x_base + pair_cols * stride_x_d, mask=pair_mask & mask, other=0.0).to(tl.float32)
        w_pair = tl.load(W_ptr + pair_cols, mask=pair_mask & mask, other=0.0).to(tl.float32)
        normed_pair = x_pair / rms * (1.0 + w_pair)

        rotated = tl.where(is_first_half, -normed_pair, normed_pair)

        result = normed * cos_val + rotated * sin_val

        tl.store(out_base + cols * stride_x_d, result.to(Out_ptr.dtype.element_ty), mask=mask)


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused GemmaRMSNorm + RoPE for Q and K.

    Args:
        q: [B, S, H_q, D]
        k: [B, S, H_kv, D]
        q_norm_weight: [D]
        k_norm_weight: [D]
        cos: [B, S, D] or [B, 1, S, D]
        sin: [B, S, D] or [B, 1, S, D]
        eps: RMSNorm epsilon.

    Returns:
        (q_out, k_out) with same shapes as inputs.
    """
    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape

    if cos.dim() == 4:
        cos = cos.squeeze(1)
    if sin.dim() == 4:
        sin = sin.squeeze(1)

    q_flat = q.reshape(-1, D).contiguous()
    k_flat = k.reshape(-1, D).contiguous()
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D > 1024:
        BLOCK_D = 1024

    total_q = B * S * H_q
    total_k = B * S * H_kv

    _fused_qk_norm_rope_kernel[(total_q,)](
        q_flat, q_out, q_norm_weight.contiguous(),
        cos, sin,
        B, S, H_q, D, eps,
        q_flat.stride(0), q_flat.stride(1),
        cos.stride(0), cos.stride(1), cos.stride(2),
        BLOCK_D=BLOCK_D,
    )

    _fused_qk_norm_rope_kernel[(total_k,)](
        k_flat, k_out, k_norm_weight.contiguous(),
        cos, sin,
        B, S, H_kv, D, eps,
        k_flat.stride(0), k_flat.stride(1),
        cos.stride(0), cos.stride(1), cos.stride(2),
        BLOCK_D=BLOCK_D,
    )

    return q_out.view(B, S, H_q, D), k_out.view(B, S, H_kv, D)


__all__ = ["fused_qk_norm_rope"]
