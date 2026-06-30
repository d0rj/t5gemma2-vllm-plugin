# SPDX-License-Identifier: Apache-2.0
"""Fused flash attention for T5Gemma2.

Supports softcapping, sliding window, GQA, packed-sequence segments,
and merged self+cross attention (decoder) -- all without materialising
dense O(S^2) masks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _flash_t5gemma2_fwd_kernel(
    Q, K, V, Out,
    Key_Mask,
    Segment_Ids,
    Q_Start_Pos,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_km_b,
    stride_seg_b,
    stride_qsp_b,
    seq_len_q,
    seq_len_k,
    self_len,
    softcap,
    sliding_window,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SLIDING_WINDOW: tl.constexpr,
    HAS_CROSS: tl.constexpr,
    HAS_KEY_MASK: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr,
    HAS_Q_START_POS: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hq = tl.program_id(2)
    pid_hkv = pid_hq // GQA_GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_hq * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < seq_len_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if HAS_Q_START_POS:
        q_start_pos = tl.load(Q_Start_Pos + pid_b * stride_qsp_b)
    else:
        q_start_pos = 0

    if IS_CAUSAL and (not HAS_CROSS):
        kv_end = tl.minimum(q_start_pos + (pid_m + 1) * BLOCK_M, seq_len_k)
    else:
        kv_end = seq_len_k

    if HAS_SEGMENTS:
        q_seg = tl.load(
            Segment_Ids + pid_b * stride_seg_b + offs_m,
            mask=offs_m < seq_len_q,
            other=-1,
        )

    num_blocks_n = tl.cdiv(kv_end, BLOCK_N)
    for block_n_idx in range(num_blocks_n):
        start_n = block_n_idx * BLOCK_N
        offs_n_curr = start_n + offs_n
        kv_valid = offs_n_curr < seq_len_k

        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_hkv * stride_kh
            + offs_n_curr[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_hkv * stride_vh
            + offs_n_curr[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)

        qk = tl.dot(q.to(tl.float32), k.to(tl.float32), allow_tf32=False)
        qk = qk * sm_scale

        if HAS_SOFTCAP:
            qk = tl.extra.cuda.libdevice.tanh(qk / softcap) * softcap

        q_pos = q_start_pos + offs_m[:, None]
        k_pos = offs_n_curr[None, :]

        valid = (offs_m[:, None] < seq_len_q) & (k_pos < seq_len_k)

        if HAS_KEY_MASK:
            km = tl.load(
                Key_Mask + pid_b * stride_km_b + offs_n_curr,
                mask=kv_valid,
                other=0,
            )
            valid = valid & (km[None, :] != 0)

        if HAS_SEGMENTS:
            k_seg = tl.load(
                Segment_Ids + pid_b * stride_seg_b + offs_n_curr,
                mask=kv_valid,
                other=-2,
            )
            seg_match = q_seg[:, None] == k_seg[None, :]
            if HAS_CROSS:
                seg_match = seg_match | (k_pos >= self_len)
            valid = valid & seg_match

        if IS_CAUSAL:
            if HAS_CROSS:
                is_self = k_pos < self_len
                causal_ok = (~is_self) | (k_pos <= q_pos)
                valid = valid & causal_ok
            else:
                valid = valid & (k_pos <= q_pos)

        if HAS_SLIDING_WINDOW:
            if IS_CAUSAL:
                if HAS_CROSS:
                    is_self_sw = k_pos < self_len
                    window_ok = (~is_self_sw) | ((q_pos - k_pos) < sliding_window)
                else:
                    window_ok = (q_pos - k_pos) < sliding_window
            else:
                dist = q_pos - k_pos
                left_win = (sliding_window + 1) // 2
                right_win = (sliding_window // 2) + 1
                window_ok = ((dist >= 0) & (dist < left_win)) | (
                    (dist < 0) & ((-dist) < right_win)
                )
            valid = valid & window_ok

        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p, v.to(tl.float32), allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_hq * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    out_mask = offs_m[:, None] < seq_len_q
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    ],
    key=["BLOCK_DMODEL", "IS_CAUSAL", "HAS_SOFTCAP", "HAS_CROSS"],
)
@triton.jit
def _flash_t5gemma2_fwd_kernel_autotuned(
    Q, K, V, Out,
    Key_Mask,
    Segment_Ids,
    Q_Start_Pos,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_km_b,
    stride_seg_b,
    stride_qsp_b,
    seq_len_q,
    seq_len_k,
    self_len,
    softcap,
    sliding_window,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_SLIDING_WINDOW: tl.constexpr,
    HAS_CROSS: tl.constexpr,
    HAS_KEY_MASK: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr,
    HAS_Q_START_POS: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    _flash_t5gemma2_fwd_kernel(
        Q, K, V, Out,
        Key_Mask,
        Segment_Ids,
        Q_Start_Pos,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_km_b,
        stride_seg_b,
        stride_qsp_b,
        seq_len_q,
        seq_len_k,
        self_len,
        softcap,
        sliding_window,
        sm_scale,
        IS_CAUSAL,
        HAS_SOFTCAP,
        HAS_SLIDING_WINDOW,
        HAS_CROSS,
        HAS_KEY_MASK,
        HAS_SEGMENTS,
        HAS_Q_START_POS,
        GQA_GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL,
    )


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def flash_t5gemma2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_mask: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    q_start_pos: torch.Tensor | None = None,
    softcap: float = 0.0,
    sliding_window: int = 0,
    is_causal: bool = False,
    self_len: int = 0,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """Flash attention for T5Gemma2 with softcapping, sliding window, GQA.

    Args:
        q: [B, H_q, S_q, D]
        k: [B, H_kv, S_k, D]
        v: [B, H_kv, S_k, D]
        key_mask: [B, S_k] int or bool, 0/False = masked.
        segment_ids: [B, S_q] int, packed-sequence segment IDs.
        q_start_pos: [B] int.  Offset added to every query position before
            applying causal / sliding-window masks.  Use this when ``q`` only
            contains the new tokens (e.g. vLLM decode) while ``k`` contains the
            full cached sequence.  ``q_start_pos = seq_len - q_len`` per batch
            item.  Default (None) is equivalent to 0.
        softcap: tanh(logits/cap)*cap; 0 disables.
        sliding_window: window size; 0 means full attention.
        is_causal: enable causal masking.
        self_len: boundary for merged self+cross (decoder).
            K[:, :, :self_len] = self-keys, K[:, :, self_len:] = cross-keys.
            0 means no cross-attention.
        sm_scale: softmax scale (typically head_dim ** -0.5).

    Returns:
        out: [B, H_q, S_q, D]
    """
    batch_size, num_heads_q, seq_len_q, head_dim = q.shape
    _, num_heads_kv, seq_len_k, _ = k.shape
    assert num_heads_q % num_heads_kv == 0
    gqa_group = num_heads_q // num_heads_kv

    out = torch.empty_like(q)

    HAS_SOFTCAP = softcap != 0.0
    HAS_SLIDING_WINDOW = sliding_window > 0
    HAS_CROSS = self_len > 0
    HAS_KEY_MASK = key_mask is not None
    HAS_SEGMENTS = segment_ids is not None
    HAS_Q_START_POS = q_start_pos is not None

    if not HAS_KEY_MASK:
        key_mask = torch.empty(0, device=q.device, dtype=torch.int32)
        stride_km_b = 0
    else:
        key_mask = key_mask.to(torch.int32).contiguous()
        stride_km_b = key_mask.stride(0)

    if not HAS_SEGMENTS:
        segment_ids = torch.empty(0, device=q.device, dtype=torch.int32)
        stride_seg_b = 0
    else:
        segment_ids = segment_ids.to(torch.int32).contiguous()
        stride_seg_b = segment_ids.stride(0)

    if not HAS_Q_START_POS:
        q_start_pos = torch.empty(0, device=q.device, dtype=torch.int32)
        stride_qsp_b = 0
    else:
        q_start_pos = q_start_pos.to(torch.int32).contiguous()
        stride_qsp_b = q_start_pos.stride(0)

    def grid(META):
        return (
            triton.cdiv(seq_len_q, META["BLOCK_M"]),
            batch_size,
            num_heads_q,
        )

    _flash_t5gemma2_fwd_kernel_autotuned[grid](
        q, k, v, out,
        key_mask,
        segment_ids,
        q_start_pos,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_km_b,
        stride_seg_b,
        stride_qsp_b,
        seq_len_q,
        seq_len_k,
        self_len,
        softcap,
        sliding_window,
        sm_scale,
        IS_CAUSAL=is_causal,
        HAS_SOFTCAP=HAS_SOFTCAP,
        HAS_SLIDING_WINDOW=HAS_SLIDING_WINDOW,
        HAS_CROSS=HAS_CROSS,
        HAS_KEY_MASK=HAS_KEY_MASK,
        HAS_SEGMENTS=HAS_SEGMENTS,
        HAS_Q_START_POS=HAS_Q_START_POS,
        GQA_GROUP_SIZE=gqa_group,
        BLOCK_DMODEL=head_dim,
    )

    return out


__all__ = ["flash_t5gemma2_attention"]
