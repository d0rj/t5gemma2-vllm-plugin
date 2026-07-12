from __future__ import annotations

import pytest
import torch

from t5gemma2_vllm_plugin.kernels.flash_t5gemma2_attention import (
    flash_t5gemma2_attention,
    flash_t5gemma2_cached_single_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("position_values", [list(range(6)), [5]])
def test_cached_single_attention_matches_padded_reference(
    position_values: list[int],
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_query_heads = 4
    num_kv_heads = 1
    head_dim = 256
    capacity = 32
    cross_tokens = 7

    self_key = torch.randn(
        capacity, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    self_value = torch.randn_like(self_key)
    cross_key = torch.randn_like(self_key)
    cross_value = torch.randn_like(self_key)
    positions = torch.tensor(position_values, device=device, dtype=torch.long)
    query = torch.randn(
        len(position_values),
        num_query_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    self_tokens = max(position_values) + 1
    scale = head_dim**-0.5

    reference = flash_t5gemma2_attention(
        query.unsqueeze(0).transpose(1, 2),
        torch.cat([self_key[:self_tokens], cross_key[:cross_tokens]])
        .unsqueeze(0)
        .transpose(1, 2),
        torch.cat([self_value[:self_tokens], cross_value[:cross_tokens]])
        .unsqueeze(0)
        .transpose(1, 2),
        key_mask=torch.ones(
            1,
            self_tokens + cross_tokens,
            device=device,
            dtype=torch.int32,
        ),
        q_start_pos=torch.tensor(
            [position_values[0]], device=device, dtype=torch.int32
        ),
        is_causal=True,
        self_len=self_tokens,
        sm_scale=scale,
    ).transpose(1, 2).squeeze(0)

    actual = flash_t5gemma2_cached_single_attention(
        query,
        self_key,
        self_value,
        cross_key,
        cross_value,
        positions,
        torch.tensor([cross_tokens], device=device, dtype=torch.int32),
        sm_scale=scale,
    )

    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)
