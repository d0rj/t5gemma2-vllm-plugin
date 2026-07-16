from __future__ import annotations

import torch
from torch import nn

from t5gemma2_vllm_plugin.speculators import (
    _share_unscaled_embedding_weight,
    _uses_scaled_embedding,
)


class _ScaledEmbedding(nn.Embedding):
    scalar_embed_scale = 4.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.scalar_embed_scale


def test_share_embedding_weight_preserves_draft_operation() -> None:
    target = _ScaledEmbedding(8, 4)
    draft = nn.Embedding(8, 4)
    with torch.no_grad():
        target.weight.copy_(torch.arange(32).view(8, 4))

    _share_unscaled_embedding_weight(draft, target)
    input_ids = torch.tensor([1, 3])

    assert draft.weight is target.weight
    assert _uses_scaled_embedding(target)
    assert not _uses_scaled_embedding(draft)
    torch.testing.assert_close(
        target(input_ids), draft(input_ids) * target.scalar_embed_scale
    )
