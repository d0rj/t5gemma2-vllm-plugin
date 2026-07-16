from __future__ import annotations

import torch
from torch import nn

from t5gemma2_vllm_plugin.vllm_dflare import DFlareDraftModel


class _FakeDFlareInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.loaded_names: list[str] = []

    def load_weights(self, weights) -> set[str]:
        items = list(weights)
        self.loaded_names.extend(name for name, _ in items)
        return set(self.loaded_names)


def test_dflare_loader_marks_own_embedding_and_lm_head() -> None:
    # Construct only the small part of the object needed by load_weights. A
    # full DFlare model requires a distributed vLLM device configuration.
    model = DFlareDraftModel.__new__(DFlareDraftModel)
    nn.Module.__init__(model)
    model.model = _FakeDFlareInnerModel()
    model.lm_head = nn.Linear(3, 2, bias=False)
    model.has_own_embed_tokens = False
    model.has_own_lm_head = False

    model.load_weights(
        [
            ("embed_tokens.weight", torch.ones(4, 3)),
            ("lm_head.weight", torch.ones(2, 3)),
        ]
    )

    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
    assert model.model.loaded_names == ["embed_tokens.weight"]
    torch.testing.assert_close(model.lm_head.weight, torch.ones(2, 3))
