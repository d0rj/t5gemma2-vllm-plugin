from __future__ import annotations

import asyncio
from types import SimpleNamespace

from t5gemma2_vllm_plugin import _patch_text_encoder_renderer


class _Tokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return f"decoded:{','.join(map(str, token_ids))}"


class _Renderer:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="t5gemma2")
    )
    tokenizer = _Tokenizer()

    def _process_multimodal(self, prompt, mm_data, **kwargs):
        return prompt, mm_data, kwargs

    async def _process_multimodal_async(self, prompt, mm_data, **kwargs):
        return prompt, mm_data, kwargs


def _prompt(*, with_text: bool):
    encoder = {"prompt_token_ids": [2, 10, 11]}
    if with_text:
        encoder["prompt"] = "the original prompt"
    return {"encoder_prompt": encoder, "decoder_prompt": None}


def test_sync_renderer_routes_plain_prompt_to_text_encoder() -> None:
    from vllm.renderers.base import BaseRenderer

    _patch_text_encoder_renderer()
    result = BaseRenderer._process_enc_dec(_Renderer(), _prompt(with_text=True))

    assert result[0] == [0]
    assert result[1] == {"text": "the original prompt"}
    assert result[2]["skip_mm_cache"] is False


def test_async_renderer_decodes_token_prompt_for_text_encoder() -> None:
    from vllm.renderers.base import BaseRenderer

    _patch_text_encoder_renderer()
    result = asyncio.run(
        BaseRenderer._process_enc_dec_async(
            _Renderer(), _prompt(with_text=False), skip_mm_cache=True
        )
    )

    assert result[0] == [0]
    assert result[1] == {"text": "decoded:2,10,11"}
    assert result[2]["skip_mm_cache"] is True
