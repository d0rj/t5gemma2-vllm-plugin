"""Standalone vLLM plugin for T5Gemma 2 generation and DFlash serving."""

from __future__ import annotations

from typing import Any

from .config import T5Gemma2Config
from .processing import (
    T5Gemma2DummyInputsBuilder,
    T5Gemma2ProcessingInfo,
    T5Gemma2Processor,
)
from .vllm_adapter import T5Gemma2VllmForConditionalGeneration
from .speculators import register_speculators


def _register_hf_config(model_type: str, config_cls: type[Any]) -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.register(model_type, config_cls, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            pass


def _register_vllm_model(architecture: str, model_cls: type[Any]) -> None:
    from vllm import ModelRegistry

    try:
        ModelRegistry.register_model(architecture, model_cls)
    except (KeyError, ValueError):
        pass


def _register_processor(model_cls: type[Any]) -> None:
    from vllm.multimodal import MULTIMODAL_REGISTRY

    MULTIMODAL_REGISTRY.register_processor(
        T5Gemma2Processor,
        info=T5Gemma2ProcessingInfo,
        dummy_inputs=T5Gemma2DummyInputsBuilder,
    )(model_cls)


def _patch_text_encoder_renderer() -> None:
    """Feed ordinary API prompts to T5Gemma's text encoder.

    vLLM currently treats a plain prompt for encoder-decoder models as a
    decoder prompt unless the encoder is represented as a multimodal input.
    T5Gemma2's text encoder is exposed through a synthetic ``text`` modality,
    so adapt only this model type at the renderer boundary.
    """
    from vllm.renderers.base import BaseRenderer

    original_sync = BaseRenderer._process_enc_dec
    if getattr(original_sync, "_t5gemma2_text_encoder_patch", False):
        return
    original_async = BaseRenderer._process_enc_dec_async

    def is_t5gemma2(renderer: Any) -> bool:
        config = renderer.model_config.hf_config
        return getattr(config, "model_type", None) == "t5gemma2"

    def encoder_text(renderer: Any, prompt: dict[str, Any]) -> str:
        encoder_prompt = prompt["encoder_prompt"]
        text = encoder_prompt.get("prompt")
        if text is not None:
            return text
        return renderer.tokenizer.decode(
            encoder_prompt["prompt_token_ids"],
            skip_special_tokens=False,
        )

    def patched_sync(self, prompt, *, skip_mm_cache=False):
        if not is_t5gemma2(self) or prompt["decoder_prompt"] is not None:
            return original_sync(self, prompt, skip_mm_cache=skip_mm_cache)
        return self._process_multimodal(
            [0],
            {"text": encoder_text(self, prompt)},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
            skip_mm_cache=skip_mm_cache,
        )

    async def patched_async(self, prompt, *, skip_mm_cache=False):
        if not is_t5gemma2(self) or prompt["decoder_prompt"] is not None:
            return await original_async(self, prompt, skip_mm_cache=skip_mm_cache)
        return await self._process_multimodal_async(
            [0],
            {"text": encoder_text(self, prompt)},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
            skip_mm_cache=skip_mm_cache,
        )

    patched_sync._t5gemma2_text_encoder_patch = True  # type: ignore[attr-defined]
    patched_async._t5gemma2_text_encoder_patch = True  # type: ignore[attr-defined]
    BaseRenderer._process_enc_dec = patched_sync
    BaseRenderer._process_enc_dec_async = patched_async


def register() -> None:
    """Register T5Gemma 2 generation support with vLLM.

    vLLM calls this function through the ``vllm.general_plugins`` entry point.
    It is intentionally idempotent because vLLM may load general plugins in
    multiple processes.
    """

    _register_hf_config("t5gemma2", T5Gemma2Config)
    _register_vllm_model(
        "T5Gemma2ForConditionalGeneration",
        T5Gemma2VllmForConditionalGeneration,
    )
    _register_processor(T5Gemma2VllmForConditionalGeneration)
    _patch_text_encoder_renderer()
    register_speculators()


__all__ = ["T5Gemma2VllmForConditionalGeneration", "register"]
