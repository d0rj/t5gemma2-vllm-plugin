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


__all__ = ["T5Gemma2VllmForConditionalGeneration", "register"]
