"""vLLM text processor for T5Gemma 2 encoder-decoder serving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.utils.collection_utils import is_list_of

from .config import T5Gemma2Config, get_t5gemma2_text_config


class T5Gemma2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> T5Gemma2Config:
        return self.ctx.get_hf_config(T5Gemma2Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # vLLM represents encoder inputs through its single-modality budget,
        # even for a text-only encoder-decoder model.
        return {"text": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del mm_counts
        encoder = get_t5gemma2_text_config(self.get_hf_config(), is_encoder=True)
        return {"text": min(seq_len, encoder.max_position_embeddings)}

    def get_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()


class TextProcessorItems(ProcessorBatchItems[str]):
    def __init__(self, data: str | list[str] | None) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class TextDataParser(MultiModalDataParser):
    def _parse_text_data(
        self,
        data: ModalityData[str],
    ) -> ModalityDataItems | None:
        if data is None or not len(data):
            return TextProcessorItems(None)
        if isinstance(data, str) or is_list_of(data, str):
            return TextProcessorItems(data)
        raise TypeError(f"Text data must be str or list[str], got {type(data)}")

    def _get_subparsers(self) -> Mapping[str, object]:
        return {"text": self._parse_text_data}


class T5Gemma2DummyInputsBuilder(BaseDummyInputsBuilder[T5Gemma2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        del mm_counts
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> dict:
        del mm_options
        if mm_counts.get("text", 0) == 0:
            return {}
        return {"text": " ".join(["word"] * seq_len)}


class T5Gemma2Processor(EncDecMultiModalProcessor[T5Gemma2ProcessingInfo]):
    """Use the request prompt only as encoder input; decoder starts at BOS."""

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        if mm_items.get_count("text", strict=False):
            # Dummy profiling supplies the encoder text as a multimodal item.
            # Keep one sentinel token for PromptReplacement below.
            return [0]
        return prompt

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        del prompt, mm_items
        config = self.info.get_hf_config()
        start_id = getattr(config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = config.bos_token_id
        return [int(start_id)]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs
        tokenizer = self.info.get_tokenizer()
        if isinstance(prompt, (list, tuple)):
            input_ids = torch.tensor([prompt], dtype=torch.long)
        else:
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                **tok_kwargs,
            )["input_ids"]
        result = {"input_ids": input_ids}
        if "texts" in mm_data:
            texts = mm_data["texts"]
            text = texts[0] if texts else ""
            result["encoder_input_ids"] = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"]
        return BatchFeature(result)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_inputs, hf_processor_mm_kwargs
        return {
            "encoder_input_ids": MultiModalFieldConfig.batched("text"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs, out_mm_kwargs
        if not mm_items.get_count("text", strict=False):
            return []

        text_items = mm_items.get_items("text", TextProcessorItems)
        tokenizer = self.info.get_tokenizer()
        num_tokens = len(
            tokenizer.encode(text_items.get(0), add_special_tokens=True)
        )
        return [
            PromptReplacement(
                modality="text",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


__all__ = [
    "T5Gemma2DummyInputsBuilder",
    "T5Gemma2ProcessingInfo",
    "T5Gemma2Processor",
]
