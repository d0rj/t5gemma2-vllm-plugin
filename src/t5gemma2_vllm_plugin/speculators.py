"""vLLM registration shims for T5Gemma2 DFlash-family speculators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def _copy_dflash_fields(
    *,
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
    architecture: str,
    draft_arch: str,
) -> None:
    pre_trained_config["architectures"] = [architecture]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        # vLLM's DFlash adapter stores target layer ids one lower than the
        # speculators config ids used by gpu_model_runner hidden-state capture.
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
        "draft_arch": draft_arch,
    }

    for key in (
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ):
        if key in config_dict:
            pre_trained_config[key] = config_dict[key]


def _register_speculators_config_shims() -> None:
    from vllm.transformers_utils.configs.speculators import base as spec_base
    from vllm.transformers_utils.configs.speculators.algos import (
        SUPPORTED_SPECULATORS_TYPES,
    )

    def update_dspark(config_dict: dict[str, Any], pre_trained_config: dict[str, Any]) -> None:
        _copy_dflash_fields(
            config_dict=config_dict,
            pre_trained_config=pre_trained_config,
            architecture="DSparkDraftModel",
            draft_arch="dspark",
        )

    def update_dflare(config_dict: dict[str, Any], pre_trained_config: dict[str, Any]) -> None:
        _copy_dflash_fields(
            config_dict=config_dict,
            pre_trained_config=pre_trained_config,
            architecture="DFlareDraftModel",
            draft_arch="dflare",
        )

    SUPPORTED_SPECULATORS_TYPES.setdefault("dspark", update_dspark)
    SUPPORTED_SPECULATORS_TYPES.setdefault("dflare", update_dflare)

    original = spec_base.SpeculatorsConfig.build_vllm_speculative_config
    if getattr(original, "_t5gemma2_dflash_family_patch", False):
        return

    def patched_build_vllm_speculative_config(
        cls,
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        result = original.__func__(cls, config_dict)
        if result.get("method") in {"dspark", "dflare"}:
            result["method"] = "dflash"
        return result

    patched_build_vllm_speculative_config._t5gemma2_dflash_family_patch = True  # type: ignore[attr-defined]
    spec_base.SpeculatorsConfig.build_vllm_speculative_config = classmethod(
        patched_build_vllm_speculative_config
    )


class _MarkovHead(nn.Module):
    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # vLLM runtime samples one position at a time. Keep the parameter
            # shape load-compatible; recurrent stateful acceleration can be
            # added later if rnn heads are needed in serving.
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def bias_for_step(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        prev_emb = self.prev_embeddings(prev_token_ids).to(self.markov_w2.weight.dtype)
        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            prev_emb = gate * prev_emb
        elif self.head_type == "rnn":
            # Serve rnn heads with the vanilla bias as a safe compatibility
            # fallback until a stateful DSpark sampler is wired in.
            pass
        return self.markov_w2(prev_emb)


class _ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def _build_dspark_model_class() -> type[nn.Module]:
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    class DSparkDraftModel(DFlashQwen3ForCausalLM):
        def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            markov_rank = int(getattr(self.config, "markov_rank", 256) or 0)
            head_type = getattr(self.config, "markov_head_type", "vanilla")
            self.markov_head: _MarkovHead | None = None
            if markov_rank > 0:
                self.markov_head = _MarkovHead(
                    verifier_vocab_size=self.config.vocab_size,
                    draft_vocab_size=self.config.draft_vocab_size,
                    markov_rank=markov_rank,
                    hidden_size=self.config.hidden_size,
                    head_type=head_type,
                )

            self.confidence_head: _ConfidenceHead | None = None
            if getattr(self.config, "enable_confidence_head", True):
                with_markov = getattr(self.config, "confidence_head_with_markov", True)
                input_dim = self.config.hidden_size + (markov_rank if with_markov else 0)
                self.confidence_head = _ConfidenceHead(input_dim)

        def compute_logits_with_prev_tokens(
            self,
            hidden_states: torch.Tensor,
            prev_token_ids: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.logits_processor(self.lm_head, hidden_states)
            if self.markov_head is not None:
                logits = logits + self.markov_head.bias_for_step(
                    prev_token_ids=prev_token_ids,
                    hidden_states=hidden_states,
                )
            if self.draft_id_to_target_id is None:
                return logits

            base = torch.arange(self.config.draft_vocab_size, device=logits.device)
            targets = base + self.draft_id_to_target_id
            target_logits = logits.new_full(
                (logits.shape[0], self.config.vocab_size),
                float("-inf"),
            )
            target_logits[:, targets] = logits
            return target_logits

        def load_weights(self, weights: Any):
            dspark_weights = []
            dflash_weights = []
            for name, weight in weights:
                if name.startswith(("markov_head.", "confidence_head.")):
                    dspark_weights.append((name, weight))
                else:
                    dflash_weights.append((name, weight))
            super().load_weights(dflash_weights)
            if dspark_weights:
                params = dict(self.named_parameters())
                for name, weight in dspark_weights:
                    if name in params:
                        params[name].data.copy_(weight.to(params[name].device))

    return DSparkDraftModel


def _patch_init_speculator() -> None:
    import vllm.v1.worker.gpu.spec_decode as spec_decode_init

    original: Callable = spec_decode_init.init_speculator
    if getattr(original, "_t5gemma2_dflash_family_patch", False):
        return

    def patched_init_speculator(vllm_config, device):
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.method == "dflash":
            hf_config = speculative_config.draft_model_config.hf_config
            dflash_config = getattr(hf_config, "dflash_config", None) or {}
            if dflash_config.get("draft_arch") == "dspark":
                from t5gemma2_vllm_plugin.vllm_dspark import DSparkSpeculator

                return DSparkSpeculator(vllm_config, device)
            if dflash_config.get("draft_arch") == "dflare":
                from t5gemma2_vllm_plugin.vllm_dflare import DFlareSpeculator

                return DFlareSpeculator(vllm_config, device)
        return original(vllm_config, device)

    patched_init_speculator._t5gemma2_dflash_family_patch = True  # type: ignore[attr-defined]
    spec_decode_init.init_speculator = patched_init_speculator


def _patch_current_dflash_proposer() -> None:
    """Patch the vLLM >=0.23 DFlash proposer factory used by GPUModelRunner."""
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    from vllm.v1.spec_decode.dflash import DFlashProposer

    current = gpu_model_runner.DFlashProposer
    if getattr(current, "_t5gemma2_dflash_family_patch", False):
        return

    class T5Gemma2DFlashFamilyProposer(DFlashProposer):
        def __init__(self, vllm_config, device, runner=None):
            super().__init__(vllm_config, device, runner)
            hf_config = self.draft_model_config.hf_config
            dflash_config = getattr(hf_config, "dflash_config", None) or {}
            self._draft_arch = dflash_config.get("draft_arch", "llama")
            self._dspark_prev_token_ids: torch.Tensor | None = None

            if self._draft_arch == "dflare":
                num_target_layers = len(dflash_config.get("target_layer_ids") or [])
                target_hidden_size = (
                    getattr(hf_config, "target_hidden_size", None) or self.hidden_size
                )
                self.hidden_size = num_target_layers * target_hidden_size
                self.hidden_states = torch.zeros(
                    self.max_num_tokens,
                    self.hidden_size,
                    dtype=self.dtype,
                    device=device,
                )

        def set_inputs_first_pass(
            self,
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        ):
            if self._draft_arch == "dspark":
                self._dspark_prev_token_ids = next_token_ids
            return super().set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                cad,
                num_rejected_tokens_gpu,
            )

        def _sample_draft_tokens(self, hidden_states, sampling_metadata):
            if self._draft_arch != "dspark":
                return super()._sample_draft_tokens(hidden_states, sampling_metadata)
            if self._dspark_prev_token_ids is None:
                raise RuntimeError("DSpark anchor token ids were not initialized")

            batch_size = self._dspark_prev_token_ids.shape[0]
            steps = self.num_speculative_tokens
            hidden_steps = hidden_states.view(batch_size, steps, -1)
            previous = self._dspark_prev_token_ids
            sampled_steps = []
            probability_steps = []
            compute = getattr(self.model, "compute_logits_with_prev_tokens")
            for step in range(steps):
                logits = compute(hidden_steps[:, step], previous)
                if (
                    not self._enable_probabilistic_draft_probs
                    or sampling_metadata.all_greedy
                ):
                    sampled = logits.argmax(dim=-1)
                    probabilities = None
                else:
                    sampled, probabilities = self._sample_from_logits(
                        logits, sampling_metadata
                    )
                sampled_steps.append(sampled)
                if probabilities is not None:
                    probability_steps.append(probabilities)
                previous = sampled

            tokens = torch.stack(sampled_steps, dim=1).reshape(-1)
            if probability_steps:
                probabilities = torch.stack(probability_steps, dim=1)
                probabilities = probabilities.reshape(-1, probabilities.shape[-1])
            else:
                probabilities = None
            return tokens, probabilities

    T5Gemma2DFlashFamilyProposer._t5gemma2_dflash_family_patch = True
    gpu_model_runner.DFlashProposer = T5Gemma2DFlashFamilyProposer


def register_speculators() -> None:
    from vllm import ModelRegistry

    _register_speculators_config_shims()
    _patch_init_speculator()
    _patch_current_dflash_proposer()

    try:
        ModelRegistry.register_model("DSparkDraftModel", _build_dspark_model_class())
    except (KeyError, ValueError):
        pass

    try:
        from t5gemma2_vllm_plugin.vllm_dflare import DFlareDraftModel

        ModelRegistry.register_model("DFlareDraftModel", DFlareDraftModel)
    except (KeyError, ValueError):
        pass
