"""DFlare vLLM model/runtime shims for DFlash-family speculative decoding."""

from __future__ import annotations

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal.inputs import NestedTensors
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class DFlareQwen3Model(DFlashQwen3Model):
    """DFlash query model with DFlare per-layer context fusion."""

    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        if hasattr(self, "fc"):
            del self.fc
        self.use_aux_hidden_state = True
        dflash_config = getattr(self.config, "dflash_config", {}) or {}
        self.num_target_layers = len(dflash_config.get("target_layer_ids") or [])
        if self.num_target_layers <= 0:
            raise ValueError("DFlare requires dflash_config.target_layer_ids")

        self.layer_fusion_weights = nn.Parameter(
            torch.empty(self.config.num_hidden_layers, self.num_target_layers)
        )
        nn.init.constant_(self.layer_fusion_weights, 0.0)
        for draft_idx in range(self.config.num_hidden_layers):
            target_idx = min(
                self.num_target_layers - 1,
                int((draft_idx / max(self.config.num_hidden_layers, 1)) * self.num_target_layers),
            )
            self.layer_fusion_weights.data[draft_idx, target_idx] = 2.0

        for layer in self.layers:
            attn = layer.self_attn
            bias = attn.qkv_proj.bias is not None
            attn.k_proj_target = nn.Linear(
                self.config.hidden_size,
                attn.kv_size,
                bias=bias,
            )
            attn.v_proj_target = nn.Linear(
                self.config.hidden_size,
                attn.kv_size,
                bias=bias,
            )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _fuse_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        num_ctx, width = context_states.shape
        hidden_size = self.config.hidden_size
        expected = self.num_target_layers * hidden_size
        if width != expected:
            raise ValueError(
                f"DFlare expected concatenated context width {expected}, got {width}"
            )
        target_states = context_states.view(num_ctx, self.num_target_layers, hidden_size)
        fusion_probs = torch.softmax(self.layer_fusion_weights, dim=1).to(
            target_states.dtype
        )
        return torch.einsum("nth,lt->nlh", target_states, fusion_probs)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        fused_context = self._fuse_context_states(context_states)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            layer_context = fused_context[:, layer_idx, :]
            normed_context = torch.empty_like(layer_context)
            ops.rms_norm(
                normed_context,
                layer_context,
                self.hidden_norm.weight.data,
                attn.q_norm.variance_epsilon,
            )
            k = attn.k_proj_target(normed_context)
            v = attn.v_proj_target(normed_context)
            k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim))
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)

            k_flat = k.reshape(k.shape[0], -1)
            # Rotate K in-place. The first tensor argument is unused by the
            # cache update path; passing K for both outputs keeps this compatible
            # with vLLM's RoPE wrapper.
            _, k_flat = attn.rotary_emb(context_positions, k_flat, k_flat)
            k = k_flat.view_as(k)

            if context_slot_mapping is None:
                continue

            inner_attn = attn.attn
            inner_attn.impl.do_kv_cache_update(
                inner_attn,
                k,
                v,
                inner_attn.kv_cache,
                context_slot_mapping,
            )

    def load_weights(self, weights):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or f"{weight_name}_target" in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlareDraftModel(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlareQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights):
        model_weights = []
        direct_weights = []
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
                direct_weights.append((name, loaded_weight))
                continue
            if "lm_head" in name:
                direct_weights.append((name, loaded_weight))
                continue
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights.append((name, loaded_weight))

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        self.model.load_weights(
            (name, weight)
            for name, weight in model_weights
            if not any(substr in name for substr in skip_substrs)
        )

        loader_params = dict(self.named_parameters())
        for name, weight in direct_weights:
            if any(substr in name for substr in skip_substrs):
                continue
            if name not in loader_params:
                continue
            param = loader_params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)


class DFlareSpeculator(DFlashSpeculator):
    """DFlash scheduler with a wider concatenated aux-hidden context buffer."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        num_target_layers = len(dflash_config.get("target_layer_ids") or [])
        target_hidden_size = getattr(hf_config, "target_hidden_size", None) or self.hidden_size
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            num_target_layers * target_hidden_size,
            dtype=self.dtype,
            device=device,
        )
