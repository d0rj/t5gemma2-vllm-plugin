"""DSpark vLLM runtime built on top of vLLM's DFlash speculator."""

from __future__ import annotations

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class DSparkSpeculator(DFlashSpeculator):
    """DFlash block proposer with sequential DSpark Markov logit bias."""

    def _sample_dspark_step(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_step: int,
    ) -> torch.Tensor:
        compute = getattr(self.model, "compute_logits_with_prev_tokens", None)
        if compute is None:
            logits = self.model.compute_logits(hidden_states)
        else:
            logits = compute(hidden_states, prev_token_ids)

        if self.draft_logits is None:
            return logits.argmax(dim=-1)

        draft_step_tensor = torch.full_like(idx_mapping, draft_step, dtype=torch.int32)
        return gumbel_sample(
            logits,
            idx_mapping,
            self.temperature,
            self.seeds,
            positions + 1,
            apply_temperature=True,
            output_processed_logits=self.draft_logits,
            output_processed_logits_col=draft_step_tensor,
            use_fp64=self.use_fp64_gumbel,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        steps = self.num_speculative_steps
        num_sample = num_reqs * steps
        sample_indices = self.sample_indices[:num_sample].view(num_reqs, steps)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, steps)
        sample_idx_mapping = self.sample_idx_mapping[:num_sample].view(num_reqs, steps)

        req_offsets = torch.arange(
            num_reqs,
            dtype=torch.long,
            device=self.input_buffers.input_ids.device,
        ) * self.num_query_per_req
        prev_token_ids = self.input_buffers.input_ids[req_offsets]
        sampled = []
        for step in range(steps):
            hidden_step = last_hidden_states[sample_indices[:, step]]
            token_step = self._sample_dspark_step(
                hidden_states=hidden_step,
                prev_token_ids=prev_token_ids,
                positions=sample_pos[:, step],
                idx_mapping=sample_idx_mapping[:, step],
                draft_step=step,
            )
            sampled.append(token_step)
            prev_token_ids = token_step

        self.draft_tokens[:num_reqs] = torch.stack(sampled, dim=1)
