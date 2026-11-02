# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

if TYPE_CHECKING:
    from typing import Any

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.core.sched.output import SamplingMetadata


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP SP layout."""

    def _lwd_build_first_pass_embeds(
        self,
        num_tokens: int,
        num_input_tokens: int,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor | None:
        """Reject token-id fallback for continued prefill with prefix caching."""
        enabled = self.vllm_config.cache_config.enable_prefix_caching
        needs_prompt = False
        runner = self.runner
        if enabled and runner is not None:
            batch = runner.input_batch
            computed = batch.num_computed_tokens_cpu
            if len(computed) < len(batch.req_ids):
                raise RuntimeError("LWD prefix MTP is missing computed-token offsets")
            needs_prompt = any(
                int(computed[i]) + 1 < int(batch.num_prompt_tokens[i]) for i in range(len(batch.req_ids))
            )
            if needs_prompt and getattr(runner, "pcp_manager", None) is not None:
                raise RuntimeError("LWD prefix MTP cannot use PCP embedding layout")
        result = super()._lwd_build_first_pass_embeds(
            num_tokens,
            num_input_tokens,
            next_token_ids,
            token_indices_to_sample,
            cad,
        )
        if enabled and needs_prompt and result is None:
            raise RuntimeError("LWD prefix MTP requires authoritative prompt embeddings")
        return result

    def _run_merged_draft(
        self,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor | None,
        target_positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        multi_steps_attn_metadata: list[Any],
        num_tokens: int,
        is_prefill: bool | None = None,
        # Base contract since the probabilistic-draft change; the LWD path
        # does not consume it but must accept and forward it.
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        """Shard external embeddings in the active draft context before forward."""
        context = get_forward_context()
        # flash_comm_v1_enabled is only set by the SP fast path; without SP
        # (the LWD 1+8 TP layout) it is absent and the sharding is skipped.
        if (
            inputs_embeds is not None
            and getattr(context, "flash_comm_v1_enabled", False)
            and not self.is_multimodal_model
        ):
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_input_tokens:
                    raise ValueError("LWD MTP SP expects full-token inputs_embeds before slicing")
                # Embeddings are already replicated; reduce-scatter would sum
                # their values again, so only pad and slice the token rows.
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
        return super()._run_merged_draft(
            num_input_tokens,
            batch_size,
            token_indices_to_sample,
            target_positions,
            inputs_embeds,
            multi_steps_attn_metadata,
            num_tokens,
            is_prefill,
            sampling_metadata,
        )
