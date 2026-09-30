# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class LwdMTPProposer(AscendEagleProposer):
    """Adapt externally supplied LWD embeddings to the MTP SP layout."""

    def _lwd_build_first_pass_embeds(
        self,
        num_tokens,
        num_input_tokens,
        next_token_ids,
        token_indices_to_sample,
        cad,
    ):
        # Prefix reuse skips input rows permanently. Never fall back to the
        # cloud's placeholder token IDs when a resumed prompt needs embeddings.
        enabled = self.vllm_config.cache_config.enable_prefix_caching
        needs_prompt = False
        if enabled:
            batch = self.runner.input_batch
            if len(batch.num_computed_tokens_cpu) < len(batch.req_ids):
                raise RuntimeError("LWD MTP is missing computed-token offsets")
            needs_prompt = any(
                int(batch.num_computed_tokens_cpu[i]) + 1
                < int(batch.num_prompt_tokens[i])
                for i in range(len(batch.req_ids))
            )
            if needs_prompt and getattr(self.runner, "pcp_manager", None) is not None:
                raise RuntimeError("LWD prefix MTP cannot use PCP embedding layout")
        result = super()._lwd_build_first_pass_embeds(
            num_tokens, num_input_tokens, next_token_ids,
            token_indices_to_sample, cad,
        )
        if enabled and needs_prompt and result is None:
            raise RuntimeError(
                "LWD prefix MTP requires authoritative prompt embeddings"
            )
        return result

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        """Shard external embeddings in the active draft context before forward."""
        context = get_forward_context()
        if inputs_embeds is not None and context.flash_comm_v1_enabled and not self.is_multimodal_model:
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
        )
