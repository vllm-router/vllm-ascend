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
        """前缀缓存守卫(设计 §5.1,移植 7f2269447 按本分支现状适配)。

        PC 开启时,命中请求的前 H 行永久不上行,续算请求
        (``computed + 1 < prompt``)的 prompt 段 embeddings 只能来自
        权威暂存;基类返回 None 会让调用方回退到占位 token-id 查表,
        draft 状态从根上静默污染 —— 两处 fail-fast(禁止回退):

        * 续算请求与 PCP 并存:PCP all-gather 布局使右移对齐假设不成立;
        * 续算请求存在而基类返回 None:占位查表不是合法降级。
        """
        enabled = self.vllm_config.cache_config.enable_prefix_caching
        needs_prompt = False
        runner = self.runner
        if enabled and runner is not None:
            batch = runner.input_batch
            computed = batch.num_computed_tokens_cpu
            if len(computed) < len(batch.req_ids):
                raise RuntimeError(
                    "LWD prefix MTP is missing computed-token offsets"
                )
            needs_prompt = any(
                int(computed[i]) + 1 < int(batch.num_prompt_tokens[i])
                for i in range(len(batch.req_ids))
            )
            if needs_prompt and getattr(runner, "pcp_manager", None) is not None:
                raise RuntimeError(
                    "LWD prefix MTP cannot use PCP embedding layout"
                )
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
