# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` enables prefill_only.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend import envs
from vllm_ascend.worker.lwd_cloud.lwd_draft_cache import LwdDraftCache
from vllm_ascend.worker.lwd_cloud.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
)
from vllm_ascend.worker.lwd_token_rank import LwdTokenRankEncoder
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class LwdCloudModelRunner(NPUModelRunner):
    """NPUModelRunner + prefill_only LWD cloud-side runner logic."""

    def __init__(self, vllm_config, device, worker=None):
        super().__init__(vllm_config, device)
        self.worker = worker  # LwdCloudWorker ref (UP recv futures live there)
        self._lwd_pending_down_payload = None
        self._lwd_token_rank_encoder = LwdTokenRankEncoder()
        self._lwd_draft_cache = (
            LwdDraftCache()
            if self.use_async_scheduling and self.speculative_config is not None
            and self.speculative_config.method == "mtp"
            and self.pcp_size == 1 and self.dcp_size == 1 else None
        )
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None
        # 用空值表示尚未准备真实批次，区分 dummy 前向。
        self._lwd_hash_step_tokens = None
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        lwd_config = getattr(vllm_config, "lwd_config", None)
        if num_hash_layers and lwd_config is not None and lwd_config.enabled and lwd_config.mode == "prefill_only":
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (parallel.data_parallel_size != 1 or parallel.prefill_context_parallel_size != 1
                    or parallel.decode_context_parallel_size != 1):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers, config.num_experts_per_tok, config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens, num_hash_layers, config.num_experts_per_tok,
                dtype=torch.int32, device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)

    def _update_states(self, scheduler_output):
        # Called inside synchronize_input_prep(), before remove_request()
        # destroys the previous mapping and before any new graph replay.
        self._lwd_rewind_staged_reads(scheduler_output)
        if self._lwd_draft_cache is not None:
            self._lwd_draft_cache.preserve(
                scheduler_output, self.requests,
                self.input_batch.prev_req_id_to_index, self._draft_token_ids,
            )
        return super()._update_states(scheduler_output)

    def _lwd_rewind_staged_reads(self, scheduler_output) -> None:
        """重算回退:调度水位退到 stager 读指针之下 = 云侧 KV 抢占重算。

        原生抢占释放全部 KV 并把 ``num_computed_tokens`` 归零
        (``Scheduler._preempt_request``);pinned host 载荷活到请求结束
        (design v7 §6.1),整条 prompt 可本地重读,只有读指针要回退 ——
        与 ``NPUModelRunner._update_states`` 里 KV-load-failure 的
        rewind 守卫同型。必须在 harvest 之前完成(本方法就是):harvest
        的恒等式校验要求 ``consumed == num_computed``(见
        ``_lwd_stage_one_request``),抢占后不回退会在那里当场炸。

        只按"回退"触发(consumed > 上报水位),不看 resumed_req_ids:
        正常爬坡里调度水位恒 ≥ 读指针(spec 拒绝的回退发生在 decode
        段,到不了 prompt 段),误触发面为零。非端点 rank 没有暂存池,
        直接跳过。
        """
        worker = self.worker
        stager = getattr(worker, "_lwd_stager", None) if worker is not None else None
        if stager is None:
            return
        req_data = scheduler_output.scheduled_cached_reqs
        for req_id, new_computed in zip(
            req_data.req_ids, req_data.num_computed_tokens
        ):
            consumed = stager.consumed_rows(req_id)
            if consumed > new_computed:
                stager.rewind_consumed(req_id, new_computed)
                logger.info(
                    "[Lwd][cloud-runner] REWIND req=%s: host read pointer "
                    "%d -> %d (preemption recompute; payload retained)",
                    req_id, consumed, new_computed,
                )

    def _prepare_input_ids(self, scheduler_output, num_reqs,
                           total_num_scheduled_tokens, cu_num_tokens):
        super()._prepare_input_ids(
            scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens,
        )
        self._lwd_restore_drafts(scheduler_output, num_reqs, cu_num_tokens)

    def _lwd_restore_drafts(self, scheduler_output, num_reqs, cu_num_tokens):
        cache = self._lwd_draft_cache
        if cache is None or not cache.entries:
            return
        rows, indices = [], []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            if req_id not in cache.entries:
                continue
            tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            row = cache.take(req_id, self.requests[req_id], len(tokens))
            # Consume the generation even when the scheduler drops all drafts.
            if not tokens or self.prev_positions.np[idx] >= 0:
                continue
            if row is None:
                raise RuntimeError(f"LWD draft snapshot is invalid for {req_id}")
            end = int(cu_num_tokens[idx])
            begin = int(cu_num_tokens[idx - 1]) if idx else 0
            if end - begin < len(tokens) + 1:
                raise RuntimeError(f"LWD draft segment is too short for {req_id}")
            rows.append(row)
            indices.extend(range(end - len(tokens), end))
        if rows:
            # Same stream as native input assembly and subsequent metadata /
            # embedding construction. No device-to-host copy or host wait.
            dst = torch.tensor(indices, dtype=torch.int64, pin_memory=self.pin_memory).to(
                self.device, non_blocking=True,
            )
            self.input_ids.gpu.scatter_(
                0, dst, torch.cat(rows).to(dtype=self.input_ids.gpu.dtype),
            )

    def _get_drafter(self):
        """Use the LWD MTP embedding adapter only in prefill_only mode."""
        lwd_config = getattr(self.vllm_config, "lwd_config", None)
        if (
            lwd_config is not None
            and lwd_config.enabled
            and lwd_config.mode == "prefill_only"
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        return super()._get_drafter()

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        """UP 收割与 mrope 准入写入在 super() 之前,embeds 覆写在之后。

        mrope 时序契约(多模态迁移复核发现的源分支缺陷修复):super()
        内部的 _calc_mrope_positions 当步即把 req_state.mrope_positions
        的窗口行拷上 GPU(model_runner_v1.py:1033-1045)——positions 若
        晚于 super() 写入,当步行永远不被自己的前向看到。wire_version 3
        起 positions 随 RequestNotify 全量一次下发,新请求在首个调度步
        经 ``_lwd_prime_mrope_positions`` 一次性写入(本函数开头)。

        embeds 保持在 super() 之后:基类填充循环 + copy_to_gpu 只看到
        零缓冲,随后把收到的 UP embeds 直写 inputs_embeds.gpu 当步调度
        窗口(后写为权威),经旁流拷贝的完成事件在流上有序——非阻塞。
        """
        if self._lwd_enabled():
            # 新准入请求的全量 mrope positions(随 SO 的 NewRequestData
            # 到达):准入时一次性写入 req_state 并自推 delta。
            for d in scheduler_output.scheduled_new_reqs:
                payload = getattr(d, "lwd_mrope_positions", b"") or b""
                if payload:
                    # LWD 请求的 prompt_token_ids 为 None;prompt 长度以
                    # 占位 embeds 的形状为准(随 NewRequestData 到达)。
                    plen = (
                        d.prompt_embeds_shape[0]
                        if d.prompt_embeds_shape
                        else 0
                    )
                    if plen <= 0:
                        raise RuntimeError(
                            f"[Lwd][cloud-runner] mrope prime req={d.req_id}"
                            f": prompt_embeds_shape missing (版本错配)"
                        )
                    self._lwd_prime_mrope_positions(d.req_id, payload, plen)
        dev = win_idx = None
        if self._lwd_enabled():
            dev, win_idx = self._lwd_harvest_up_batches(num_scheduled_tokens)
        embeds_map = None
        if self._lwd_enabled():
            # 跳过基类填充循环对 LWD 占位行的逐行拷贝(gpu_model_runner
            # .py:952-996):该循环把 CPU 装配缓冲的当步窗口(此刻恒为
            # 零值)做 84MB 主机 memcpy + H2D 灌入 inputs_embeds,随后
            # 注入覆写同一窗口(后写为权威)——纯浪费,实测 super 段
            # ~17-20ms/chunk。清空映射即整体跳过;注入/释放逻辑在
            # super() 返回后恢复引用。LWD 云侧请求全是占位行,无真实
            # prompt-embeds 消费者被误伤。
            embeds_map = self.input_batch.req_prompt_embeds
            self.input_batch.req_prompt_embeds = {}
        try:
            out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            if embeds_map is not None:
                self.input_batch.req_prompt_embeds = embeds_map
        if self._lwd_enabled():
            self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
            self._lwd_release_consumed_prompt_embeds()
            self._lwd_inject_remote_embeds(dev, win_idx, num_scheduled_tokens)
        if self.lwd_hash_state is not None:
            self._lwd_prepare_hash_batch(num_scheduled_tokens)
        return out

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens):
        """Skip remote expert staging for decode, preserving graph buffers."""
        num_reqs = self.input_batch.num_reqs
        counts = num_scheduled_tokens[:num_reqs]
        computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_reqs]
        self._lwd_hash_step_tokens = sum(int(n) for n in counts)
        has_prompt = any(
            int(count) > 0 and int(start) < int(prompt_len)
            for count, start, prompt_len in zip(counts, computed, prompt_lengths)
        )
        # Clear the full mask, including graph padding and previous larger
        # batches. Keep its address and the captured merge graph unchanged.
        self._lwd_hash_prompt_mask.zero_()
        if not has_prompt:
            # The merge selects local lookup for every row. Stale expert IDs
            # are only unselected values, never indices into an expert table.
            return
        batch, mask = self.lwd_hash_state.build_batch(
            self.input_batch.req_ids, counts, computed, prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[:batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[:mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens) -> None:
        """重建 prompt-embeds 分支的扁平 is_token_ids 掩码。

        根因(async spec decode 乐观值):runner 的 num_computed_tokens_cpu
        假设上步草稿全被接受(比真实提交位置多 r),而原生扁平化
        (model_runner_v1.py 的 token_indices = positions + idx*M)用乐观
        positions 从二维掩码取值——decode 行整体右移 r 位,每段末尾
        r 行越出 True 标记区读到 False/陈旧值,被 prompt-embeds 分支
        当作 embeds 行而不做本地嵌入,保留 inputs_embeds.gpu 的陈旧
        垃圾;这些行恰是被采样行(logits_indices 取段末 1+k 行),
        垃圾 logits 直接污染已提交 token。原生/phase/单请求不受影响
        (掩码只在 prompt-embeds 分支被消费,那些形态走 MM 分支)。

        LWD 语义里只有注入行(prompt 段)应是 embeds 行(False),
        decode/草稿行恒为 token id(True)。此处按调度状态确定性重建,
        不依赖任何乐观值修正:decode 请求的 computed 乐观只会偏大,
        prefill_rows 恒为 0,天然安全。保留 CPU 视图修复供父类路径使用
        (.np 与 .cpu 同存储)；异步路径在 _preprocess 中按修正后的
        设备位置重新生成 GPU 掩码，并直接消费设备 embedding。
        """
        if not self.input_batch.req_prompt_embeds:
            # 批内无 embeds 缓冲(MM/text 分支,掩码无人读)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(
                n, max(int(num_prompt[i]) - int(computed[i]), 0)
            )
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        """在异步投机位置修正后刷新 decode 掩码，再准备模型输入。"""
        if (self.use_async_spec_decode and self.enable_prompt_embeds
                and intermediate_tensors is None and self.pcp_size == 1):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (get_pp_group().is_first_rank
                    and not self.model_config.is_encoder_decoder
                    and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    def _lwd_preprocess_prompt_embeds(self, scheduler_output, num_input_tokens):
        """Consume the corrected device mask on the LWD cloud decoder path."""
        count = scheduler_output.total_num_scheduled_tokens
        is_token_ids = self.is_token_ids.gpu[:count]
        # Remote prompt rows do not carry meaningful token IDs.
        token_ids = self.input_ids.gpu[:count].masked_fill(~is_token_ids, 0)
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        embeds = self.inputs_embeds.gpu[:count]
        embeds.copy_(torch.where(is_token_ids.unsqueeze(-1), tokens_to_embeds, embeds))

        # Match the parent's position layout and clear padding for graph replay.
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > count:
                self.positions[count:num_input_tokens].zero_()
        return (
            None,
            self.inputs_embeds.gpu[:num_input_tokens],
            positions,
            None,
            self._init_model_kwargs(),
            None,
        )

    def _lwd_refresh_decode_embedding_mask(self, scheduler_output):
        """根据设备上的实际位置重建掩码，避免 decode 行复用旧 embedding。

        异步上传各请求的 prompt 长度，复用设备上的请求索引展开到 token 行。
        掩码直接写入设备缓冲区，由 LWD prompt-embeds 路径在设备上消费；
        不回传 positions，也不刷新 CPU 掩码。
        """
        num_reqs = len(self.input_batch.req_ids)
        count = scheduler_output.total_num_scheduled_tokens
        prompt_lens_gpu = self._lwd_prompt_lens_gpu
        assert prompt_lens_gpu is not None
        prompt_lens_gpu[:num_reqs].copy_(
            self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        prompt_lens_rows = prompt_lens_gpu[self.req_indices.gpu[:count]]
        self.is_token_ids.gpu[:count] = self.positions[:count] >= prompt_lens_rows

    # 在云侧模拟前向前清除真实批次标记，防止复用上次请求路由。
    def _dummy_run(self, *args, **kwargs):
        """清除真实批次的 Hash 路由标记，再执行模拟前向。"""
        # 用空值表示尚未准备真实批次，区分 dummy 前向。
        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(self, num_tokens_padded, input_ids=None, positions=None,
                       intermediate_tensors=None, inputs_embeds=None, **model_kwargs):
        """按 SP 布局切分 embedding，并将完整 Hash 路由数据写入前向上下文。"""
        context = get_forward_context()
        if (inputs_embeds is not None and intermediate_tensors is None
                and context.flash_comm_v1_enabled and not self.supports_mm_inputs):
            # 直接传入 embedding 会绕过原生嵌入层，需要在此补上序列切分。
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_tokens_padded:
                    raise ValueError("LWD SP expects full-token inputs_embeds before slicing")
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
            # 位置、注意力元数据和 Hash 路由表保持完整 token 布局。
        if self.lwd_hash_state is not None:
            # 先初始化路由索引缓存，确保补齐行使用合法零索引。
            self._lwd_hash_token_ids.zero_()
            if self._lwd_hash_step_tokens is None:
                # Synthetic forwards have no remote prompt; route dummy token 0.
                self._lwd_hash_prompt_mask.zero_()
            else:
                count = self._lwd_hash_step_tokens
                if count > num_tokens_padded:
                    raise ValueError("LWD Hash routing rows exceed model input rows")
                # Native input preparation retains real decode IDs even when
                # model input_ids=None because the model consumes embeddings.
                self._lwd_hash_token_ids[:count].copy_(self.input_ids.gpu[:count])
            context.input_ids = self._lwd_hash_token_ids[:num_tokens_padded]
            context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens_padded]
            context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens_padded]
        return super()._model_forward(
            num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs,
        )

    def _lwd_enabled(self) -> bool:
        cfg = getattr(self.vllm_config, "lwd_config", None)
        return bool(cfg is not None and cfg.enabled)

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        if self.lwd_hash_state is not None:
            for req_id in list(self.lwd_hash_state.requests):
                request = self.requests.get(req_id)
                if request is None or request.num_computed_tokens >= request.num_prompt_tokens:
                    self.lwd_hash_state.discard(req_id)
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def _lwd_harvest_up_batches(self, num_scheduled_tokens):
        """UP 收割(super()._prepare_inputs 之前调用),arrival 驱动(design v8)。

        对本步所有"还在 prefill"的请求(num_computed < prompt_len 且本步
        排了行),把 ``consumed .. consumed+w`` 从 per-request host buffer
        搬进**共享窗口缓冲**(双缓冲按步轮换),端点一次 TP 广播 —— 无
        seqno、无 meta:行数完全由原生 ``num_scheduled_tokens`` 驱动,
        注入落点恒为 ``num_computed``。

        mrope positions 自 wire_version 3 起随 RequestNotify 全量一次
        下发、准入时写入 req_state(见 ``_lwd_prime_mrope_positions``),
        本路径不再有 per-chunk positions 工作。

        返回 ``(up_flat, win_idx)``;本步没有要注入的行时返回
        ``(None, None)``。
        """
        worker = self.worker
        if worker is None:
            return None, None
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()
        # 收集本步要注入的 prompt 行(越过 prompt 末尾的行是 decode
        # token,走 token-ids 路径,不注入)
        plan: list[tuple[str, int]] = []  # (rid, w)
        for idx, rid in enumerate(self.input_batch.req_ids):
            if idx >= len(num_scheduled_tokens):
                break
            w = int(num_scheduled_tokens[idx])
            ncomp = int(computed[idx])
            nprog = int(num_prompt[idx])
            if ncomp >= nprog or w <= 0:
                continue
            w = min(w, nprog - ncomp)
            if w > 0:
                plan.append((rid, w))
        if not plan:
            return None, None
        # 本步各请求的取数起点(供注入侧切专家 ID 尾段):每步重开。
        self._lwd_step_tail_base: dict[str, int] = {}
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        _tp = get_tp_group()
        _lwd_cfg = worker.parallel_config.lwd_config
        _is_endpoint = worker.rank == _lwd_cfg.edge_npu_count
        stager = getattr(worker, "_lwd_stager", None)
        if _is_endpoint and stager is None:
            # 端点必须有暂存池(引擎开关保证);缺失即部署错配,fail-fast。
            raise RuntimeError(
                "[Lwd][cloud-runner] endpoint rank without host stager "
                "(host staging is mandatory in this deployment)"
            )
        total_rows = sum(w for _r, w in plan)
        if _is_endpoint:
            dev, win_idx = self._lwd_window_device(total_rows * hidden_size)
            stream = self._lwd_h2d_stream()
            # WAR 门:旁流开写这块缓冲前,等它上一次被计算流消费(inject +
            # broadcast)完;按步轮换下该事件正常早已就绪,wait 只是顺序
            # 记号,不阻 CPU。
            used_ev = self._lwd_win_used_ev[win_idx]
            _t0 = time.monotonic()
            with torch.npu.stream(stream):
                if used_ev is not None:
                    stream.wait_event(used_ev)
                offset = 0
                for rid, w in plan:
                    self._lwd_stage_one_request(
                        stager, rid, w, dev, offset, hidden_size
                    )
                    offset += w * hidden_size
                ready_ev = torch.npu.Event()
                ready_ev.record(stream)
            # 事件门:计算流在随后的 broadcast/inject 前等拷贝完成 ——
            # 只是流间顺序记号,CPU 不停;拷贝已在旁流与上一步 forward
            # 并行执行。
            torch.npu.current_stream().wait_event(ready_ev)
            for rid, w in plan:
                stager.consume(rid, w)
            logger.info(
                "[Lwd][cloud-runner] CONSUME rows=%d reqs=%d "
                "host_MB=%.1f held=%d stage=%.2fms",
                total_rows, len(plan),
                stager.host_bytes / 2 ** 20, stager.request_count,
                (time.monotonic() - _t0) * 1000,
            )
        else:
            # 非端点 rank 没有 host 池:只按同样的**行数**分配接收缓冲,
            # 等端点广播过来(行数必须一致,W4)。
            dev = torch.empty(
                total_rows * hidden_size, dtype=torch.bfloat16, device="npu"
            )
            win_idx = None
        if _tp.world_size > 1 and dev.numel() > 0:
            # demo 同款:直接用框架 TP 通信域(建组顺序由框架保证,
            # PD 分离路径已验证),launch+wait 同流原子对。
            work = dist.broadcast(
                dev, src=_tp.ranks[0], group=_tp.device_group, async_op=True,
            )
            work.wait()  # 桥接广播完成到当前(计算)流,后续 copy 有序
        return dev, win_idx

    def _lwd_h2d_stream(self):
        """H2D 专用拷贝流:与计算流并行,把窗口拷贝藏进上一步 forward。"""
        stream = getattr(self, "_lwd_h2d_copy_stream", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._lwd_h2d_copy_stream = stream
        return stream

    def _lwd_window_device(self, numel: int) -> tuple[torch.Tensor, int]:
        """窗口暂存设备缓冲(§4.5):``budget × H`` 大小,**双缓冲轮换**。

        双缓冲是旁流 H2D 的前提:拷贝在独立拷贝流上发射(与上一步
        forward 并行),计算流消费前只等一个 ready event;隔批轮换 +
        消费事件(``_lwd_win_used_ev``)保证旁流不会在计算流还没读完时
        覆写同一块(WAR)。
        """
        bufs = getattr(self, "_lwd_win_dev", None)
        if bufs is None or bufs[0].numel() < numel:
            cap = max(
                numel,
                int(getattr(self.worker, "_lwd_up_hidden_size", 0) or 0)
                * int(
                    getattr(
                        getattr(self.vllm_config, "scheduler_config", None),
                        "max_num_batched_tokens", 0,
                    ) or 0
                ),
            )
            bufs = [
                torch.empty(
                    max(cap, numel), dtype=torch.bfloat16, device=self.device
                )
                for _ in range(2)
            ]
            self._lwd_win_dev = bufs
            self._lwd_win_flip = 0
            self._lwd_win_used_ev = [None, None]
            logger.info(
                "[Lwd][host-stage] win_dev allocated rows_capacity=%d "
                "(%.1fMiB) x2 (double-buffered)",
                bufs[0].numel() // max(self.model_config.get_hidden_size(), 1),
                bufs[0].numel() * 2 / 2 ** 20,
            )
        idx = self._lwd_win_flip
        self._lwd_win_flip ^= 1
        return bufs[idx][:numel], idx

    def _lwd_stage_one_request(
        self, stager, rid: str, w: int, buf, elem_offset: int,
        hidden_size: int,
    ) -> None:
        """单请求窗口:校验指针恒等式 + 一次异步 H2D 到共享缓冲的切片内。

        恒等式(踩过静默精度事故,硬校验宁可当场炸):取数从
        ``consumed`` 顺序读,注入落点恒等于 ``num_computed``,两者必须
        相等 —— 一旦错开(同一批被搬两次、窗口起点算错),模型拿到错位
        embedding,表现是输出截断/空白、不报错、不乱码。W2(下发量 ≤
        已写入量)由 stager 取数时用**推进前**的指针权威判定
        (``LwdReqEmbed.read_range``)。
        """
        idx = self.input_batch.req_id_to_index.get(rid)
        num_computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        consumed_before = stager.consumed_rows(rid)
        expect = int(num_computed[idx]) if idx < len(num_computed) else 0
        if consumed_before != expect:
            raise RuntimeError(
                f"[Lwd][cloud-runner] STAGE POSITION MISMATCH req={rid}: "
                f"host read pointer consumed={consumed_before} != "
                f"computed={expect} (rows={w}); the embeddings would be "
                f"injected at a position that does not match the data read"
            )
        host_start, numel, _tail = stager.segment_for(rid, w)
        if idx < len(num_prompt):
            room = int(num_prompt[idx]) - consumed_before
            if room > 0 and w > room:
                raise RuntimeError(
                    f"[Lwd][cloud-runner] STAGE OVERRUN req={rid}: "
                    f"staging {w} rows at consumed={consumed_before} but "
                    f"the prompt is only {int(num_prompt[idx])} rows"
                )
        # 记录本步的取数起点(供注入侧切专家 ID 尾段;consumed 指针在
        # harvest 末尾才推进,注入侧不能直接读它)
        self._lwd_step_tail_base[rid] = consumed_before
        buf[elem_offset: elem_offset + numel].copy_(
            stager.take_req(rid).host[host_start: host_start + numel],
            non_blocking=True,
        )

    def _lwd_inject_remote_embeds(self, dev, win_idx, num_scheduled_tokens) -> None:
        """Device-side inject(super() 之后):把收割到的 UP embeds 直写
        ``inputs_embeds.gpu`` 各请求的当步调度窗口。

        arrival 驱动(design v8):注入落点恒为 ``num_computed``(没有
        chunk 起点要算);dev 是本步所有暂存行的紧邻拼接(行序 =
        input_batch 的请求序,与 harvest 的 plan 同序)。The flattened
        output offsets reproduce the native fill loop's accumulation.
        The base fill loop is skipped for LWD placeholder rows (see
        ``_prepare_inputs``); our device write is authoritative.

        专家 ID(DSv4)不上设备,直接从**该请求自己的** host buffer 尾段
        按同一窗口切出来记账(§4.6);窗口起点用 harvest 记录的**推进前**
        读指针(``_lwd_step_tail_base``)——直接读 consumed 会拿到推进后
        的值,取到下一个窗口的专家 ID(真机踩过)。
        """
        if dev is None:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        num_computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        stager = self.worker._lwd_stager if self.worker is not None else None

        # Flattened output offset per request (native fill loop 同款累计)
        out_offset: dict[str, int] = {}
        off = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            out_offset[req_id] = off
            off += (
                int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens)
                else 0
            )
        embeds = dev.view(-1, hidden_size)
        row = 0
        for idx, req_id in enumerate(self.input_batch.req_ids):
            if idx >= len(num_scheduled_tokens):
                break
            w = int(num_scheduled_tokens[idx])
            ncomp = int(num_computed[idx])
            nprog = int(num_prompt[idx])
            if ncomp >= nprog or w <= 0:
                continue
            w = min(w, nprog - ncomp)
            if w <= 0:
                continue
            if self.lwd_hash_state is not None and stager is not None:
                # 专家 ID 按**同一窗口**推进(§4.6);暂存路径的尾段在
                # host 上是逐请求的权威副本,起点用推进前的读指针。
                top_k = config.num_experts_per_tok
                staged_req = stager.take_req(req_id)
                if staged_req is None:
                    raise ValueError(
                        f"[Lwd][cloud-runner] expert tail for {req_id} "
                        f"unavailable: not staged"
                    )
                t_base = self._lwd_step_tail_base.get(req_id)
                if t_base is None:
                    raise ValueError(
                        f"[Lwd][cloud-runner] expert tail base for "
                        f"{req_id} missing (harvest did not stage it)"
                    )
                t_start, _t_end = staged_req.tail_range_at(t_base, w)
                seg = staged_req.host[t_start: t_start + w * 4
                                      * num_hash_layers * top_k]
                ids = self._lwd_decode_expert_ids(
                    seg, w, num_hash_layers, top_k,
                )
                self.lwd_hash_state.add_chunk(
                    req_id, nprog, ncomp, ids,
                )
            out = out_offset.get(req_id, 0)
            # stream 上直接写 inputs_embeds.gpu 的调度窗口
            gpu_embeds[out : out + w].copy_(
                embeds[row : row + w], non_blocking=True
            )
            # 不再 D2H 写 CPU 装配缓冲:draft provider 已改为
            # 直读 inputs_embeds.gpu 同窗口(本行即权威副本),
            # 消除每 chunk 84MB 的 D2H 同步长尾
            self.input_batch.is_token_ids[idx, ncomp : ncomp + w] = False
            row += w
        # 生命周期收尾:暂存路径登记"计算流已消费完这块窗口缓冲"(旁流
        # 双缓冲的 WAR 门);非暂存路径 record_stream 防分配器提前复用。
        if win_idx is not None:
            ev = torch.npu.Event()
            ev.record()
            self._lwd_win_used_ev[win_idx] = ev
        else:
            dev.record_stream(torch.npu.current_stream())

    @staticmethod
    def _lwd_decode_expert_ids(seg, w: int, num_layers: int, top_k: int):
        """把窗口的专家 ID 字节段解成 ``[w, L, K]`` int32。

        与 ``unpack_hash_payload`` 逐位同款:线序是 uint8 字节流,按 int32
        视图解释(不能按 bf16 数值解释)。窗口化后段是 host 上的切片,
        所以这里只做视图转换,不再触发设备往返。
        """
        return (
            seg.contiguous()
            .view(torch.uint8)
            .to(torch.int32)
            .view(w, num_layers, top_k)
        )

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _lwd_prime_mrope_positions(
        self, req_id: str, payload: bytes, prompt_len: int
    ) -> None:
        """准入时一次性写入全量 mrope positions(wire_version 3):

        positions 随 RequestNotify 到来([3, prompt_len] int64 原始字节,
        边侧准入时一次算好),frombuffer 零转置直写 req_state,并即席
        自推 delta(= positions.max()+1−prompt_len,与边侧逐值一致)供
        decode 期原生 _calc_mrope_positions 现算 completion 段位置。

        req_state.mrope_positions 由 ids=None 直通初始化(arange 纯文
        本位置)——多模态请求在这里被全量覆盖,纯文本请求永不走本路径
        (边侧不给它发 positions)。
        """
        req_state = self.requests.get(req_id)
        if req_state is None:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope prime req={req_id}: "
                f"req_state not found ( admission ordering broken)"
            )
        positions = torch.frombuffer(bytearray(payload), dtype=torch.int64)
        if positions.numel() != 3 * prompt_len:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope prime req={req_id}: payload "
                f"{positions.numel()} elems != 3 * prompt_len "
                f"{prompt_len} (边云版本错配,须同版本部署)"
            )
        positions = positions.view(3, prompt_len)
        req_state.mrope_positions[:, :prompt_len] = positions
        req_state.mrope_position_delta = (
            int(positions.max().item()) + 1 - prompt_len
        )
        logger.info(
            "[Lwd][cloud-runner] mrope primed: req=%s prompt=%d delta=%d",
            req_id, prompt_len, req_state.mrope_position_delta,
        )

    def _sample(self, logits, spec_decode_metadata):
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        from vllm.distributed.parallel_state import get_tp_group

        # 只有云 TP 组首卡(= DOWN 通道端点 rank)采集/发送;
        # 其余 rank 无通道 peer,采集即弃也一并省掉(8 卡冗余)。
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # 批序/掩码也一并快照:batch queue 重叠时,下一步的
            # _prepare_inputs 可能在 collect 前重排 input_batch,
            # 现读会拿到批序B去切批序A的张量(高并发错位根因)
            captured = (
                state[5], state[1], state[2], state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        # VLLM_ASCEND_LWD_DISABLE_DOWN=1 诊断:不做 collect(云引擎改走
        # token_ids 直通通告),payload 为 None 时 worker 自然跳过 DOWN send
        if (captured is not None
                and self._lwd_captured_sampler_output is not None
                and not envs.VLLM_ASCEND_LWD_DISABLE_DOWN):
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0], captured[1], captured[2], captured[3],
                captured[4], captured[5], self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(self):
        """worker 层取走本步 DOWN payload(单槽覆盖写,每步必被取走)。"""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self, sample_hidden_states, logits, spec_decode_metadata,
        scheduler_output, batch_req_ids, discard_mask_np, sampler_output,
    ):
        """生产级 DOWN 采集(rank-replay):hidden 组包 + 全局秩 + num_accepted。

        廉价计算:bf16 直比(logits 原生 bf16,与 cast 后逐位等价)、
        批全覆盖时整行直接算(不做高级索引取材)、ranks/counts
        拼单个设备张量;meta 不落 pinned——设备张量本体随 payload
        传递,由 executor 输出线程在就绪事件同步后收割拼装,计算
        关键路径零新增同步。批序/掩码取捕获时刻快照(防 batch queue
        重叠重排)。返回:
          (hidden_packet, meta_dev, seg_lens_host, ready_event, req_ids)
        或 None(本步无在途请求)。
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask 来自捕获时刻快照(与 logits/hidden
        # 同批序),不读活的 input_batch
        valid = ~discard_mask_np[: len(batch_req_ids)]
        is_spec = spec_decode_metadata is not None

        rows_list, ranks_list, accepted = [], [], []
        hidden_packet: torch.Tensor | None = None
        lg = logits  # bf16 原生,直接比较(cast 无精度增益)
        full_cover = all(valid[: len(batch_req_ids)])
        if not is_spec:
            idx = [i for i in range(len(batch_req_ids)) if valid[i]]
            if not idx:
                return None
            lg_sel = lg if full_cover else lg[idx]
            sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
            # 与边侧稳定排序一致：并列候选按 token ID 升序编码为唯一排名。
            ranks_list.append(self._lwd_token_rank_encoder.encode(lg_sel, sm_sel))
            rows_list.extend(sample_hidden_states[i : i + 1] for i in idx)
            accepted.extend([1] * len(idx))
        else:
            # 全零同步 spec 路径(向量化):
            # 段长按 host 侧 scheduled_spec_decode_tokens 推(1+draft_len),
            # 按完整段打包(含被拒行,边侧按 num_accepted 取有效前缀);
            # 段长取 spec_decode_metadata.num_draft_tokens(host list,
            # 运行期真实布局,与 sample_hidden_states 段结构一致);
            # 不用 scheduler_output.scheduled_spec_decode_tokens
            # (调度输入,可能与实际运行不一致)
            seg_lens = [
                d + 1 for d in spec_decode_metadata.num_draft_tokens
            ]
            # counts 从同一个 sampled 张量 device 推导(与秩/行同源,
            # 天然按批位对齐);seg_lens/counts 只收 valid 请求,
            # 保证 [ranks|counts|seg_lens] 三段长度一致
            counts_all = (sampled != -1).sum(dim=1)
            # host 一趟建索引,替代逐请求 device 循环(原实现每请求
            # ~6 次小 kernel 发射,rank 段随并发涨到 20+ms;向量化后
            # 发射数与 N 无关)。索引语义与原循环逐位等价:
            # vidx/vlen = 有效请求批位/实际行数(按总段长封顶,
            # = 原 rows_i);req_idx/pos_idx = 每行的请求位/行内位
            # (原 sampled[i, :rows_i]);row_sel = 有效行全局行号。
            seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
            seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
            total_rows = sample_hidden_states.shape[0]
            rows_per_req = np.minimum(
                seg_lens_np, np.maximum(total_rows - seg_starts, 0)
            )
            vidx_np = np.nonzero(
                valid[: len(batch_req_ids)] & (rows_per_req >= 1)
            )[0].astype(np.int64)
            if vidx_np.size == 0:
                return None
            vlen_np = rows_per_req[vidx_np].astype(np.int64)
            starts_v = seg_starts[vidx_np].astype(np.int64)
            n_rows = int(vlen_np.sum())
            req_idx_np = np.repeat(vidx_np, vlen_np)
            pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(
                np.cumsum(vlen_np) - vlen_np, vlen_np
            )
            row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
            # 全部索引一次 pinned 异步 H2D(pageable 同步 H2D 会在
            # 主流上等设备排空,见 seg_lens 同款教训)
            buf = self._lwd_h2d_stage_i64(
                np.concatenate(
                    [req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]
                )
            )
            n_valid = vidx_np.size
            req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
            sel_d = buf[2 * n_rows : 3 * n_rows]
            vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
            vlen_d = buf[3 * n_rows + n_valid :]
            sampled_rows = sampled[req_d, pos_d]
            if n_rows == total_rows:
                # 全 valid 无封顶:行即连续前缀,免 gather
                lg_rows = lg[:n_rows]
                hidden_packet = sample_hidden_states[:n_rows]
            else:
                lg_rows = lg[sel_d]
                hidden_packet = sample_hidden_states[sel_d]
            # 投机与非投机共用排名规则；拒绝后的 -1 行只保留占位，不交付。
            ranks_list.append(self._lwd_token_rank_encoder.encode(lg_rows, sampled_rows))
            # 过期 sampled 位(spec 未运行的请求可能残留上个 spec 步的
            # token):accepted/秩/段长一律按实际 hidden 行数封顶——
            # 行数才是真实采样位置的真相
            counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(
                torch.int32
            )
            seg_lens_list = vlen_np.tolist()
        if hidden_packet is None:
            if not rows_list:
                return None
            hidden_packet = torch.cat(rows_list)
        n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
        seg_lens_host: list[int] | None = None
        if is_spec:
            meta_dev = torch.cat(ranks_list + [counts_dev])
            # seg_lens 不上设备(host 侧列表随 payload 携带,输出线程
            # 收割时拼装到 meta 尾部;非 spec 路径 seg_lens 在
            # meta_dev 内)。线布局:[ranks|counts|seg_lens]
            seg_lens_host = seg_lens_list
        else:
            counts_dev = torch.ones(n_req, dtype=torch.int32,
                                    device=logits.device)
            seg_lens_dev = counts_dev
            meta_dev = torch.cat(ranks_list + [counts_dev, seg_lens_dev])
        # meta 不随 collect 落 pinned:异步 D2H 在本平台走边流,主流
        # 就绪事件围栏不住(真机实锤:引擎读到上一代 meta 影像)。
        # payload 改携带设备张量本体,由 executor 输出线程收割拼装
        # (拷贝流 wait_event 就绪事件 + 完成事件与拷贝同流),计算
        # 关键路径零新增同步。就绪事件随 payload 传递(每步独立对象,
        # record 完成才离手):不能写 worker 共享属性由输出线程现读
        # ——属性每步被覆写,存在"已赋值未 record"的窗口(record
        # 在流满载时阻塞可达毫秒级),sync 未 record 的事件会立即放行。
        _t_pack = time.monotonic()
        ready_event = torch.npu.Event()
        ready_event.record()
        # [Lwd][perf] 采集分段:rank=秩+行收集;pinned=meta 组装(拷贝
        # 在输出线程收割,本段应≈0)
        logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f pinned=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (_t_pack - _t0) * 1000,
            (time.monotonic() - _t_pack) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            meta_dev,
            seg_lens_host,
            ready_event,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
        )

    def _lwd_h2d_stage_i64(self, arr: "np.ndarray") -> torch.Tensor:
        """int64 host 索引数组的异步 H2D 暂存(pinned 4 槽轮换 +
        non_blocking):替代 torch.tensor(arr, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(MTP 下即 draft propose
        尾巴)。环深覆盖 batch_queue 深度 + 引擎滞后,防止下步 host
        覆写时上步拷贝尚未执行。"""
        n = arr.size
        ring = getattr(self, "_lwd_h2d_ring", None)
        if ring is None or ring[0][0].numel() < n:
            m = max(n, 1024)
            ring = [
                (
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(m, dtype=torch.int64, device=self.device),
                )
                for _ in range(4)
            ]
            self._lwd_h2d_ring = ring
            self._lwd_h2d_ring_idx = 0
        idx = self._lwd_h2d_ring_idx
        self._lwd_h2d_ring_idx = (idx + 1) % len(ring)
        stage, dev = ring[idx]
        stage.numpy()[:n] = arr
        dev[:n].copy_(stage[:n], non_blocking=True)
        return dev[:n]
