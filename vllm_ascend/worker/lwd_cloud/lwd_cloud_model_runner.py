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
from vllm.v1.lwd_control.control_communication.lwd_id_adapter import parse_edge_id
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend import envs
from vllm_ascend.worker.lwd_cloud.lwd_draft_cache import LwdDraftCache
from vllm_ascend.worker.lwd_cloud.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
    hash_payload_numel,
    unpack_hash_payload,
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
        if self._lwd_draft_cache is not None:
            self._lwd_draft_cache.preserve(
                scheduler_output, self.requests,
                self.input_batch.prev_req_id_to_index, self._draft_token_ids,
            )
        return super()._update_states(scheduler_output)

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
        """UP 收割与 mrope 写入在 super() 之前,embeds 覆写在之后。

        mrope 时序契约(多模态迁移复核发现的源分支缺陷修复):super()
        内部的 _calc_mrope_positions 当步即把 req_state.mrope_positions
        的窗口行拷上 GPU(model_runner_v1.py:1033-1045)——mrope 写入
        若晚于 super()(源实现形态),当步 chunk 的 mrope 行永远不被
        自己的前向看到(所有 chunk 用 ids=None 直通初始化的 arange
        值,图像行必错)。故收割(端点 host 门 + TP 组内现场广播)与
        mrope/delta 写入提到 super() 之前。

        embeds 保持在 super() 之后:基类填充循环 + copy_to_gpu 只看到
        零缓冲,随后把收到的 UP embeds 直写 inputs_embeds.gpu 当步调度
        窗口(后写为权威),经 future 完成事件在流上有序——非阻塞,
        recv 等待另有 [Lwd][perf] up-recv-wait 计时可见。
        """
        harvested = None
        if self._lwd_enabled():
            harvested = self._lwd_harvest_up_batches(num_scheduled_tokens)
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
            self._lwd_inject_remote_embeds(harvested, num_scheduled_tokens)
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

    def _lwd_harvest_up_batches(self, num_scheduled_tokens) -> list:
        """UP 批收割(super()._prepare_inputs 之前调用,时序契约见
        _prepare_inputs):端点过 host 门 + TP 组内现场广播收主帧
        embeds,完成窗口核验与 mrope 行写入 req_state(末 chunk 自推
        delta),返回 [(seqno, meta, up_flat)] 供 super() 后 embeds
        覆写消费。

        The recv uses the host gate + consume-time TP broadcast (each
        frame its own launch+wait atomic pair on the current stream —
        the prefill_only_demo_br discipline; 跨流排序不可依赖是此前
        广播段交付残缺的根因)。

        mrope positions 自 wire_version 2 起经控制面随 meta 到达(逐
        请求 [3,n] int64 字节,frombuffer 零转置直写 req_state),不再
        经数据面 aux 帧与 TP 组二次广播;纯文本请求空字节、零路径
        变化。"""
        worker = self.worker
        if worker is None:
            return []
        posted = getattr(worker, "_lwd_up_recv_futures", None)
        if not posted:
            return []
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()

        harvested = []
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        _tp = get_tp_group()
        # 端点判定 engine 感知:本 engine TP 组内轮转出哪个卡是本通信域
        # 的端点;多云 DP 下不能用云级固定端点——那会命中别的引擎首卡,
        # 本引擎被误判为非端点,把空 buffer 广播给全组注入(乱码)。
        engine_index = getattr(worker, "_lwd_engine_index", 0)
        for key in list(posted.keys()):
            edge_id, batch_seqno = key
            item = worker._lwd_up_recv_futures.pop(key, None)
            if item is None:
                continue
            future, meta = item
            _t_wait = time.monotonic()
            from vllm_ascend.distributed import lwd_wire
            from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

            _is_endpoint = lwd_wire.is_lwd_channel_endpoint(
                LwdChannelType.UP, edge_id=edge_id, cloud_id=None,
                engine_index=engine_index,
            )
            num_tokens = sum(len(t) for t in meta.token_ids)
            config = self.model_config.hf_config
            num_hash_layers = hash_layer_count(config)
            numel = hash_payload_numel(
                num_tokens, hidden_size, num_hash_layers,
                getattr(config, "num_experts_per_tok", 0),
            )
            mrope_positions = getattr(meta, "mrope_positions", None)
            if not mrope_positions:
                # None(旧批结构)与空 list(幽灵/drain 批漏填)同归回填:
                # 空列表若直接进长度核验会必崩,而 has_mrope 全 False 时
                # 空字节占位本就零路径变化;真带 mrope 的请求缺字节在
                # 下方注入处 fail-fast(边云版本错配)。
                mrope_positions = [b""] * len(meta.req_ids)
            if not (len(meta.req_ids) == len(meta.token_ids)
                    == len(meta.prompt_offsets) == len(meta.has_mrope)
                    == len(mrope_positions)):
                raise ValueError("LWD requires one prompt offset and mrope flag per request")
            # UP 第二段(TP 组内广播)按 prefill_only_demo_br 范式在消费
            # 时刻、本卡计算流上原子完成:launch + handle.wait() 同一流
            # 上下文(torch_npu 的 wait 桥接到调用时当前流),不依赖两个
            # HCCL op 之间的跨流排序——那是此前广播段交付残缺的根因。
            if _is_endpoint:
                # 端点:先过 host 就绪门(demo 的 wait gate):done_event
                # 轮询通过即 P2P 数据已完整落 buffer;超时显式报错而非
                # 静默乱码。门通过后再广播,广播读到的必然是完整数据。
                res = future.wait(timeout=60.0)
                up_flat = res.tensor
                if up_flat is None:
                    continue
                # 跨流生命周期登记(收割时刻):端点 recv buffer 由通道流
                # 分配(_wire_recv 在通道流上下文 torch.empty),收割后的
                # 消费(TP 广播读 + 后续注入 copy)全在计算流。不登记
                # record_stream,分配器可在计算流 copy 尚未执行时把块交给
                # 下一 seqno 的 irecv 覆写——深异步队列下表现为跨 seqno
                # 串数据(预测开头重复/乱码)。登记后复用即被正确排序。
                # 非端点 rank 的 torch.empty 在计算流上下文分配(同流
                # 使用),无此问题,不用登记。
                up_flat.record_stream(torch.npu.current_stream())
            else:
                # 非端点:现场分配接收 buffer,等端点广播转发。
                up_flat = torch.empty(
                    numel, dtype=torch.bfloat16, device="npu"
                )
            if _tp.world_size > 1:
                # 端点随 edge_id 在本引擎 C 卡内轮转:广播源 = 本通信域
                # 的云端点 rank,而非固定 TP0(不同边/引擎对应不同端点
                # 卡,见 lwd_wire 建组)。
                _, cloud_endpoint = lwd_wire.get_lwd_channel_endpoint_rank(
                    edge_id=edge_id, cloud_id=None, engine_index=engine_index,
                )
                work = dist.broadcast(
                    up_flat,
                    src=cloud_endpoint,
                    group=_tp.device_group,
                    async_op=True,
                )
                work.wait()  # 桥接广播完成到当前(计算)流,后续 copy 有序
            logger.info(
                "[Lwd][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
                batch_seqno, (time.monotonic() - _t_wait) * 1000,
            )
            logger.info(
                "[Lwd][cloud-runner] UP embeds harvested seqno=%s rows=%d "
                "reqs=%s mrope_reqs=%s",
                batch_seqno, num_tokens, meta.req_ids,
                sum(meta.has_mrope),
            )
            # 窗口核验 + mrope 行写入(当步生效的前提:super() 内的
            # _calc_mrope_positions 随即按窗读 req_state)
            ghost_ids = set(getattr(meta, "ghost_req_ids", None) or [])
            mrope_rows = 0
            _mrope_cpu_ms = _mrope_delta_ms = 0.0
            for k, (req_id, token_ids) in enumerate(
                zip(meta.req_ids, meta.token_ids)
            ):
                n = len(token_ids)
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    if req_id in ghost_ids:
                        # 幽灵条目(请求已 abort,行在载荷里不可撤):
                        # 照收不注入——跳过全部核验与状态写入;
                        # positions 字节随 meta 条目一并跳过(aux 帧
                        # 已退役,无游标推进义务)
                        continue
                    # fail-fast:embeds 已收齐但请求不在本步 batch——继续
                    # 走下去该请求将用未注入的脏 embeds 解码(静默乱码),
                    # 必须当场暴露而非丢弃。出现即调度/数据面时序契约
                    # 被破坏,需要修的是上游而不是这里。
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    # fail-fast:本 chunk 行数必须恰等于该请求本步调度 token
                    # 数。不等(调度窗口 ≠ 边侧 chunk,如预算挤压截断)时
                    # 注入会越窗踩邻请求行/留下未注入尾巴——静默乱码源,
                    # 直接暴露。
                    scheduled = (
                        int(num_scheduled_tokens[idx])
                        if idx < len(num_scheduled_tokens) else 0
                    )
                    if n != scheduled:
                        raise RuntimeError(
                            f"[Lwd][cloud-runner] INJECT window mismatch "
                            f"req={req_id} seqno={batch_seqno}: chunk rows="
                            f"{n} != scheduled={scheduled} (computed="
                            f"{int(computed[idx])} prompt="
                            f"{int(num_prompt[idx])})"
                        )
                    start = int(meta.prompt_offsets[k])
                    prompt_len = int(num_prompt[idx])
                    if start != int(computed[idx]) or not 0 <= start <= start + n <= prompt_len:
                        raise ValueError(
                            f"LWD prompt chunk offset mismatch for {req_id}: "
                            f"start={start}, computed={int(computed[idx])}, rows={n}, prompt={prompt_len}"
                        )
                    if meta.has_mrope[k]:
                        # positions 字节随 meta 经控制面到达([3,n] int64),
                        # frombuffer 成 CPU 张量零转置直写 req_state;
                        # 空字节 = 边云版本错配,fail-fast
                        payload = mrope_positions[k]
                        if not payload:
                            raise RuntimeError(
                                f"[Lwd][cloud-runner] mrope payload missing "
                                f"req={req_id} seqno={batch_seqno}: "
                                f"has_mrope=True 但控制面 positions 为空"
                                f"(边云版本错配,须同版本部署)"
                            )
                        cpu_ms, delta_ms = self._lwd_inject_mrope_positions(
                            req_id, start, n,
                            torch.frombuffer(
                                bytearray(payload), dtype=torch.int64
                            ).view(3, n),
                            prompt_len,
                        )
                        _mrope_cpu_ms += cpu_ms
                        _mrope_delta_ms += delta_ms
                        mrope_rows += n
            if mrope_rows > 0:
                # [Lwd][perf] mrope 注入探针:cpu=positions 字节 frombuffer
                # +CPU 直写 req_state(控制面送达,无设备往返);delta=末
                # chunk max() 自推,逐请求累加
                logger.info(
                    "[Lwd][perf] cloud mrope-inject seqno=%s rows=%d "
                    "cpu=%.2f delta=%.2fms",
                    batch_seqno, mrope_rows,
                    _mrope_cpu_ms, _mrope_delta_ms,
                )
            harvested.append((batch_seqno, meta, up_flat))
        return harvested

    def _lwd_inject_remote_embeds(self, harvested, num_scheduled_tokens) -> None:
        """Device-side inject(super() 之后):把收割到的 UP embeds 直写
        ``inputs_embeds.gpu`` 各请求的当步调度窗口。

        The flattened output offsets reproduce the native fill loop's
        accumulation (per-request scheduled segment start), so rows land
        exactly where the prompt-embeds branch expects them.  The CPU
        draft first-pass provider reads the SAME window straight from
        ``inputs_embeds.gpu`` (llm_base_proposer), so the CPU assembly
        buffer is no longer written at all — the per-chunk 84MB D2H is
        gone.  The base fill loop is skipped for LWD placeholder rows
        (see ``_prepare_inputs``); our device write is authoritative.
        """
        if not harvested:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        # Flattened output offset per request (native fill loop 同款累计):
        # 每个请求的调度段在扁平 token 序列中的起点。
        out_offset: dict[str, int] = {}
        off = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            out_offset[req_id] = off
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        for batch_seqno, meta, up_flat in harvested:
            num_tokens = sum(len(t) for t in meta.token_ids)
            config = self.model_config.hf_config
            num_hash_layers = hash_layer_count(config)
            expert_ids = None
            if num_hash_layers:
                embeds, expert_ids = unpack_hash_payload(
                    up_flat, num_tokens, hidden_size, num_hash_layers,
                    config.num_experts_per_tok,
                )
            else:
                embeds = up_flat.view(num_tokens, hidden_size)
            logger.info(
                "[Lwd][cloud-runner] UP embeds injected seqno=%s rows=%s "
                "reqs=%s",
                batch_seqno, embeds.shape[0], meta.req_ids,
            )
            row = 0
            ghost_ids = set(getattr(meta, "ghost_req_ids", None) or [])
            for meta_idx, (req_id, token_ids) in enumerate(zip(meta.req_ids, meta.token_ids)):
                n = len(token_ids)
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    if req_id in ghost_ids:
                        # 幽灵条目:行照收不注入(abort 在飞 chunk,
                        # 保 HCCL numel 配对);row 游标照常推进
                        row += n
                        continue
                    # 与收割期同款防线(双阶段各自独立核验)
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if n > 0:
                    start = int(meta.prompt_offsets[meta_idx])
                    prompt_len = int(num_prompt[idx])
                    if self.lwd_hash_state is not None:
                        if expert_ids is None:
                            raise ValueError(f"Missing LWD V4 expert payload for {req_id}")
                        self.lwd_hash_state.add_chunk(
                            req_id, prompt_len, start, expert_ids[row:row + n],
                        )
                    out = out_offset.get(req_id, 0)
                    # stream 上直接写 inputs_embeds.gpu 的调度窗口
                    gpu_embeds[out : out + n].copy_(
                        embeds[row : row + n], non_blocking=True
                    )
                    # 不再 D2H 写 CPU 装配缓冲:draft provider 已改为
                    # 直读 inputs_embeds.gpu 同窗口(本行即权威副本),
                    # 消除每 chunk 84MB 的 D2H 同步长尾
                    self.input_batch.is_token_ids[idx, start : start + n] = False
                row += n
            # 跨流生命周期登记:端点 recv buffer 由通道流分配与复用
            # (同尺寸 chunk 下分配器几乎总给同一块),本步 copy 在计算流。
            # 不登记 record_stream,分配器可在本批 copy 尚未执行时把块交给
            # 下一 seqno 的 irecv 覆写——深异步队列下表现为跨 seqno 串
            # 数据(多请求乱码)。登记后复用即被正确排序。
            up_flat.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _lwd_inject_mrope_positions(
        self,
        req_id: str,
        start: int,
        n: int,
        chunk_positions: torch.Tensor,
        prompt_len: int,
    ) -> tuple[float, float]:
        """写入一个 chunk 的 mrope positions([3,n] int64,CPU——控制面
        RangeItem.mrope_positions 字节 frombuffer 而来)到请求缓存的
        [3, prompt] 窗口;末 chunk 落地时自推 delta(=
        positions.max()+1-prompt_len,与边侧逐值一致)供 decode 期
        原生 _calc_mrope_positions 现算 completion 段位置。

        req_state.mrope_positions 由 ids=None 直通初始化(arange 纯文
        本位置)——多模态请求的窗口被逐 chunk 覆盖,纯文本请求永不走
        本路径(边侧不给它发 positions)。

        返回 (cpu_ms, delta_ms):CPU 直写与末 chunk delta 自推的耗时
        打点,由调用方按 seqno 聚合。"""
        req_state = self.requests.get(req_id)
        assert chunk_positions.shape == (3, n), (
            f"mrope payload shape {tuple(chunk_positions.shape)} != "
            f"(3, {n}) (req={req_id})"
        )
        _t_cpu = time.monotonic()
        req_state.mrope_positions[:, start : start + n] = chunk_positions
        cpu_ms = (time.monotonic() - _t_cpu) * 1000
        delta_ms = 0.0
        if start + n >= prompt_len:
            _t_delta = time.monotonic()
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = (
                int(positions.max().item()) + 1 - prompt_len
            )
            delta_ms = (time.monotonic() - _t_delta) * 1000
            logger.info(
                "[Lwd][cloud-runner] mrope complete: req=%s prompt=%d "
                "delta=%d",
                req_id, prompt_len, req_state.mrope_position_delta,
            )
        return cpu_ms, delta_ms

    def _sample(self, logits, spec_decode_metadata):
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        # 端点随 edge_id 轮转后,不同 edge 的 DOWN 端点落在不同云 rank,
        # 因此每卡都可能是某些 edge 的端点——采集改在全部 TP rank 上进行,
        # 发送侧再按 edge_id 过滤到对应端点(见 LwdCloudWorker.sample_tokens)。
        captured = None
        if self.execute_model_state is not None:
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
        """worker 层取走本步 DOWN payload 列表(per-edge,单槽覆盖写)。"""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self, sample_hidden_states, logits, spec_decode_metadata,
        scheduler_output, batch_req_ids, discard_mask_np, sampler_output,
    ):
        """生产级 DOWN 采集(rank-replay),按来源边拆分回传。

        混在多边 batch 里的结果按 ``wrapped_req_id`` 解析出 ``edge_id``,
        同一条边的 hidden/秩/counts/seg_lens 归到一起,返回 per-edge
        载荷列表 ``[(edge_id, hidden_packet, meta_dev, seg_lens_host,
        ready_event, req_ids)]``;单边场景自然退化为单元素列表,与原
        实现行为一致。

        廉价计算:bf16 直比(logits 原生 bf16,与 cast 后逐位等价)、
        批全覆盖时整行直接算(不做高级索引取材)、ranks/counts
        拼单个设备张量;meta 不落 pinned——设备张量本体随 payload
        传递,由 executor 输出线程在就绪事件同步后收割拼装,计算
        关键路径零新增同步。批序/掩码取捕获时刻快照(防 batch queue
        重叠重排)。
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask 来自捕获时刻快照(与 logits/hidden
        # 同批序),不读活的 input_batch
        valid = ~discard_mask_np[: len(batch_req_ids)]
        is_spec = spec_decode_metadata is not None

        # edge_id -> {rows, ranks, req_ids, counts, seg_lens_host}
        buckets: dict[int, dict[str, list]] = {}
        lg = logits  # bf16 原生,直接比较(cast 无精度增益)


        def _req_edge_id(req_id: str) -> int:
            """严格解析来源边:未包装 id(无 e<num>- 前缀)在
            parse_edge_id 里静默回退 edge 0——collect 分桶场景这是
            错路由(别边/本地流量被并入 edge 0 的 DOWN 载荷,边侧按
            行序错位交付 = 乱码),必须当场暴露而非静默归边。"""
            edge_id, raw = parse_edge_id(req_id)
            if raw == req_id:
                raise RuntimeError(
                    f"[Lwd][cloud-runner] DOWN collect got unwrapped "
                    f"req_id={req_id}: cannot determine source edge "
                    f"(cloud-reuse requires wrapped ids)"
                )
            return edge_id

        def _bucket(edge_id: int) -> dict[str, list]:
            b = buckets.get(edge_id)
            if b is None:
                b = {"rows": [], "ranks": [], "req_ids": [],
                     "counts": [], "seg_lens_host": []}
                buckets[edge_id] = b
            return b

        if not is_spec:
            idx = [i for i in range(len(batch_req_ids)) if valid[i]]
            if not idx:
                return None
            full_cover = all(valid[: len(batch_req_ids)])
            lg_sel = lg if full_cover else lg[idx]
            sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
            # 与边侧稳定排序一致：并列候选按 token ID 升序编码为唯一排名。
            ranks_all = self._lwd_token_rank_encoder.encode(lg_sel, sm_sel)
            for k, i in enumerate(idx):
                edge_id = _req_edge_id(batch_req_ids[i])
                b = _bucket(edge_id)
                b["rows"].append(sample_hidden_states[i : i + 1])
                b["ranks"].append(ranks_all[k : k + 1])
                b["req_ids"].append(batch_req_ids[i])
                # counts 恒 1:打包段按边一次 torch.ones(n),不逐请求发射
        else:
            # 全零同步 spec 路径(向量化 × per-edge 分桶):
            # 段长取 spec_decode_metadata.num_draft_tokens(host list,
            # 运行期真实布局,与 sample_hidden_states 段结构一致);
            # counts 用框架每步已算好的 num_accepted_tokens(纯 device);
            # host 一趟建索引,替代逐请求 device 循环(原实现每请求
            # ~6 次小 kernel 发射,rank 段随并发涨到 20+ms;向量化后
            # 发射数只随边数,与请求数无关)。索引语义与原循环逐位
            # 等价:vidx/vlen = 有效请求批位/实际行数(按总段长封顶);
            # req_idx/pos_idx = 每行的请求位/行内位;row_sel = 有效行
            # 全局行号。过期 sampled 位一律按实际 hidden 行数封顶。
            seg_lens = [
                d + 1 for d in spec_decode_metadata.num_draft_tokens
            ]
            counts_all = (sampled != -1).sum(dim=1)
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
            # 按来源边分桶(D2 回传拆分):每条边独立索引/秩/hidden 包,
            # 每边一次 pinned 异步 H2D(pageable 同步 H2D 会在主流上
            # 等设备排空,见 seg_lens 同款教训)
            edge_per_req = np.array(
                [_req_edge_id(batch_req_ids[i]) for i in vidx_np]
            )
            edge_ids_sorted = sorted(set(edge_per_req.tolist()))
            for edge_id in edge_ids_sorted:
                sel_req = np.nonzero(edge_per_req == edge_id)[0]
                vidx_e = vidx_np[sel_req]
                vlen_e = vlen_np[sel_req]
                starts_e = starts_v[sel_req]
                n_rows = int(vlen_e.sum())
                req_idx_np = np.repeat(vidx_e, vlen_e)
                pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(
                    np.cumsum(vlen_e) - vlen_e, vlen_e
                )
                row_sel_np = np.repeat(starts_e, vlen_e) + pos_idx_np
                buf = self._lwd_h2d_stage_i64(
                    np.concatenate(
                        [req_idx_np, pos_idx_np, row_sel_np, vidx_e, vlen_e]
                    ),
                    # 每条边占一槽,深度随单步边数放大,
                    # 防同步内环槽复用覆写尚在飞行的一步
                    min_depth=len(edge_ids_sorted) * 4,
                )
                n_valid = vidx_e.size
                req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
                sel_d = buf[2 * n_rows : 3 * n_rows]
                vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
                vlen_d = buf[3 * n_rows + n_valid :]
                sampled_rows = sampled[req_d, pos_d]
                lg_rows = lg[sel_d]
                hidden_e = sample_hidden_states[sel_d]
                # 投机与非投机共用排名规则；拒绝后的 -1 行只保留占位，不交付。
                rank = self._lwd_token_rank_encoder.encode(lg_rows, sampled_rows)
                count = torch.minimum(counts_all[vidx_d], vlen_d).to(
                    torch.int32
                )
                b = _bucket(edge_id)
                b["rows"].append(hidden_e)
                b["ranks"].append(rank)
                b["req_ids"].extend(batch_req_ids[i] for i in vidx_e)
                b["counts"].append(count)
                # seg_lens 不上设备(torch.tensor(list, device=...) 是
                # 同步 H2D):host 侧列表随 payload 携带,输出线程收割
                # 时拼装到 meta 尾部(见打包段)。
                b["seg_lens_host"].extend(vlen_e.tolist())

        if not buckets:
            return None

        _t_pack = time.monotonic()
        payloads = []
        for edge_id in sorted(buckets):
            b = buckets[edge_id]
            hidden_packet = torch.cat(b["rows"])
            if is_spec:
                counts_dev = torch.cat(b["counts"]).to(torch.int32)
            else:
                # 非 spec counts 恒 1:一次成型,不逐请求攒小 kernel
                counts_dev = torch.ones(
                    len(b["req_ids"]), dtype=torch.int32, device=hidden_packet.device
                )
            meta_dev = torch.cat(b["ranks"] + [counts_dev])
            if is_spec:
                # seg_lens 不上设备:host 侧列表随 payload 携带,输出线程
                # 收割时拼装到 meta 尾部;非 spec 路径 seg_lens 在
                # meta_dev 内。线布局:[ranks|counts|seg_lens]
                seg_lens_host = b["seg_lens_host"]
            else:
                meta_dev = torch.cat([meta_dev, counts_dev])  # seg_lens≡counts=1
                seg_lens_host = None
            payloads.append(
                (edge_id, hidden_packet, meta_dev, seg_lens_host, None,
                 list(b["req_ids"]))
            )
        # meta 不随 collect 落 pinned:异步 D2H 在本平台走边流,主流
        # 就绪事件围栏不住(真机实锤:引擎读到上一代 meta 影像)。
        # payload 改携带设备张量本体,由 executor 输出线程收割拼装
        # (拷贝流 wait_event 就绪事件 + 完成事件与拷贝同流),计算
        # 关键路径零新增同步。就绪事件随 payload 传递(每步独立对象,
        # record 完成才离手;本步所有边共享同一事件,record 在全部
        # meta_dev 组装完成之后):不能写 worker 共享属性由输出线程
        # 现读——属性每步被覆写,存在"已赋值未 record"的窗口(record
        # 在流满载时阻塞可达毫秒级),sync 未 record 的事件会立即放行。
        ready_event = torch.npu.Event()
        ready_event.record()
        payloads = [
            (edge_id, hidden_packet, meta_dev, seg_lens_host,
             ready_event, req_ids)
            for edge_id, hidden_packet, meta_dev, seg_lens_host, _, req_ids
            in payloads
        ]
        # [Lwd][perf] 采集分段:rank=秩+行收集;pinned=meta 组装(拷贝
        # 在输出线程收割,本段应≈0)
        logger.info(
            "[Lwd][perf] collect edges=%d rows=%d rank=%.2f pinned=%.2f "
            "total=%.2fms",
            len(payloads),
            sum(p[1].shape[0] for p in payloads),
            (_t_pack - _t0) * 1000,
            (time.monotonic() - _t_pack) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return payloads

    def _lwd_h2d_stage_i64(self, arr: "np.ndarray", min_depth: int = 4) -> torch.Tensor:
        """int64 host 索引数组的异步 H2D 暂存(pinned 轮换 +
        non_blocking):替代 torch.tensor(arr, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(MTP 下即 draft propose
        尾巴)。环深随单步边数放大(min_depth):每条边一步占一槽,
        防同槽被下一条边/下一步覆写时异步拷贝尚未执行。槽复用安全:
        主机覆写 stage 前等该槽上一次 H2D 的完成事件(同
        lwd_edge_worker._lwd_embed_ids_h2d 的教训:
        CachingHostAllocator 管不到持有张量的显式主机覆写)。"""
        n = arr.size
        depth = max(4, min_depth)
        ring = getattr(self, "_lwd_h2d_ring", None)
        if ring is None or ring[0][0].numel() < n or len(ring) < depth:
            m = max(n, 1024)
            ring = [
                [
                    torch.empty(m, dtype=torch.int64, pin_memory=True),
                    torch.empty(m, dtype=torch.int64, device=self.device),
                    None,  # 该槽上一次 H2D 的完成事件
                ]
                for _ in range(depth)
            ]
            self._lwd_h2d_ring = ring
            self._lwd_h2d_ring_idx = 0
        idx = self._lwd_h2d_ring_idx
        self._lwd_h2d_ring_idx = (idx + 1) % len(ring)
        stage, dev, last_ev = ring[idx]
        if last_ev is not None:
            last_ev.synchronize()
        stage.numpy()[:n] = arr
        dev[:n].copy_(stage[:n], non_blocking=True)
        ev = torch.npu.Event()
        ev.record()
        ring[idx][2] = ev
        return dev[:n]
