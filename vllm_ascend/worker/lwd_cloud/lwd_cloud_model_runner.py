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

        **host 暂存打开时(design v7 §5.7)**:端点不再从 recv future 直接取
        整批设备 tensor,而是向 stager 取**每个请求自己的** host buffer,把
        ``consumed .. consumed + num_scheduled_tokens[req]`` 这一段 H2D 拼进
        ``win_dev``,再广播 ``win_dev``(仍然只有一次广播)。

        搬多少**由本步调度量决定**,不由边侧 chunk 决定 —— 边云 chunk 边界
        从此无关(§5.3.2),边侧 chunk 化退化为纯粹的流量整形。

        mrope positions 自 wire_version 2 起经控制面随 meta 到达(逐
        请求 [3,n] int64 字节,frombuffer 零转置直写 req_state),不再
        经数据面 aux 帧与 TP 组二次广播;纯文本请求空字节、零路径
        变化。窗口化后 positions 必须**按窗口切**(见 §3.7):布局是
        行主序 [3, n],窗口 [c, c+w) 是每行各切 [c*8, (c+w)*8) 共 3 段
        再按行拼接,不能按连续字节偏移切。
        """
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
        _lwd_cfg = worker.parallel_config.lwd_config
        _is_endpoint = worker.rank == _lwd_cfg.edge_npu_count
        stager = getattr(worker, "_lwd_stager", None)
        for batch_seqno in list(posted.keys()):
            item = worker._lwd_up_recv_futures.pop(batch_seqno, None)
            if item is None:
                continue
            future, meta = item
            _t_wait = time.monotonic()
            num_tokens = sum(len(t) for t in meta.token_ids)
            config = self.model_config.hf_config
            numel = hash_payload_numel(
                num_tokens, hidden_size, hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            if not (len(meta.req_ids) == len(meta.token_ids)
                    == len(meta.prompt_offsets) == len(meta.has_mrope)
                    == len(meta.mrope_positions)):
                raise ValueError("LWD requires one prompt offset and mrope flag per request")
            # 本步要搬的行数(design v7 §5.3/§5.7):每个请求搬
            #     consumed .. consumed + num_scheduled_tokens[req]
            # 这一段(就是它本步要算的那几行)。比整批小得多,而且各请求
            # 的 host buffer 是独立的,所以下面每个请求各一次 H2D 拷贝。
            # 非端点 rank 没有 host 池:它们只按同样的**行数**分配接收缓冲,
            # 等端点广播过来(行数必须一致,W4)。
            if _is_endpoint and stager is not None:
                up_flat = self._lwd_stage_requests_to_device(
                    stager, meta, batch_seqno, num_scheduled_tokens,
                    hidden_size,
                )
            elif _is_endpoint:
                # 端点但暂存关着:先过 host 就绪门(demo 的 wait gate)。
                # done_event 轮询通过即 P2P 数据已完整落 buffer;超时显式
                # 报错而非静默乱码。门通过后再广播,广播读到的必然是完整数据。
                res = future.wait(timeout=60.0)
                up_flat = res.tensor
                if up_flat is None:
                    continue
            else:
                # 非端点:现场分配接收 buffer,等端点广播转发。行数按本步
                # 被排量算(与端点一致)。
                rows = 0
                for k, rid in enumerate(meta.req_ids):
                    idx = self.input_batch.req_id_to_index.get(rid)
                    if idx is not None and idx < len(num_scheduled_tokens):
                        rows += int(num_scheduled_tokens[idx])
                up_flat = torch.empty(
                    rows * hidden_size, dtype=torch.bfloat16, device="npu"
                )
            if _tp.world_size > 1:
                # demo 同款:直接用框架 TP 通信域(建组顺序由框架保证,
                # PD 分离路径已验证),不再使用自建 _LWD_EMBED_BCAST_GROUP。
                work = dist.broadcast(
                    up_flat,
                    src=_tp.ranks[0],
                    group=_tp.device_group,
                    async_op=True,
                )
                work.wait()  # 桥接广播完成到当前(计算)流,后续 copy 有序
            logger.info(
                "[Lwd][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
                batch_seqno, (time.monotonic() - _t_wait) * 1000,
            )
            logger.info(
                "[Lwd][cloud-runner] UP embeds harvested seqno=%s batch_rows=%d "
                "reqs=%s mrope_reqs=%s moved_rows=%d",
                batch_seqno, num_tokens, meta.req_ids,
                sum(meta.has_mrope), up_flat.numel() // max(hidden_size, 1),
            )
            # 行数核验 + mrope 行写入(当步生效的前提:super() 内的
            # _calc_mrope_positions 随即按窗读 req_state)
            mrope_rows = 0
            _mrope_cpu_ms = _mrope_delta_ms = 0.0
            for k, (req_id, token_ids) in enumerate(
                zip(meta.req_ids, meta.token_ids)
            ):
                n = len(token_ids)
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
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
                    # design v7 §5.7:本步要取的就是
                    #     consumed .. consumed + num_scheduled_tokens[req]
                    # 其中 consumed = num_computed − chunk_offset。这两个量
                    # 每个 rank 都有,所以各 rank 算出的区间必然相同(W1)。
                    scheduled = (
                        int(num_scheduled_tokens[idx])
                        if idx < len(num_scheduled_tokens) else 0
                    )
                    start = int(meta.prompt_offsets[k])
                    prompt_len = int(num_prompt[idx])
                    c = self._lwd_window_start(int(computed[idx]), start)
                    if not (0 <= start + c <= start + c + scheduled
                            <= prompt_len):
                        raise ValueError(
                            f"LWD prompt chunk offset mismatch for {req_id}: "
                            f"chunk_start={start}, consumed={c}, "
                            f"scheduled={scheduled}, "
                            f"computed={int(computed[idx])}, "
                            f"prompt={prompt_len}"
                        )
                    if meta.has_mrope[k]:
                        # positions 字节随 meta 经控制面到达([3,n] int64),
                        # 窗口化后按窗口切(§3.7),frombuffer 成 CPU 张量后
                        # 零转置直写 req_state;空字节 = 边云版本错配,
                        # fail-fast
                        payload = meta.mrope_positions[k]
                        if not payload:
                            raise RuntimeError(
                                f"[Lwd][cloud-runner] mrope payload missing "
                                f"req={req_id} seqno={batch_seqno}: "
                                f"has_mrope=True 但控制面 positions 为空"
                                f"(边云版本错配,须同版本部署)"
                            )
                        chunk_positions = torch.frombuffer(
                            bytearray(
                                self._lwd_slice_mrope_window(
                                    payload, n, c, scheduled
                                )
                            ),
                            dtype=torch.int64,
                        ).view(3, scheduled)
                        cpu_ms, delta_ms = self._lwd_inject_mrope_positions(
                            req_id, start + c, scheduled,
                            chunk_positions,
                            int(num_prompt[idx]),
                        )
                        _mrope_cpu_ms += cpu_ms
                        _mrope_delta_ms += delta_ms
                        mrope_rows += scheduled
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

    # ------------------------------------------------------------------ #
    # host 暂存:窗口化取数(§4.5)                                        #
    # ------------------------------------------------------------------ #
    def _lwd_window_start(self, computed_i: int, chunk_offset: int) -> int:
        """该请求在本 chunk 内**已消费**的行数(窗口起点 ``c``)。

        **必须由所有 rank 各自从 Request 状态算出同一个值**(§3.4 W1):

            c = num_computed − chunk_offset

        ``num_computed`` 与 ``chunk_offset`` 都在 SchedulerOutput / input_batch
        里、每个 rank 都有,所以各 TP rank 必然得出同一个 ``c``。**不能**从
        worker 的 stager 台账取 —— 那个池只有端点 rank 有(§4.1),非端点取到的
        永远是 0,一旦某个请求跨云步被部分消费,端点与非端点就会把同一批行
        注入到不同窗口(W3 的跨 rank 版本失守,静默错位)。

        非暂存路径 ``chunk_offsets`` 为空、``computed_i == chunk_offset``,
        于是 ``c == 0`` —— 与改造前"整批一次"逐字一致。
        """
        c = int(computed_i) - int(chunk_offset)
        if c < 0:
            raise RuntimeError(
                f"[Lwd][cloud-runner] window start negative: computed="
                f"{computed_i} < chunk_offset={chunk_offset}"
            )
        return c

    def _lwd_slice_mrope_window(
        self, payload: bytes, chunk_rows: int, c: int, w: int
    ) -> bytes:
        """按窗口切 mrope positions 原始字节(§3.7 的红框)。

        控制面载荷是 ``[3, chunk_rows]`` int64 的连续字节(行主序,3 行在外)。
        窗口 ``[c, c+w)`` **不是**一段连续字节,而是每行各切
        ``[c*8, (c+w)*8)`` 共 3 段、按行拼接:

            payload[r*chunk_rows*8 + c*8 : r*chunk_rows*8 + (c+w)*8]  r=0,1,2

        按"``c*8`` 起 ``w*24`` 长"去切是错的(会把 3 行混进一段)。
        """
        if c == 0 and w == chunk_rows:
            return payload
        stride = chunk_rows * 8
        _row_bytes = w * 8
        expected = stride * 3
        if len(payload) != expected:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope payload size {len(payload)} != "
                f"3*{chunk_rows}*8={expected} (chunk rows vs bytes mismatch)"
            )
        return b"".join(
            payload[r * stride + c * 8: r * stride + c * 8 + _row_bytes]
            for r in range(3)
        )

    def _lwd_window_device(self, numel: int) -> torch.Tensor:
        """窗口暂存设备缓冲(§4.5):``budget × H`` 大小,终身复用。

        一次性按调度上限分配,避免每步 torch.empty;跨批复用必须
        record_stream(否则分配器会把块交给下一批,表现为跨 seqno 串数据)。
        """
        buf = getattr(self, "_lwd_win_dev", None)
        if buf is None or buf.numel() < numel:
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
            buf = torch.empty(
                max(cap, numel), dtype=torch.bfloat16, device=self.device
            )
            self._lwd_win_dev = buf
            logger.info(
                "[Lwd][host-stage] win_dev allocated rows_capacity=%d "
                "(%.1fMiB)",
                buf.numel() // max(self.model_config.get_hidden_size(), 1),
                buf.numel() * 2 / 2 ** 20,
            )
        return buf[:numel]

    def _lwd_stage_requests_to_device(
        self, stager, meta, seqno, num_scheduled_tokens, hidden_size,
    ) -> torch.Tensor:
        """把本步每个请求要算的那几行从 host 搬回设备(design v7 §5.7)。

        每请求搬 ``consumed .. consumed + num_scheduled_tokens[req]`` 这一段
        —— 正好是它本步要算的行。**各请求独立**,因为它们的 host buffer 本来
        就各占一块(§4.1),所以是 P 次拷贝而不是一次。

        **必须异步**(non_blocking=True):源是 pinned 内存,异步拷贝安全;
        同步拷贝会让 CPU 阻塞到每次 PCIe 往返完成,每个窗口制造一个
        CPU-NPU 串行气泡,并发下每步前向白白多出十几 ms(真机实录 ITL
        抬升)。异步化后拷贝在计算流上流水线化,CPU 不停。

        落点 = ``prefix_sum(本步行数)``:所有 TP rank 都用同一份
        ``num_scheduled_tokens`` 算,所以段布局必然一致(W4)。

        拷贝落在**当前(计算)流**上、紧接随后的 broadcast 之前 —— 同流
        launch 序保证广播读到的是搬完的数据,不引入跨流排序。

        源 buffer 的存活由 stager 保证:per-request buffer 活到请求结束
        (§6.1),扩容 realloc 的旧块也延迟到那时才释放(见
        ``LwdReqEmbed._retired``),异步拷贝飞行期间不会碰到已释放内存。

        搬完即推进 ``consumed``(§4.2):"搬回设备"就是"本步要算",两者同步
        推进(指针推进只是 CPU 侧记账,已入流的拷贝不受影响);真正的释放
        仍然等到请求结束(§6.1)。
        """
        dev = self._lwd_window_device(
            sum(
                int(num_scheduled_tokens[self.input_batch.req_id_to_index[rid]])
                for rid in meta.req_ids
                if rid in self.input_batch.req_id_to_index
            ) * hidden_size
        )
        offset = 0
        num_computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        for rid in meta.req_ids:
            idx = self.input_batch.req_id_to_index.get(rid)
            if idx is None:
                continue
            n_rows = int(num_scheduled_tokens[idx])
            if n_rows <= 0:
                continue
            # ---- 内容与位置必须对齐(此前从未校验,踩过静默精度事故)----
            #
            # 注入侧把本批行写到 prompt 位置 ``start + c``,其中
            # ``c = num_computed − start``,所以落点**恒等于** ``num_computed``;
            # 而这里从 ``consumed`` 顺序取数。两者只有相等时,"读到的内容"
            # 才正是"要写到的位置"要的内容。一旦错开(例如同一批被搬了两次、
            # 或某个 chunk 的窗口起点算错),模型会拿到错位的 embedding ——
            # 表现是输出**被截断/空白**,不报错、不乱码,极难定位。
            # 所以在这里把恒等式变成硬校验,宁可当场炸。
            consumed_before = stager.consumed_rows(rid)
            expect = int(num_computed[idx]) if idx < len(num_computed) else 0
            if consumed_before != expect:
                raise RuntimeError(
                    f"[Lwd][cloud-runner] STAGE POSITION MISMATCH req={rid} "
                    f"seqno={seqno}: host read pointer consumed="
                    f"{consumed_before} != computed={expect} "
                    f"(rows={n_rows}); the embeddings would be injected at a "
                    f"position that does not match the data that was read"
                )
            host_start, numel, _tail = stager.segment_for(rid, n_rows)
            if idx < len(num_prompt):
                room = int(num_prompt[idx]) - consumed_before
                if room > 0 and n_rows > room:
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] STAGE OVERRUN req={rid} "
                        f"seqno={seqno}: staging {n_rows} rows at "
                        f"consumed={consumed_before} but the prompt is only "
                        f"{int(num_prompt[idx])} rows"
                    )
            dev[offset: offset + numel].copy_(
                stager.take_req(rid).host[host_start: host_start + numel],
                non_blocking=True,
            )
            offset += numel
        dev.record_stream(torch.npu.current_stream())
        for rid in meta.req_ids:
            idx = self.input_batch.req_id_to_index.get(rid)
            if idx is None:
                continue
            n_rows = int(num_scheduled_tokens[idx])
            if n_rows > 0:
                stager.consume(rid, n_rows)
        logger.info(
            "[Lwd][cloud-runner] CONSUME seqno=%s rows=%d segs=%d "
            "host_MB=%.1f held=%d",
            seqno, offset // max(hidden_size, 1), len(meta.req_ids),
            stager.host_bytes / 2 ** 20, stager.request_count,
        )
        return dev

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

        **按请求取数(design v7 §5.7)**:``up_flat`` 是本步逐请求行**紧邻
        拼接**(``meta.req_ids`` 序),只含 embeds;每请求的起点 c =
        ``num_computed − chunk_offset``,落点 = ``prompt_offsets[k] + c``
        (W1,每个 rank 各自算,结果必然相同)。专家 ID(DSv4)不再上设备,
        直接按同一行区间从**该请求自己的** host buffer 尾段切出来记账
        (§4.6)——省掉"尾段上设备再 D2H 回来"那次往返。
        """
        if not harvested:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        # 窗口起点由 computed − chunk_offset 在**每个 rank 上各自**算出
        # (W1:所有 TP rank 必须得到同一个 c;stager 台账只有端点有,不能用)。
        num_computed = self.input_batch.num_computed_tokens_cpu
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        # Flattened output offset per request (native fill loop 同款累计):
        # 每个请求的调度段在扁平 token 序列中的起点。
        out_offset: dict[str, int] = {}
        off = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            out_offset[req_id] = off
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        stager = self.worker._lwd_stager if self.worker is not None else None
        for batch_seqno, meta, up_flat in harvested:
            num_tokens = sum(len(t) for t in meta.token_ids)
            # 本步逐请求行数 = 该请求本步被调度的行数(design v7 §5.3)。
            # 非暂存路径(开关关着)整批一次搬,行数就是它自己的 chunk 行数。
            rows_for_req: dict[str, int] = {}
            for k, rid in enumerate(meta.req_ids):
                idx = self.input_batch.req_id_to_index.get(rid)
                if idx is None:
                    continue
                rows_for_req[rid] = (
                    int(num_scheduled_tokens[idx])
                    if idx < len(num_scheduled_tokens) else 0
                )
            expert_tail = None
            expert_is_host_tail = False
            if num_hash_layers and stager is not None:
                # 暂存路径:尾段在 host 上是权威副本,**按请求**切片
                # (§4.6),不再上设备、也不再需要整批的尾段偏移。
                expert_is_host_tail = True
            elif num_hash_layers:
                # 非暂存路径(开关关闭):整批设备载荷,尾段由原
                # ``unpack_hash_payload`` 一次拆出 —— 与改造前逐字等价。
                embeds_dev, expert_ids_all = unpack_hash_payload(
                    up_flat, num_tokens, hidden_size, num_hash_layers,
                    config.num_experts_per_tok,
                )
                up_flat = embeds_dev.reshape(-1)
                expert_tail = expert_ids_all
            embeds = up_flat.view(-1, hidden_size)
            # 这两个字典合起来就是"每个请求写到哪、写了多少":
            #   rows    = 本步行数(w)
            #   at      = 注入的 prompt 起点(start + c,恒等于 num_computed)
            # 精度问题(截断/空白)只能来自这两者与实际数据不一致,所以把它们
            # 并列打出来,不必再从别的日志反推。
            logger.info(
                "[Lwd][cloud-runner] UP embeds injected seqno=%s moved_rows=%s "
                "reqs=%s rows=%s at=%s",
                batch_seqno, embeds.shape[0], meta.req_ids, rows_for_req,
                {
                    rid: (
                        int(meta.prompt_offsets[k])
                        + self._lwd_window_start(
                            int(num_computed[self.input_batch.req_id_to_index[rid]]),
                            int(meta.prompt_offsets[k]),
                        )
                        if rid in self.input_batch.req_id_to_index
                        else -1
                    )
                    for k, rid in enumerate(meta.req_ids)
                },
            )
            row = 0
            for meta_idx, (req_id, token_ids) in enumerate(zip(meta.req_ids, meta.token_ids)):
                n = len(token_ids)
                w = int(rows_for_req.get(req_id, 0))
                if w <= 0:
                    continue
                idx = self.input_batch.req_id_to_index.get(req_id)
                if idx is None:
                    # 与收割期同款防线(双阶段各自独立核验)
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                        f"{batch_seqno} rows={n}: req not in input_batch "
                        f"(batch={self.input_batch.req_ids})"
                    )
                if w > 0:
                    start = int(meta.prompt_offsets[meta_idx])
                    c = self._lwd_window_start(
                        int(num_computed[idx]), start,
                    )
                    # ---- W2 的落点不在这里 ----
                    # 曾经在这里用 `recv_rows - consumed` 再核验一遍,结果第一批
                    # 就被误判越界(真机实录:recv_rows=15 consumed=15 却报
                    # "only 0 have arrived")。原因与调度器侧那处完全相同:
                    # ``consumed`` 是 host 侧的**读指针**,而携带它的那次取数
                    # (`_lwd_stage_requests_to_device` → `stager.segment_for`)
                    # 已经在本步 harvest 阶段**先行推进过**它,所以注入侧看到的
                    # 必然是"已经搬完"的值,减法恒为 0。
                    #
                    # W2(下发量 ≤ 已写入量)由 stager 在做取数时用**推进前**的
                    # 指针权威判定:`LwdReqEmbed.read_range` 读到未写入区间即抛
                    # `LwdHostStageGeometryError`,所以这里不再重复。
                    # 调度器侧另有一道"下发量 ≤ 已到达量"的核验(数据面事实)。
                    if self.lwd_hash_state is not None:
                        if expert_tail is None:
                            raise ValueError(f"Missing LWD V4 expert payload for {req_id}")
                        # 专家 ID 按**同一窗口**推进(§4.6)。
                        # 两条路径的尾段落点不同,此处显式区分:
                        #  * 暂存路径:host 上是逐请求拼接的 int32 尾段,
                        #    该请求起点由行表给出,窗口取 [c, c+w) 行;
                        #  * 非暂存路径:`unpack_hash_payload` 已把它解成
                        #    [total_rows, L, K] int32,直接按行号切。
                        top_k = config.num_experts_per_tok
                        if expert_is_host_tail:
                            # 暂存路径:专家 ID 在**该请求自己的** host buffer
                            # 尾段里(design v7 §4.1:每请求独立存储),所以
                            # 起点由那个请求自己的 tail_range 给出 —— 不再需要
                            # "整批布局里第几个请求"这种推导(它正是拆存储要
                            # 消灭的东西)。
                            staged_req = stager.take_req(req_id)
                            if staged_req is None:
                                raise ValueError(
                                    f"[Lwd][cloud-runner] expert tail for "
                                    f"{req_id} unavailable: not staged"
                                )
                            t_start, _t_end = staged_req.tail_range(w)
                            seg = staged_req.host[t_start: t_start + w * 4
                                                  * num_hash_layers * top_k]
                            ids = self._lwd_decode_expert_ids(
                                seg, w, num_hash_layers, top_k,
                            )
                        else:
                            seg = expert_tail[row: row + w]
                            ids = seg.reshape(w, num_hash_layers, top_k)
                        self.lwd_hash_state.add_chunk(
                            req_id, int(num_prompt[idx]), start + c, ids,
                        )
                    out = out_offset.get(req_id, 0)
                    # stream 上直接写 inputs_embeds.gpu 的调度窗口
                    gpu_embeds[out : out + w].copy_(
                        embeds[row : row + w], non_blocking=True
                    )
                    # 不再 D2H 写 CPU 装配缓冲:draft provider 已改为
                    # 直读 inputs_embeds.gpu 同窗口(本行即权威副本),
                    # 消除每 chunk 84MB 的 D2H 同步长尾
                    self.input_batch.is_token_ids[idx, start + c : start + c + w] = False
                row += w
            # 跨流生命周期登记:窗口缓冲由计算流写、下一批复用,必须登记
            # (不登记 = 跨 seqno 串数据,与端点 buffer 同款事故);
            # 端点整批 buffer 那处登记在收割期完成。
            up_flat.record_stream(torch.npu.current_stream())

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
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = (
                int(positions.max().item()) + 1 - prompt_len
            )
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
        批全覆盖时整行直接算(不做高级索引取材)、ranks/counts/seg_lens
        拼单个设备张量;meta 经 pinned(4 轮换)在主流末尾异步拷贝,
        worker 属性上紧随记录就绪事件,由响应入队处 synchronize 后
        引擎侧解码——计算关键路径零新增同步。批序/掩码取捕获时刻
        快照(防 batch queue 重叠重排)。返回:
          (hidden_packet, pinned_view, None, req_ids, None)
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
            # seg_lens 不上设备(host 直写 pinned 尾段):与 D2H 前缀
            # 区域不相交,且引擎读 pinned 发生在 ready_event 同步
            # (更晚)之后,无竞态。
            seg_lens_host = seg_lens_list
        else:
            counts_dev = torch.ones(n_req, dtype=torch.int32,
                                    device=logits.device)
            seg_lens_dev = counts_dev
            meta_dev = torch.cat(ranks_list + [counts_dev, seg_lens_dev])
        # pinned 拷贝排主流末尾(异步),紧随记录就绪事件;
        # 由 worker 响应入队处在发送前 synchronize——
        # "响应发出 ⟹ pinned 就绪"成为硬保证(同步在输出线程,
        # 不在计算关键路径)。
        _t_pack = time.monotonic()
        n_dev = meta_dev.numel()
        n_meta = n_dev + (len(seg_lens_host) if seg_lens_host is not None else 0)
        pinned = self._lwd_meta_pinned(n_meta)
        pinned[:n_dev].copy_(meta_dev, non_blocking=True)
        if seg_lens_host is not None:
            pinned.numpy()[n_dev:n_meta] = seg_lens_host
        self.worker._lwd_meta_ready_event = torch.npu.Event()
        self.worker._lwd_meta_ready_event.record()
        # [Lwd][perf] 采集分段:rank=秩+行收集;pinned=meta 拼包+拷贝
        logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f pinned=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (_t_pack - _t0) * 1000,
            (time.monotonic() - _t_pack) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            pinned[:n_meta],
            None,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
            None,
        )

    def _lwd_meta_pinned(self, n: int):
        """轮换 pinned 缓冲(深度 4 > batch_queue 深度 2 + 引擎滞后 1):
        避免下一/N+2 步的边流拷贝覆盖引擎尚未读完的上一步 meta。"""
        ring = getattr(self, "_lwd_pinned_ring", None)
        if ring is None or ring[0].numel() < n:
            ring = [
                torch.empty(max(n, 4096), dtype=torch.int32, pin_memory=True)
                for _ in range(4)
            ]
            self._lwd_pinned_ring = ring
            self._lwd_pinned_ring_idx = 0
        idx = self._lwd_pinned_ring_idx
        self._lwd_pinned_ring_idx = (idx + 1) % len(ring)
        return ring[idx]

    def _lwd_h2d_stage_i64(self, arr: "np.ndarray") -> torch.Tensor:
        """int64 host 索引数组的异步 H2D 暂存(pinned 4 槽轮换 +
        non_blocking):替代 torch.tensor(arr, device=...) 的 pageable
        同步拷贝——后者在主流上等设备排空(MTP 下即 draft propose
        尾巴)。环深与 meta ring 同理:覆盖 batch_queue 深度 + 引擎滞后,
        防止下步 host 覆写时上步拷贝尚未执行。"""
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
