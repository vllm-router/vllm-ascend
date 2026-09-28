# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` enables prefill_only.
"""

from __future__ import annotations

import time

import torch
from vllm.logger import logger
from vllm.v1.core.sched.output import GrammarOutput  # noqa: F401  (type)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized

from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest

# ops 须先于 model_runner 链初始化，否则 device_op 与 ops 包循环导入
#（对齐 worker.py 的导入顺序）
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.lwd_cloud.lwd_cloud_model_runner import LwdCloudModelRunner
from vllm_ascend.worker.worker import NPUWorker


class LwdCloudWorker(NPUWorker):
    """NPUWorker + LWD data-plane wiring (cloud side only).

    This subclass is selected by ``platform.py`` only when the LWD
    deployment enables LWD on a cloud process, so no role/mode checks
    are needed here — ``self.enable_lwd`` (set by NPUWorker.__init__
    from vllm_config.lwd_config) is the single switch."""

    def _init_worker_distributed_environment(self) -> None:
        """覆写原生入口(worker.py):ascend 侧并行组按 Lwd 布局构建。

        vllm 侧分组由 parallel_state.initialize_model_parallel 的 Lwd
        分支完成;ascend 侧原生 init_ascend_model_parallel 按 (dp, pp,
        pcp, tp) 均匀网格切分,表达不了非对称边云拓扑,故以
        init_lwd_ascend_model_parallel 替代(构建 MC2 等组后注入)。
        其余步骤与原生 worker.py 保持一致。"""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_lwd_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def init_device(self):
        super().init_device()
        # per-instance DOWN seqno counters (worker layer, send-time
        # alloc). Keyed by instance: each cloud instance's DOWN channel
        # has its own seqno space starting at 0. A single-instance
        # cloud process holds exactly one entry (instance 0 until the
        # control plane stamps batches).
        self._lwd_down_next_seqno: dict[int, int] = {}
        # (instance_id, seqno) -> posted UP recv futures (consumed by
        # the runner's device-side inject)
        self._lwd_up_recv_futures: dict[tuple[int, int], tuple] = {}
        # Identity of THIS cloud instance (§4.1): derived from the
        # global rank (authoritative — the rank's position inside the
        # instance-major layout IS the instance), validated against
        # --instance-parallel-start-rank when the command declares one.
        self._lwd_own_instance = self._lwd_resolve_own_instance()
        if not self.enable_lwd:
            return
        if self.use_v2_model_runner:
            # The LWD hooks live on the V1 model runner; the V2 runner
            # (separate class) lacks them entirely — fail fast at bring-up
            # instead of AttributeError at first request.
            raise RuntimeError(
                "prefill_only (LWD) data plane requires the V1 model runner; "
                "use_v2_model_runner is not supported"
            )
        # Swap in the LWD model runner BEFORE any model load/usage.
        # The cloud feeds on edge-computed prompt embeddings (no token
        # ids on its input), so the native prompt-embeds path must be
        # enabled for the runner to allocate inputs_embeds buffers.
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(
            self.vllm_config, self.device, worker=self
        )
        from vllm_ascend.distributed import lwd_wire

        lwd_wire.init_lwd_duplex_channels()
        self._register_lwd_prompt_embeds_provider()

    def _lwd_resolve_own_instance(self) -> int:
        """Instance index of this cloud worker process.

        Derived from the global rank under the §4.1 layout (edge ranks
        first; cloud instance i's block spans
        ``[E + i*D*C, E + (i+1)*D*C)``); when the launch command
        declared ``--instance-parallel-start-rank`` the two must agree —
        a mismatch means two commands collided on the same rank block
        (or the edge/cloud counts disagree), so fail fast."""
        pc = self.parallel_config
        lwd = pc.lwd_config
        d = pc.data_parallel_size
        e_total = lwd.edge_rank_count(d)
        block = d * lwd.cloud_npu_count_per_instance_dp(d)
        if self.rank < e_total or block <= 0:
            return 0
        derived = (self.rank - e_total) // block
        declared = lwd.instance_parallel_start_rank
        if declared is not None and int(declared) != derived:
            raise RuntimeError(
                f"cloud rank {self.rank} derives instance {derived} from "
                f"the §4.1 layout but the command declares "
                f"--instance-parallel-start-rank {declared}; check "
                "edge/cloud NPU counts and the start-rank assignment"
            )
        return derived

    def load_model(self):
        """加载后打实际切片:层数/首末层名/pp 切层参数,启动期即可裁决
        半模型嫌疑(全量且从 layer 0 起 = 正常;减半/起始非 0 = pp 切错)。"""
        super().load_model()
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(
            backbone, "decoder_layers", None
        )
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules()
            if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — 全量应覆盖 layer 0 起的全部层",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size, pc.tensor_parallel_size, self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output):
        # [Lwd][perf] worker 开工时刻(与引擎 rpc-enqueue ts 对减 =
        # 下达-开工延迟;各卡开工离散 = 集合对齐成本)
        logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank, time.monotonic(),
        )
        if self.enable_lwd:
            # cloud: post the exact-size UP irecv for every incoming
            # LWD_EMBED batch (control info rides scheduler_output.lwd_batch).
            self._lwd_up_post_recvs(scheduler_output)

        # cloud: finished requests trigger local cleanup (collector
        # bookkeeping + UP chunk table); finished_req_ids is the
        # engine's own liveness signal.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        get_lwd_comm_service().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] 云侧每步 exec_model 计时:total=前向全程
        # (forward 主段,sample/collect 在 sample_tokens 另有分项)
        logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output) -> None:
        """Post the UP recv for an incoming LWD_EMBED batch.

        UP 两段通信按 prefill_only_demo_br 范式拆到两个时刻:
          * post 时刻(本函数):仅端点 rank 经通道 P2P posted recv
            (尽早 posted 以匹配边侧 isend);非端点只登记 meta 占位;
          * 消费时刻(runner 注入侧):各卡在*自己的计算流上*现场发起
            TP 组内 broadcast 并紧随 work.wait()——torch_npu 下两个
            HCCL op 之间的跨流排序不可依赖(实测 post 时刻发起的广播
            抢跑 irecv、交付残缺),launch+wait 同流上下文原子对才是
            demo 验证过的可靠形态。

        All control info rides the SchedulerOutput (edge -> cloud
        control plane fills it in):batch.seqno 为边侧派发号(跨机段
        配对键),recv 尺寸 = sum(len(token_ids)) x H。"""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or not meta.req_ids:
            return
        hidden_size = self.model_config.get_hidden_size()
        num_tokens = sum(len(t) for t in meta.token_ids)
        if num_tokens <= 0:
            return

        lwd = self.parallel_config.lwd_config
        # Stamp/identity agreement: the batch's instance stamp (edge
        # scheduler's pick) must route onto THIS instance's wire pair —
        # the recv is always posted on the instance this rank actually
        # belongs to. An unstamped (legacy) batch stamps 0; a mismatched
        # non-zero stamp is an edge misroute and is surfaced loudly.
        instance_id = self._lwd_own_instance
        stamped = int(getattr(batch, "instance_id", 0) or 0)
        if stamped not in (0, instance_id):
            logger.error(
                "[Lwd][MISROUTE] batch seqno=%d stamped instance=%d but "
                "this cloud process is instance=%d — posting on own "
                "instance (check edge dispatch / instance start-ranks)",
                batch.seqno, stamped, instance_id,
            )
        from vllm_ascend.distributed import lwd_wire

        try:
            _, cloud_endpoint = lwd_wire.lwd_pair_ranks(instance_id)
        except RuntimeError:
            cloud_endpoint = lwd.edge_npu_count
        is_endpoint = self.rank == cloud_endpoint
        if is_endpoint:
            future = get_lwd_comm_service().submit_recv(
                LwdCommRequest(
                    channel=LwdChannelType.UP,
                    op="recv",
                    num_elements=num_tokens * hidden_size,
                    seqno=batch.seqno,
                    instance_id=instance_id,
                )
            )
        else:
            # 非端点:不碰跨机通道;广播接收推迟到注入时刻现场发起,
            # None 即"消费时现场广播"标记。
            future = None
        self._lwd_up_recv_futures[(instance_id, batch.seqno)] = (future, meta)
        logger.info(
            "[Lwd][cloud-worker] UP recv posted seqno=%d instance=%d "
            "reqs=%d tokens=%d endpoint=%s",
            batch.seqno, instance_id, len(meta.req_ids), num_tokens,
            is_endpoint,
        )

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN:发送 hidden 包(通道异步、流内有序);pinned 视图
        # 挂输出内层(就绪由 get_output 的 wait_stream(主流) 保证),
        # 引擎侧解码发布 meta。
        if self.enable_lwd:
            payload = self.model_runner.take_lwd_pending_down_payload()
            if payload is not None and output is not None:
                hidden, pinned, _event, req_ids, _accepted = payload
                instance_id = self._lwd_own_instance
                seqno = self._lwd_next_down_seqno(instance_id)
                logger.info(
                    "[Lwd][cloud-worker] DOWN send seqno=%d instance=%d "
                    "numel=%d",
                    seqno, instance_id, hidden.numel(),
                )
                get_lwd_comm_service().submit_send(
                    LwdCommRequest(
                        channel=LwdChannelType.DOWN,
                        op="send",
                        num_elements=hidden.numel(),
                        tensor=hidden,
                        seqno=seqno,
                        instance_id=instance_id,
                    )
                )
                # async 包装器下挂到内层,否则 get_output() 解包丢失
                target = getattr(output, "_model_runner_output", output)
                target.lwd_down_carrier = (
                    pinned, req_ids, hidden.numel(), seqno
                )
                # [Lwd][perf] 云侧每步计时:sample=采样+采集(含秩计算);
                # send=DOWN 提交(快照clone+isend+bridge wait);total=全步
                logger.info(
                    "[Lwd][perf] cloud-step seqno=%d sample=%.2f send=%.2f "
                    "total=%.2fms rows=%d",
                    seqno,
                    (_t_sample - _t0) * 1000,
                    (time.monotonic() - _t_sample) * 1000,
                    (time.monotonic() - _t0) * 1000,
                    hidden.shape[0] if hidden is not None else 0,
                )
        return output

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def _lwd_cloud_flush_finished(self, finished_req_ids) -> None:
        """Cloud side (streaming): a finished request's DOWN stream simply
        STOPS (its last step packet already went out with that step).
        No FIN packet -- request termination is signaled by the control
        plane (v2.6).  Here we drop the collector bookkeeping —
        finished_req_ids is the engine's own liveness signal, so this
        cleanup does not depend on the control plane.  Also releases the
        request's prompt-embeds assembly buffer (backstop for abort
        mid-prefill; normal prefill completion frees it in the runner's
        ``_lwd_release_consumed_prompt_embeds``)."""
        embeds_map = self.model_runner.input_batch.req_prompt_embeds
        req_id_to_index = self.model_runner.input_batch.req_id_to_index
        logger.info(
            "[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids)
        )
        for req_id in finished_req_ids:
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_remote_embeds`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str):
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(
            _lwd_prompt_embeds_provider
        )

    def _lwd_next_down_seqno(self, instance_id: int = 0) -> int:
        """Allocate the instance's next DOWN seqno (its own channel-local
        space starting at 0 — the instance axis keeps every instance's
        counter independent)."""
        seqno = self._lwd_down_next_seqno.get(int(instance_id), 0)
        self._lwd_down_next_seqno[int(instance_id)] = seqno + 1
        return seqno
