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

import os
import threading
import time
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.v1.core.sched.output import GrammarOutput  # noqa: F401  (type)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest

# ops 须先于 model_runner 链初始化，否则 device_op 与 ops 包循环导入
#（对齐 worker.py 的导入顺序）
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.lwd_cloud.lwd_cloud_model_runner import LwdCloudModelRunner
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel
from vllm_ascend.worker.worker import NPUWorker

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue


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
        # Payload scramble: bind the two static factors once; the per-flight
        # seed is derived by the comm layer from each request's own fields.
        self._build_payload_scrambler()
        # per-edge DOWN seqno counter (worker layer, send-time alloc);
        # 多边各边独立 seqno 流。
        self._lwd_down_next_seqno: dict[int, int] = {}
        # (edge_id, seqno) -> (posted UP recv future, batch meta)
        self._lwd_up_recv_futures: dict[tuple[int, int], tuple] = {}
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
        # [Lwd][pre-recv] 提前收:本 rank 对哪些边是 UP 端点(端点随边轮转,
        # 一个云 rank 可能是边 A 的端点、边 B 的非端点)。只有端点 rank 持
        # 跨机 P2P 通道、才提前挂真实 irecv 并上报收完水位;非端点 rank 的
        # 数据在注入时刻由 TP 组内广播取到,既不挂 recv 也不上报。
        # hint MQ 由执行器在建 worker 之前逐边建好并写进环境变量;done MQ
        # 由本处逐边建 writer,handle 随 READY 握手回传引擎。
        self._init_lwd_early_recv_state()
        _, _, _self_cloud = lwd_wire.self_ids()
        self._lwd_early_recv_edges: list[int] = [
            edge_id
            for edge_id, cloud_id in lwd_wire.lwd_pairs()
            if cloud_id == _self_cloud
            and lwd_wire.is_lwd_channel_endpoint(
                LwdChannelType.UP, edge_id=edge_id, cloud_id=None)
        ]
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        # 每个云 rank 都打一行:谁该起通信线程一目了然
        logger.info(
            "[Lwd][pre-recv] worker rank=%d endpoint_edges=%s hidden=%d",
            self.rank, self._lwd_early_recv_edges, self._lwd_up_hidden_size,
        )
        if self._lwd_early_recv_edges:
            self._start_lwd_early_recv_thread()
        self._register_lwd_prompt_embeds_provider()

    def _build_payload_scrambler(self) -> None:
        """Initialize the process-wide scrambler when enabled (else no-op).

        Opt-in: a missing ``payload_scramble`` block, or one without
        ``enable: true``, leaves scramble off -- no error.  With ``enable:
        true`` the two static factors are required; a missing one fails fast
        rather than silently deriving a seed that differs from the peer's.
        """
        if not self.enable_lwd:
            return
        scramble_cfg = get_ascend_config().payload_scramble
        if not scramble_cfg.get("enable", False):
            return
        from vllm_ascend.distributed.lwd_comm.payload_scramble import (
            get_payload_scrambler,
        )

        get_payload_scrambler(
            device_id=scramble_cfg.get("device_id"),
            custom_factor=scramble_cfg.get("custom_factor"),
        )

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
        """Bind the UP recv for an incoming LWD_EMBED batch.

        UP 两段通信按 prefill_only_demo_br 范式拆到两个时刻:
          * post 时刻(本函数):仅端点 rank 经通道 P2P 挂 recv;非端点只登记
            meta 占位;
          * 消费时刻(runner 注入侧):各卡在*自己的计算流上*现场发起 TP 组内
            broadcast 并紧随 work.wait()——torch_npu 下两个 HCCL op 之间的跨流
            排序不可依赖(实测 post 时刻发起的广播抢跑 irecv、交付残缺),
            launch+wait 同流上下文原子对才是 demo 验证过的可靠形态。

        提前收(early recv):端点那条 irecv **已由通信线程按引擎提示提前挂好**
        (RangeNotify 到达即挂,不等本 SO),本处只是取用那条 future;闸门
        (``_lwd_pre_recv_ready``)保证放行时它一定已挂出。提示万一没到,本处用
        同一份信息现场补挂(见 ``get_or_post_early_recv``),不会漏挂。

        All control info rides the SchedulerOutput (edge -> cloud
        control plane fills it in):batch.seqno 为边侧派发号(跨机段
        配对键,按 (edge_id, seqno) 分域),recv 尺寸包含 embedding 及
        可选的 Hash 专家 ID。"""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or not meta.req_ids:
            return
        num_tokens = sum(len(t) for t in meta.token_ids)
        if num_tokens <= 0:
            return
        edge_id = meta.edge_id
        if edge_id in self._lwd_early_recv_edges:
            # 端点:取通信线程提前挂好的 recv(没挂好就现场补挂)
            future = self.get_or_post_early_recv(edge_id, batch.seqno,
                                                 num_tokens)
        else:
            # 非端点:不碰跨机通道;广播接收推迟到注入时刻现场发起,
            # None 即"消费时现场广播"标记。
            future = None
        # per-(edge, seqno) 分桶:多边 seqno 各自独立从 0 起,不能按 seqno 合并。
        self._lwd_up_recv_futures[(edge_id, batch.seqno)] = (future, meta)
        logger.debug(
            "[Lwd][pre-recv] POST edge=%d seqno=%d reqs=%d tokens=%d "
            "endpoint=%s ts=%.3f",
            edge_id, batch.seqno, len(meta.req_ids), num_tokens,
            future is not None, time.monotonic(),
        )

    # ------------------------------------------------------------------ #
    # 提前收:通信线程(提示下发 -> 挂 recv -> 上报水位)                     #
    # ------------------------------------------------------------------ #
    #
    # 锁纪律不变量(0826 四环死锁教训,迁移时修正源版瑕疵):
    # ``_early_recv_lock`` 锁内**只做 dict/set 操作**;``submit_recv`` 一律
    # 锁外。锁外 submit 与消费点补挂的竞态由 POSTING 状态解决(见
    # ``_early_recv_pump`` / ``get_or_post_early_recv``):同键绝不允许两条
    # irecv 同时在挂——同一通道上两条 irecv 会抢发送端那一条 isend。

    def _init_lwd_early_recv_state(self) -> None:
        """提前收收单(与引擎下发的提示、上报的水位同一套键)。

        key = (edge_id, seqno)(channel 恒 UP 不进键)。条目生命周期:提示
        到达即 submit_recv 并登记;消费点 ``get_or_post_early_recv`` 只是
        **取用并标记已消费**(不 pop:上报线程还要继续轮询它推进水位);
        "既已上报又已消费"才移除。

        **在飞上限(本 rank 全局池 G)**:一条已挂 recv 占一块整批大小的
        接收 buffer;与边侧 pending 闸门同源 —— ``LwdEdgeScheduler``
        ``_LWD_MAX_PENDING_CHUNKS``(按 cloud_id 分域,默认 2)把每边
        领先量封顶,本 rank 最坏在飞 = 本 rank 服务的边数 × cap。上限之外
        的提示先排队(``_early_recv_pending``,按到达序,只存字段不占显存),
        前面的批随 SO 被消费腾出名额后立刻补挂。归还点必须是"随 SO 被
        消费"(=已下发)而不是"收完":闸门放行第 N 批的条件是水位 ≥ N,
        放行后 execute_model 立刻要取这条 future;名额若拖到收完才还,
        N+1 的 recv 可能还没挂上,闸门就会跑到 recv 前面。
        """
        self._early_recv_max_inflight = int(
            os.environ.get("VLLM_ASCEND_LWD_EARLY_RECV_MAX_INFLIGHT", "4"))
        self._early_recv_futures: dict[tuple[int, int], LwdCommFuture] = {}
        # key -> Event:泵已取出提示、正在锁外 submit(POSTING 状态)
        self._early_recv_posting: dict[tuple[int, int], threading.Event] = {}
        # 已被消费点取走的 key:防止通信线程在提示迟到时再挂一条重复 recv
        self._early_recv_consumed: set[tuple[int, int]] = set()
        # 已上报完成的 key
        self._early_recv_reported: set[tuple[int, int]] = set()
        # 名额已满时排队的提示(key -> hint,插入序即到达序)
        self._early_recv_pending: dict[tuple[int, int], dict] = {}
        self._early_recv_lock = threading.Lock()
        # 队列卡住诊断的限频时间戳
        self._early_recv_stuck_log_at = 0.0
        # 下行提示 MQ(edge_id -> reader)/ 上行完成上报 MQ(edge_id -> writer)
        self._irecv_hint_mqs: dict[int, "MessageQueue"] = {}
        self._irecv_done_mqs: dict[int, "MessageQueue"] = {}
        # 公开别名:WorkerProc 在 worker 构造后取它,把 handle 随 READY 回传,
        # 引擎据此挂 done reader(见 vllm/v1/executor/multiproc_executor.py)。
        self.irecv_done_mqs: dict[int, "MessageQueue"] = {}
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(
        self, edge_id: int, seqno: int, num_tokens: int
    ) -> LwdCommRequest:
        """UP recv 请求的唯一构造点(提示路径与消费点补挂共用同一形状)。

        HCCL P2P 要求两端 numel 严格相等:主帧 = hash 载荷公式(embeds + 可选
        专家 ID 字节)。尺寸公式只在本处算一次,两侧路径不会各算一遍算歪。
        aux 恒 0:mrope positions 已走控制面(LwdRangeItem.mrope_positions,
        wire_version>=2),数据面 aux 帧退役——若残留非零,预挂的 aux irecv
        永远收不到数据,水位停摆。aux 机制保留在通道层供 deepstack 等扩展。
        """
        return LwdCommRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=hash_payload_numel(
                num_tokens, self._lwd_up_hidden_size,
                hash_layer_count(self.model_config.hf_config),
                getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
            ),
            seqno=int(seqno),
            edge_id=int(edge_id),
            aux_num_elements=0,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """逐边重建 hint MQ reader、建 done MQ writer,起提前收通信线程。"""
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        for edge_id in self._lwd_early_recv_edges:
            mq = attach_hint_mq(edge_id)
            if mq is None:
                logger.warning(
                    "[Lwd][pre-recv] hint MQ handle missing for edge=%d "
                    "(executor not armed?); recv falls back to consume-point",
                    edge_id,
                )
                continue
            self._irecv_hint_mqs[edge_id] = mq
            done_mq = create_report_mq()
            self._irecv_done_mqs[edge_id] = done_mq
            self.irecv_done_mqs[edge_id] = done_mq
        logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQs=%s",
            self.rank, sorted(self._irecv_hint_mqs),
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def _lwd_early_recv_loop(self) -> None:
        """通信线程主体:取提示 -> 提前挂 recv;轮询 future -> 上报收完。

        只做四件极轻的事:``dequeue`` / ``submit_recv`` / ``future.done()`` /
        ``enqueue`` —— 从不 ``wait()`` 任何 future、也不碰模型,所以不与
        busy_loop 的 HCCL 使用冲突(同通道上一端 irecv 一端跨线程 wait 是
        HCCL 不允许的)。
        """
        # 当前设备是线程级状态,通信线程要自己设一次(否则 irecv 落到 0 卡)
        torch.npu.set_device(self.device)
        logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQs=%s",
            self.rank, self.device, sorted(self._irecv_hint_mqs),
        )
        hint_mqs = list(self._irecv_hint_mqs.items())
        while True:
            # 逐边轮取(每边每轮至多 1 条):防单边暴量饿死他边 reader
            for _edge_id, mq in hint_mqs:
                try:
                    hint = mq.dequeue(timeout=0)
                except TimeoutError:
                    continue
                except Exception:
                    # 关停时 ring 被取消等:记一条、小睡,别热转(daemon 线程
                    # 随进程退出)
                    logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                    time.sleep(0.01)
                    continue
                self.start_early_irecv(hint)
            self._report_irecv_completions()
            self._warn_if_early_recv_stuck()
            # 封顶 CPU:ring 读端在"刚读过"的时间窗内是自旋,光给 0 超时并不
            # 封顶 —— 这行 sleep 才是,代价是提示/水位各约 1ms 的可见延迟。
            time.sleep(0.001)

    _LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S = 5.0

    def _warn_if_early_recv_stuck(self) -> None:
        """队列挂不上时每 5s 一条告警:排队有提示、名额却全被占住。

        名额被占住 = 那几条已挂 recv 既没被消费也没释放(水位因此不前进、
        闸门不再放行 prefill)。这条日志把"谁占着名额"直接写出来,避免出现
        "prefill 不动了但不知道为什么"。
        """
        with self._early_recv_lock:
            if (
                not self._early_recv_pending
                or len(self._early_recv_futures) + len(self._early_recv_posting)
                < self._early_recv_max_inflight
            ):
                return
            now = time.monotonic()
            if (
                now - self._early_recv_stuck_log_at
                < self._LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S
            ):
                return
            self._early_recv_stuck_log_at = now
            oldest = next(iter(self._early_recv_pending))
            holders = sorted(self._early_recv_futures)
        logger.warning(
            "[Lwd][pre-recv] queue stuck: oldest queued (edge,seqno)=%s, "
            "%d slot(s) held by (edge,seqno)=%s (slots held = those batches "
            "posted recv but were never consumed; downstream dispatch is "
            "the pacer)",
            oldest, len(holders), holders,
        )

    def start_early_irecv(self, hint: dict) -> None:
        """收下一条引擎提示;名额没满就立刻挂 recv,满了先排队。

        幂等:同一 (edge_id, seqno) 的重复提示、或消费点已经处理过的,都不再
        接受 —— 同一通道上两条 irecv 会抢发送端那一条 isend。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None:
            logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        edge_id, seqno, _num_tokens = fields
        key = (edge_id, seqno)
        with self._early_recv_lock:
            if (
                key in self._early_recv_futures
                or key in self._early_recv_posting
                or key in self._early_recv_consumed
                or key in self._early_recv_pending
            ):
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _early_recv_quota(self, edge_id: int) -> int:
        """该边在本 rank 的软配额:ceil(G / E_live),保底 1。"""
        e_live = max(1, len(self._lwd_early_recv_edges))
        return max(1, -(-self._early_recv_max_inflight // e_live))

    def _pick_pending_locked(self) -> tuple[tuple[int, int], dict]:
        """在锁内按软配额挑一条排队提示(调用前须保证 pending 非空)。

        工作守恒 + 保底:优先选"本边在飞数 < 配额"的队首;都超配额时允许
        借额(选队首),但若某边有排队且在飞数 == 0,只许选该边的队首
        (保底 1 条,防一边暴量把提前收变成单边专属)。
        """
        inflight_by_edge: dict[int, int] = {}
        for e, _s in self._early_recv_futures:
            inflight_by_edge[e] = inflight_by_edge.get(e, 0) + 1
        for e, _s in self._early_recv_posting:
            inflight_by_edge[e] = inflight_by_edge.get(e, 0) + 1
        # 保底:有排队却在飞为 0 的边最优先
        for (e, _s), hint in self._early_recv_pending.items():
            if inflight_by_edge.get(e, 0) == 0:
                return (e, _s), hint
        # 配额内先到先挂
        for (e, _s), hint in self._early_recv_pending.items():
            if inflight_by_edge.get(e, 0) < self._early_recv_quota(e):
                return (e, _s), hint
        # 全部超配额:借额,选全局队首(到达序)
        return next(iter(self._early_recv_pending.items()))

    def _early_recv_pump(self) -> None:
        """把排队的提示按到达序挂成 recv,直到用满在飞名额。

        名额 = "已挂 recv(含 POSTING)、还没随 SO 被消费"的条数(每条占
        一块整批接收 buffer),所以这里的上限就是显存上限;
        ``get_or_post_early_recv`` 消费一条后调用本函数补挂下一条,提示
        队列因此不会因为上限而丢。

        锁纪律:锁内只做 dict 操作(取出提示、占 POSTING 位);
        ``submit_recv`` 在锁外(它可能取通道层锁,锁内调用是 0826 死锁的
        同款形态)。POSTING 占位使消费点不会在同一键上重复补挂。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        while True:
            with self._early_recv_lock:
                if (
                    not self._early_recv_pending
                    or len(self._early_recv_futures)
                    + len(self._early_recv_posting)
                    >= self._early_recv_max_inflight
                ):
                    return
                key, hint = self._pick_pending_locked()
                del self._early_recv_pending[key]
                ev = threading.Event()
                self._early_recv_posting[key] = ev
            # ---- 锁外 submit ----
            edge_id, seqno, num_tokens = hint_fields(hint)
            future = get_lwd_comm_service().submit_recv(
                self._lwd_build_up_recv_request(edge_id, seqno, num_tokens)
            )
            with self._early_recv_lock:
                self._early_recv_posting.pop(key, None)
                # 消费点若在 submit 期间到达,已在 POSTING 上等待,
                # 登记 future 后由它取走;幂等集合不变
                self._early_recv_futures[key] = future
                ev.set()
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            logger.debug(
                "[Lwd][pre-recv] SUBMIT edge=%d seqno=%d num_tokens=%d "
                "inflight=%d queued=%d ts=%.3f",
                edge_id, seqno, num_tokens, inflight, queued,
                time.monotonic(),
            )

    def _report_irecv_completions(self) -> None:
        """把已收完的 recv 以 (edge_id, channel, seqno) 上报到该边 done MQ。

        每 (edge, channel) 完成序 = seqno 序,所以引擎侧只维护一个最大
        水位就能精确判就绪。已被消费点取走的条目继续轮询:消费意味着前向
        正在用这份数据,``done()`` 很快转真,水位照样前进;条目"既上报过
        又被消费过"才移除。
        """
        if not self._irecv_done_mqs:
            return
        with self._early_recv_lock:
            pending = [
                (key, future)
                for key, future in self._early_recv_futures.items()
                if key not in self._early_recv_reported
            ]
        for (edge_id, seqno), future in pending:
            if not future.done():
                continue
            done_mq = self._irecv_done_mqs.get(edge_id)
            if done_mq is None:
                continue
            logger.debug(
                "[Lwd][pre-recv] DONE edge=%d seqno=%d ts=%.3f",
                edge_id, seqno, time.monotonic(),
            )
            # 上报:引擎每步排空 done MQ 推进该边水位
            done_mq.enqueue((edge_id, LwdChannelType.UP, seqno))
            logger.debug(
                "[Lwd][pre-recv] REPORT edge=%d seqno=%d ts=%.3f",
                edge_id, seqno, time.monotonic(),
            )
            with self._early_recv_lock:
                key = (edge_id, seqno)
                self._early_recv_reported.add(key)
                if key in self._early_recv_consumed:
                    self._early_recv_futures.pop(key, None)
                    self._early_recv_reported.discard(key)
                    self._early_recv_consumed.discard(key)
                    released = True
                else:
                    released = False
            if released:
                self._early_recv_pump()

    def get_or_post_early_recv(
        self, edge_id: int, seqno: int, num_tokens: int
    ) -> LwdCommFuture | None:
        """取通信线程提前挂好的 UP recv future;没有就现场补挂一条并登记。

        非端点 rank 无跨机通道(数据在注入时刻由 TP 广播取到),返回 None。
        条目不在本处移除:上报线程要继续轮询它推进水位,只有"既上报又消费"
        才移除,所以水位不会因为被消费而断档。

        本处是**名额归还点**:SO 一到,这一条的 buffer 马上要交给前向,名额
        立刻转给还在排队的提示(提示还排着队时先就地挂上,闸门/前向不可能
        跑到 recv 前面)。

        POSTING 竞态协议:泵已取出提示正在锁外 submit 时,本处在该键的
        Event 上短等(submit 是 µs~ms 级内存操作,等待实际为零);超时仍无
        future = 泵线程异常,fail-fast 而不是再挂一条(同键两条 irecv 会抢
        同一条 isend)。
        """
        if edge_id not in self._lwd_early_recv_edges:
            return None

        key = (edge_id, seqno)
        future = None
        posting_ev = None
        hint = None
        with self._early_recv_lock:
            # 名额归还点(先于取用):SO 已下发,这条 buffer 即刻交给前向
            self._early_recv_consumed.add(key)
            future = self._early_recv_futures.get(key)
            if future is None:
                posting_ev = self._early_recv_posting.get(key)
                if posting_ev is None:
                    # 名额满时该提示可能还在排队:本批已下发,必须挂上
                    hint = self._early_recv_pending.pop(key, None)
        if future is None and posting_ev is not None:
            posting_ev.wait(timeout=5.0)
            with self._early_recv_lock:
                future = self._early_recv_futures.get(key)
            if future is None:
                raise RuntimeError(
                    f"[Lwd][pre-recv] posting stuck edge={edge_id} "
                    f"seqno={seqno}: pump submitted no future within 5s"
                )
        if future is None:
            # 提示彻底没到(或到得比本 SO 还晚/在排队):用同一份信息现场
            # 补挂,保证上报链不缺号,水位能继续前进。锁外 submit,锁内登记。
            logger.debug(
                "[Lwd][pre-recv] submit at consume edge=%d seqno=%d "
                "(hint %s)", edge_id, seqno,
                "still queued" if hint is not None else "never arrived",
            )
            future = get_lwd_comm_service().submit_recv(
                self._lwd_build_up_recv_request(edge_id, seqno, num_tokens)
            )
            with self._early_recv_lock:
                self._early_recv_futures[key] = future
        with self._early_recv_lock:
            if key in self._early_recv_reported:
                # 完成已上报:消费即条目终点(上报线程不会再碰它)
                self._early_recv_futures.pop(key, None)
                self._early_recv_reported.discard(key)
                self._early_recv_consumed.discard(key)
            inflight = len(self._early_recv_futures)
            queued = len(self._early_recv_pending)
        # 名额已归还:补挂排队中的提示
        self._early_recv_pump()
        logger.debug(
            "[Lwd][pre-recv] stats edge=%d seqno=%d inflight=%d queued=%d",
            edge_id, seqno, inflight, queued,
        )
        return future

    def drop_edge_early_recv(self, edge_id: int) -> None:
        """边失联清理挂点:清该边收单/排队提示/台账,名额回池。

        已挂出的 HCCL recv 无法取消(buffer 随进程生命周期);条目摘除后
        上报线程不再轮询它。本轮无调用方(当前线尚无 edge-loss 检测点),
        供后续检测落地时接线。
        """
        with self._early_recv_lock:
            for table in (
                self._early_recv_futures,
                self._early_recv_posting,
                self._early_recv_pending,
            ):
                for key in [k for k in table if k[0] == edge_id]:
                    del table[key]
            self._early_recv_consumed = {
                k for k in self._early_recv_consumed if k[0] != edge_id
            }
            self._early_recv_reported = {
                k for k in self._early_recv_reported if k[0] != edge_id
            }

    # 注:take_lwd_up_embeds(host 阻塞收割)不随移植恢复——本线上 UP
    # 消费已收进 runner(_lwd_inject_remote_embeds:端点过门 + TP 组内
    # 现场广播),worker 层只保留 post 侧;aux(mrope)帧同路径消费。

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: "GrammarOutput") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN:发送 hidden 包(通道异步、流内有序);meta 设备
        # 张量随输出内层传递(收割拼装在响应入队处、就绪事件同步之后),
        # 引擎侧解码发布 meta。
        if self.enable_lwd:
            payloads = self.model_runner.take_lwd_pending_down_payload()
            if payloads:
                from vllm.distributed.parallel_state import get_tp_group
                from vllm_ascend.distributed import lwd_wire

                # 只有 driver(云 TP0)的输出会回传引擎,carrier 只在 driver
                # 上挂载;但 seqno 必须在所有 rank 上按同一顺序分配,保证
                # 各卡 per-edge 计数一致(端点轮转后不同 edge 由不同卡发)。
                is_driver = get_tp_group().is_first_rank
                carriers = []
                ready_event = None
                for (edge_id, hidden, meta_dev, seg_lens_host, ev,
                     req_ids) in payloads:
                    # 全 rank 同序递增,保持 seqno 全局一致(与端点卡对齐)
                    seqno = self._lwd_next_down_seqno(edge_id)
                    # 端点随 edge_id 轮转:仅本卡是该 edge 的 DOWN 端点才发送
                    if lwd_wire.is_lwd_channel_endpoint(
                        LwdChannelType.DOWN, edge_id=edge_id, cloud_id=None
                    ):
                        logger.info(
                            "[Lwd][cloud-worker] DOWN send edge=%d seqno=%d "
                            "numel=%d",
                            edge_id, seqno, hidden.numel(),
                        )
                        get_lwd_comm_service().submit_send(
                            LwdCommRequest(
                                channel=LwdChannelType.DOWN,
                                op="send",
                                num_elements=hidden.numel(),
                                tensor=hidden,
                                seqno=seqno,
                                edge_id=edge_id,
                            )
                        )
                    if is_driver:
                        # 未收割形态(meta_dev 仍在设备上);executor 输出
                        # 线程收割拼装后重写为成品 5 元组(edge_id,
                        # cpu_tensor, req_ids, numel, seqno)——线布局
                        # 不变,引擎零改动
                        carriers.append(
                            (edge_id, meta_dev, seg_lens_host, req_ids,
                             hidden.numel(), seqno)
                        )
                        # 本步所有边共享同一步的就绪事件
                        ready_event = ev
                if is_driver and output is not None:
                    # async 包装器下挂到内层,否则 get_output() 解包丢失
                    target = getattr(output, "_model_runner_output", output)
                    target.lwd_down_carrier = carriers
                    # 就绪事件随输出对象传递(同进程输出线程消费);
                    # Event 不可序列化,enqueue 前由 executor 同步并摘除
                    if ready_event is not None:
                        target.lwd_meta_ready_event = ready_event
                    # [Lwd][perf] 云侧每步计时:sample=采样+采集(含秩计算);
                    # send=DOWN 提交(快照clone+isend+bridge wait);total=全步
                    logger.info(
                        "[Lwd][perf] cloud-step edges=%d sample=%.2f send=%.2f "
                        "total=%.2fms",
                        len(carriers),
                        (_t_sample - _t0) * 1000,
                        (time.monotonic() - _t_sample) * 1000,
                        (time.monotonic() - _t0) * 1000,
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
            if self.model_runner.lwd_hash_state is not None:
                self.model_runner.lwd_hash_state.discard(req_id)
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

    def _lwd_next_down_seqno(self, edge_id: int = 0) -> int:
        seqno = self._lwd_down_next_seqno.get(edge_id, 0)
        self._lwd_down_next_seqno[edge_id] = seqno + 1
        return seqno
