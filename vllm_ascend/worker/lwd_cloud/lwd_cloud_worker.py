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

import threading
import time
from dataclasses import replace

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
from vllm_ascend import envs
from vllm_ascend.worker.lwd_cloud.lwd_cloud_model_runner import LwdCloudModelRunner
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel
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
        # Payload scramble: bind the two static factors once; the per-flight
        # seed is derived by the comm layer from each request's own fields.
        self._build_payload_scrambler()
        # channel-global DOWN seqno counter (worker layer, send-time alloc)
        self._lwd_down_next_seqno = 0
        # [Lwd][pre-recv] 提前收:端点 rank(= 唯一持跨机 P2P 通道的云 rank,
        # 与 lwd_wire 同款约定)才提前挂真实 irecv 并上报收完水位;其余 rank
        # 的数据在注入时刻由 TP 组内广播取到,既不挂 recv 也不上报。
        self._lwd_up_hidden_size = 0
        self._lwd_early_recv_endpoint = False
        self._init_lwd_early_recv_state()
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
        # 端点 rank 才有跨机通道:把它的提前收通信线程起起来(hint MQ 已由
        # 执行器在建 worker 之前建好并写进环境变量;done MQ 由本处建 writer,
        # handle 随 READY 握手回传引擎)。其余 rank 只登记"非端点"。
        self._lwd_early_recv_endpoint = (
            self.rank == self.parallel_config.lwd_config.edge_npu_count
        )
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        # 每个云 rank 都打一行:谁该起通信线程一目了然
        logger.info(
            "[Lwd][pre-recv] worker rank=%d endpoint=%s hidden=%d",
            self.rank, self._lwd_early_recv_endpoint, self._lwd_up_hidden_size,
        )
        if self._lwd_early_recv_endpoint:
            self._lwd_init_host_stager(self._lwd_up_hidden_size)
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
            # host 暂存:请求结束 ⇒ 释放它的 embedding(design v7 §6.1 的
            # 唯一常规释放点;不是"用完就放")。
            self._lwd_release_finished(scheduler_output.finished_req_ids)

        get_lwd_comm_service().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] 云侧每步 exec_model 计时:total=前向全程
        # (forward 主段,sample/collect 在 sample_tokens 另有分项);
        # tokens/reqs 与集中式 [Perf] exec 打点同口径,便于纯 decode 步对表
        logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms tokens=%d reqs=%d",
            (time.monotonic() - _t0) * 1000,
            scheduler_output.total_num_scheduled_tokens,
            len(scheduler_output.num_scheduled_tokens),
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output) -> None:
        """处理本步 SO 上的 abort 名单(退化 EMBED 批,seqno=-1)。

        arrival 驱动后,UP recv 全部由通信线程按引擎提示提前挂出
        (RangeNotify → hint),本路径不再按 SO 投递;runner 按
        input_batch 直接收割,也不再经 ``_lwd_up_recv_futures``。
        """
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None:
            return
        # abort 名单(§3.6.1):退化批里 req_ids 为空,只有这里会读它。
        # 冻结被 abort 请求的剩余行 —— 否则那一份 pinned 会一直占着。
        self._lwd_handle_abort_notice(getattr(meta, "aborted_req_ids", None))

    # ------------------------------------------------------------------ #
    # 提前收:通信线程(提示下发 -> 挂 recv -> 上报水位)                     #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """提前收收单(与引擎下发的提示、上报的水位同一套键)。

        key = (channel, seqno)。条目生命周期:提示到达即 submit_recv 并
        登记;分段 D2H 完成(publish)+ 上报后即移除(发布即摘 ——
        arrival 驱动后没有 SO 消费点,runner 按 input_batch 直接收割)。

        边侧 pending 闸门(``VLLM_ASCEND_LWD_MAX_PENDING_CHUNKS``,默认 6)
        按"未消费 chunk(seqno)"限制边侧领先量,云侧在飞上限必须不小于它,
        否则 recv 挂不满、闸门放宽不生效;上限之外的提示先排队
        (``_early_recv_pending``,按到达序,只存字段不占显存),前面的批
        发布后立刻补挂。
        """
        self._early_recv_max_inflight = envs.VLLM_ASCEND_LWD_MAX_INFLIGHT_RECV
        self._early_recv_futures: dict[tuple[LwdChannelType, int], LwdCommFuture] = {}
        self._early_recv_lock = threading.Lock()
        # 名额已满时排队的提示(key -> hint,插入序即到达序)
        self._early_recv_pending: dict[tuple[LwdChannelType, int], dict] = {}
        # 队列卡住诊断的限频时间戳
        self._early_recv_stuck_log_at = 0.0
        # 下行提示 MQ(reader)/ 上行完成上报 MQ(writer)
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        # 公开别名:WorkerProc 在 worker 构造后取它,把 handle 随 READY 回传,
        # 引擎据此挂 done reader(见 vllm/v1/executor/multiproc_executor.py)。
        self.irecv_done_mq = None
        self._lwd_early_recv_thread: threading.Thread | None = None
        # ------------------------------------------------------------------ #
        # host 暂存(design v7:按请求拆分存储 / 双指针 / 请求结束才释放)        #
        # ------------------------------------------------------------------ #
        # 行表不再单独登记:提交期由 ``plan_recv_segments`` 一次建成
        # (分段 + 到达计划 + 批描述符),prompt 全长由 hint / SO meta 的
        # ``prompt_lens`` 提供。
        # req_id -> (prompt_rows, num_hash_layers, num_experts_per_tok)。
        # **按请求存**(不再按 seqno):一个请求可能跨多个批到达,元数据只有
        # 一份,按请求挂才对得上。
        self._lwd_req_meta: dict[str, tuple[int, int, int]] = {}
        # req_id -> 上次上报的**累计**已收行数(只用于跳过"没前进"的请求)。
        # 注意:上报的是累计值而不是增量 —— 引擎侧台账用 max() 合并,那是为
        # 累计值设计的口径;报增量会让台账停在"本块行数"上(真机踩过)。
        self._lwd_last_total: dict[str, int] = {}
        # 已交付(收完 + 被某个 SO 消费)的最高 seqno。通道完成序 == seqno 序,
        # 所以"已交付集合"恒为前缀,一个高水位即可精确表示。
        # 用途:同一 seqno 因"窗口跨多步"被二次下发时,**不得再挂 recv**
        # (边上只发一帧,多挂一条会让整条通道错位一帧,且不报错)。
        self._early_recv_last_delivered: int = -1
        # 开关默认关:关着时下面所有暂存分支都是 no-op,行为与改造前逐字一致。
        self._lwd_host_stage_enabled = bool(envs.VLLM_ASCEND_LWD_HOST_STAGE)
        self._lwd_stager = None

    def _lwd_init_host_stager(self, hidden_size: int) -> None:
        """建 host 暂存池(端点 rank、开关打开时)。

        槽数可由环境变量覆盖;默认按"能同时在飞的 recv 条数"推导 —— 那是设备
        槽真正需要的数量(design v7 §4.5:D2H 一完成即归还,所以驱动槽数的是
        在飞 recv 数,不是未消费的请求数)。

        **没有 host 上限**(v7 明确不设):内存按"全存到请求结束"预算,不够时
        再上 LRU + 边侧重发(后话)。所以准入只剩设备槽/接收缓冲一个闸门。
        """
        if not (self._lwd_host_stage_enabled and self._lwd_early_recv_endpoint):
            return
        from vllm_ascend.worker.lwd_cloud.lwd_host_stage import LwdHostStager

        slots = envs.VLLM_ASCEND_LWD_HOST_STAGE_SLOTS or (
            self._early_recv_max_inflight
        )
        # slot 尺寸上限:一个批的最大 token 数 × 每行载荷。取不到调度上限时留 0
        # —— 池按首个批的实际 numel 懒建,后续更大的批自动扩容,只是少了
        # "提前按上限预留"这一层保护。
        # 每行载荷必须用 hash 载荷公式(embeds + 可选专家 ID 尾段):DSv4
        # 的接收 numel 按 hash_payload_numel 计(_lwd_build_up_recv_request),
        # 这里若只算 hidden,每个批都会被 allocate 的几何校验拒掉,水位停摆。
        sched_cfg = getattr(self.vllm_config, "scheduler_config", None)
        max_tokens = getattr(sched_cfg, "max_num_batched_tokens", 0) or 0
        max_numel = hash_payload_numel(
            max_tokens,
            hidden_size,
            hash_layer_count(self.model_config.hf_config),
            getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
        )
        self._lwd_stager = LwdHostStager(
            device=self.device,
            slots=slots,
            max_numel=max_numel,
        )
        logger.info(
            "[Lwd][host-stage] stager ready slots=%d max_numel=%d "
            "(max_tokens=%d hidden=%d, no host cap)",
            slots, max_numel, max_tokens, hidden_size,
        )

    def _lwd_build_up_recv_request(
        self, seqno: int, num_tokens: int, aux_rows: int,
        fields: tuple | None = None,
    ) -> LwdCommRequest:
        """UP recv 请求的唯一构造点(提示路径与消费点补挂共用同一形状)。

        HCCL P2P 要求两端 numel 严格相等:主帧 = hash 载荷公式(embeds + 可选
        专家 ID 字节)。aux 帧机制保留供 deepstack 等未来载荷(mrope
        positions 自 wire_version 2 起改走控制面,调用方恒传 aux_rows=0)。
        尺寸公式只在本处算一次,两侧路径不会各算一遍算歪。

        host 暂存打开时,本处同时完成两件事(design v7 §4.3 改版):
          1. 按行表向暂存池**建行表 + 分段**(``plan_recv_segments``):
             per-request pinned buffer 按 prompt 全长一次分好,返回逐段
             D2H 落点;
          2. 申请设备乒乓槽(``allocate``)。
        通道层随后把接收载荷**逐段异步 D2H** 直接落进各请求的 buffer,
        通信线程零 CPU 拷贝(H2H 拆分退役)。槽满抛 ``LwdHostStageSlotBusy``
        (调用方排队重试,提示不丢弃);行表/几何缺失抛
        ``LwdHostStageGeometryError``(协议级错误,不可重试)。

        ``fields`` = ``hint_fields()`` 的 8 元组;消费点补挂路径用 SO 的
        batch_meta 重建同形状元组。
        """
        num_elements = hash_payload_numel(
            num_tokens, self._lwd_up_hidden_size,
            hash_layer_count(self.model_config.hf_config),
            getattr(self.model_config.hf_config, "num_experts_per_tok", 0),
        )
        request = LwdCommRequest(
            channel=LwdChannelType.UP,
            op="recv",
            num_elements=num_elements,
            seqno=int(seqno),
            # aux 第二载荷与主帧同 seqno;mrope 退役后恒 0(零 aux 流量),
            # 未来载荷(如 deepstack)显式传行数即可
            aux_num_elements=int(aux_rows) * 3,
        )
        stager = self._lwd_stager
        if stager is None:
            return request
        from vllm_ascend.worker.lwd_cloud.lwd_host_stage import (
            LwdHostStageGeometryError,
        )

        if fields is None or not fields[3] or not fields[7]:
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: staged recv needs the per-request row "
                f"table with prompt_lens (req_ids/rows/prompt_offsets/"
                f"prompt_lens); got fields={None if fields is None else 'incomplete'}"
            )
        (_seqno, _nt, _aux, req_ids, rows, _offsets, prompt_offsets,
         prompt_lens) = fields
        hf = self.model_config.hf_config
        segments = stager.plan_recv_segments(
            seqno=int(seqno),
            numel=num_elements,
            hidden_size=self._lwd_up_hidden_size,
            req_ids=req_ids,
            num_tokens_per_req=rows,
            prompt_lens=prompt_lens,
            row_base=list(prompt_offsets) if prompt_offsets else None,
            num_hash_layers=hash_layer_count(hf),
            num_experts_per_tok=getattr(hf, "num_experts_per_tok", 0),
        )
        slot_view, slot = stager.allocate(seqno=int(seqno), numel=num_elements)
        return replace(
            request,
            recv_buffer=slot_view,
            stage_segments=segments,
            recv_slot=slot,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """重建 hint MQ reader、建 done MQ writer,起提前收通信线程。

        起线程前后各打一行:"starting" 在起之前(证明这段代码走到了)、"up"
        在线程体里设完设备之后(证明线程真的活起来了)。只有前一行 = 线程在
        设置设备时挂了(异常栈在 stderr);两行都在但没 SUBMIT = 没收到提示
        (hint MQ 没挂上,日志里会写 attached/none)。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        self._irecv_hint_mq = attach_hint_mq()
        self._irecv_done_mq = create_report_mq()
        self.irecv_done_mq = self._irecv_done_mq
        logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank, "attached" if self._irecv_hint_mq is not None else "none",
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
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank, self.device,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mq = self._irecv_hint_mq
        while True:
            if hint_mq is not None:
                try:
                    hint = hint_mq.dequeue(timeout=0.0005)
                except TimeoutError:
                    pass
                except Exception:
                    # 关停时 ring 被取消等:记一条、小睡,别热转(daemon 线程
                    # 随进程退出)
                    logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                    time.sleep(0.01)
                else:
                    self.start_early_irecv(hint)
            self._report_irecv_completions()
            self._warn_if_early_recv_stuck()
            # ring 读端在"刚读过"的时间窗内是 sched_yield 自旋(busy_loop_s=1s),
            # 光给 0.5ms 超时并不封顶 CPU —— 这行 sleep 才是封顶,代价是
            # 提示/水位各约 2ms 的可见延迟。
            time.sleep(0.002)

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
                or len(self._early_recv_futures) < self._early_recv_max_inflight
            ):
                return
            now = time.monotonic()
            if (
                now - self._early_recv_stuck_log_at
                < self._LWD_EARLY_RECV_STUCK_LOG_INTERVAL_S
            ):
                return
            self._early_recv_stuck_log_at = now
            oldest = next(iter(self._early_recv_pending))[1]
            holders = sorted(key[1] for key in self._early_recv_futures)
        logger.warning(
            "[Lwd][pre-recv] queue stuck: oldest queued seqno=%d, %d slot(s) "
            "held by seqno=%s (slots held = those batches posted recv but "
            "were never consumed; downstream dispatch is the pacer)",
            oldest, len(holders), holders,
        )

    def start_early_irecv(self, hint: dict) -> None:
        """收下一条引擎提示;名额没满就立刻挂 recv,满了先排队。

        幂等:同一 (channel, seqno) 的重复提示、或**已交付过的**
        (``_early_recv_last_delivered`` 高水位拦截,发布即摘后这是唯一
        的判据),都不再接受 —— 同一通道上两条 irecv 会抢发送端那一条
        isend。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None:
            logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        seqno = fields[0]
        key = (hint["channel"], seqno)
        with self._early_recv_lock:
            if (
                seqno <= self._early_recv_last_delivered
                or key in self._early_recv_futures
                or key in self._early_recv_pending
            ):
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _early_recv_pump(self) -> None:
        """把排队的提示按到达序挂成 recv,直到用满在飞名额。

        名额 = "已挂 recv、还没发布"的条数。host 暂存打开时,设备槽在
        分段 D2H 完成(publish)时即归还,在飞条目只剩 future 簿记;发布
        后本函数随即被调用补挂下一条,提示队列因此不会因为上限而丢。

        **host 暂存打开时名额性质变了**(§5 不变量 6):recv buffer 不再是每批
        一块、活到被消费,而是从池里借一个乒乓槽、D2H 一完成就归还。所以除了
        "在飞条数 < 上限",还必须**池真的有空槽且 host 字节未超上限**才挂 ——
        否则先排队(只存字段、不占任何资源),等前面的批被消费腾出 host 额度。
        """
        from vllm_ascend.distributed.lwd_comm.pre_recv import hint_fields

        while True:
            stager = self._lwd_stager
            with self._early_recv_lock:
                if (
                    not self._early_recv_pending
                    or len(self._early_recv_futures)
                    >= self._early_recv_max_inflight
                ):
                    return
                key, hint = next(iter(self._early_recv_pending.items()))
                fields = hint_fields(hint)
                if fields is None:
                    # 坏提示已在 start_early_irecv 处告警过;这里再兜一次,
                    # 防 hint_fields 语义收紧后老条目卡在队列里。
                    del self._early_recv_pending[key]
                    continue
                seqno, num_tokens, aux_rows = fields[0], fields[1], fields[2]
                from vllm_ascend.worker.lwd_cloud.lwd_host_stage import (
                    LwdHostStageError,
                    LwdHostStageSlotBusy,
                )

                try:
                    request = self._lwd_build_up_recv_request(
                        seqno, num_tokens, aux_rows, fields
                    )
                except LwdHostStageSlotBusy:
                    # 池里没有空闲槽:提示**留在队列原地**,下一轮再试
                    # (与"名额满只排队"同款纪律,绝不丢弃)。
                    return
                except LwdHostStageError:
                    # 几何/行表错误是协议级问题,不可重试(S4:曾被当成
                    # 池满无限排队,水位停摆)。丢弃该提示并告警 —— 同版本
                    # 部署下不应出现;出现时该批数据永不到达,涉及的请求随
                    # 客户端超时终结,至少不再堵死后续提示。
                    logger.exception(
                        "[Lwd][pre-recv] seqno=%s: build recv request "
                        "failed (geometry/protocol), hint dropped", seqno,
                    )
                    del self._early_recv_pending[key]
                    continue
                del self._early_recv_pending[key]
                future = get_lwd_comm_service().submit_recv(request)
                self._early_recv_futures[key] = future
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            logger.info(
                "[Lwd][pre-recv] SUBMIT channel=%s seqno=%d num_tokens=%d "
                "aux_rows=%d inflight=%d queued=%d ts=%.3f",
                key[0].value, seqno, num_tokens, aux_rows, inflight, queued,
                time.monotonic(),
            )

    def _report_irecv_completions(self) -> None:
        """把已收完的 recv 以 (channel, seqno) 上报到 done MQ。

        每通道完成序 = seqno 序。发布即摘:数据落进 per-request pinned
        buffer 后,future 簿记就没有读者了(arrival 驱动后 runner 按
        input_batch 直接收割,不再有 SO 消费点)。

        **三段式**(design v7 §5.2):
          ① ``future.done()``          —— HCCL 收完(主帧 + aux 帧)
          ② ``future.staged_done()``   —— 分段 D2H 完成,逐请求的 pinned
             buffer 里已各有一份完整段(通道层逐段异步拷入,零 CPU 拆分)
          ③ 发布 + 上报 —— ``publish_received`` 推进逐请求高水位、
             **归还设备槽**、写 done MQ(**捎带本批逐请求已收行数**)

        这一步**只做 query(),从不 wait()** —— 纯 CPU 轮询是本线程与
        busy_loop 不冲突的前提。

        上报为什么必须捎带行数:调度器按"每个请求收到多少行"决定下一步下发
        多少(design v7 §5.3),而它只看到"某条 seqno 到了"。行数随同一条消息
        过去,引擎侧累加成 ``recv_rows[req]`` —— 不动数据面协议。
        """
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        with self._early_recv_lock:
            pending = list(self._early_recv_futures.items())
        for key, future in pending:
            if not future.done():
                continue
            # ① 收完:这条 recv 的数据(主帧 + aux 帧)已完整落地设备缓冲
            logger.info(
                "[Lwd][pre-recv] DONE-RECV channel=%s seqno=%d ts=%.3f",
                key[0].value, key[1], time.monotonic(),
            )
            if not future.staged_done():
                # ② 已收完、暂存未就绪:继续轮询,不阻塞本线程
                continue
            arrived = self._lwd_commit_received(future)
            # ③ 上报:水位推进 + 逐请求已收行数(两者同时送达)
            done_mq.enqueue((key, arrived) if arrived else key)
            logger.info(
                "[Lwd][pre-recv] REPORT channel=%s seqno=%d arrived=%s ts=%.3f",
                key[0].value, key[1], arrived or "-", time.monotonic(),
            )
            with self._early_recv_lock:
                # 发布即摘:数据已落 host per-request buffer,future 簿记
                # 再无读者(arrival 驱动后 runner 直接按 input_batch 收割,
                # 不再有 SO 消费点)。已交付水位在这里推进,
                # ``start_early_irecv`` 用它拦截迟到/重复的提示。
                self._early_recv_futures.pop(key, None)
                if key[1] > self._early_recv_last_delivered:
                    self._early_recv_last_delivered = key[1]
            self._early_recv_pump()

    def _lwd_commit_received(self, future: LwdCommFuture) -> list:
        """分段 D2H 完成后的收尾:发布高水位、归还设备槽、算待上报行数。

        只在 host 暂存打开时做实事;关着时立即返回空(消费点仍走"取 future
        的 tensor"老路径)。**不再搬数据**:分段 D2H 由通道层在接收流上
        直接落进 per-request buffer(§4.3 改版,H2H 拆分退役)。

        Returns:
            ``[(req_id, rows), ...]`` —— 逐请求**累计**已收行数。
        """
        stager = self._lwd_stager
        if stager is None:
            return []
        seqno = int(future.request.seqno)
        rows = stager.publish_received(seqno)
        # 设备槽已归还;丢掉 future 对设备帧的引用,让池能立刻复用那块内存
        # (不丢的话引用计数还在,而 future 不会马上回收)。
        future.release_device_tensor()
        # ---- 逐请求上报**累计已收行数**(不是增量)----
        #
        # 引擎侧 `_lwd_note_arrived` 用的是 ``ledger[req] = max(旧值, 上报值)``,
        # 那套写法**专门为"上报累计值"设计**:幂等、抗重放、不会被重复上报推过
        # 头。这里曾经改报"增量"(``total - already``),两边语义就对不上:引擎
        # 拿增量去 max,台账便停在"本块行数"上 ——
        # 真机实录:一个分 1168 + 2342 两块到达的请求,worker 内部 recv_rows
        # 已是 3510(日志可证),上报的增量 2342 却让引擎台账停在 2342,调度器
        # W2 据此误判越界并把引擎打崩。
        #
        # 报累计值同时省掉一本 `_lwd_reported_rows` 账(少一个可能分叉的量):
        # 未前进的请求直接跳过,不产生消息。
        arrived: list[tuple[str, int]] = []
        last_total: dict[str, int] = getattr(self, "_lwd_last_total", None) or {}
        self._lwd_last_total = last_total
        hf = self.model_config.hf_config
        for rid, n, offset in rows:
            total = stager.recv_rows(rid)
            if total <= last_total.get(rid, 0):
                continue
            last_total[rid] = total
            arrived.append((rid, total))
            self._lwd_req_meta.setdefault(
                rid, (offset + n, hash_layer_count(hf),
                      getattr(hf, "num_experts_per_tok", 0)),
            )
        if arrived:
            logger.info(
                "[Lwd][host-stage] ARRIVED seqno=%d cumulative=%s host_MB=%.1f",
                seqno, arrived, stager.host_bytes / 2 ** 20,
            )
        return arrived

    def _lwd_handle_abort_notice(self, aborted_req_ids) -> None:
        """冻结被 abort 请求的剩余行,并**在请求结束时**释放其 host 存储。

        调度器侧已经把这些请求从台账里摘掉了(它们的剩余行永远不会被请求),
        但那本台账**只活在调度器进程里**;worker 的暂存池不知道,就会把那些
        行的 pinned 一直留到请求被显式释放为止。

        本处只做一件事:``abort_request`` 冻结剩余行(已消费的行不回滚 —— 那是
        历史事实,它的 KV 已算、embeddings 已注入)。**不在本处释放存储**:
        design v7 §6.1 的释放点是"请求结束",由 ``_lwd_release_finished`` 在
        finish 路径上做 —— 两条路分开,abort 只是让它更早到达那个点。

        幂等:重复送达只是把已冻结的再冻一次(``mark_aborted`` 自己判重)。
        """
        stager = self._lwd_stager
        if stager is None or not aborted_req_ids:
            return
        for req_id in aborted_req_ids:
            if stager.take_req(req_id) is None:
                continue
            req = stager.take_req(req_id)
            stager.abort_request(req_id)
            logger.info(
                "[Lwd][host-stage] ABORT-SYNC req=%s consumed=%d/%d "
                "dropped_rows=%d",
                req_id, req.consumed, req.prompt_rows,
                max(req.prompt_rows - req.consumed, 0),
            )

    def _lwd_release_finished(self, finished_req_ids) -> int:
        """请求结束时释放它的 host embedding(design v7 §6.1)。

        这是本方案唯一的常规释放点:embedding **不随"用完"释放**,而是留到
        请求结束(finish / 长度截断 / abort 都是同一条路)。返回回收字节数。

        ``finished_req_ids`` 是引擎的存活信号(每个 SO 都带),所以这里不需要
        额外协议 —— 与既有的 ``_lwd_cloud_flush_finished`` 同一处触发。
        """
        stager = self._lwd_stager
        if stager is None or not finished_req_ids:
            return 0
        freed = 0
        for req_id in finished_req_ids:
            got = stager.release_request(req_id)
            if got:
                freed += got
                self._lwd_req_meta.pop(req_id, None)
                self._lwd_last_total.pop(req_id, None)
                logger.info(
                    "[Lwd][host-stage] RELEASED req=%s bytes=%.1fKiB "
                    "host_MB=%.1f held_reqs=%d",
                    req_id, got / 1024, stager.host_bytes / 2 ** 20,
                    stager.request_count,
                )
        if freed:
            # 该请求所属的批描述符可能已经没人持有了,顺手收掉
            for seqno in list(stager.staged_seqnos()):
                stager.release_if_drained(seqno)
        return freed

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
            payload = self.model_runner.take_lwd_pending_down_payload()
            if payload is not None and output is not None:
                (hidden, meta_dev, seg_lens_host, ready_event,
                 req_ids) = payload
                seqno = self._lwd_next_down_seqno()
                logger.info(
                    "[Lwd][cloud-worker] DOWN send seqno=%d numel=%d",
                    seqno, hidden.numel(),
                )
                get_lwd_comm_service().submit_send(
                    LwdCommRequest(
                        channel=LwdChannelType.DOWN,
                        op="send",
                        num_elements=hidden.numel(),
                        tensor=hidden,
                        seqno=seqno,
                    )
                )
                # async 包装器下挂到内层,否则 get_output() 解包丢失
                target = getattr(output, "_model_runner_output", output)
                # 未收割形态(meta_dev 仍在设备上);executor 输出线程
                # 收割拼装后重写为成品 4 元组(cpu_tensor, req_ids,
                # numel, seqno)——线布局不变,引擎零改动
                target.lwd_down_carrier = (
                    meta_dev, seg_lens_host, req_ids, hidden.numel(), seqno,
                )
                # 就绪事件随输出对象传递(同进程输出线程消费);
                # Event 不可序列化,enqueue 前由 executor 同步并摘除
                target.lwd_meta_ready_event = ready_event
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

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno
