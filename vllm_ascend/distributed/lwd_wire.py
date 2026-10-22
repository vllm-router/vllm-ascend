# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge), one pair
per cloud engine (multi-DP: M pairs):
  * channel HCCL communicators: dedicated ``new_group`` groups over the
    endpoint rank pairs (shared-edge layout: edge rank 0 <-> cloud
    engine i's TP0 at global rank E + i*C);
  * one NPU stream per (engine, channel) pair (wire ops are bridged
    onto it, keeping the compute stream free);
  * init-time warmup + echo handshake of every pair in a fixed
    global order (engine order, UP then DOWN): the edge sends a known
    payload on UP, the cloud engine echoes it back on DOWN, and the
    edge verifies it byte-for-byte within a timeout. This moves the
    first-use HCCL rendezvous out of the pipeline *and* proves the two
    endpoints are correctly paired and reachable in both directions.

Only prefill_only mode builds these groups; other modes never reach
here (gated by ``LwdConfig.is_prefill_only``).
"""

from __future__ import annotations

import os
import sys
import threading
import time

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_world_group
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

# (engine_index, channel) -> (device_group, peer_global_rank)
_LWD_CHANNEL_GROUPS: dict[tuple[int, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[int, LwdChannelType], "torch.npu.Stream"] = {}
# 每 engine 一对端点:(edge_rank=0, cloud_rank=E + i*C),i = engine_index
_LWD_ENDPOINTS: list[tuple[int, int]] = []
_INITIALIZED = False


def init_lwd_duplex_channels() -> None:
    """Create the duplex channels: one UP/DOWN pair per cloud engine.

    Called once per worker process when ``is_prefill_only`` is on.
    Idempotent.

    Endpoint ranks (shared-edge layout, world = E + M*C):
      edge rank 0  <->  cloud engine i's TP0 (global rank E + i*C).
    Multi-engine requires the CLI counts (``--edge-npu-count`` /
    ``--cloud-npu-count`` / ``--cloud-dp-size``); the PP-group fallback
    only resolves the legacy 1E1C pair (M=1).
    Only the endpoint ranks ever touch the channels; interior ranks
    (extra cloud TP ranks, other engines' ranks) join the group
    collective but get peer=-1.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    t_groups = time.perf_counter()
    from vllm.config import get_current_vllm_config

    lwd_cfg = get_current_vllm_config().parallel_config.lwd_config
    if lwd_cfg.enable_lwd and lwd_cfg.edge_npu_count > 0:
        # 共享边布局:edge [0, E),engine i 占 [E+i*C, E+(i+1)*C),
        # 端点对 (0, E+i*C)
        num_engines = lwd_cfg.cloud_dp_size
        endpoints = [
            (0, lwd_cfg.cloud_engine_rank_base(i)) for i in range(num_engines)
        ]
    else:
        pp_group = get_pp_group()
        if pp_group.world_size != 2:
            raise RuntimeError(
                "prefill_only duplex channels cannot resolve edge/cloud "
                "endpoint ranks: lwd_config endpoint ranks unset (check "
                "--edge-npu-count/--cloud-npu-count) and the PP group "
                "does not span exactly the edge/cloud pair "
                f"(pp world_size={pp_group.world_size})"
            )
        endpoints = [(pp_group.ranks[0], pp_group.ranks[1])]
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for engine_index, (edge_rank, cloud_rank) in enumerate(endpoints):
        ranks = [edge_rank, cloud_rank]
        for channel in LwdChannelType:
            # new_group 是 world collective:全 rank 必须同序参与每一次
            # 建组(固定顺序:engine 序,每 engine 先 UP 后 DOWN)
            group = dist.new_group(ranks, backend=backend)
            # P2P peer = 组内另一个 rank,只在两个端点 rank 上有意义;
            # 其余 rank(云其余 TP 卡、其他 engine 的卡)不碰这条通道。
            if my_rank == edge_rank:
                peer = cloud_rank
            elif my_rank == cloud_rank:
                peer = edge_rank
            else:
                peer = -1
            _LWD_CHANNEL_GROUPS[(engine_index, channel)] = (group, peer)
            _LWD_CHANNEL_STREAMS[(engine_index, channel)] = torch.npu.Stream()
    _LWD_ENDPOINTS.clear()
    _LWD_ENDPOINTS.extend(endpoints)
    # embeds 组内广播不再自建通信域(仅指为组内 broadcast 额外自建的
    # 同域 new_group 在 torch_npu 下实测零交付,疑建组顺序错位;
    # UP/DOWN P2P 组不受此结论影响,已由 pg5 探针验证),改用框架
    # TP device_group(demo 同款,建组顺序由框架保证);广播在注入时刻
    # 于计算流上发起,与前向 TP 集合同流 FIFO,天然不串台。
    _INITIALIZED = True
    logger.info(
        "[lwd-wire] duplex channels created: engines=%d endpoints=%s "
        "(my_rank=%d) groups=%.1fms",
        len(endpoints), endpoints, my_rank,
        (time.perf_counter() - t_groups) * 1000.0,
    )
    # 每对通道都 warmup:把首用 HCCL rendezvous 挪到初始化期,逐对按
    # 固定全局顺序(engine 序,先 UP 后 DOWN),保证两端到场次序一致。
    warmup_lwd_duplex_channels()


def _lwd_wait_with_timeout(work, timeout_s: float, what: str) -> None:
    """等待一个 torch Work 完成;超时抛 TimeoutError,HCCL 异常原样抛出。

    ``Work.wait()`` 没有 timeout 参数,因此用"守护线程 + Event"实现:线程只
    负责 wait 并把异常回传,主线程负责超时判定。超时后raise。
    """
    done = threading.Event()
    errors: list[BaseException] = []

    def _waiter() -> None:
        try:
            work.wait()
        except BaseException as exc:  # noqa: BLE001 - 通信错误必须回传
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=_waiter, name="lwd-hs", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"[Lwd][handshake] timeout {timeout_s:.0f}s: {what}")
    if errors:
        raise RuntimeError(
            f"[Lwd][handshake] failed: {what}: {errors[0]!r}"
        ) from errors[0]


def _lwd_echo_handshake(
    engine_index: int, edge_rank: int, cloud_rank: int, timeout_s: float
) -> dict[str, float]:
    """单个 engine pair 的 echo 往返:边发已知值 → 云原样回 → 边逐位比对。

    判据只在边侧 recv:HCCL ``isend`` 完成只表示本地发送缓冲可复用,
    不代表对端已收到。云侧超时取 ``2x``,用于边侧失败后的对端自清。


    """
    magic = 0x4C5744  # "LWD"
    payload_dtype = torch.int32
    payload_elems = 3
    my_rank = dist.get_rank()
    up_group, up_peer = _LWD_CHANNEL_GROUPS[(engine_index, LwdChannelType.UP)]
    down_group, down_peer = _LWD_CHANNEL_GROUPS[(engine_index, LwdChannelType.DOWN)]
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    # 可变容器:主线程写阶段,看门狗线程读(只用于日志,允许轻微竞争)
    phase = ["post"]
    budget_s = timeout_s if my_rank == edge_rank else timeout_s * 2
    finished = threading.Event()

    """HCCL 建链会阻塞在 ``isend``/``irecv`` 或``.cpu()`` 同步里, 
    而 ``HCCL_CONNECT_TIMEOUT``的合法下限是 120s, 无法用它压到 30s。
    看门狗到期先打一条 ``FAILED phase=… (watchdog …)`` 然后 ``os._exit(1)``;
    它不调用任何 NPU API, 因此不影响流语义。"""
    def _watchdog() -> None:
        if finished.wait(budget_s):
            return
        logger.error(
            "[Lwd][handshake] engine=%d rank=%d FAILED phase=%s "
            "(watchdog %.0fs, blocked in HCCL) up_peer=%d down_peer=%d",
            engine_index, my_rank, phase[0], budget_s, up_peer, down_peer,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    threading.Thread(target=_watchdog, name="lwd-wd", daemon=True).start()

    def _mark(name: str) -> None:
        marks[name] = (time.monotonic() - t0) * 1000.0

    def _fmt(value: float | None) -> str:
        """阶段耗时格式化：未到达的阶段打印 "-"。"""
        return "-" if value is None else f"{value:.1f}"

    try:
        if my_rank == edge_rank:
            send = torch.tensor(
                [magic, engine_index, 0],
                dtype=payload_dtype,
                device="npu",
            )
            recv = torch.zeros(
                payload_elems, dtype=payload_dtype, device="npu"
            )
            send_work = dist.isend(send, dst=up_peer, group=up_group)
            recv_work = dist.irecv(recv, src=down_peer, group=down_group)
            _mark("post")
            phase[0] = "recv"
            _lwd_wait_with_timeout(
                recv_work, timeout_s,
                f"engine={engine_index} dir=DOWN peer={down_peer} rank={my_rank}",
            )
            _mark("recv")
            phase[0] = "send"
            _lwd_wait_with_timeout(
                send_work, timeout_s, f"engine={engine_index} dir=UP-send"
            )
            _mark("send")
            phase[0] = "verify"
            if not torch.equal(recv.cpu(), send.cpu()):
                raise RuntimeError(
                    f"[Lwd][handshake] content mismatch engine={engine_index} "
                    f"expect={send.tolist()} got={recv.tolist()}"
                )
            _mark("verify")
        else:
            recv = torch.zeros(
                payload_elems, dtype=payload_dtype, device="npu"
            )
            recv_work = dist.irecv(recv, src=up_peer, group=up_group)
            _mark("post")
            phase[0] = "recv"
            _lwd_wait_with_timeout(
                recv_work, timeout_s * 2,
                f"engine={engine_index} dir=UP peer={up_peer} rank={my_rank}",
            )
            _mark("recv")
            phase[0] = "verify"
            got = [int(v) for v in recv.cpu().tolist()]
            if got[0] != magic or got[1] != engine_index:
                raise RuntimeError(
                    f"[Lwd][handshake] bad payload engine={engine_index} got={got}"
                )
            _mark("verify")
            phase[0] = "send"
            send_work = dist.isend(recv, dst=down_peer, group=down_group)
            _lwd_wait_with_timeout(
                send_work, timeout_s * 2,
                f"engine={engine_index} dir=DOWN-send peer={down_peer}",
            )
            _mark("send")
    except BaseException:
        logger.error(
            "[Lwd][handshake] engine=%d FAILED phase=%s post=%s recv=%s "
            "send=%s verify=%s (ms)",
            engine_index, phase[0],
            _fmt(marks.get("post")), _fmt(marks.get("recv")),
            _fmt(marks.get("send")), _fmt(marks.get("verify")),
        )
        raise
    finally:
        finished.set()
    logger.debug(
        "[Lwd][handshake] engine=%d round-trip=%.2fms connect=%.2fms",
        engine_index, (time.monotonic() - t0) * 1000, marks.get("post", 0.0),
    )
    return marks


def warmup_lwd_duplex_channels() -> None:
    """建链预热 + echo 校验。

    1. 首用 HCCL rendezvous 前移:每条 (engine, channel) 都真实收发一次;
    2. 固定全局顺序(engine 序,UP 先 DOWN),保证两端到场次序一致;
    3. echo 校验:边发已知值、云原样回、边逐位比对,超时(默认 30s)即失败;
    4. 末尾 world barrier:确认全部 rank 到达。
    """
    timeout_s = float(ascend_envs.VLLM_ASCEND_LWD_HCCL_HANDSHAKE_TIMEOUT_S)
    my_rank = dist.get_rank()
    t_echo_start = time.perf_counter()
    connect_ms = 0.0
    for engine_index, (edge_rank, cloud_rank) in enumerate(_LWD_ENDPOINTS):
        # 非端点 rank:只参加 new_group/barrier(world collective),不碰 P2P。
        if my_rank not in (edge_rank, cloud_rank):
            continue
        marks = _lwd_echo_handshake(
            engine_index, edge_rank, cloud_rank, timeout_s
        )
        connect_ms += marks.get("post", 0.0)
    # 两段分开计时:echo 是真实往返耗时,barrier 是"等最慢 rank 到齐"的耗时。
    echo_ms = (time.perf_counter() - t_echo_start) * 1000.0
    t_barrier = time.perf_counter()
    get_world_group().barrier()
    barrier_ms = (time.perf_counter() - t_barrier) * 1000.0
    logger.info(
        "[Lwd][handshake] OK engines=%d rank=%d connect=%.1fms "
        "echo=%.1fms barrier=%.1fms",
        len(_LWD_ENDPOINTS), my_rank, connect_ms, echo_ms, barrier_ms,
    )


def get_lwd_channel_device_group(
    channel: LwdChannelType, engine_index: int = 0
) -> dist.ProcessGroup:
    group, _ = _LWD_CHANNEL_GROUPS[(engine_index, channel)]
    return group


def get_lwd_channel_peer(channel: LwdChannelType, engine_index: int = 0) -> int:
    """Global rank of this process's P2P peer on the channel."""
    _, peer = _LWD_CHANNEL_GROUPS[(engine_index, channel)]
    return peer


def get_lwd_channel_stream(
    channel: LwdChannelType, engine_index: int = 0
) -> "torch.npu.Stream":
    return _LWD_CHANNEL_STREAMS[(engine_index, channel)]
