# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge), one pair
per cloud engine (multi-DP: M pairs):
  * channel HCCL communicators: dedicated ``new_group`` groups over the
    endpoint rank pairs (shared-edge layout: edge rank 0 <-> cloud
    engine i's TP0 at global rank E + i*C);
  * one NPU stream per (engine, channel) pair (wire ops are bridged
    onto it, keeping the compute stream free);
  * init-time warmup of every pair in a fixed global order (engine
    order, UP then DOWN), moving the first-use HCCL rendezvous out of
    the pipeline, where the two sides could otherwise arrive on
    different channels and deadlock.

Only prefill_only mode builds these groups; other modes never reach
here (gated by ``LwdConfig.is_prefill_only``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_world_group
from vllm.logger import logger

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
        "(my_rank=%d)",
        len(endpoints), endpoints, my_rank,
    )
    # 每对通道都 warmup:把首用 HCCL rendezvous 挪到初始化期,逐对按
    # 固定全局顺序(engine 序,先 UP 后 DOWN),保证两端到场次序一致。
    warmup_lwd_duplex_channels()


def warmup_lwd_duplex_channels() -> None:
    """Pre-establish every engine pair's P2P links at init time.

    A tiny payload is exchanged on each channel in a fixed global order
    (engine 序,每 engine 先 UP 后 DOWN), so both sides rendezvous
    identically regardless of when the first real message is posted.
    """
    my_rank = dist.get_rank()
    for engine_index, (edge_rank, cloud_rank) in enumerate(_LWD_ENDPOINTS):
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            group, peer = _LWD_CHANNEL_GROUPS[(engine_index, channel)]
            am_sender = (
                (channel is LwdChannelType.UP and my_rank == edge_rank)
                or (channel is LwdChannelType.DOWN and my_rank == cloud_rank)
            )
            am_receiver = (
                (channel is LwdChannelType.UP and my_rank == cloud_rank)
                or (channel is LwdChannelType.DOWN and my_rank == edge_rank)
            )
            # Non-endpoint ranks participate in new_group/barrier (world
            # collectives) but skip the P2P ops entirely.
            if not (am_sender or am_receiver):
                continue
            payload = torch.zeros(8, dtype=torch.bfloat16, device="npu")
            if am_sender:
                logger.info(
                    "[lwd-warmup] SEND post engine=%d channel=%s my_rank=%d "
                    "peer=%d", engine_index, channel, my_rank, peer)
                handle = dist.isend(payload, dst=peer, group=group)
            else:
                logger.info(
                    "[lwd-warmup] RECV post engine=%d channel=%s my_rank=%d "
                    "src=%d", engine_index, channel, my_rank, peer)
                handle = dist.irecv(payload, src=peer, group=group)
            handle.wait()
            logger.info(
                "[lwd-warmup] DONE engine=%d channel=%s my_rank=%d",
                engine_index, channel, my_rank)
    get_world_group().barrier()
    logger.info(
        "[lwd-wire] duplex channels warmed up (%d engine pair(s))",
        len(_LWD_ENDPOINTS),
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
