# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer.

Owns the physical data-plane channels.  In the single-edge/single-cloud
case this is exactly the two direction-only channels (UP edge->cloud,
DOWN cloud->edge).  In multi-edge/multi-cloud deployments the same
machinery is replicated per ``(edge, cloud)`` pair:

  * one HCCL communicator + one NPU stream per ``(edge, cloud, channel)``;
  * per-channel seqno stream stays independent (see ``lwd_comm``);
  * init-time warmup of every channel in a fixed global order (moves the
    first-use HCCL rendezvous out of the pipeline, where two sides could
    otherwise arrive on different channels and deadlock).

Only prefill_only mode builds these groups; other modes never reach
here (gated by ``LwdConfig.is_prefill_only``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

# (edge_id, cloud_id, channel) -> (device_group, peer_global_rank)
_LWD_CHANNEL_GROUPS: dict[
    tuple[int, int, LwdChannelType], tuple[dist.ProcessGroup, int]
] = {}
# (edge_id, cloud_id, channel) -> stream
_LWD_CHANNEL_STREAMS: dict[
    tuple[int, int, LwdChannelType], "torch.npu.Stream"
] = {}
# (edge_id, cloud_id) -> (edge_global_rank, cloud_global_rank)
_LWD_ENDPOINTS: dict[tuple[int, int], tuple[int, int]] = {}
_LWD_PAIRS: list[tuple[int, int]] = []
# 本进程身份(由 role registry + 全局 rank 推导),供 None 维度回填
_SELF_ROLE: str | None = None
_SELF_EDGE_ID = 0
_SELF_CLOUD_ID = 0
_INITIALIZED = False


def dump_tensor(tag: str, tensor: "torch.Tensor") -> None:
    """调试:打印数据面张量摘要,供边云两端成对对比数值。"""
    import numpy as np

    flat = tensor.detach().to("cpu", torch.float32).numpy().reshape(-1)
    with np.printoptions(threshold=np.inf, linewidth=10000, precision=8):
        logger.info(
            "%s shape=%s dtype=%s head10=%s tail10=%s sum=%.6f mean=%.8f",
            tag, tuple(tensor.shape), tensor.dtype,
            flat[:10], flat[-10:], float(flat.sum()), float(flat.mean()),
        )


def _load_registry():
    """加载角色注册表;未配置/加载失败退化单边一云域。

    单边一云端点 rank 沿用 prefill_only 原有推导:连续 edge-first 布局
    edge [0, E), cloud [E, E+C),端点取 (0, E)。

    云侧复用 registry 路径两处来源:CLI ``--role-registry``(镜像在
    ``parallel_config.lwd_config.role_registry``)与 additional_config
    的 ``lwd_config.role_registry_path``;命中即复用控制面装配期的
    进程级单例(保证全场同一份)。"""
    from vllm.config import get_current_vllm_config

    vllm_cfg = get_current_vllm_config()
    lwd_cfg = vllm_cfg.parallel_config.lwd_config
    from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
        LwdRoleRegistry,
        get_role_registry,
        load_role_registry,
    )

    if lwd_cfg.enable_lwd and lwd_cfg.edge_npu_count > 0:
        edge_rank, cloud_rank = 0, lwd_cfg.edge_npu_count
    else:
        pp_group = get_pp_group()
        if pp_group.world_size != 2:
            raise RuntimeError(
                "prefill_only duplex channels cannot resolve edge/cloud "
                "endpoint ranks: lwd_config endpoint ranks unset and the PP "
                "group does not span exactly the edge/cloud pair "
                f"(pp world_size={pp_group.world_size})"
            )
        edge_rank, cloud_rank = pp_group.ranks[0], pp_group.ranks[1]
    fallback = LwdRoleRegistry.single_pair(edge_rank, cloud_rank)
    registry_path = (
        getattr(vllm_cfg.lwd_config, "role_registry_path", "") or ""
    ) or (
        getattr(
            vllm_cfg.parallel_config.lwd_config, "role_registry", ""
        ) or ""
    )
    if not registry_path:
        return fallback
    # 控制面已加载则复用单例(全场同一份);worker 进程首载走 load
    singleton = get_role_registry()
    if singleton is not None:
        return singleton
    return load_role_registry(registry_path, fallback)


def _compute_self_ids(registry) -> None:
    global _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID
    my_rank = dist.get_rank()
    role, edge_id, cloud_id = registry.self_ids(my_rank)
    if role is not None:
        _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID = role, edge_id, cloud_id
    else:
        # 非边非云 rank(理论不出现):保持单边一云默认身份
        _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID = None, 0, 0


def init_lwd_duplex_channels() -> None:
    """Create the duplex channels for every ``(edge, cloud)`` pair.

    Called once per worker process when ``is_prefill_only`` is on.
    Idempotent.  ``dist.new_group`` is a world collective — every rank
    must participate in every creation in the same fixed order, so pairs
    are iterated sorted and channels in ``LwdChannelType`` enum order.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    registry = _load_registry()
    _compute_self_ids(registry)

    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()

    pairs = registry.pairs()
    _LWD_PAIRS[:] = pairs
    for edge_id, cloud_id in pairs:
        edge_rank = registry.edge_rank(edge_id)
        # 端点随 edge_id 在该云各 rank 间轮转(分摊跨机 P2P 压力),
        # 而非固定取云 TP0 首卡。
        cloud_rank = registry.cloud_endpoint_rank(edge_id, cloud_id)
        _LWD_ENDPOINTS[(edge_id, cloud_id)] = (edge_rank, cloud_rank)
        ranks = [edge_rank, cloud_rank]
        for channel in LwdChannelType:
            group = dist.new_group(ranks, backend=backend)
            if my_rank == edge_rank:
                peer = cloud_rank
            elif my_rank == cloud_rank:
                peer = edge_rank
            else:
                peer = -1
            _LWD_CHANNEL_GROUPS[(edge_id, cloud_id, channel)] = (group, peer)
            _LWD_CHANNEL_STREAMS[(edge_id, cloud_id, channel)] = torch.npu.Stream()

    _INITIALIZED = True
    logger.info(
        "[lwd-wire] duplex channels created over %d pair(s) %s "
        "(my_rank=%d self=%s e%d c%d)",
        len(pairs), pairs, my_rank, _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID,
    )
    warmup_lwd_duplex_channels()


def _resolve_pair(edge_id: int | None, cloud_id: int | None) -> tuple[int, int]:
    """None 维度用本进程身份回填,得到完整 (edge, cloud) 通信域。"""
    if edge_id is None:
        edge_id = _SELF_EDGE_ID
    if cloud_id is None:
        cloud_id = _SELF_CLOUD_ID
    if (edge_id, cloud_id) not in _LWD_ENDPOINTS:
        raise RuntimeError(
            f"[lwd-wire] unknown (edge, cloud) pair ({edge_id}, {cloud_id}); "
            f"known pairs={list(_LWD_ENDPOINTS)}"
        )
    return edge_id, cloud_id


def warmup_lwd_duplex_channels() -> None:
    """Pre-establish every channel's P2P link at init time.

    Pairs and channels are warmed up in a fixed global order (sorted pairs,
    UP then DOWN), so both sides rendezvous identically regardless of when
    the first real message is posted.
    """
    my_rank = dist.get_rank()
    for edge_id, cloud_id in _LWD_PAIRS:
        edge_rank, cloud_rank = _LWD_ENDPOINTS[(edge_id, cloud_id)]
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            group, peer = _LWD_CHANNEL_GROUPS[(edge_id, cloud_id, channel)]
            am_sender = (
                (channel is LwdChannelType.UP and my_rank == edge_rank)
                or (channel is LwdChannelType.DOWN and my_rank == cloud_rank)
            )
            am_receiver = (
                (channel is LwdChannelType.UP and my_rank == cloud_rank)
                or (channel is LwdChannelType.DOWN and my_rank == edge_rank)
            )
            if not (am_sender or am_receiver):
                continue
            payload = torch.zeros(8, dtype=torch.bfloat16, device="npu")
            if am_sender:
                logger.info(
                    "[lwd-warmup] SEND pair=(%d,%d) channel=%s my_rank=%d "
                    "peer=%d group_ranks=%s",
                    edge_id, cloud_id, channel, my_rank, peer,
                    dist.get_process_group_ranks(group))
                handle = dist.isend(payload, dst=peer, group=group)
            else:
                logger.info(
                    "[lwd-warmup] RECV pair=(%d,%d) channel=%s my_rank=%d "
                    "src=%d group_ranks=%s",
                    edge_id, cloud_id, channel, my_rank, peer,
                    dist.get_process_group_ranks(group))
                handle = dist.irecv(payload, src=peer, group=group)
            handle.wait()
            logger.info(
                "[lwd-warmup] DONE pair=(%d,%d) channel=%s my_rank=%d",
                edge_id, cloud_id, channel, my_rank)
    logger.info("[lwd-wire] duplex channels warmed up")


def lwd_channels_initialized() -> bool:
    return _INITIALIZED


def get_lwd_channel_device_group(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> dist.ProcessGroup:
    pair = _resolve_pair(edge_id, cloud_id)
    group, _ = _LWD_CHANNEL_GROUPS[(pair[0], pair[1], channel)]
    return group


def get_lwd_channel_peer(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> int:
    """Global rank of this process's P2P peer on the channel."""
    pair = _resolve_pair(edge_id, cloud_id)
    _, peer = _LWD_CHANNEL_GROUPS[(pair[0], pair[1], channel)]
    return peer


def get_lwd_channel_endpoint_rank(
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> tuple[int, int]:
    """该 (edge, cloud) 通信域两端端点的全局 rank (edge_rank, cloud_rank)。

    cloud_rank 已随 edge_id 轮转(见 LwdRoleRegistry.cloud_endpoint_rank),
    供 UP 的 TP 广播选源等需要明确端点 rank 的上层使用。
    """
    pair = _resolve_pair(edge_id, cloud_id)
    return _LWD_ENDPOINTS[(pair[0], pair[1])]


def get_lwd_channel_stream(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> "torch.npu.Stream":
    pair = _resolve_pair(edge_id, cloud_id)
    return _LWD_CHANNEL_STREAMS[(pair[0], pair[1], channel)]


def is_lwd_channel_endpoint(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> bool:
    """本进程是否为该通道通信域的端点(边入口/云 TP0)。

    非端点 rank(云内其它 TP rank)peer=-1,仅参与建组/广播,不碰跨机 P2P。
    """
    pair = _resolve_pair(edge_id, cloud_id)
    _, peer = _LWD_CHANNEL_GROUPS[(pair[0], pair[1], channel)]
    return peer != -1


def destroy_lwd_duplex_channels() -> None:
    global _INITIALIZED
    for (group, _) in list(_LWD_CHANNEL_GROUPS.values()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[lwd-wire] failed to destroy group")
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS.clear()
    _LWD_PAIRS.clear()
    _INITIALIZED = False
