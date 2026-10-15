# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer.

Owns the physical data-plane channels.  In the single-edge/single-cloud
case this is exactly the two direction-only channels (UP edge->cloud,
DOWN cloud->edge).  In multi-edge/multi-cloud deployments the same
machinery is replicated per ``(edge, cloud, engine)`` triple (the cloud
side may run several DP engines, each owning its own channel pair):

  * one HCCL communicator + one NPU stream per
    ``(edge, cloud, engine, channel)``;
  * per-(triple, channel) seqno stream stays independent (see
    ``lwd_comm``);
  * init-time warmup of every channel in a fixed global order
    (lexicographic ``(edge, cloud, engine)``, UP then DOWN — moves the
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

# (edge_id, cloud_id, engine_index, channel) -> (device_group, peer_global_rank)
_LWD_CHANNEL_GROUPS: dict[
    tuple[int, int, int, LwdChannelType], tuple[dist.ProcessGroup, int]
] = {}
# (edge_id, cloud_id, engine_index, channel) -> stream
_LWD_CHANNEL_STREAMS: dict[
    tuple[int, int, int, LwdChannelType], "torch.npu.Stream"
] = {}
# (edge_id, cloud_id, engine_index) -> (edge_global_rank, cloud_global_rank)
_LWD_ENDPOINTS: dict[tuple[int, int, int], tuple[int, int]] = {}
_LWD_TRIPLES: list[tuple[int, int, int]] = []
# 本进程身份(由 role registry + 全局 rank 推导),供 None 维度回填
_SELF_ROLE: str | None = None
_SELF_EDGE_ID = 0
_SELF_CLOUD_ID = 0
_SELF_ENGINE_ID = 0
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
    global _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID, _SELF_ENGINE_ID
    my_rank = dist.get_rank()
    role, edge_id, cloud_id = registry.self_ids(my_rank)
    if role is not None:
        _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID = role, edge_id, cloud_id
    else:
        # 非边非云 rank(理论不出现):保持单边一云默认身份
        _SELF_ROLE, _SELF_EDGE_ID, _SELF_CLOUD_ID = None, 0, 0
    # 本进程所属云 engine:本 rank 落在该云哪一段连续 ranks 内
    # (engine i 占 ranks[i*C:(i+1)*C])。registry 未升级(无 engine 维
    # API)/非云 rank 时恒 0,退化为单 (e, c, 0) 域。
    _SELF_ENGINE_ID = 0
    if _SELF_ROLE == "cloud":
        engine_ranks = getattr(registry, "cloud_engine_ranks", None)
        engine_ids = getattr(registry, "engine_ids", None)
        if callable(engine_ranks) and callable(engine_ids):
            for i in engine_ids(_SELF_CLOUD_ID):
                if my_rank in engine_ranks(_SELF_CLOUD_ID, i):
                    _SELF_ENGINE_ID = i
                    break


def _registry_triples(registry) -> list[tuple[int, int, int]]:
    """全部 (edge_id, cloud_id, engine_index) 通信域,全局固定字典序。

    registry 未升级(无 triples API)时由 pairs() 展开 engine=0,
    行为与 (edge, cloud) 键逐值一致。"""
    triples_fn = getattr(registry, "triples", None)
    if callable(triples_fn):
        return sorted(triples_fn())
    return [(e, c, 0) for (e, c) in registry.pairs()]


def _engine_endpoint_rank(registry, edge_id: int, cloud_id: int,
                          engine_index: int) -> int:
    """云端点 rank:engine 感知(端点在该引擎的 C 个 rank 内随 edge_id
    轮转);registry 未升级时回退 cloud 级轮转(等价 engine0)。"""
    ep_fn = getattr(registry, "engine_endpoint_rank", None)
    if callable(ep_fn):
        return ep_fn(edge_id, cloud_id, engine_index)
    return registry.cloud_endpoint_rank(edge_id, cloud_id)


def init_lwd_duplex_channels() -> None:
    """Create the duplex channels for every ``(edge, cloud, engine)`` triple.

    Called once per worker process when ``is_prefill_only`` is on.
    Idempotent.  ``dist.new_group`` is a world collective — every rank
    must participate in every creation in the same fixed order, so
    triples are iterated in lexicographic order and channels in
    ``LwdChannelType`` enum order (UP then DOWN).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    registry = _load_registry()
    _compute_self_ids(registry)

    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()

    triples = _registry_triples(registry)
    _LWD_TRIPLES[:] = triples
    for edge_id, cloud_id, engine_index in triples:
        edge_rank = registry.edge_rank(edge_id)
        # 端点随 edge_id 在该引擎的 C 个 rank 间轮转(分摊跨机 P2P
        # 压力),而非固定取引擎 TP0 首卡。
        cloud_rank = _engine_endpoint_rank(
            registry, edge_id, cloud_id, engine_index)
        _LWD_ENDPOINTS[(edge_id, cloud_id, engine_index)] = (
            edge_rank, cloud_rank)
        ranks = [edge_rank, cloud_rank]
        for channel in LwdChannelType:
            group = dist.new_group(ranks, backend=backend)
            if my_rank == edge_rank:
                peer = cloud_rank
            elif my_rank == cloud_rank:
                peer = edge_rank
            else:
                peer = -1
            _LWD_CHANNEL_GROUPS[(edge_id, cloud_id, engine_index, channel)] = (group, peer)
            _LWD_CHANNEL_STREAMS[(edge_id, cloud_id, engine_index, channel)] = torch.npu.Stream()

    _INITIALIZED = True
    logger.info(
        "[lwd-wire] duplex channels created over %d triple(s) %s "
        "(my_rank=%d self=%s e%d c%d i%d)",
        len(triples), triples, my_rank, _SELF_ROLE, _SELF_EDGE_ID,
        _SELF_CLOUD_ID, _SELF_ENGINE_ID,
    )
    warmup_lwd_duplex_channels()


def _resolve_triple(
    edge_id: int | None,
    cloud_id: int | None,
    engine_index: int | None = None,
) -> tuple[int, int, int]:
    """None 维度用本进程身份回填,得到完整 (edge, cloud, engine) 通信域。"""
    if edge_id is None:
        edge_id = _SELF_EDGE_ID
    if cloud_id is None:
        cloud_id = _SELF_CLOUD_ID
    if engine_index is None:
        engine_index = _SELF_ENGINE_ID
    if (edge_id, cloud_id, engine_index) not in _LWD_ENDPOINTS:
        raise RuntimeError(
            f"[lwd-wire] unknown (edge, cloud, engine) triple "
            f"({edge_id}, {cloud_id}, {engine_index}); "
            f"known triples={list(_LWD_ENDPOINTS)}"
        )
    return edge_id, cloud_id, engine_index


def warmup_lwd_duplex_channels() -> None:
    """Pre-establish every channel's P2P link at init time.

    Triples and channels are warmed up in a fixed global order
    (lexicographic triples, UP then DOWN), so both sides rendezvous
    identically regardless of when the first real message is posted.
    """
    my_rank = dist.get_rank()
    for edge_id, cloud_id, engine_index in _LWD_TRIPLES:
        edge_rank, cloud_rank = _LWD_ENDPOINTS[(edge_id, cloud_id, engine_index)]
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            group, peer = _LWD_CHANNEL_GROUPS[(edge_id, cloud_id, engine_index, channel)]
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
                    "[lwd-warmup] SEND triple=(%d,%d,%d) channel=%s "
                    "my_rank=%d peer=%d group_ranks=%s",
                    edge_id, cloud_id, engine_index, channel, my_rank,
                    peer, dist.get_process_group_ranks(group))
                handle = dist.isend(payload, dst=peer, group=group)
            else:
                logger.info(
                    "[lwd-warmup] RECV triple=(%d,%d,%d) channel=%s "
                    "my_rank=%d src=%d group_ranks=%s",
                    edge_id, cloud_id, engine_index, channel, my_rank,
                    peer, dist.get_process_group_ranks(group))
                handle = dist.irecv(payload, src=peer, group=group)
            handle.wait()
            logger.info(
                "[lwd-warmup] DONE triple=(%d,%d,%d) channel=%s my_rank=%d",
                edge_id, cloud_id, engine_index, channel, my_rank)
    get_world_group().barrier()
    logger.info("[lwd-wire] duplex channels warmed up")


def lwd_channels_initialized() -> bool:
    return _INITIALIZED


def get_lwd_channel_device_group(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
    engine_index: int | None = None,
) -> dist.ProcessGroup:
    triple = _resolve_triple(edge_id, cloud_id, engine_index)
    group, _ = _LWD_CHANNEL_GROUPS[(triple[0], triple[1], triple[2], channel)]
    return group


def get_lwd_channel_peer(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
    engine_index: int | None = None,
) -> int:
    """Global rank of this process's P2P peer on the channel."""
    triple = _resolve_triple(edge_id, cloud_id, engine_index)
    _, peer = _LWD_CHANNEL_GROUPS[(triple[0], triple[1], triple[2], channel)]
    return peer


def get_lwd_channel_endpoint_rank(
    edge_id: int | None = None,
    cloud_id: int | None = None,
    engine_index: int | None = None,
) -> tuple[int, int]:
    """该 (edge, cloud, engine) 通信域两端端点的全局 rank
    (edge_rank, cloud_rank)。

    cloud_rank 已随 edge_id 在该引擎的 C 个 rank 内轮转(见
    LwdRoleRegistry.engine_endpoint_rank),供 UP 的 TP 广播选源等
    需要明确端点 rank 的上层使用。
    """
    triple = _resolve_triple(edge_id, cloud_id, engine_index)
    return _LWD_ENDPOINTS[(triple[0], triple[1], triple[2])]


def get_lwd_channel_stream(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
    engine_index: int | None = None,
) -> "torch.npu.Stream":
    triple = _resolve_triple(edge_id, cloud_id, engine_index)
    return _LWD_CHANNEL_STREAMS[(triple[0], triple[1], triple[2], channel)]


def is_lwd_channel_endpoint(
    channel: LwdChannelType,
    edge_id: int | None = None,
    cloud_id: int | None = None,
    engine_index: int | None = None,
) -> bool:
    """本进程是否为该通道通信域的端点(边入口/云该引擎的端点卡)。

    非端点 rank(引擎内其它 TP rank、其它引擎的 rank)peer=-1,
    仅参与建组/广播,不碰跨机 P2P。
    """
    triple = _resolve_triple(edge_id, cloud_id, engine_index)
    _, peer = _LWD_CHANNEL_GROUPS[(triple[0], triple[1], triple[2], channel)]
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
    _LWD_TRIPLES.clear()
    _INITIALIZED = False
