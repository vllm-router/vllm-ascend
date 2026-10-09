# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only duplex data-plane wire layer (multi-instance).

Owns the two physical channels (UP edge->cloud, DOWN cloud->edge) per
cloud instance (design doc §4.3/§4.4):

  * channel HCCL communicators: one dedicated ``new_group`` pair over
    each (instance, dp) edge/cloud rank pair of the §4.1 layout.
    ``new_group`` is a world collective, so EVERY rank participates in
    EVERY creation, in the same fixed order (instance ascending, dp
    ascending, UP then DOWN) — a divergent order deadlocks the group
    creation itself;
  * one NPU stream per (channel, instance): wire ops are bridged onto
    it, keeping the compute stream free. The dp axis shares one stream
    per instance (per-rank topology: each edge process is a single dp,
    so the axis never multiplies here; the future shared-edge form
    distinguishes in-instance peers by ``src_dst`` on the same
    communicator, matching the demo topology's stream keying);
  * init-time warmup of every owned channel pair in the same fixed
    global order (moves the first-use HCCL rendezvous out of the
    pipeline, where the two sides could otherwise arrive on different
    channels and deadlock). The edge endpoint warms ALL N instance
    pairs (it time-shares them); a cloud endpoint warms only its own
    instance's pair.

Only prefill_only mode builds these groups; other modes never reach
here (gated by ``LwdConfig.is_prefill_only``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

# (channel, instance_id, dp_rank) -> (device_group, peer_global_rank).
# Only the two endpoint ranks of a pair hold a record; interior ranks
# (extra cloud TP ranks) join the group collective but never touch the
# channels and hold nothing.
_LWD_CHANNEL_GROUPS: dict[
    tuple[LwdChannelType, int, int], tuple[dist.ProcessGroup, int]
] = {}
# (channel, instance_id) -> dedicated NPU stream.
_LWD_CHANNEL_STREAMS: dict[tuple[LwdChannelType, int], "torch.npu.Stream"] = {}
# (channel, instance_id, dp_rank) -> (edge_rank, cloud_rank) — global
# layout knowledge, populated on EVERY rank (endpoint or not) so any
# caller can resolve pair membership (e.g. a cloud worker asking
# whether it is its instance's endpoint) without re-deriving arithmetic.
_LWD_CHANNEL_PAIRS: dict[tuple[LwdChannelType, int, int], tuple[int, int]] = {}
# dp index of THIS process (per-rank topology: one dp per edge/cloud
# process; the getters resolve records with it implicitly).
_MY_DP: int = 0
_INITIALIZED = False


def dump_tensor(tag: str, tensor: "torch.Tensor") -> None:
    """调试:张量摘要打印(head10/tail10/sum/mean/shape/dtype),
    供边云两端成对对比数值。"""
    import numpy as np

    flat = tensor.detach().to("cpu", torch.float32).numpy().reshape(-1)
    with np.printoptions(threshold=np.inf, linewidth=10000, precision=8):
        logger.info(
            "%s shape=%s dtype=%s head10=%s tail10=%s sum=%.6f mean=%.8f",
            tag, tuple(tensor.shape), tensor.dtype,
            flat[:10], flat[-10:], float(flat.sum()), float(flat.mean()),
        )


def _dp_of_rank(
    rank: int, e_per: int, e_total: int, d: int, c_per: int
) -> int:
    """dp index of a global rank under the §4.1 layout (edge ranks
    first, then instance-major cloud blocks)."""
    if rank < e_total:
        return (rank // e_per) if e_per > 0 else 0
    if c_per > 0 and d > 0:
        return ((rank - e_total) % (d * c_per)) // c_per
    return 0


def init_lwd_duplex_channels() -> None:
    """Create the duplex channel pairs for every (instance, dp) edge/
    cloud rank pair of the §4.1 layout.

    Called once per worker process when ``is_prefill_only`` is on.
    Idempotent.

    Endpoint ranks are resolved from existing deployment info, in
    order:
      1. ``lwd_config`` NPU counts — one pair per (instance i, engine j):
         ``(edge_rank, cloud_first(i, j))`` where
         ``cloud_first(i, j) = E + i*D*C + j*C`` and ``edge_rank`` is 0
         for the shared edge (E==1, D>1), otherwise ``j*E_per``;
      2. PP-group convention fallback (demo 1E1C topology, counts
         unset): a 2-rank PP group spans exactly the edge/cloud pair
         (rank[0]=edge, rank[1]=cloud), recorded as instance 0, engine 0.

    Only the two endpoint ranks of each pair ever touch its channels;
    interior ranks (extra cloud TP ranks) join every group collective
    but hold no record.
    """
    global _INITIALIZED, _MY_DP
    if _INITIALIZED:
        return
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    lwd_cfg = parallel_config.lwd_config
    my_rank = dist.get_rank()

    # (instance, dp) -> (edge_rank, cloud_rank), in the fixed creation
    # order: instance ascending, dp ascending.
    pair_list: list[tuple[int, int, tuple[int, int]]] = []
    if lwd_cfg.enable_lwd and lwd_cfg.edge_npu_count > 0:
        # 云 DP 轴以 cloud_dp_size 为准(边侧本地 data_parallel_size
        # 可能为 1);实例维度保留 B 的 instance_parallel_size。
        d = lwd_cfg.cloud_dp_size
        n = lwd_cfg.instance_parallel_size
        e_per = lwd_cfg.edge_npu_count_per_dp(d)
        e_total = lwd_cfg.edge_rank_count(d)
        c_per = lwd_cfg.cloud_npu_count_per_instance_dp(d)
        # 共享边(E==1 且 D>1):唯一边卡被所有 engine 复用,端点边 rank
        # 恒为 0;否则按 dp 切边卡段(j*E_per)。
        shared_edge = d > 1 and e_total == 1
        for i in range(n):
            for j in range(d):
                edge_rank = 0 if shared_edge else j * e_per
                pair_list.append(
                    (
                        i,
                        j,
                        (edge_rank, lwd_cfg.lwd_cloud_first_rank(i, j, d)),
                    )
                )
        _MY_DP = _dp_of_rank(my_rank, e_per, e_total, d, c_per)
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
        pair_list.append((0, 0, (pp_group.ranks[0], pp_group.ranks[1])))
        _MY_DP = 0

    backend = dist.get_backend(get_world_group().device_group)
    for i, j, ranks in pair_list:
        for channel in LwdChannelType:
            # Both channels span the same rank pair; direction is given
            # by who sends. new_group is a world collective — every rank
            # must participate in every creation, in the same order.
            group = dist.new_group(list(ranks), backend=backend)
            _LWD_CHANNEL_PAIRS[(channel, i, j)] = ranks
            if my_rank in ranks:
                peer = ranks[1] if my_rank == ranks[0] else ranks[0]
                _LWD_CHANNEL_GROUPS[(channel, i, j)] = (group, peer)
                stream_key = (channel, i, j)
                if stream_key not in _LWD_CHANNEL_STREAMS:
                    _LWD_CHANNEL_STREAMS[stream_key] = torch.npu.Stream(priority=-1)
    # embeds 组内广播不再自建通信域(自建 new_group 在 torch_npu 下实测
    # 零交付,疑建组顺序错位),改用框架 TP device_group(demo 同款,
    # 建组顺序由框架保证);广播在注入时刻于计算流上发起,与前向 TP
    # 集合同流 FIFO,天然不串台。
    _INITIALIZED = True
    logger.info(
        "[lwd-wire] duplex channels created over %d (instance, dp) pair(s) "
        "(my_rank=%d my_dp=%d, owned_records=%d)",
        len(pair_list),
        my_rank,
        _MY_DP,
        len(_LWD_CHANNEL_GROUPS),
    )
    # Every owned pair is warmed up in the fixed creation order: the
    # first-use HCCL rendezvous is moved to init time, where both sides
    # are guaranteed to arrive in the same order.
    warmup_lwd_duplex_channels()


def warmup_lwd_duplex_channels() -> None:
    """Pre-establish every owned channel pair's P2P links at init time.

    A tiny payload is exchanged on each channel in the fixed global
    order (instance ascending, dp ascending, UP then DOWN — the
    tables' insertion order), so both sides rendezvous identically
    regardless of when the first real message is posted. The edge
    endpoint (which owns all N instance pairs) warms every one of them;
    a cloud endpoint warms only its own instance's pair. Non-endpoint
    ranks participate in new_group/barrier (world collectives) but
    skip the P2P ops.
    """
    my_rank = dist.get_rank()
    for (channel, i, j), (group, peer) in _LWD_CHANNEL_GROUPS.items():
        edge_rank, cloud_rank = _LWD_CHANNEL_PAIRS[(channel, i, j)]
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
                "[lwd-warmup] SEND post channel=%s instance=%d engine=%d "
                "my_rank=%d peer=%d group_ranks=%s",
                channel, i, j, my_rank, peer,
                dist.get_process_group_ranks(group))
            handle = dist.isend(payload, dst=peer, group=group)
        else:
            logger.info(
                "[lwd-warmup] RECV post channel=%s instance=%d engine=%d "
                "my_rank=%d src=%d group_ranks=%s",
                channel, i, j, my_rank, peer,
                dist.get_process_group_ranks(group))
            handle = dist.irecv(payload, src=peer, group=group)
        handle.wait()
        logger.info(
            "[lwd-warmup] DONE channel=%s instance=%d engine=%d my_rank=%d",
            channel, i, j, my_rank,
        )
    get_world_group().barrier()
    logger.info(
        "[lwd-wire] duplex channels warmed up (UP + DOWN, %d pair(s))",
        len({(i, j) for (_, i, j) in _LWD_CHANNEL_GROUPS}),
    )


def lwd_channels_initialized() -> bool:
    return _INITIALIZED


def _owned_key(
    channel: LwdChannelType, instance_id: int, engine_index: int | None
) -> tuple[LwdChannelType, int, int]:
    key = (
        channel,
        int(instance_id),
        _MY_DP if engine_index is None else int(engine_index),
    )
    if key not in _LWD_CHANNEL_GROUPS:
        raise RuntimeError(
            f"no lwd duplex channel record for channel={channel} "
            f"instance={instance_id} engine={key[2]}: this rank "
            f"({dist.get_rank()}) is not an endpoint of that pair "
            "(or the wire layer is not initialized)"
        )
    return key


def get_lwd_channel_device_group(
    channel: LwdChannelType,
    instance_id: int = 0,
    engine_index: int | None = None,
) -> dist.ProcessGroup:
    key = _owned_key(channel, instance_id, engine_index)
    return _LWD_CHANNEL_GROUPS[key][0]


def get_lwd_channel_peer(
    channel: LwdChannelType,
    instance_id: int = 0,
    engine_index: int | None = None,
) -> int:
    """Global rank of this process's P2P peer on the (instance, engine) channel."""
    key = _owned_key(channel, instance_id, engine_index)
    return _LWD_CHANNEL_GROUPS[key][1]


def get_lwd_channel_stream(
    channel: LwdChannelType,
    instance_id: int = 0,
    engine_index: int | None = None,
) -> "torch.npu.Stream":
    key = (
        channel,
        int(instance_id),
        _MY_DP if engine_index is None else int(engine_index),
    )
    stream = _LWD_CHANNEL_STREAMS.get(key)
    if stream is None:
        raise RuntimeError(
            f"no lwd channel stream for channel={channel} "
            f"instance={instance_id} engine={key[2]}: this rank owns no "
            "endpoint of that pair (streams exist on endpoints only)"
        )
    return stream


def lwd_pair_ranks(
    instance_id: int = 0, engine_index: int | None = None
) -> tuple[int, int]:
    """``(edge_rank, cloud_rank)`` of one (instance, engine) channel pair.

    Layout knowledge available on EVERY rank (endpoint or not); the
    cloud endpoint check is ``rank == lwd_pair_ranks(i, j)[1]``."""
    key = (
        LwdChannelType.UP,
        int(instance_id),
        _MY_DP if engine_index is None else int(engine_index),
    )
    ranks = _LWD_CHANNEL_PAIRS.get(key)
    if ranks is None:
        raise RuntimeError(
            f"no lwd channel pair record for instance={instance_id} "
            f"engine={key[2]} (wire layer not initialized?)"
        )
    return ranks


def lwd_my_dp() -> int:
    """dp index of this process under the §4.1 layout."""
    return _MY_DP


def destroy_lwd_duplex_channels() -> None:
    global _INITIALIZED, _MY_DP
    for (channel, i, j), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning(
                "[lwd-wire] failed to destroy group for %s instance=%d dp=%d",
                channel.value, i, j,
            )
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_CHANNEL_PAIRS.clear()
    _MY_DP = 0
    _INITIALIZED = False
