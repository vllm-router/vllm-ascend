# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UP and DOWN channels of the LWD edge-cloud data plane.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge):
  * channel HCCL communicators: two dedicated ``new_group`` groups per DP
    connection, over its endpoint pair (``LwdChannelEndpoints``: the first
    edge rank and the first cloud rank);
  * one NPU stream per channel of each DP connection the local rank
    endpoints (channel ops are bridged onto it, keeping the compute stream
    free);
  * init-time warmup of every connection's channels in a fixed global
    order (moves the first-use HCCL rendezvous out of the pipeline, where
    the two sides could otherwise arrive on different channels and
    deadlock); the warmup payload is an identity probe frame verified by
    the receiver (see ``lwd_channel_probe``).

In a single_instance topology the edge process serves every DP
connection, so its endpoint rank holds the channel pairs of all of them;
a cloud endpoint rank holds exactly its own DP's pair.

Design: docs/design/lwd_data_plane.md in the vLLM repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_channel_probe import (
    LWD_PROBE_TIMEOUT_S,
    lwd_probe_aggregate,
    lwd_probe_frame,
    lwd_probe_verify,
    lwd_probe_wait,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

# (dp_idx, channel) -> (device_group, peer_global_rank), endpoint ranks only
_LWD_CHANNEL_GROUPS: dict[tuple[int, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[int, LwdChannelType], torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[LwdChannelEndpoints, ...] = ()
_LWD_CHANNELS_INITIALIZED = False


def lwd_init_channels() -> None:
    """Create the UP and DOWN channels of every DP connection. Idempotent.

    Called once by every worker of the shared world after its distributed
    environment is up. Group creation is a world collective, so every rank
    creates the groups of every connection in the order of
    ``LwdParallelGroupsSpec``; only the two endpoint ranks keep them.
    """
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    if _LWD_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for endpoints in spec.channel_endpoints:
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        for channel in LwdChannelType:
            # Both channels span the same rank pair; the direction is given
            # by who sends.
            group = dist.new_group(ranks, backend=backend)
            if my_rank in ranks:
                peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
                _LWD_CHANNEL_GROUPS[(endpoints.dp_idx, channel)] = (group, peer)
                _LWD_CHANNEL_STREAMS[(endpoints.dp_idx, channel)] = torch.npu.Stream()
    _LWD_ENDPOINTS = spec.channel_endpoints_of(my_rank)
    # The cloud-internal embeds broadcast does not build its own group: a
    # dedicated new_group delivered nothing under torch_npu in testing
    # (suspected group creation order mismatch), so it uses the framework TP
    # device_group, whose creation order the framework guarantees. The
    # broadcast is issued on the compute stream at injection time, FIFO with
    # the forward TP collectives, so the two cannot interleave.
    _LWD_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] data-plane channels created for %s (my_rank=%d, endpoints=%s)",
        spec.channel_endpoints,
        my_rank,
        _LWD_ENDPOINTS,
    )
    # Every connection's two channels, always warmed up: the first-use HCCL
    # rendezvous is moved to init time, where both sides are guaranteed to
    # arrive in the same fixed order.
    lwd_warm_up_channels()


class _ProbeOutcome(Enum):
    """The three possible endings of one channel probe."""

    OK = "ok"
    MISMATCH = "mismatch"  # op completed, frame wrong: keep probing
    TIMEOUT = "timeout"  # channel wedged: stop posting ops


@dataclass(frozen=True)
class _ChannelProbeResult:
    outcome: _ProbeOutcome
    detail: str = ""


def _lwd_probe_timeout_detail(dp_idx: int, channel: LwdChannelType, peer: int) -> str:
    """Format the shared TIMEOUT detail for a channel probe."""
    return f"[dp_idx={dp_idx} channel={channel.value}] probe TIMEOUT after {LWD_PROBE_TIMEOUT_S:g}s (peer={peer})"


def _lwd_probe_channel(
    group: dist.ProcessGroup,
    peer: int,
    dp_idx: int,
    channel: LwdChannelType,
    is_sender: bool,
) -> _ChannelProbeResult:
    """Post one probe op on a channel and check the result.

    Returns one of three outcomes: ``OK``, ``MISMATCH`` (the op completed but
    the frame is wrong -- keep probing), or ``TIMEOUT`` (the channel is wedged
    -- stop posting ops).  The backend check runs before any op is posted, so
    a degraded (gloo) group fails here instead of probing a non-HCCL link.
    """
    backend = str(dist.get_backend(group))
    if backend != "hccl":
        raise RuntimeError(f"channel group backend must be hccl, got {backend}")
    my_rank = dist.get_rank()
    if is_sender:
        # The frame is the identity proof; the sender only needs the op done.
        payload = lwd_probe_frame(my_rank, dp_idx, channel)
        handle = dist.isend(payload, dst=peer, group=group)
        if not lwd_probe_wait(handle):
            return _ChannelProbeResult(_ProbeOutcome.TIMEOUT, _lwd_probe_timeout_detail(dp_idx, channel, peer))
        return _ChannelProbeResult(_ProbeOutcome.OK)
    # Zero-filled buffer: a no-op irecv or a stale frame fails the [0] magic
    # check (magic is 90, so an all-zero frame is never a legal one).
    payload = torch.zeros_like(lwd_probe_frame(peer, dp_idx, channel))
    handle = dist.irecv(payload, src=peer, group=group)
    if not lwd_probe_wait(handle):
        return _ChannelProbeResult(_ProbeOutcome.TIMEOUT, _lwd_probe_timeout_detail(dp_idx, channel, peer))
    # The poll above only observed completion; the CPU must sync before
    # reading the received buffer.
    torch.npu.synchronize()
    problem = lwd_probe_verify(payload, peer, dp_idx, channel)
    if problem:
        return _ChannelProbeResult(
            _ProbeOutcome.MISMATCH,
            f"[dp_idx={dp_idx} channel={channel.value}] frame mismatch (peer={peer}): {problem}",
        )
    return _ChannelProbeResult(_ProbeOutcome.OK)


def lwd_warm_up_channels() -> None:
    """Pre-establish and verify every connection's channel P2P links at init.

    Each channel exchanges one identity probe frame over the existing
    isend/irecv round-trip in the fixed global order (connection order, UP
    then DOWN, zero added P2P); the receiver compares it field by field
    against the frame it derives from the same public facts.  Failures are
    aggregated over the CPU/gloo world group so every rank raises with the
    same verdict instead of hanging on the 1836s HCCL timeout; non-endpoint
    ranks only join the aggregation and the closing barrier.

    Failure policy: a verify failure is recorded and probing continues --
    the peer's op did complete, and HCCL P2P has no tags, so the op
    count/order must stay invariant for later pairing.  A poll TIMEOUT
    breaks at once (the peer's op cannot complete either, so it times out
    symmetrically) and skips the rollback destroy, which could hang on a
    wedged communicator and forfeit the fast fail; the platform restarts
    the process.  An unexpected raise inside the channel probe is treated
    the same way, because posting further ops on unknown state only
    manufactures untraceable paths.
    """
    failures: list[str] = []
    aborted = False
    skipped: list[str] = []
    if _LWD_ENDPOINTS:
        my_rank = dist.get_rank()
        planned = [
            (endpoints, channel) for endpoints in _LWD_ENDPOINTS for channel in (LwdChannelType.UP, LwdChannelType.DOWN)
        ]
        for index, (endpoints, channel) in enumerate(planned):
            group, peer = _LWD_CHANNEL_GROUPS[(endpoints.dp_idx, channel)]
            sender = endpoints.edge_rank if channel is LwdChannelType.UP else endpoints.cloud_rank
            try:
                result = _lwd_probe_channel(group, peer, endpoints.dp_idx, channel, my_rank == sender)
            except Exception as exc:
                # The state is unknown (OOM, broken NPU context, a degraded
                # backend): probing on it only manufactures untraceable
                # paths, so treat it like a TIMEOUT and stop posting ops.
                # BaseException keeps propagating.
                failures.append(f"[dp_idx={endpoints.dp_idx} channel={channel.value}] probe raised: {exc!r}")
                aborted = True
                skipped = [f"dp{ep.dp_idx}/{ch.value}" for ep, ch in planned[index + 1 :]]
                break
            if result.outcome is _ProbeOutcome.OK:
                logger.info(
                    "[LWD] data-plane warmup done: dp_idx=%d channel=%s my_rank=%d peer=%d",
                    endpoints.dp_idx,
                    channel,
                    my_rank,
                    peer,
                )
                continue
            failures.append(result.detail)
            if result.outcome is _ProbeOutcome.TIMEOUT:
                # The channel is wedged: the peer times out symmetrically,
                # so stop posting ops entirely.
                aborted = True
                skipped = [f"dp{ep.dp_idx}/{ch.value}" for ep, ch in planned[index + 1 :]]
                break
    if skipped:
        logger.warning(
            "[LWD] channel probe aborted; %d channel(s) left unprobed: %s",
            len(skipped),
            ", ".join(skipped),
        )
    # Verdict over the CPU/gloo group, never through the suspected HCCL
    # channels; every rank (endpoints and non-endpoints) must reach this.
    if not lwd_probe_aggregate(not failures, get_world_group().cpu_group):
        detail = "; ".join(failures) or "remote rank(s) reported probe failure"
        logger.error("[LWD] channel probe failed: %s", detail)
        if not aborted:
            # All ops completed and the groups are healthy: roll back for a
            # clean restart.  (On TIMEOUT/raise the groups' state is unknown
            # and they are deliberately left for process exit to reclaim.)
            lwd_destroy_channels()
        raise RuntimeError(f"[LWD] channel probe failed: {detail}")
    get_world_group().barrier()


def lwd_channels_initialized() -> bool:
    return _LWD_CHANNELS_INITIALIZED


def _lwd_channel(dp_idx: int, channel: LwdChannelType) -> tuple[dist.ProcessGroup, int]:
    try:
        return _LWD_CHANNEL_GROUPS[(dp_idx, channel)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no {channel.value} channel of "
            f"dp_idx {dp_idx}; only the endpoint ranks of that DP connection do"
        ) from None


def lwd_get_channel_device_group(channel: LwdChannelType, dp_idx: int = 0) -> dist.ProcessGroup:
    group, _ = _lwd_channel(dp_idx, channel)
    return group


def lwd_get_channel_peer(channel: LwdChannelType, dp_idx: int = 0) -> int:
    """Global rank of this process's P2P peer on the DP connection's channel."""
    _, peer = _lwd_channel(dp_idx, channel)
    return peer


def lwd_get_channel_stream(channel: LwdChannelType, dp_idx: int = 0) -> torch.npu.Stream:
    _lwd_channel(dp_idx, channel)
    return _LWD_CHANNEL_STREAMS[(dp_idx, channel)]


def lwd_destroy_channels() -> None:
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    for (dp_idx, channel), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[LWD] failed to destroy the data-plane group for dp%d %s", dp_idx, channel.value)
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = ()
    _LWD_CHANNELS_INITIALIZED = False
