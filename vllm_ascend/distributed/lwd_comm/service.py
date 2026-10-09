# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdCommService: singleton entry point of the duplex comm layer.

Compute side only ever calls ``submit_send`` / ``submit_recv`` and
consumes ``LwdCommFuture`` notifications.  ``poll_completions`` is meant
to be invoked from an existing host loop head (worker busy loop);
it drives keepalive reclamation via lazy reap.

Ported (simplified) from the demo branch's ``lwd_comm/service.py``.
"""

from __future__ import annotations

import threading
import time

from vllm.logger import logger

from vllm_ascend.distributed.lwd_comm.channel import LwdChannel
from vllm_ascend.distributed.lwd_comm.future import LwdCommFuture
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest


class LwdCommService:
    """Process-wide singleton.  All public methods are thread-safe."""

    _instance: "LwdCommService | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LwdCommService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # HCCL ordering is per physical (instance, engine, channel,
        # direction): send and recv have INDEPENDENT seqno streams (each
        # peer's send order must match the other side's recv-post
        # order), and each (cloud instance, cloud engine) pair owns a
        # dedicated wire pair whose seqno space starts at 0 — so the
        # FIFO key is
        # (instance_id, engine_index, channel, op, effective_peer).
        # Merging any axis would make seqno streams collide across
        # instances/engines/directions.  The peer axis is redundant in
        # the per-rank topology (one endpoint per (channel, instance,
        # engine) pair) but keeps the shared-edge form (one process,
        # several peers per pair) correct.
        self._channels: dict[
            tuple[int, int, LwdChannelType, str, int], LwdChannel
        ] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit_send(self, request: LwdCommRequest) -> LwdCommFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdCommRequest) -> LwdCommFuture:
        """Submit a recv.  The irecv is posted immediately — "early" is a
        property of when you submit, not a separate API."""
        assert request.op == "recv", "submit_recv requires op='recv'"
        return self._submit(request)

    # ------------------------------------------------------------------ #
    # Completion driving                                                  #
    # ------------------------------------------------------------------ #

    def poll_completions(self) -> int:
        """Reap every channel once.  Call from an existing loop head;
        costs one head-of-line ``event.query()`` per channel with
        pending traffic."""
        with self._lock:
            channels = list(self._channels.values())
        completed = 0
        for channel in channels:
            completed += len(channel.reap())
        return completed

    def skip_seqno(
        self,
        channel_type: LwdChannelType,
        seqno: int,
        op: str = "recv",
        instance_id: int = 0,
        engine_index: int = 0,
    ) -> None:
        """Mark a seqno as never-to-arrive (aborted request) on the given
        channel+direction FIFO of one (instance, engine).  Recv managers'
        drop path passes op="recv"; the sender-side abort glue passes
        op="send".  Both peers must skip the same seqnos on the same
        (instance, engine)."""
        with self._lock:
            peer = _wire_peer_or_none(channel_type, instance_id, engine_index)
            channel = self._channels.get(
                (
                    int(instance_id),
                    int(engine_index),
                    channel_type,
                    op,
                    peer,
                )
            )
        if channel is not None:
            channel.skip_seqno(seqno)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            channels = list(self._channels.values())
        try:
            for channel in channels:
                channel.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._channels.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _submit(self, request: LwdCommRequest) -> LwdCommFuture:
        instance_id = int(getattr(request, "instance_id", 0) or 0)
        engine_index = int(getattr(request, "engine_index", 0) or 0)
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("lwd-comm service is shutting down")
            peer = request.src_dst
            if peer is None:
                peer = _resolve_wire_peer(
                    request.channel, instance_id, engine_index
                )
            key = (
                instance_id,
                engine_index,
                request.channel,
                request.op,
                peer,
            )
            channel = self._channels.get(key)
            if channel is None:
                channel = LwdChannel(
                    request.channel,
                    request.op,
                    instance_id=instance_id,
                    engine_index=engine_index,
                )
                self._channels[key] = channel
                logger.info(
                    "[lwd-comm] created channel %s/%s (instance=%d "
                    "engine=%d peer=%s)",
                    request.channel.value,
                    request.op,
                    instance_id,
                    engine_index,
                    peer,
                )
        return channel.submit(request)


def _resolve_wire_peer(
    channel: LwdChannelType, instance_id: int, engine_index: int = 0
) -> int:
    """Default peer of (channel, instance, engine) from the wire table."""
    from vllm_ascend.distributed import lwd_wire

    if not lwd_wire.lwd_channels_initialized():
        raise RuntimeError(
            "lwd duplex channels are not initialized; call "
            "init_lwd_duplex_channels() before submitting comm requests"
        )
    return lwd_wire.get_lwd_channel_peer(
        channel, instance_id, engine_index
    )


def _wire_peer_or_none(
    channel: LwdChannelType, instance_id: int, engine_index: int = 0
) -> int | None:
    """Peer used at channel-creation time, for skip_seqno lookups; None
    when the wire layer is not up (no channel could exist either)."""
    try:
        return _resolve_wire_peer(channel, instance_id, engine_index)
    except RuntimeError:
        return None


def get_lwd_comm_service() -> LwdCommService:
    return LwdCommService.instance()
