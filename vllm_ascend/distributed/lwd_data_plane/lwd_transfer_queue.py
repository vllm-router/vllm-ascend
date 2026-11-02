# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdTransferQueue: one strict FIFO of transfers per channel and op.

Simplified port of the demo branch's ``lwd_comm/channel.py`` for
the prefill_only data plane (two direction-only channels,
single-tensor payloads).

Mechanics preserved from the demo:
  * send payload snapshot at submit-entry (graph replay / staging buffer
    reuse safety), keepalive released by lazy reap — no background
    thread;
  * per-channel seqno reorder buffer: ops are posted to HCCL only once
    all lower seqnos have been submitted, so the send order and the
    peer's recv-post order always agree (HCCL P2P has no tags);
  * every channel op is bridged onto the channel stream via
    ``handle.wait()`` (non-blocking HCCL semantics) followed by an event
    record; the next op waits for the predecessor's event.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane import lwd_channels
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest


def _lwd_scramble_seed_for(request: LwdTransferRequest) -> int | None:
    """Derive the per-flight seed from the request's own fields, or None.

    The seed's three inputs (channel / seqno / num_elements) already live on
    the request, so the data plane derives it here instead of having the
    worker pre-compute and duplicate them.  Returns None when scramble is off
    (no process-wide scrambler has been built).
    """
    from vllm_ascend.distributed.lwd_data_plane.lwd_payload_scrambler import (
        lwd_get_payload_scrambler,
        lwd_payload_scrambler_initialized,
    )

    if not lwd_payload_scrambler_initialized():
        return None
    return lwd_get_payload_scrambler().seed_for(
        channel=request.channel.value,
        seqno=request.seqno,
        num_elements=request.num_elements,
    )


class LwdTransferQueue:
    """One FIFO of pending requests for one DP connection, channel and op."""

    def __init__(self, dp_idx: int, channel_type: LwdChannelType, op: str) -> None:
        self.dp_idx = dp_idx
        self.channel_type = channel_type
        self.op = op  # one FIFO carries exactly one direction's seqno stream
        self._pending: deque[LwdTransferFuture] = deque()
        self._lock = threading.Lock()
        # The pending-queue lock alone cannot prevent two host threads
        # from interleaving HCCL calls of separate requests.
        self._submission_lock = threading.Lock()
        self._next_seqno: int | None = None
        self._held: dict[int, tuple[LwdTransferRequest, LwdTransferFuture]] = {}
        # Abort holes: seqnos the peer will never submit (aborted
        # requests).  Skipped when advancing so one abort cannot wedge
        # every later request on this channel.
        self._skipped: set[int] = set()

    # ------------------------------------------------------------------ #
    # Abort holes                                                         #
    # ------------------------------------------------------------------ #

    def skip_seqno(self, seqno: int) -> None:
        """Mark a seqno as never-to-arrive (aborted request) and advance
        past it.  Both peers must skip the same seqno (driven by the
        control-plane abort on both sides) — skipping a seqno the peer
        later submits is a pairing error and raises at submission."""
        with self._submission_lock:
            if self._next_seqno is None:
                self._next_seqno = 0
            if seqno < self._next_seqno:
                return  # already passed
            # A held (out-of-order) entry for an aborted seqno must never
            # be executed: discard it and fail its deferred future so any
            # waiter fails promptly instead of waiting out the timeout.
            held = self._held.pop(seqno, None)
            if held is not None:
                held[1]._finalize(error=RuntimeError(f"seqno {seqno} aborted on channel {self.channel_type.value}"))
            self._skipped.add(seqno)
            self._advance_past_skipped()

    def _advance_past_skipped(self) -> None:
        """Caller holds the submission lock."""
        while self._next_seqno in self._skipped:
            self._skipped.discard(self._next_seqno)
            self._next_seqno += 1
            while self._next_seqno in self._held:
                held_request, held_future = self._held.pop(self._next_seqno)
                self._execute_next(held_request, into=held_future)
                self._next_seqno += 1

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        """Execute the channel op and enqueue the future."""
        if request.op != self.op:
            raise RuntimeError(
                f"op mismatch on {self.channel_type.value} FIFO: channel carries {self.op!r}, got {request.op!r}"
            )
        finalized = self._reap()
        self._finalize_many(finalized)
        if request.op == "send":
            request = self._snapshot_send(request)
        with self._submission_lock:
            if request.seqno is not None:
                return self._submit_sequenced(request)
            return self._execute_next(request)

    @staticmethod
    def _snapshot_send(request: LwdTransferRequest) -> LwdTransferRequest:
        """Give the data plane ownership of the send payload."""
        assert request.tensor is not None, "send requires tensor"
        owned = request.tensor.detach().clone()
        seed = _lwd_scramble_seed_for(request)
        if seed is not None:
            # Scramble the communication-owned snapshot (never the model
            # tensor the caller still owns).
            from vllm_ascend.distributed.lwd_data_plane.lwd_payload_scrambler import (
                lwd_get_payload_scrambler,
            )

            lwd_get_payload_scrambler().scramble(owned, seed=seed)
        aux_owned = request.aux_tensor.detach().clone() if request.aux_tensor is not None else None
        return replace(request, tensor=owned, aux_tensor=aux_owned)

    # ------------------------------------------------------------------ #
    # Sequenced submission (reorder buffer)                               #
    # ------------------------------------------------------------------ #

    def _submit_sequenced(self, request: LwdTransferRequest) -> LwdTransferFuture:
        """Post in per-channel seqno order.  Caller holds the submission
        lock."""
        seqno = request.seqno
        assert seqno is not None
        if self._next_seqno is None:
            # Contract: per-channel seqno counters start at 0.
            self._next_seqno = 0
        self._advance_past_skipped()
        if seqno in self._skipped or seqno < self._next_seqno:
            # A skipped (aborted) seqno can still arrive when the peer's
            # abort raced its send — do NOT crash the process; complete
            # the request with an error result so the caller's readiness
            # gate fails this request promptly.
            logger.warning(
                "[lwd-comm] dropping submission of skipped/duplicate seqno %d on channel %s %s (next expected %d)",
                seqno,
                self.channel_type.value,
                self.op,
                self._next_seqno,
            )
            future = LwdTransferFuture(request, [], None)
            future._finalize(
                error=RuntimeError(f"seqno {seqno} was skipped (aborted) on channel {self.channel_type.value}")
            )
            return future
        if seqno > self._next_seqno:
            if seqno in self._held:
                # Duplicate submission of an already-held seqno: return
                # the existing deferred future instead of overwriting it
                # (its waiter would otherwise spin until timeout).
                return self._held[seqno][1]
            future = LwdTransferFuture.deferred(request)
            self._held[seqno] = (request, future)
            logger.debug(
                "[lwd-comm] held out-of-order %s %s seqno=%d (next=%d)",
                self.channel_type.value,
                request.op,
                seqno,
                self._next_seqno,
            )
            return future
        future = self._execute_next(request)
        self._next_seqno += 1
        # TODO(lwd): confirm whether this drain should also step over skipped
        # seqnos, as _advance_past_skipped does: a held seqno right after a
        # skipped one is only posted on the next submit or skip_seqno call.
        while self._next_seqno in self._held:
            held_request, held_future = self._held.pop(self._next_seqno)
            self._execute_next(held_request, into=held_future)
            self._next_seqno += 1
        return future

    def _execute_next(
        self,
        request: LwdTransferRequest,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        """Execute one channel op as the channel's new tail."""
        with self._lock:
            predecessor = self._pending[-1] if self._pending else None
        future = self._execute(request, predecessor, into=into)
        with self._lock:
            self._pending.append(future)
        return future

    # ------------------------------------------------------------------ #
    # Reaping (head-of-line query)                                        #
    # ------------------------------------------------------------------ #

    def reap(self) -> list[LwdTransferFuture]:
        """Pop and finalize every completed request from the head.

        FIFO completion is in-order: if the head is not done, nothing
        behind it can be either, so one ``event.query()`` per call per
        channel suffices.
        """
        finalized = self._reap()
        self._finalize_many(finalized)
        return finalized

    def shutdown(self, timeout: float | None = None) -> list[LwdTransferFuture]:
        """Wait for pending operations before releasing owned buffers."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._submission_lock:
            if self._held:
                logger.warning(
                    "[lwd-comm] shutdown on channel %s with %d held (never-posted) request(s); lowest missing seqno=%s",
                    self.channel_type.value,
                    len(self._held),
                    self._next_seqno,
                )
                # Fail the deferred futures before dropping them: their
                # waiters must not hang until the readiness-gate timeout.
                for _, held_future in self._held.values():
                    held_future._finalize(
                        error=RuntimeError(
                            f"channel {self.channel_type.value} shut down with the request still held (never posted)"
                        )
                    )
                self._held.clear()
            with self._lock:
                pending = list(self._pending)
            for future in pending:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                future.wait(timeout=remaining)
            finalized = self._reap()
            self._finalize_many(finalized)
            return finalized

    def _reap(self) -> list[LwdTransferFuture]:
        done: list[LwdTransferFuture] = []
        with self._lock:
            while self._pending and self._pending[0].done():
                done.append(self._pending.popleft())
        return done

    @staticmethod
    def _finalize_many(futures: list[LwdTransferFuture]) -> None:
        # TODO(lwd): confirm whether a transfer that completed in ERROR should
        # raise here: no caller holds send futures, so a failed send is
        # finalized silently.
        for future in futures:
            future._finalize()

    # ------------------------------------------------------------------ #
    # Channel ops                                                         #
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        req: LwdTransferRequest,
        predecessor: LwdTransferFuture | None,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        """Issue the channel op and record device and host completion."""
        tensor: torch.Tensor | None = None
        aux_tensor: torch.Tensor | None = None
        keepalive: Any = None
        staged_event: torch.npu.Event | None = None
        stream = self._stream()
        # Capture the producer stream BEFORE entering the channel-stream
        # context: inside the `with` block torch.npu.current_stream() IS
        # the channel stream, so waiting on it there would be a no-op
        # self-wait and the send could read a half-written snapshot.
        producer_stream = torch.npu.current_stream()
        # The channel op MUST be issued on the channel stream: seqno
        # ordering only holds if every op on this FIFO lands on the same
        # device stream (HCCL P2P has no tags; matching order = device
        # arrival order, not host submission order).
        with torch.npu.stream(stream):
            self._order_after(predecessor)
            if req.op == "send":
                assert req.tensor is not None, "send requires tensor"
                # The snapshot was produced on the producer stream; order
                # the channel stream after it.
                stream.wait_stream(producer_stream)
                handles = self._post_send(req)
                keepalive = (req.tensor, req.aux_tensor)
            else:
                tensor, aux_tensor, handles = self._post_recv(req)
            done_event = self._bridge_and_record(handles)
            seed = _lwd_scramble_seed_for(req) if req.op == "recv" else None
            if seed is not None:
                # `_bridge_and_record` waits on the HCCL handle, so the recv
                # payload is already complete here; XOR is its own inverse,
                # so applying it in place undoes the sender's scramble.  The
                # completion event is re-recorded AFTER the descramble so the
                # consumer's wait() cannot observe the tensor mid-scramble.
                from vllm_ascend.distributed.lwd_data_plane.lwd_payload_scrambler import (
                    lwd_get_payload_scrambler,
                )

                assert tensor is not None, "recv requires tensor"
                lwd_get_payload_scrambler().descramble(tensor, seed=seed)
                done_event = torch.npu.Event()
                done_event.record()
            if req.op == "recv" and req.stage_segments is not None:
                assert tensor is not None and done_event is not None
                staged_event = self._stage_recv(tensor, done_event, req.stage_segments)
        future_request = replace(req, tensor=None) if req.op == "send" else req
        if into is None:
            return LwdTransferFuture(
                request=future_request,
                handles=handles,
                done_event=done_event,
                tensor=tensor,
                keepalive=keepalive,
                aux_tensor=aux_tensor,
                staged_event=staged_event,
            )
        into._bind(
            handles=handles,
            done_event=done_event,
            tensor=tensor,
            keepalive=keepalive,
            aux_tensor=aux_tensor,
            staged_event=staged_event,
        )
        return into

    def _stream(self) -> Any:
        return lwd_channels.lwd_get_channel_stream(self.channel_type, self.dp_idx)

    def _copy_stream(self) -> torch.npu.Stream:
        stream = getattr(self, "_d2h_copy_stream", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._d2h_copy_stream = stream
        return stream

    def _stage_recv(
        self,
        tensor: torch.Tensor,
        done_event: torch.npu.Event,
        segments: list[tuple[Any, int, int]],
    ) -> torch.npu.Event:
        """Copy staged segments after HCCL and descramble complete."""
        copy_stream = self._copy_stream()
        copy_stream.wait_event(done_event)
        with torch.npu.stream(copy_stream):
            for dst, src_offset, numel in segments:
                dst.copy_(tensor[src_offset : src_offset + numel], non_blocking=True)
            staged_event = torch.npu.Event()
            staged_event.record()
        return staged_event

    @staticmethod
    def _validate_stage_segments(req: LwdTransferRequest) -> None:
        if req.stage_segments is None:
            return
        for dst, src_offset, numel in req.stage_segments:
            if src_offset < 0 or numel < 0 or src_offset + numel > req.num_elements:
                raise ValueError(
                    f"invalid stage segment offset={src_offset} numel={numel} for recv num_elements={req.num_elements}"
                )
            if dst.numel() != numel:
                raise ValueError(f"stage destination numel {dst.numel()} != segment numel {numel}")

    def _order_after(self, predecessor: LwdTransferFuture | None) -> None:
        if predecessor is None:
            return
        event = predecessor._done_event
        if event is not None:
            self._stream().wait_event(event)

    def _post_send(self, req: LwdTransferRequest) -> list[Any]:
        group = lwd_channels.lwd_get_channel_device_group(self.channel_type, self.dp_idx)
        peer = req.peer_rank
        if peer is None:
            peer = lwd_channels.lwd_get_channel_peer(self.channel_type, self.dp_idx)
        tensor = req.tensor
        assert tensor is not None
        logger.debug(
            "[lwd-comm] SEND post channel=%s my_rank=%d peer=%s group_ranks=%s shape=%s dtype=%s op=isend",
            self.channel_type,
            dist.get_rank(),
            peer,
            dist.get_process_group_ranks(group),
            list(tensor.shape),
            tensor.dtype,
        )
        # UP (edge -> cloud) takes two hops: this channel carries only the
        # edge -> cloud endpoint P2P, so it waits for the endpoint alone
        # instead of every cloud TP rank; the cloud endpoint then broadcasts
        # within the cloud TP group (see _post_recv). DOWN (cloud leader ->
        # edge) is a single P2P.
        # The aux frame (mrope positions) is a second P2P op right after the
        # main one under the same seqno, so skip/drain semantics are
        # unchanged; the cloud runner broadcasts it within the cloud TP group
        # at consume time as well (see lwd_cloud_model_runner).
        if self.channel_type == LwdChannelType.UP:
            handles = [dist.isend(tensor.contiguous(), dst=peer, group=group)]
            if req.aux_tensor is not None:
                handles.append(dist.isend(req.aux_tensor.contiguous(), dst=peer, group=group))
            return handles
        if req.aux_tensor is not None:
            raise RuntimeError("aux payload is only supported on the UP channel")
        return [dist.isend(tensor.contiguous(), dst=peer, group=group)]

    def _post_recv(self, req: LwdTransferRequest) -> tuple[torch.Tensor, torch.Tensor | None, list[Any]]:
        group = lwd_channels.lwd_get_channel_device_group(self.channel_type, self.dp_idx)
        peer = req.peer_rank
        if peer is None:
            peer = lwd_channels.lwd_get_channel_peer(self.channel_type, self.dp_idx)
        logger.debug(
            "[lwd-comm] RECV post channel=%s my_rank=%d src=%s group_ranks=%s num_elements=%d aux_elements=%d op=irecv",
            self.channel_type,
            dist.get_rank(),
            peer,
            dist.get_process_group_ranks(group),
            req.num_elements,
            req.aux_num_elements,
        )
        # Exact-size buffer: HCCL P2P requires matching numel on both ends.
        buffer = req.recv_buffer
        if buffer is None:
            buffer = torch.empty(req.num_elements, dtype=torch.bfloat16, device="npu")
        elif buffer.numel() != req.num_elements:
            raise RuntimeError(
                f"recv_buffer numel {buffer.numel()} != requested num_elements {req.num_elements} "
                f"on channel {self.channel_type.value} seqno={req.seqno}"
            )
        self._validate_stage_segments(req)
        aux_buffer = None
        if self.channel_type == LwdChannelType.UP and req.aux_num_elements > 0:
            # aux frame size = num_tokens * 3 (mrope positions); the dtype is
            # the request's aux_dtype (int64 by default), and the numel must
            # match the sender's exactly.
            aux_buffer = torch.empty(
                req.aux_num_elements,
                dtype=req.aux_dtype or torch.int64,
                device="npu",
            )
            return (
                buffer,
                aux_buffer,
                [
                    dist.irecv(buffer, src=peer, group=group),
                    dist.irecv(aux_buffer, src=peer, group=group),
                ],
            )
        if req.aux_num_elements > 0:
            raise RuntimeError("aux payload is only supported on the UP channel")
        return buffer, None, [dist.irecv(buffer, src=peer, group=group)]

    def _bridge_and_record(self, handles: list[Any]):
        """Bridge HCCL completion onto the channel stream and record an
        event right behind it (CPU returns immediately).  Caller already
        holds the channel stream as current, so bridging binds to the
        right stream."""
        if not handles:
            return None
        _t = time.monotonic()
        for handle in handles:
            handle.wait()
        # [Lwd][perf] temporary probe: checks that handle.wait() really
        # returns immediately on the CPU, as documented; a large dur means
        # posting the send/recv blocks the host (the lockstep criterion).
        # log_analyze_tools/lwd_backlog_probe.py (mm branch) parses this line,
        # including the LwdChannelType repr in ch=; keep the format.
        logger.debug(
            "[Lwd][perf] bridge-wait op=%s ch=%s dur=%.2fms",
            self.op,
            self.channel_type,
            (time.monotonic() - _t) * 1000,
        )
        logger.debug(
            "[lwd-comm] DONE channel=%s my_rank=%d handles=%d",
            self.channel_type,
            dist.get_rank(),
            len(handles),
        )
        event = torch.npu.Event()
        event.record()
        return event
