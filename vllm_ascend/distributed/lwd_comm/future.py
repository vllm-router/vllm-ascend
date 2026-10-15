# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdCommFuture: completion handle for one duplex comm request.

Ported (simplified) from the demo branch's ``lwd_comm/future.py``:
pure-CPU ``done()`` via ``event.query()``, blocking ``wait()`` for the
debug/timeout path, in-place ``_bind`` for deferred (out-of-order
sequenced) futures, and exactly-once ``_finalize`` releasing the send
keepalive.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from vllm_ascend.distributed.lwd_comm.types import (
    LwdCommRequest,
    LwdCommResult,
    LwdCommStatus,
)


class LwdCommFuture:
    """Tracks one posted (or held) wire op.

    Completion means the cross-node P2P op has finished on the device,
    witnessed by ``done_event`` recorded on the channel stream right
    behind the bridged HCCL op.
    """

    def __init__(
        self,
        request: LwdCommRequest,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
        staged_tensor: torch.Tensor | None = None,
        slot: int = -1,
    ) -> None:
        self._request = request
        self._handles = handles
        self._done_event = done_event
        self._tensor = tensor
        self._aux_tensor = aux_tensor
        self._keepalive = keepalive
        # Host-staging (see docs/design/lwd_host_staging.md §4.3):
        #   staged_event  - recorded on the channel stream AFTER the D2H,
        #                   so ``query()`` true means the pinned host copy
        #                   is complete. Strictly later than ``done_event``.
        #   staged_tensor - the pinned host destination (owned by the
        #                   caller's stager, not by this future).
        #   slot          - device ping-pong slot the recv landed in;
        #                   diagnostic only, the caller owns pool state.
        self._staged_event = staged_event
        self._staged_tensor = staged_tensor
        self._slot = slot
        self._status = LwdCommStatus.PENDING
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        self._finalized = False

    # ------------------------------------------------------------------ #
    # Deferred (out-of-order sequenced) support                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def deferred(cls, request: LwdCommRequest) -> "LwdCommFuture":
        """A future for a held request; bound in place when its seqno
        turn arrives."""
        return cls(request, [], None)

    def _bind(
        self,
        *,
        handles: list[Any],
        done_event: "torch.npu.Event | None",
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: "torch.npu.Event | None" = None,
        staged_tensor: torch.Tensor | None = None,
        slot: int = -1,
    ) -> None:
        with self._done_cond:
            self._handles = handles
            self._done_event = done_event
            self._tensor = tensor
            self._aux_tensor = aux_tensor
            self._keepalive = keepalive
            self._staged_event = staged_event
            self._staged_tensor = staged_tensor
            self._slot = slot
            self._done_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def request(self) -> LwdCommRequest:
        return self._request

    def done(self) -> bool:
        """Pure CPU query; never blocks, never touches the device."""
        with self._lock:
            if self._status is not LwdCommStatus.PENDING:
                return True
            event = self._done_event
        if event is None:
            return False
        try:
            if event.query():
                with self._lock:
                    if self._status is LwdCommStatus.PENDING:
                        self._status = LwdCommStatus.OK
                        self._done_cond.notify_all()
                return True
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdCommStatus.PENDING:
                    self._status = LwdCommStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True
        return False

    def wait(self, timeout: float | None = None) -> LwdCommResult:
        """Block until completion (readiness-gate path)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._done_cond:
            while self._status is LwdCommStatus.PENDING:
                if self._done_event is None:
                    # Deferred future not yet bound; wait for _bind.
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"lwd-comm wait timed out (unbound) on "
                            f"{self._request.channel.value} seqno="
                            f"{self._request.seqno}"
                        )
                    self._done_cond.wait(timeout=remaining or 0.001)
                    continue
                if self._done_event.query():
                    self._status = LwdCommStatus.OK
                    break
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"lwd-comm wait timed out on "
                        f"{self._request.channel.value} seqno="
                        f"{self._request.seqno}"
                    )
                self._done_cond.wait(timeout=min(0.001, remaining or 0.001))
            if self._status is LwdCommStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._request.channel.value} "
                    f"seqno={self._request.seqno}"
                ) from self._error
        return LwdCommResult(
            status=self._status, tensor=self._tensor,
            aux_tensor=self._aux_tensor,
        )

    def result(self) -> LwdCommResult:
        with self._lock:
            status, tensor, aux_tensor, error = (
                self._status, self._tensor, self._aux_tensor, self._error,
            )
            staged_tensor = self._staged_tensor
        if status is LwdCommStatus.ERROR:
            raise RuntimeError(
                f"lwd-comm op failed on {self._request.channel.value} "
                f"seqno={self._request.seqno}"
            ) from error
        return LwdCommResult(
            status=status, tensor=tensor, aux_tensor=aux_tensor,
            staged_tensor=staged_tensor,
        )

    # ------------------------------------------------------------------ #
    # Host staging (see docs/design/lwd_host_staging.md §4.3)              #
    # ------------------------------------------------------------------ #

    @property
    def slot(self) -> int:
        """Device ping-pong slot this recv landed in (-1 when unused)."""
        return self._slot

    @property
    def staged_tensor(self) -> torch.Tensor | None:
        """Pinned host destination, or None when staging is off."""
        return self._staged_tensor

    def staging_requested(self) -> bool:
        return self._staged_event is not None

    def staged_done(self) -> bool:
        """Pure-CPU query of the D2H completion event.

        ``True`` implies the pinned host copy is complete (the event is
        recorded behind the copy on the channel stream). Always ``True``
        when staging was not requested, so callers can use it as a plain
        readiness gate.
        """
        event = self._staged_event
        if event is None:
            return True
        try:
            return bool(event.query())
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdCommStatus.PENDING:
                    self._status = LwdCommStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True

    def wait_staged(self, timeout: float | None = None) -> LwdCommResult:
        """Block until the host-staged copy is complete (debug path only).

        The comm thread never calls this (it must not block); it exists
        for tests and for a caller that wants the staged payload directly.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        event = self._staged_event
        if event is None:
            return self.wait(timeout=timeout)
        while not self.staged_done():
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"lwd-comm staging timed out on "
                    f"{self._request.channel.value} seqno="
                    f"{self._request.seqno}"
                )
            time.sleep(min(0.001, remaining or 0.001))
        return self.result()

    # ------------------------------------------------------------------ #
    # Device-side ordering for consumers                                  #
    # ------------------------------------------------------------------ #

    def wait_for_comm(self) -> None:
        """Order the *current* stream after the channel-stream completion
        event (pure device-side; CPU returns immediately)."""
        event = self._done_event
        if event is not None:
            torch.npu.current_stream().wait_event(event)

    def release_device_tensor(self) -> None:
        """Drop the device-frame reference after its D2H copy completed.

        With host staging the device buffer belongs to the caller's
        ping-pong pool and is reused by the next receive; holding it from
        a completed future would only keep the slot's device memory alive
        (the allocator cannot reuse it while a reference exists).  The
        comm thread calls this exactly once, right after it has committed
        the staged copy - never while a consumer could still read the
        device frame.

        Safe no-op when staging was not requested (then the device frame
        IS the delivered payload and ``result()`` must keep serving it).
        """
        if self._staged_event is None:
            return
        with self._done_cond:
            self._tensor = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _finalize(self, error: BaseException | None = None) -> None:
        """Exactly-once terminal bookkeeping: release the send keepalive.
        With ``error`` the future completes in ERROR state immediately
        (abort/skip path) so waiters fail promptly instead of timing out."""
        with self._done_cond:
            if self._finalized:
                return
            self._finalized = True
            self._keepalive = None
            self._handles = []
            if error is not None:
                self._status = LwdCommStatus.ERROR
                self._error = error
                self._done_cond.notify_all()
                return
            if self._status is LwdCommStatus.PENDING:
                self._status = (
                    LwdCommStatus.OK
                    if (self._done_event is not None and self._done_event.query())
                    else self._status
                )
