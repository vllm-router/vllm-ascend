# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Per-request host staging for cloud-side LWD prompt embeddings.

Each receive uses an NPU slot followed by segmented D2H copies into pinned
buffers owned by individual requests. A buffer is allocated at full prompt
length and remains alive until the request finishes. Arrival and consumption
are independent watermarks::

    0 <= consumed <= recv_rows <= prompt_rows

The module has no module-level torch dependency. Buffers are duck-typed through
``numel()`` and slicing, allowing the state machine to run without an NPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from vllm.lwd_logger import lwd_logger


class LwdBuffer(Protocol):
    def numel(self) -> int: ...

    def __getitem__(self, key: slice) -> LwdBuffer: ...


LwdBufferFactory = Callable[[int], LwdBuffer]


class LwdHostStageError(RuntimeError):
    """Base class for host-staging contract violations."""


class LwdHostStageSlotBusy(LwdHostStageError):
    """All device receive slots are owned; the caller must retry later."""


class LwdHostStageGeometryError(LwdHostStageError):
    """Payload, prompt, or buffer coordinates are inconsistent."""


@dataclass
class LwdReqEmbed:
    """Full-prompt host storage and watermarks for one request."""

    req_id: str
    host: LwdBuffer
    numel: int
    prompt_rows: int
    hidden_size: int
    prefix_nonce: int = 0
    tail_off: int = -1
    tail_numel: int = 0
    num_hash_layers: int = 0
    num_experts_per_tok: int = 0
    recv_rows: int = 0
    consumed: int = 0
    aborted: bool = False
    _written_hi: int = 0
    _stage_floor: int = 0
    _refill_lo: int | None = None
    _refill_hi: int | None = None
    _refill_completed: bool = False

    def seed_prefix(self, hit_rows: int) -> None:
        """Seed the watermarks at a prefix-cache hit boundary."""
        if hit_rows <= 0 or hit_rows >= self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: seed_prefix({hit_rows}) outside (0, prompt_rows={self.prompt_rows})"
            )
        if self._stage_floor or self.recv_rows or self._written_hi or self.consumed:
            raise LwdHostStageGeometryError(f"req={self.req_id}: seed_prefix({hit_rows}) after arrivals/consumption")
        self._stage_floor = hit_rows
        self.recv_rows = hit_rows
        self._written_hi = hit_rows
        self.consumed = hit_rows

    def note_arrival(self, offset: int, rows: int) -> None:
        """Publish a contiguous or repeated arrival range."""
        if offset < 0 or rows < 0:
            raise LwdHostStageGeometryError(f"req={self.req_id}: negative arrival offset={offset} rows={rows}")
        end = offset + rows
        if end > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: announced rows [{offset}, {end}) exceed prompt_rows={self.prompt_rows}"
            )
        if offset < self._stage_floor:
            self._note_refill_arrival(offset, rows)
        if offset > self._written_hi:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: arrival gap [{self._written_hi}, {offset}) before chunk [{offset}, {end})"
            )
        if end > self._written_hi:
            self._written_hi = end
        previous = self.recv_rows
        self.recv_rows = max(self.recv_rows, self._written_hi)
        lwd_logger.debug(
            "[Lwd][host-stage] ARRIVE req=%s offset=%d rows=%d recv_rows=%d->%d",
            self.req_id,
            offset,
            rows,
            previous,
            self.recv_rows,
        )

    def _note_refill_arrival(self, offset: int, rows: int) -> None:
        end = offset + rows
        if end > self._stage_floor:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: refill rows [{offset}, {end}) overrun stage floor S={self._stage_floor}"
            )
        if self._refill_hi is None:
            self._refill_lo = offset
            self._refill_hi = offset
        elif offset > self._refill_hi:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: refill gap [{self._refill_hi}, {offset}) before chunk [{offset}, {end})"
            )
        if end > self._refill_hi:
            self._refill_hi = end
        if not self._refill_completed and self._refill_hi >= self._stage_floor:
            self._refill_completed = True

    def available(self) -> int:
        return max(self.recv_rows - self.consumed, 0)

    def advanced_from_start(self) -> int:
        return self.consumed

    def read_range(self, n_rows: int) -> tuple[int, int]:
        """Return the embedding element range for the next read window."""
        if n_rows < 1:
            raise LwdHostStageGeometryError(f"req={self.req_id}: read_range needs n_rows >= 1, got {n_rows}")
        end_row = self.consumed + n_rows
        if end_row > self.recv_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read {n_rows} rows at consumed={self.consumed} "
                f"exceeds received rows {self.recv_rows}"
            )
        if end_row > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read {n_rows} rows at consumed={self.consumed} "
                f"exceeds prompt_rows={self.prompt_rows}"
            )
        if self.consumed < self._stage_floor and (
            self._refill_hi is None or (self._refill_hi < self._stage_floor and end_row > self._refill_hi)
        ):
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read [{self.consumed}, {end_row}) hits unfilled refill hole "
                f"(refill_hi={self._refill_hi}, floor S={self._stage_floor})"
            )
        start = self.consumed * self.hidden_size
        return start, start + n_rows * self.hidden_size

    def tail_range(self, n_rows: int) -> tuple[int, int]:
        return self.tail_range_at(self.consumed, n_rows)

    def tail_range_at(self, consumed_base: int, n_rows: int) -> tuple[int, int]:
        """Return the expert-ID tail range from an explicit read pointer."""
        if self.tail_numel <= 0:
            return 0, 0
        per_row = 4 * self.num_hash_layers * self.num_experts_per_tok
        start = self.tail_off + consumed_base * per_row
        return start, start + n_rows * per_row

    def note_consumed(self, n_rows: int) -> None:
        """Advance the read pointer after a successful device dispatch."""
        if n_rows < 1:
            raise LwdHostStageGeometryError(f"req={self.req_id}: consume needs n_rows >= 1, got {n_rows}")
        end_row = self.consumed + n_rows
        if end_row > self.recv_rows or end_row > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: consumed end {end_row} exceeds received={self.recv_rows} "
                f"or prompt_rows={self.prompt_rows}"
            )
        self.consumed = end_row

    def rewind_consumed(self, to_rows: int) -> None:
        """Move the read pointer back without changing arrival watermarks."""
        if to_rows < 0 or to_rows > self.consumed:
            raise LwdHostStageGeometryError(f"req={self.req_id}: rewind target {to_rows} outside [0, {self.consumed}]")
        self.consumed = to_rows

    def skip_consumed(self, to_rows: int) -> None:
        """Skip rows covered by a re-admission prefix-cache hit."""
        if to_rows < self.consumed or to_rows > self.recv_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: skip target {to_rows} outside "
                f"[consumed={self.consumed}, recv_rows={self.recv_rows}]"
            )
        if to_rows > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: skip target {to_rows} exceeds prompt_rows={self.prompt_rows}"
            )
        self.consumed = to_rows

    def mark_aborted(self) -> None:
        self.aborted = True

    def done(self) -> bool:
        return self.aborted or self.consumed >= self.prompt_rows


@dataclass
class LwdStagedBatch:
    """Request and source-layout metadata for one received batch."""

    seqno: int
    req_ids: list[str]
    num_tokens_per_req: list[int]
    hidden_size: int
    num_hash_layers: int = 0
    num_experts_per_tok: int = 0
    src_offsets: list[int] = field(default_factory=list)
    src_tail_offsets: list[int] = field(default_factory=list)
    created_req_ids: list[str] = field(default_factory=list)
    request_hosts: dict[str, LwdReqEmbed] = field(default_factory=dict)
    recv_bytes: int = 0
    split_ms: float = 0.0

    @property
    def total_rows(self) -> int:
        return sum(self.num_tokens_per_req)


class LwdHostStager:
    """Own NPU receive slots and full-prompt pinned request buffers."""

    def __init__(
        self,
        device: str = "npu",
        slots: int = 2,
        max_numel: int = 0,
        empty_factory: LwdBufferFactory | None = None,
        device_factory: LwdBufferFactory | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        self.device = device
        self.slots = slots
        self.max_numel = max_numel
        self._empty_factory = empty_factory
        self._device_factory = device_factory
        self._lock = threading.Lock()
        self._slot_buffers: dict[int, LwdBuffer] = {}
        self._slot_owner: dict[int, int] = {slot: -1 for slot in range(slots)}
        self._seqno_slot: dict[int, int] = {}
        self._batches: dict[int, LwdStagedBatch] = {}
        self._reqs: dict[str, LwdReqEmbed] = {}
        self._ended_req_ids: set[str] = set()
        self._end_pending: dict[str, set[int]] = {}
        self._end_after: dict[str, int] = {}
        self._end_waiting: set[str] = set()
        self._last_delivered = -1
        self._batch_reqs: dict[int, list[str]] = {}
        self._arrival_plan: dict[int, list[tuple[LwdReqEmbed, int, int]]] = {}
        self._refill_done: list[tuple[str, int]] = []
        self._host_bytes = 0
        self._released_requests = 0
        self._aborted_requests = 0

    @property
    def host_bytes(self) -> int:
        with self._lock:
            return self._host_bytes

    @property
    def staged_count(self) -> int:
        with self._lock:
            return len(self._batches)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._reqs)

    @property
    def released_requests(self) -> int:
        with self._lock:
            return self._released_requests

    @property
    def aborted_requests(self) -> int:
        with self._lock:
            return self._aborted_requests

    def busy_slots(self) -> list[int]:
        with self._lock:
            return [slot for slot, owner in self._slot_owner.items() if owner >= 0]

    def staged_seqnos(self) -> list[int]:
        with self._lock:
            return list(self._batches)

    def req_ids(self) -> list[str]:
        with self._lock:
            return list(self._reqs)

    def take_req(self, req_id: str) -> LwdReqEmbed | None:
        with self._lock:
            return self._reqs.get(req_id)

    def recv_rows(self, req_id: str) -> int:
        with self._lock:
            req = self._reqs.get(req_id)
            return req.recv_rows if req is not None else 0

    def consumed_rows(self, req_id: str) -> int:
        with self._lock:
            req = self._reqs.get(req_id)
            return req.consumed if req is not None else 0

    def stage_floor(self, req_id: str) -> int:
        with self._lock:
            req = self._reqs.get(req_id)
            return req._stage_floor if req is not None else 0

    def take_refill_done(self) -> list[tuple[str, int]]:
        with self._lock:
            done = self._refill_done
            self._refill_done = []
        return done

    def allocate(self, seqno: int, numel: int) -> tuple[LwdBuffer, int]:
        """Bind a free device slot to one receive sequence number."""
        with self._lock:
            if numel < 0 or (self.max_numel and numel > self.max_numel):
                raise LwdHostStageGeometryError(f"seqno={seqno}: numel {numel} outside slot capacity {self.max_numel}")
            if seqno in self._seqno_slot:
                raise LwdHostStageGeometryError(f"seqno={seqno}: receive slot is already allocated")
            free_slots = [slot for slot, owner in self._slot_owner.items() if owner < 0]
            if not free_slots:
                raise LwdHostStageSlotBusy(
                    f"seqno={seqno}: all {self.slots} device slot(s) busy (held by seqno={sorted(self._seqno_slot)})"
                )
            slot = free_slots[0]
            buf = self._slot_buffers.get(slot)
            if buf is None or buf.numel() < numel:
                buf = self._make_device_buffer(self.max_numel or numel)
                self._slot_buffers[slot] = buf
            self._slot_owner[slot] = seqno
            self._seqno_slot[seqno] = slot
        return buf[:numel], slot

    def abandon(self, seqno: int) -> None:
        """Roll back a plan and allocation whose receive was never posted."""
        with self._lock:
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1
            self._arrival_plan.pop(seqno, None)
            self._batch_reqs.pop(seqno, None)
            batch = self._batches.pop(seqno, None)
            self._complete_end_plan(seqno, delivered=False)
            if batch is None:
                return
            for req_id in batch.created_req_ids:
                req = self._reqs.get(req_id)
                if req is not None and batch.request_hosts.get(req_id) is req and req._written_hi == req._stage_floor:
                    self._reqs.pop(req_id)
                    self._host_bytes -= (req.numel + req.tail_numel) * 2

    def plan_recv_segments(
        self,
        seqno: int,
        numel: int,
        hidden_size: int,
        req_ids: list[str],
        num_tokens_per_req: list[int],
        prompt_lens: list[int],
        row_base: list[int] | None = None,
        num_hash_layers: int = 0,
        num_experts_per_tok: int = 0,
        refills: list[bool] | None = None,
        prefix_nonces: list[int] | None = None,
    ) -> list[tuple[LwdBuffer, int, int]]:
        """Allocate request storage and return ``(destination, source, size)`` D2H segments."""
        started = time.monotonic()
        rows = list(num_tokens_per_req)
        bases = self._validate_plan(
            seqno,
            numel,
            hidden_size,
            req_ids,
            rows,
            prompt_lens,
            row_base,
            num_hash_layers,
            num_experts_per_tok,
        )
        if refills is None:
            refills = [False] * len(rows)
        elif len(refills) < len(rows):
            refills = list(refills) + [False] * (len(rows) - len(refills))
        elif len(refills) > len(rows):
            raise LwdHostStageGeometryError(f"seqno={seqno}: {len(refills)} refill flags vs {len(rows)} requests")
        if prefix_nonces is None:
            prefix_nonces = [0] * len(rows)
        elif len(prefix_nonces) < len(rows):
            prefix_nonces = list(prefix_nonces) + [0] * (len(rows) - len(prefix_nonces))
        elif len(prefix_nonces) > len(rows):
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: {len(prefix_nonces)} prefix nonces vs {len(rows)} requests"
            )
        tail_per_row = 4 * num_hash_layers * num_experts_per_tok
        batch = LwdStagedBatch(
            seqno=seqno,
            req_ids=list(req_ids),
            num_tokens_per_req=rows,
            hidden_size=hidden_size,
            num_hash_layers=num_hash_layers,
            num_experts_per_tok=num_experts_per_tok,
            recv_bytes=numel * 2,
        )
        segments: list[tuple[LwdBuffer, int, int]] = []
        plan: list[tuple[LwdReqEmbed, int, int]] = []
        with self._lock:
            source_row = 0
            for index, req_id in enumerate(req_ids):
                host: LwdReqEmbed | None = None
                dest_row = bases[index]
                existing = self._reqs.get(req_id)
                if req_id in self._ended_req_ids:
                    if req_id in self._end_waiting:
                        self._end_waiting.discard(req_id)
                        self._end_pending[req_id].add(seqno)
                        self._end_after[req_id] = seqno
                elif existing is None or not existing.aborted:
                    if existing is None:
                        batch.created_req_ids.append(req_id)
                    host = self._request_for_chunk(
                        req_id,
                        rows[index],
                        dest_row,
                        prompt_lens[index],
                        hidden_size,
                        tail_per_row,
                        num_hash_layers,
                        num_experts_per_tok,
                        refills[index],
                        prefix_nonces[index],
                    )
                    batch.request_hosts[req_id] = host
                    plan.append((host, dest_row, rows[index]))
                segments.extend(self._d2h_segments(host, batch, source_row, dest_row, rows[index], batch.total_rows))
                source_row += rows[index]
            self._batches[seqno] = batch
            self._batch_reqs[seqno] = list(req_ids)
            self._arrival_plan[seqno] = plan
        lwd_logger.debug(
            "[Lwd][host-stage] PLAN seqno=%d dur=%.2fms reqs=%d rows=%d",
            seqno,
            (time.monotonic() - started) * 1000,
            len(req_ids),
            batch.total_rows,
        )
        return segments

    def publish_received(self, seqno: int) -> list[tuple[str, int, int]]:
        """Publish active D2H arrivals and release the receive slot."""
        with self._lock:
            plan = self._arrival_plan.pop(seqno, None)
            if plan is None:
                raise LwdHostStageGeometryError(f"seqno={seqno}: no arrival plan")
            rows: list[tuple[str, int, int]] = []
            for host, dest_row, n_rows in plan:
                if host.req_id in self._ended_req_ids or host.aborted:
                    continue
                host.note_arrival(dest_row, n_rows)
                if host._refill_completed:
                    assert host._refill_lo is not None
                    host._stage_floor = host._refill_lo
                    host._refill_lo = None
                    host._refill_hi = None
                    host._refill_completed = False
                    self._refill_done.append((host.req_id, host.prefix_nonce))
                rows.append((host.req_id, n_rows, dest_row))
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1
            self._last_delivered = max(self._last_delivered, seqno)
            self._complete_end_plan(seqno, delivered=True)
        return rows

    def batch(self, seqno: int) -> LwdStagedBatch | None:
        with self._lock:
            return self._batches.get(seqno)

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]:
        """Return ``(host_start, numel, tail_start)`` for the next read window."""
        req = self.take_req(req_id)
        if req is None:
            raise LwdHostStageGeometryError(f"req={req_id}: not staged")
        start, end = req.read_range(n_rows)
        tail_start, _ = req.tail_range(n_rows)
        return start, end - start, tail_start if req.tail_numel else -1

    def consume(self, req_id: str, n_rows: int) -> None:
        req = self.take_req(req_id)
        if req is None:
            raise LwdHostStageGeometryError(f"req={req_id}: not staged")
        req.note_consumed(n_rows)

    def rewind_consumed(self, req_id: str, to_rows: int) -> bool:
        """Rewind a known request while serializing with stager state access."""
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                return False
            req.rewind_consumed(to_rows)
            return True

    def skip_consumed(self, req_id: str, to_rows: int) -> bool:
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                return False
            req.skip_consumed(to_rows)
            return True

    def abort_request(self, req_id: str) -> bool:
        """Freeze a request and guard its old plans until they drain."""
        with self._lock:
            req = self._reqs.get(req_id)
            self._retire_request(req_id, wait_for_plan=req is None)
            if req is None:
                return False
            already = req.aborted
            req.mark_aborted()
            if not already:
                self._aborted_requests += 1
        return True

    def release_request(self, req_id: str) -> int:
        """Release host storage and guard only its unpublished old plans."""
        with self._lock:
            req = self._reqs.pop(req_id, None)
            if req is not None or req_id in self._ended_req_ids:
                self._retire_request(req_id, wait_for_plan=False)
            if req is None:
                return 0
            freed = (req.numel + req.tail_numel) * 2
            self._host_bytes -= freed
            self._released_requests += 1
        return freed

    def release_if_drained(self, seqno: int) -> bool:
        """Drop published batch metadata after its generation's hosts are released."""
        with self._lock:
            if seqno in self._arrival_plan:
                return False
            batch = self._batches.get(seqno)
            if batch is None:
                return False
            if any(self._reqs.get(req_id) is host for req_id, host in batch.request_hosts.items()):
                return False
            self._batch_reqs.pop(seqno, None)
            self._batches.pop(seqno, None)
            return True

    def batches_holding(self, req_id: str) -> list[int]:
        with self._lock:
            return [seqno for seqno, req_ids in self._batch_reqs.items() if req_id in req_ids]

    def _retire_request(self, req_id: str, wait_for_plan: bool) -> None:
        if req_id in self._ended_req_ids:
            return
        pending = {seqno for seqno in self._arrival_plan if req_id in self._batch_reqs.get(seqno, ())}
        if pending:
            self._ended_req_ids.add(req_id)
            self._end_pending[req_id] = pending
            self._end_after[req_id] = max(pending)
        elif wait_for_plan:
            self._ended_req_ids.add(req_id)
            self._end_pending[req_id] = set()
            self._end_after[req_id] = self._last_delivered
            self._end_waiting.add(req_id)

    def _complete_end_plan(self, seqno: int, delivered: bool) -> None:
        for pending in self._end_pending.values():
            pending.discard(seqno)
        for req_id in list(self._ended_req_ids):
            if req_id in self._end_waiting or self._end_pending.get(req_id):
                continue
            if delivered and self._last_delivered < self._end_after[req_id]:
                continue
            self._ended_req_ids.discard(req_id)
            self._end_pending.pop(req_id, None)
            self._end_after.pop(req_id, None)

    def _validate_plan(
        self,
        seqno: int,
        numel: int,
        hidden_size: int,
        req_ids: list[str],
        rows: list[int],
        prompt_lens: list[int],
        row_base: list[int] | None,
        num_hash_layers: int,
        num_experts_per_tok: int,
    ) -> list[int]:
        if hidden_size < 1 or num_hash_layers < 0 or num_experts_per_tok < 0:
            raise LwdHostStageGeometryError(f"seqno={seqno}: invalid hidden or tail geometry")
        if len(req_ids) != len(rows) or len(prompt_lens) != len(rows):
            raise LwdHostStageGeometryError(f"seqno={seqno}: request, row, and prompt tables must have equal length")
        bases = [0] * len(rows) if row_base is None else list(row_base)
        if len(bases) != len(rows):
            raise LwdHostStageGeometryError(f"seqno={seqno}: {len(bases)} row bases vs {len(rows)} requests")
        if any(n_rows < 0 for n_rows in rows) or any(prompt < 0 for prompt in prompt_lens):
            raise LwdHostStageGeometryError(f"seqno={seqno}: negative row count or prompt length")
        if any(offset < 0 for offset in bases):
            raise LwdHostStageGeometryError(f"seqno={seqno}: negative prompt offset")
        tail_per_row = 4 * num_hash_layers * num_experts_per_tok
        total_rows = sum(rows)
        expected = total_rows * (hidden_size + tail_per_row)
        if numel != expected:
            raise LwdHostStageGeometryError(f"seqno={seqno}: payload {numel} != rows*(hidden+tail) = {expected}")
        return bases

    def _request_for_chunk(
        self,
        req_id: str,
        n_rows: int,
        offset: int,
        prompt_rows: int,
        hidden_size: int,
        tail_per_row: int,
        num_hash_layers: int,
        num_experts_per_tok: int,
        refill: bool,
        prefix_nonce: int,
    ) -> LwdReqEmbed:
        host = self._reqs.get(req_id)
        if host is None:
            if refill:
                raise LwdHostStageGeometryError(f"req={req_id}: refill chunk for an unknown request")
            if offset + n_rows > prompt_rows:
                raise LwdHostStageGeometryError(
                    f"req={req_id}: chunk end {offset + n_rows} exceeds prompt_rows={prompt_rows}"
                )
            host = LwdReqEmbed(
                req_id=req_id,
                host=self._make_host_buffer(prompt_rows * (hidden_size + tail_per_row)),
                numel=prompt_rows * hidden_size,
                prompt_rows=prompt_rows,
                hidden_size=hidden_size,
                prefix_nonce=prefix_nonce,
                tail_off=prompt_rows * hidden_size if tail_per_row else -1,
                tail_numel=prompt_rows * tail_per_row,
                num_hash_layers=num_hash_layers,
                num_experts_per_tok=num_experts_per_tok,
            )
            self._reqs[req_id] = host
            self._host_bytes += (host.numel + host.tail_numel) * 2
            if offset > 0:
                host.seed_prefix(offset)
        elif (
            host.prompt_rows != prompt_rows
            or host.hidden_size != hidden_size
            or host.tail_numel != prompt_rows * tail_per_row
            or host.num_hash_layers != num_hash_layers
            or host.num_experts_per_tok != num_experts_per_tok
        ):
            raise LwdHostStageGeometryError(f"req={req_id}: later chunk geometry differs from the first chunk")
        else:
            if host.prefix_nonce != prefix_nonce:
                raise LwdHostStageGeometryError(
                    f"req={req_id}: prefix nonce {prefix_nonce} cannot reuse active generation "
                    f"with nonce {host.prefix_nonce} before drain"
                )
            geometry_refill = offset < host._stage_floor
            if refill != geometry_refill:
                raise LwdHostStageGeometryError(
                    f"req={req_id}: refill flag {refill} disagrees with geometry "
                    f"(offset={offset}, stage floor S={host._stage_floor})"
                )
            if geometry_refill and offset + n_rows > host._stage_floor:
                raise LwdHostStageGeometryError(
                    f"req={req_id}: refill rows [{offset}, {offset + n_rows}) overrun stage floor S={host._stage_floor}"
                )
        if offset + n_rows > host.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={req_id}: chunk rows [{offset}, {offset + n_rows}) exceed prompt_rows={host.prompt_rows}"
            )
        return host

    def _d2h_segments(
        self,
        host: LwdReqEmbed | None,
        batch: LwdStagedBatch,
        source_row: int,
        dest_row: int,
        n_rows: int,
        total_rows: int,
    ) -> list[tuple[LwdBuffer, int, int]]:
        hidden_size = batch.hidden_size
        embed_source = source_row * hidden_size
        tail_per_row = 4 * batch.num_hash_layers * batch.num_experts_per_tok
        tail_source = total_rows * hidden_size + source_row * tail_per_row if tail_per_row else -1
        batch.src_offsets.append(embed_source)
        batch.src_tail_offsets.append(tail_source)
        if host is None:
            return []
        segments = [
            (
                host.host[dest_row * hidden_size : (dest_row + n_rows) * hidden_size],
                embed_source,
                n_rows * hidden_size,
            )
        ]
        if tail_per_row:
            segments.append(
                (
                    host.host[
                        host.tail_off + dest_row * tail_per_row : host.tail_off + (dest_row + n_rows) * tail_per_row
                    ],
                    tail_source,
                    n_rows * tail_per_row,
                )
            )
        return segments

    def _make_host_buffer(self, numel: int) -> LwdBuffer:
        if self._empty_factory is not None:
            return self._empty_factory(numel)
        import torch

        return torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)

    def _make_device_buffer(self, numel: int) -> LwdBuffer:
        if self._device_factory is not None:
            return self._device_factory(numel)
        import torch

        return torch.empty(numel, dtype=torch.bfloat16, device=self.device)
