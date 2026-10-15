# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud-side host staging, per-request (LWD prefill_only, design v7).

Worker half of ``docs/design/lwd_host_staging.md`` (in the vLLM repo).  Two
hops are the design's accepted cost (Ascend cannot deliver a receive straight
into host memory):

    HCCL recv -> NPU ping-pong slot --D2H--> rotating receive pinned
              --H2H split--> per-request pinned (kept until the request ends)
              --H2D (per step, scheduled rows only)--> NPU compute

Why split into **per-request** buffers instead of slicing one big batch
buffer: the storage lifetime is now *the request's* lifetime ("keep every
embedding until the request finishes"), and one pinned allocation can only be
freed as a whole - a shared carrier would let one long request pin the entire
batch.  The split costs one host-to-host copy per batch and buys an allocation
lifetime that matches exactly what the scheduler reasons about.

Two independent counters per request (do not conflate them):

  * ``recv_rows`` - how much has *arrived* (data plane).  Drives what the
    scheduler is allowed to dispatch; reported to the engine by the worker.
  * ``consumed``  - how much has been *computed* (native watermark).  Drives
    the read pointer into the host buffer.

Invariant: ``0 <= consumed <= recv_rows <= prompt_rows``.

This module imports **no torch and no NPU runtime** at module level: the
offline checks exercise it with fake buffers, and it must stay importable on a
machine without torch_npu.  Buffers are duck-typed (``numel()``, slicing,
``copy_()``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = init_logger("vllm.lwd.host_stage")


# ---------------------------------------------------------------------- #
# Errors                                                                  #
# ---------------------------------------------------------------------- #
class LwdHostStageError(RuntimeError):
    """Base class for staging-pool contract violations."""


class LwdHostStageSlotBusy(LwdHostStageError):
    """No free device slot / receive buffer - the caller queues the hint.

    Design v7 has no host-memory cap (memory is assumed sufficient), so the
    only admission gate left is the in-flight receive count.
    """


class LwdHostStageGeometryError(LwdHostStageError):
    """numel / rows / hidden_size disagree - would corrupt every later slice."""


# ---------------------------------------------------------------------- #
# One request's host storage                                              #
# ---------------------------------------------------------------------- #
@dataclass
class LwdReqEmbed:
    """Host copy of one request's prompt embeddings, plus its two counters.

    Owned by the request: allocated when the first chunk arrives, freed when
    the request ends (design §6.1) - *not* when it is fully consumed.
    """

    req_id: str
    host: Any                     # 1-D bf16 pinned; embeds then optional tail
    numel: int                    # embeds elements = prompt_rows * hidden_size
    prompt_rows: int              # rows this request's prompt needs
    hidden_size: int
    # Expert-ID tail (DSv4).  Qwen: num_hash_layers == 0 -> tail_numel == 0.
    tail_off: int = -1
    tail_numel: int = 0
    num_hash_layers: int = 0
    num_experts_per_tok: int = 0
    #: prompt row at which each request's storage was first created
    _row_base: int = 0
    # ---- the two counters (design §4.2) ----
    recv_rows: int = 0            # arrived (data plane)
    consumed: int = 0             # computed (native watermark)
    aborted: bool = False
    #: highest prompt row written so far, to detect gaps on arrival
    _written_hi: int = 0

    # ------------------------------------------------------------------ #
    # Arrival (data plane)                                                #
    # ------------------------------------------------------------------ #
    def note_arrival(self, offset: int, rows: int) -> None:
        """Record that ``rows`` rows starting at prompt ``offset`` arrived.

        ``recv_rows`` is the high-water mark, so repeats are idempotent -
        the edge retries a full publish queue, and the same range can be
        announced twice.

        Raises:
            LwdHostStageGeometryError: if the announced range runs past the
                declared prompt length (edge/cloud version skew, or a mixed-up
                batch) - silently clamping here would later let the scheduler
                dispatch rows that were never written.
        """
        if offset < 0 or rows < 0:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: negative arrival offset={offset} rows={rows}"
            )
        end = offset + rows
        if end > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: announced rows [{offset}, {end}) exceed "
                f"prompt_rows={self.prompt_rows}"
            )
        self.recv_rows = max(self.recv_rows, end)
        if end > self._written_hi:
            self._written_hi = end

    def available(self) -> int:
        """Rows that may be dispatched now: ``recv_rows - consumed``."""
        return max(self.recv_rows - self.consumed, 0)

    def advanced_from_start(self) -> int:
        return self.consumed

    # ------------------------------------------------------------------ #
    # Consumption                                                         #
    # ------------------------------------------------------------------ #
    def read_range(self, n_rows: int) -> tuple[int, int]:
        """Element range ``[start, end)`` of the next ``n_rows`` rows.

        The window always starts where the previous one stopped, so this is
        ``consumed * H`` into the request's own buffer.

        Raises:
            LwdHostStageGeometryError: if the window would run past what has
                arrived (the scheduler must never do that - W2) or past the
                prompt.
        """
        if n_rows < 1:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read_range needs n_rows >= 1, got {n_rows}"
            )
        if self.consumed + n_rows > self.recv_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read {n_rows} rows at consumed="
                f"{self.consumed} exceeds received rows {self.recv_rows}"
            )
        if self.consumed + n_rows > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: read {n_rows} rows at consumed="
                f"{self.consumed} exceeds prompt_rows={self.prompt_rows}"
            )
        start = self.consumed * self.hidden_size
        return start, start + n_rows * self.hidden_size

    def tail_range(self, n_rows: int) -> tuple[int, int]:
        """Element range of this window's expert-ID tail (empty for Qwen)."""
        if self.tail_numel <= 0:
            return 0, 0
        per_row = 4 * self.num_hash_layers * self.num_experts_per_tok
        start = self.tail_off + self.consumed * per_row
        return start, start + n_rows * per_row

    def note_consumed(self, n_rows: int) -> None:
        """Advance the computed pointer (called after the dispatch is real)."""
        self.consumed += n_rows
        if self.consumed > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: consumed {self.consumed} exceeded "
                f"prompt_rows {self.prompt_rows}"
            )

    def mark_aborted(self) -> None:
        """Freeze the remainder; already-computed rows are history, not undone."""
        self.aborted = True

    def done(self) -> bool:
        """Nothing more will ever be read from this buffer."""
        return self.aborted or self.consumed >= self.prompt_rows


# ---------------------------------------------------------------------- #
# Batch descriptor (only to tie a receive to its requests)                #
# ---------------------------------------------------------------------- #
@dataclass
class LwdStagedBatch:
    """Bookkeeping for one received UP batch.

    The payload does **not** live here any more: each request owns its own
    pinned buffer (``LwdReqEmbed``).  What stays is the mapping from the
    batch's seqno to the requests it delivered, so the abort path can find
    "which requests came in this batch" and the row layout of the receive can
    be validated at split time.
    """

    seqno: int
    req_ids: list[str]
    num_tokens_per_req: list[int]
    hidden_size: int
    num_hash_layers: int = 0
    num_experts_per_tok: int = 0
    #: absolute element offset of each request inside the received payload
    src_offsets: list[int] = field(default_factory=list)
    #: absolute element offset of each request's expert-ID tail, -1 = none
    src_tail_offsets: list[int] = field(default_factory=list)
    recv_bytes: int = 0
    split_ms: float = 0.0

    @property
    def total_rows(self) -> int:
        return sum(self.num_tokens_per_req)


# ---------------------------------------------------------------------- #
# The pool                                                                #
# ---------------------------------------------------------------------- #
class LwdHostStager:
    """Device ping-pong slots, rotating receive buffers, per-request storage.

    Threading: ``allocate``/``commit_received`` run on the comm thread (they
    are part of posting/monitoring a receive), while ``adopt``/``consume``/
    ``release_request`` run on the worker's forward thread.  One lock guards
    the tables; buffers are handed over exactly once and only touched by their
    owner afterwards.

    Args:
        device: NPU device string for slot allocation.
        slots: Device ping-pong slots.  Must be >= the number of receives that
            can be in flight (the edge's pending window).
        max_numel: Slot size in bf16 elements (largest batch payload).
        empty_factory: ``(numel) -> buffer`` for host-side allocations;
            injectable so the offline checks run without torch.
        device_factory: ``(numel) -> buffer`` for slot allocations.
    """

    def __init__(
        self,
        device: str = "npu",
        slots: int = 2,
        max_numel: int = 0,
        empty_factory: Any | None = None,
        device_factory: Any | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        self.device = device
        self.slots = slots
        self.max_numel = max_numel
        self._empty_factory = empty_factory
        self._device_factory = device_factory

        self._lock = threading.Lock()
        self._slot_buffers: dict[int, Any] = {}
        self._slot_owner: dict[int, int] = {i: -1 for i in range(slots)}
        self._recv_buffers: dict[int, Any] = {}
        self._recv_owner: dict[int, int] = {}
        self._seqno_slot: dict[int, int] = {}
        self._batches: dict[int, LwdStagedBatch] = {}
        #: req_id -> per-request storage (kept until the request ends)
        self._reqs: dict[str, LwdReqEmbed] = {}
        #: seqno -> req_ids it delivered (so a drain can drop the descriptor)
        self._batch_reqs: dict[int, list[str]] = {}
        self._host_bytes = 0
        self._released_requests = 0
        self._aborted_requests = 0

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
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
            return [s for s, owner in self._slot_owner.items() if owner >= 0]

    def busy_recv_buffers(self) -> list[int]:
        with self._lock:
            return sorted(self._recv_owner)

    def staged_seqnos(self) -> list[int]:
        """Seqnos currently holding a batch descriptor (abort path)."""
        with self._lock:
            return list(self._batches)

    def req_ids(self) -> list[str]:
        """Requests currently holding host storage (diagnostics)."""
        with self._lock:
            return list(self._reqs)

    def take_req(self, req_id: str) -> LwdReqEmbed | None:
        with self._lock:
            return self._reqs.get(req_id)

    def recv_rows(self, req_id: str) -> int:
        with self._lock:
            req = self._reqs.get(req_id)
            return req.recv_rows if req is not None else 0

    # ------------------------------------------------------------------ #
    # Admission (comm thread): bind slot + receive buffer to a recv        #
    # ------------------------------------------------------------------ #
    def allocate(self, seqno: int, numel: int) -> tuple[Any, Any, int]:
        """Reserve a device slot and a rotating receive buffer.

        Returns ``(slot_view, recv_buffer, slot_index)`` for the recv request.
        The receive buffer is a **rotating pool**: it lives only for the D2H
        (plus the split), because the per-request buffers have their own,
        much longer lifetime.

        Raises:
            LwdHostStageSlotBusy: every slot (or every receive buffer) is in
                use.  The caller queues the hint - it is never dropped.
        """
        with self._lock:
            if self.max_numel and numel > self.max_numel:
                raise LwdHostStageGeometryError(
                    f"seqno={seqno}: numel {numel} exceeds slot capacity "
                    f"{self.max_numel}"
                )
            free_slots = [s for s, o in self._slot_owner.items() if o < 0]
            if not free_slots:
                raise LwdHostStageSlotBusy(
                    f"seqno={seqno}: all {self.slots} device slot(s) busy "
                    f"(held by seqno={sorted(self._seqno_slot)})"
                )
            slot = free_slots[0]
            buf = self._slot_buffers.get(slot)
            if buf is None or buf.numel() < numel:
                buf = self._make_device_buffer(self.max_numel or numel)
                self._slot_buffers[slot] = buf
            self._slot_owner[slot] = seqno
            self._seqno_slot[seqno] = slot
        return buf[:numel], self._take_recv_buffer(numel), slot

    def _take_recv_buffer(self, numel: int) -> Any:
        """Grab (or allocate) a rotating receive buffer, tracked by ``id()``.

        Returns the **root** buffer, never a view: the pool keys on ``id()`` so
        the same object must come back at release time.  Cached buffers are
        only reused when they are already exactly ``numel``; a larger one is
        left for a bigger batch rather than handed out as an oversized slice,
        which keeps the received payload byte-exact (its length is checked
        against the row table at split time).
        """
        key = None
        for k, buf in self._recv_buffers.items():
            if k not in self._recv_owner and buf.numel() == numel:
                key = k
                break
        if key is None:
            buf = self._make_host_buffer(numel)
            key = id(buf)
            self._recv_buffers[key] = buf
        self._recv_owner[key] = numel
        return self._recv_buffers[key]

    def _release_recv_buffer(self, recv_buffer) -> None:
        self._recv_owner.pop(id(recv_buffer), None)

    def abandon(self, seqno: int, recv_buffer: Any | None = None) -> None:
        """Roll back an ``allocate`` whose receive was never posted."""
        with self._lock:
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1
        if recv_buffer is not None:
            self._release_recv_buffer(recv_buffer)

    # ------------------------------------------------------------------ #
    # Commit (comm thread): D2H done, split into per-request buffers       #
    # ------------------------------------------------------------------ #
    def commit_received(
        self,
        seqno: int,
        recv_buffer: Any,
        numel: int,
        hidden_size: int,
        req_ids: list[str],
        num_tokens_per_req: list[int],
        num_hash_layers: int = 0,
        num_experts_per_tok: int = 0,
        row_offsets: list[int] | None = None,
    ) -> LwdStagedBatch:
        """Split the received payload into per-request buffers.

        Called by the comm thread the moment ``staged_event.query()`` turns
        true (the pinned receive buffer now holds the whole batch).  Returns
        the device slot and the receive buffer to their pools immediately -
        the payload now lives in the per-request buffers.

        Args:
            row_offsets: prompt offset of each request's rows within the
                batch (the edge announces these); defaults to the running
                sum, i.e. a fresh, contiguous chunk per request in item order.

        Raises:
            LwdHostStageGeometryError: if the announced row layout does not
                account for exactly ``numel`` elements.  Slicing on a wrong
                layout would silently pair embeddings with the wrong request.
        """
        rows = list(num_tokens_per_req)
        total_rows = sum(rows)
        tail_per_row = 4 * num_hash_layers * num_experts_per_tok
        expect = total_rows * hidden_size + (total_rows * tail_per_row)
        if expect != numel:
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: payload {numel} != rows*hidden + rows*tail = "
                f"{total_rows}*{hidden_size} + {total_rows}*{tail_per_row} = "
                f"{expect} (row table vs received size mismatch)"
            )
        if row_offsets is None:
            row_offsets = []
            acc = 0
            for n in rows:
                row_offsets.append(acc)
                acc += n
        elif len(row_offsets) != len(rows):
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: {len(row_offsets)} row offsets vs {len(rows)} "
                f"requests"
            )

        batch = LwdStagedBatch(
            seqno=seqno,
            req_ids=list(req_ids),
            num_tokens_per_req=rows,
            hidden_size=hidden_size,
            num_hash_layers=num_hash_layers,
            num_experts_per_tok=num_experts_per_tok,
            recv_bytes=int(numel) * 2,
        )
        # 载荷布局:embeds 区在前、专家 ID 区在后(与线上载荷一致)。
        # 每个请求按**它自己声明的 prompt offset**落进自己的 buffer —— 记录
        # 必须能装下多批增量(边侧把长 prompt 切成多段发时,一个请求会跨多个
        # seqno 到达),所以这里用 note_arrival + 累积推进,而不是"新建覆盖"。
        embeds_base = 0
        tail_base = total_rows * hidden_size
        with self._lock:
            for i, rid in enumerate(req_ids):
                n = rows[i]
                host = self._reqs.get(rid)
                if host is None:
                    host = LwdReqEmbed(
                        req_id=rid,
                        host=self._make_host_buffer(n * hidden_size + n * tail_per_row),
                        numel=n * hidden_size,
                        prompt_rows=n,
                        hidden_size=hidden_size,
                        tail_off=(n * hidden_size) if tail_per_row else -1,
                        tail_numel=n * tail_per_row,
                        num_hash_layers=num_hash_layers,
                        num_experts_per_tok=num_experts_per_tok,
                    )
                    self._reqs[rid] = host
                    self._host_bytes += (n * hidden_size + n * tail_per_row) * 2
                    # 该 buffer 的第 0 行对应 prompt 的哪一行 —— 批次里每个
                    # 请求都从自己的 row 0 开始,所以写入偏移必须减去这个基准
                    # （用 prompt 绝对行号会写到别人的范围里去）。
                    host._row_base = row_offsets[i]
                else:
                    # 续传增量:prompt 长度/几何必须与首批一致,否则后面的
                    # 切片全会错位(静默串数据),直接 fail-fast。
                    if host.hidden_size != hidden_size:
                        raise LwdHostStageGeometryError(
                            f"req={rid}: hidden_size {host.hidden_size} != "
                            f"{hidden_size} on a later chunk"
                        )
                    dest_row = row_offsets[i] - host._row_base
                    if dest_row < 0:
                        raise LwdHostStageGeometryError(
                            f"req={rid}: chunk starts at prompt row "
                            f"{row_offsets[i]} which is before this buffer's "
                            f"base row {host._row_base} (delivery reordering)"
                        )
                    # 首批按"那一批的行数"分配,所以后续 chunk 可能落在已分配
                    # 范围之外 —— 这时**扩容并保留已写内容**,而不是假定
                    # "首批就是整个 prompt"。扩容是显式且带告警的:它只在多
                    # chunk 交付时发生,是几何假设被打破的可观测信号。
                    need_rows = dest_row + n
                    if need_rows > host.prompt_rows:
                        old_rows = host.prompt_rows
                        grown = self._make_host_buffer(
                            need_rows * hidden_size + need_rows * tail_per_row
                        )
                        grew = old_rows * hidden_size
                        if grew:
                            grown[:grew].copy_(host.host[:grew])
                        if tail_per_row:
                            old_tail = old_rows * tail_per_row
                            grown[need_rows * hidden_size:
                                  need_rows * hidden_size + old_tail].copy_(
                                host.host[host.tail_off: host.tail_off + old_tail]
                            )
                        self._host_bytes += (
                            (need_rows - old_rows) * (hidden_size + tail_per_row) * 2
                        )
                        logger.warning(
                            "[Lwd][host-stage] req=%s buffer grown %d -> %d rows "
                            "on a later chunk (multi-chunk delivery)",
                            rid, old_rows, need_rows,
                        )
                        host.host = grown
                        host.prompt_rows = need_rows
                        # Keep the byte accounting fields in step with the
                        # grown allocation, otherwise release_request would
                        # reclaim (and report) less than was actually held.
                        host.numel = need_rows * hidden_size
                        host.tail_numel = need_rows * tail_per_row
                        host.tail_off = (
                            need_rows * hidden_size if tail_per_row else -1
                        )
                dest_row = row_offsets[i] - host._row_base
                if dest_row < 0:
                    raise LwdHostStageGeometryError(
                        f"req={rid}: chunk starts at prompt row "
                        f"{row_offsets[i]} which is before this buffer's base "
                        f"row {host._row_base} (delivery reordering)"
                    )
                dest = dest_row * hidden_size
                host.host[dest: dest + n * hidden_size].copy_(
                    recv_buffer[embeds_base: embeds_base + n * hidden_size]
                )
                batch.src_offsets.append(embeds_base)
                if tail_per_row:
                    t_off = tail_base + row_offsets[i] * tail_per_row
                    host.host[
                        host.tail_off + dest_row * tail_per_row:
                        host.tail_off + (dest_row + n) * tail_per_row
                    ].copy_(recv_buffer[t_off: t_off + n * tail_per_row])
                    batch.src_tail_offsets.append(t_off)
                else:
                    batch.src_tail_offsets.append(-1)
                embeds_base += n * hidden_size
                host.note_arrival(dest_row, n)
            self._batches[seqno] = batch
            self._batch_reqs[seqno] = list(req_ids)
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1
        self._release_recv_buffer(recv_buffer)
        return batch

    # ------------------------------------------------------------------ #
    # Consumption (forward thread)                                        #
    # ------------------------------------------------------------------ #
    def batch(self, seqno: int) -> LwdStagedBatch | None:
        with self._lock:
            return self._batches.get(seqno)

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]:
        """``(host_start, numel, tail_start)`` for this request's next window.

        One request = one contiguous range in its own buffer, so there is
        exactly one copy segment per request per step (no cross-request
        merging - the buffers are independent by construction).
        tail_start is -1 when the model carries no expert IDs.
        """
        with self._lock:
            req = self._reqs.get(req_id)
        if req is None:
            raise LwdHostStageGeometryError(f"req={req_id}: not staged")
        start, end = req.read_range(n_rows)
        t_start, _t_end = req.tail_range(n_rows)
        return start, end - start, (t_start if req.tail_numel else -1)

    def consume(self, req_id: str, n_rows: int) -> None:
        """Advance the computed pointer after the rows were really used."""
        with self._lock:
            req = self._reqs.get(req_id)
        if req is None:
            raise LwdHostStageGeometryError(f"req={req_id}: not staged")
        req.note_consumed(n_rows)

    def abort_request(self, req_id: str) -> bool:
        """Freeze a request's remainder (design §6.3).  Frees nothing by itself."""
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                return False
            already = req.aborted
            req.mark_aborted()
            if not already:
                self._aborted_requests += 1
        return True

    def release_request(self, req_id: str) -> int:
        """Free a request's host storage - called when the **request ends**.

        Design §6.1: storage lives until the request finishes, not until it
        is consumed.  Returns the bytes reclaimed (0 if unknown/already gone).
        """
        with self._lock:
            req = self._reqs.pop(req_id, None)
            if req is None:
                return 0
            freed = (req.numel + req.tail_numel) * 2
            self._host_bytes -= freed
            self._released_requests += 1
        return freed

    def release_if_drained(self, seqno: int) -> bool:
        """Drop a batch descriptor once none of its requests are still held.

        Only the descriptor goes; the payload was already owned (and freed)
        per request.  Keeps ``staged_seqnos`` (and the abort scan) bounded.
        """
        with self._lock:
            rids = self._batch_reqs.get(seqno)
            if rids is None:
                return False
            if any(rid in self._reqs for rid in rids):
                return False
            del self._batch_reqs[seqno]
            self._batches.pop(seqno, None)
            return True

    def batches_holding(self, req_id: str) -> list[int]:
        """Seqnos whose delivery included ``req_id`` (abort bookkeeping)."""
        with self._lock:
            return [s for s, rids in self._batch_reqs.items() if req_id in rids]

    # ------------------------------------------------------------------ #
    # Buffer factories                                                    #
    # ------------------------------------------------------------------ #
    def _make_host_buffer(self, numel: int) -> Any:
        if self._empty_factory is not None:
            return self._empty_factory(numel)
        import torch  # local import: keep the module importable without torch

        return torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)

    def _make_device_buffer(self, numel: int) -> Any:
        if self._device_factory is not None:
            return self._device_factory(numel)
        import torch

        return torch.empty(numel, dtype=torch.bfloat16, device=self.device)
