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
import time
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
        """Mark that rows up to ``offset + rows`` are now usable.

        Two separate things are tracked, and they must not be conflated:

        * ``recv_rows`` - the **arrival high-water mark**: how far into the
          prompt the cloud can dispatch.  It must account for *every* row that
          has landed, not just the rows of the most recently committed chunk:
          a prompt delivered as several chunks can have its earlier chunks
          arrive while the buffer is still small (they take the grow path), and
          a later chunk's own extent would then hide them.  Live failure: a
          request delivered as 1152 + 2358 rows was reported as having received
          2358 only, so the scheduler's W2 bound rejected a legitimate window.

        * ``_written_hi`` - how far *this call path* has written into the
          buffer; kept for gap detection.

        So the high-water mark is the maximum of the previous mark and the
        highest row this request has been written to at all.  Repeats are
        idempotent, which the edge's publish retries rely on.

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
        if end > self._written_hi:
            if offset > self._written_hi:
                # 到达出现空洞:[_written_hi, offset) 从未写入,而高水位会
                # 越过它 —— 调度器可能据此下发零值行。通道 FIFO 下不该发生,
                # 一旦发生必须留痕(此前这道检测因判据写错恒不触发)。
                logger.warning(
                    "[Lwd][host-stage] req=%s arrival gap: chunk covers "
                    "[%d, %d) but only %d rows were written before",
                    self.req_id, offset, end, self._written_hi,
                )
            self._written_hi = end
        # The high-water mark is the furthest row this request has reached by
        # any route - not merely the end of the range announced just now.
        prev = self.recv_rows
        self.recv_rows = max(self.recv_rows, self._written_hi)
        # 逐调用自证:现场把**实参**与**结果**一起打出来。"调用前打日志"已经
        # 让我连续几轮看错因果(日志里 dest_row 是对的,上报值却是本块行数),
        # 所以这一行必须紧贴赋值、并带上实例身份 —— 同名请求若各持一份 buffer,
        # 外部日志完全看不出来。
        logger.debug(
            "[Lwd][host-stage] ARRIVE req=%s id=%s in=(%d,%d) end=%d "
            "written_hi=%d recv_rows %d->%d prompt_rows=%d",
            self.req_id, id(self), offset, rows, end,
            self._written_hi, prev, self.recv_rows, self.prompt_rows,
        )

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
            # **这里是 W2 的权威落点**(design v7 §7):用**推进前**的
            # ``consumed`` 判断"这一步要读的区间是否已经写入"。别再在别处用
            # `recv_rows - consumed` 复核 —— 那个减法只有在指针还没被推进时
            # 才有意义,而携带它的取数路径本身就会推进它(真机踩过两次:
            # 调度器侧一次、runner 注入侧一次,都是"第一批就被误判越界")。
            raise LwdHostStageGeometryError(
                f"W2 VIOLATION req={self.req_id}: read {n_rows} rows at "
                f"consumed={self.consumed} exceeds received rows "
                f"{self.recv_rows} (rows not yet written would be read)"
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

    def tail_range_at(self, consumed_base: int, n_rows: int) -> tuple[int, int]:
        """同上,但起点显式给(推进前读指针)——注入侧在 harvest 推进
        consumed 之后才切尾段,直接读 self.consumed 会拿到下一个窗口的
        起点(真机踩过)。"""
        if self.tail_numel <= 0:
            return 0, 0
        per_row = 4 * self.num_hash_layers * self.num_experts_per_tok
        start = self.tail_off + consumed_base * per_row
        return start, start + n_rows * per_row

    def note_consumed(self, n_rows: int) -> None:
        """Advance the computed pointer (called after the dispatch is real)."""
        self.consumed += n_rows
        if self.consumed > self.prompt_rows:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: consumed {self.consumed} exceeded "
                f"prompt_rows {self.prompt_rows}"
            )

    def rewind_consumed(self, to_rows: int) -> None:
        """Roll the read pointer back to ``to_rows`` for a local recompute.

        Cloud-side KV preemption frees the blocks and zeroes the request's
        ``num_computed_tokens``; the next schedule re-dispatches the prompt
        from row 0.  The pinned payload lives until the request ends (design
        §6.1), and consumed rows are never overwritten by arrival writes
        (those land at ``>= recv_rows``), so re-reading earlier rows is
        legal - only the pointer moves.  ``recv_rows`` is untouched: the
        arrival high-water mark stays valid, and the invariant
        ``consumed <= recv_rows`` is preserved because the target never
        exceeds the current pointer.

        Raises:
            LwdHostStageGeometryError: if the target is outside
                ``[0, consumed]`` - a rewind past what was ever read would
                desynchronise the stage/inject identity for good.
        """
        if to_rows < 0 or to_rows > self.consumed:
            raise LwdHostStageGeometryError(
                f"req={self.req_id}: rewind target {to_rows} outside "
                f"[0, consumed={self.consumed}]"
            )
        self.consumed = to_rows

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
    "which requests came in this batch" and the row layout of the receive is
    validated at plan time.
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
    """Device ping-pong slots + per-request pinned storage.

    Threading: ``allocate``/``plan_recv_segments``/``publish_received`` run
    on the comm thread (they are part of posting/monitoring a receive), while
    ``adopt``/``consume``/``release_request`` run on the worker's forward
    thread.  One lock guards the tables; per-request buffers are created once
    at prompt length and only grow their *written* region afterwards, so the
    segmented D2H views handed to the channel stay valid for the whole
    receive.

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
        self._seqno_slot: dict[int, int] = {}
        self._batches: dict[int, LwdStagedBatch] = {}
        #: req_id -> per-request storage (kept until the request ends)
        self._reqs: dict[str, LwdReqEmbed] = {}
        #: seqno -> req_ids it delivered (so a drain can drop the descriptor)
        self._batch_reqs: dict[int, list[str]] = {}
        #: seqno -> [(LwdReqEmbed, dest_row, rows)] arrival plan, registered
        #: at submit time (plan_recv_segments) and published by
        #: ``publish_received`` once the segmented D2H completes
        self._arrival_plan: dict[int, list] = {}
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

    def consumed_rows(self, req_id: str) -> int:
        """该请求的存储读指针(下一个要搬去设备的 prompt 行号)。

        数据面与计算面靠一个恒等式对齐:设备上这批行被注入到 prompt 位置
        ``num_computed``(注入侧 ``start + c``,而 ``c = num_computed − start``
        把它化简掉了),而取数是从 ``consumed`` 顺序读的。两者**必须相等**,
        否则"读到的内容"和"写入的位置"就是错开的 —— 模型会拿到错位的
        embedding,表现为输出被截断/空白,且**不会有任何报错或乱码**。
        调度侧 ``_lwd_stage_requests_to_device`` 用它做这道核验。
        """
        with self._lock:
            req = self._reqs.get(req_id)
            return req.consumed if req is not None else 0

    # ------------------------------------------------------------------ #
    # Admission (comm thread): bind a device slot to a recv                #
    # ------------------------------------------------------------------ #
    def allocate(self, seqno: int, numel: int) -> tuple[Any, int]:
        """Reserve a device ping-pong slot for one receive.

        Returns ``(slot_view, slot_index)``.  The slot is released by
        ``publish_received`` once the segmented D2H has completed.

        Raises:
            LwdHostStageSlotBusy: every slot is in use.  The caller queues
                the hint - it is never dropped.
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
        return buf[:numel], slot

    def abandon(self, seqno: int) -> None:
        """Roll back an ``allocate`` whose receive was never posted."""
        with self._lock:
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1

    # ------------------------------------------------------------------ #
    # Plan (comm thread, submit time): per-request buffers + D2H segments  #
    # ------------------------------------------------------------------ #
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
    ) -> list[tuple[Any, int, int]]:
        """提交期建行表:per-request buffer 按 prompt 全长一次分好,返回
        本批的 D2H 分段 ``[(dst_pinned_view, src_elem_offset, numel)]``,
        由通道层在接收完成后逐段异步拷入。

        buffer 一次按全长分够、**没有扩容路径** —— 这是分段方案的安全前提:
        交给通道层的视图在后续 chunk 到达时不会被 realloc(否则在飞 D2H
        会写进已退役的旧 buffer)。``prompt_lens`` 因此是硬需求,缺失即
        几何错误 fail-fast(部署必须两仓同版本)。

        commit 期不再搬数据,``publish_received`` 只做高水位发布。
        """
        rows = list(num_tokens_per_req)
        _t0 = time.monotonic()
        _alloc_ms = 0.0
        total_rows = sum(rows)
        tail_per_row = 4 * num_hash_layers * num_experts_per_tok
        expect = total_rows * hidden_size + (total_rows * tail_per_row)
        if expect != numel:
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: payload {numel} != rows*hidden + rows*tail = "
                f"{total_rows}*{hidden_size} + {total_rows}*{tail_per_row} = "
                f"{expect} (row table vs received size mismatch)"
            )
        if len(prompt_lens) != len(rows):
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: {len(prompt_lens)} prompt_lens vs "
                f"{len(rows)} requests (分段 D2H 需要逐请求 prompt 全长,"
                f"两仓必须同版本部署)"
            )
        if row_base is None:
            row_base = [0] * len(rows)
        elif len(row_base) != len(rows):
            raise LwdHostStageGeometryError(
                f"seqno={seqno}: {len(row_base)} row bases vs {len(rows)} "
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
        segments: list[tuple[Any, int, int]] = []
        plan: list[tuple[Any, int, int]] = []
        with self._lock:
            acc = 0
            for i, rid in enumerate(req_ids):
                n = rows[i]
                host = self._reqs.get(rid)
                if host is None:
                    # 一次按 prompt 全长分够:后续 chunk 直接落进预留区间,
                    # 没有扩容、没有 realloc、没有在飞拷贝的视图失效问题。
                    plen = int(prompt_lens[i])
                    if row_base[i] + n > plen:
                        raise LwdHostStageGeometryError(
                            f"seqno={seqno} req={rid}: chunk end "
                            f"{row_base[i] + n} exceeds prompt_len={plen}"
                        )
                    # [Lwd][perf] pinned 分配耗时与 prompt 长度强相关
                    # (长 prompt 单次可达几十~几百 ms),且在通信线程持锁
                    # 执行 —— 长序列 TTFT 劣化的头号嫌疑,逐请求留证。
                    _ta = time.monotonic()
                    host = LwdReqEmbed(
                        req_id=rid,
                        host=self._make_host_buffer(
                            plen * hidden_size + plen * tail_per_row
                        ),
                        numel=plen * hidden_size,
                        prompt_rows=plen,
                        hidden_size=hidden_size,
                        tail_off=(plen * hidden_size) if tail_per_row else -1,
                        tail_numel=plen * tail_per_row,
                        num_hash_layers=num_hash_layers,
                        num_experts_per_tok=num_experts_per_tok,
                    )
                    _a = (time.monotonic() - _ta) * 1000
                    _alloc_ms += _a
                    if _a > 2.0:
                        logger.info(
                            "[Lwd][perf] host-alloc req=%s rows=%d "
                            "dur=%.2fms (%.1fMiB pinned)",
                            rid, plen, _a,
                            plen * (hidden_size + tail_per_row) * 2
                            / 2 ** 20,
                        )
                    self._reqs[rid] = host
                    self._host_bytes += (
                        plen * (hidden_size + tail_per_row) * 2
                    )
                    # 该 buffer 的第 0 行对应 prompt 的哪一行 = 它在提示里声明
                    # 的 prompt 起点(首次到达的那一段的起点;正常为 0)。
                    host._row_base = row_base[i]
                elif host.hidden_size != hidden_size:
                    # 续传增量:几何必须与首批一致,否则后面的切片全会错位
                    # (静默串数据),直接 fail-fast。
                    raise LwdHostStageGeometryError(
                        f"req={rid}: hidden_size {host.hidden_size} != "
                        f"{hidden_size} on a later chunk"
                    )
                # ---- 写入位置:prompt 坐标 − buffer 基准行 ----
                # ``row_base[i]`` 是本段在**该请求 prompt 内**的起点(prompt
                # 坐标);buffer 基准行是它首段的起点。两者相减得 buffer 内
                # 行号。批内布局(载荷内偏移)由行数累加得出,两套坐标不混用。
                dest_row = row_base[i] - host._row_base
                if dest_row < 0 or dest_row + n > host.prompt_rows:
                    raise LwdHostStageGeometryError(
                        f"req={rid}: chunk rows [{dest_row}, {dest_row + n}) "
                        f"out of buffer rows {host.prompt_rows} "
                        f"(delivery reordering or prompt_len mismatch)"
                    )
                segments.append(
                    (
                        host.host[
                            dest_row * hidden_size:
                            (dest_row + n) * hidden_size
                        ],
                        acc * hidden_size,
                        n * hidden_size,
                    )
                )
                batch.src_offsets.append(acc * hidden_size)
                if tail_per_row:
                    t_src = total_rows * hidden_size + acc * tail_per_row
                    segments.append(
                        (
                            host.host[
                                host.tail_off + dest_row * tail_per_row:
                                host.tail_off + (dest_row + n) * tail_per_row
                            ],
                            t_src,
                            n * tail_per_row,
                        )
                    )
                    batch.src_tail_offsets.append(t_src)
                else:
                    batch.src_tail_offsets.append(-1)
                plan.append((host, dest_row, n))
                acc += n
            self._batches[seqno] = batch
            self._batch_reqs[seqno] = list(req_ids)
            self._arrival_plan[seqno] = plan
        # [Lwd][perf] 提交期建行表耗时(通信线程持锁):dur 大且 alloc 占比
        # 高 = pinned 分配慢(长 prompt);alloc≈0 而 dur 大 = 锁/记账开销。
        logger.info(
            "[Lwd][perf] PLAN seqno=%d dur=%.2fms alloc=%.2fms reqs=%d "
            "rows=%d host_MB=%.1f",
            seqno, (time.monotonic() - _t0) * 1000, _alloc_ms,
            len(req_ids), total_rows, self._host_bytes / 2 ** 20,
        )
        return segments

    # ------------------------------------------------------------------ #
    # Publish (comm thread): D2H done, advance the high-water marks        #
    # ------------------------------------------------------------------ #
    def publish_received(self, seqno: int) -> list[tuple[str, int, int]]:
        """分段 D2H 完成后的发布:逐请求推进已收高水位 + 归还设备槽。

        返回 ``[(req_id, rows, prompt_offset)]`` 供 worker 上报。不再搬
        数据 —— 数据由通道层的分段 D2H 直接落进 per-request buffer。
        """
        with self._lock:
            plan = self._arrival_plan.pop(seqno, None)
            if plan is None:
                raise LwdHostStageGeometryError(
                    f"seqno={seqno}: no arrival plan (publish without plan; "
                    f"submit-time planning is mandatory in staged mode)"
                )
            rows = [
                (host.req_id, n, host._row_base + dest_row)
                for host, dest_row, n in plan
            ]
            for host, dest_row, n in plan:
                host.note_arrival(dest_row, n)
            slot = self._seqno_slot.pop(seqno, -1)
            if slot >= 0:
                self._slot_owner[slot] = -1
        return rows


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

    def rewind_consumed(self, req_id: str, to_rows: int) -> bool:
        """Roll a request's read pointer back for a local recompute.

        Called by the runner when the scheduler's watermark regresses below
        the staged read pointer (KV preemption zeroes num_computed_tokens).
        Returns False when the request is unknown - e.g. it finished between
        scheduling and execution under async scheduling; nothing to rewind.
        """
        with self._lock:
            req = self._reqs.get(req_id)
            if req is None:
                return False
            req.rewind_consumed(to_rows)
        return True

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
