# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Torch-free tests for per-request LWD host staging."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_host_stage.py"
SPEC = importlib.util.spec_from_file_location("lwd_host_stage_under_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
fake_vllm = types.ModuleType("vllm")
fake_logger_module = types.ModuleType("vllm.lwd_logger")
fake_logger_module.lwd_logger = logging.getLogger("lwd-host-stage-test")
fake_vllm.lwd_logger = fake_logger_module
with mock.patch.dict(
    sys.modules,
    {"vllm": fake_vllm, "vllm.lwd_logger": fake_logger_module, "torch": None},
):
    SPEC.loader.exec_module(module)

LwdHostStageGeometryError = module.LwdHostStageGeometryError
LwdHostStageSlotBusy = module.LwdHostStageSlotBusy
LwdHostStager = module.LwdHostStager

HIDDEN = 4


class FakeBuffer:
    """A sliced view that exposes coordinates without storing payload data."""

    def __init__(self, numel: int, start: int = 0) -> None:
        self._numel = numel
        self.start = start

    def numel(self) -> int:
        return self._numel

    def __getitem__(self, key: slice) -> FakeBuffer:
        assert key.step is None or key.step == 1
        start = key.start or 0
        stop = self._numel if key.stop is None else key.stop
        assert 0 <= start <= stop <= self._numel
        return FakeBuffer(stop - start, self.start + start)


def make_stager(slots: int = 2, max_numel: int = 0) -> LwdHostStager:
    return LwdHostStager(
        device="cpu",
        slots=slots,
        max_numel=max_numel,
        empty_factory=FakeBuffer,
        device_factory=FakeBuffer,
    )


def plan_chunk(
    stager: LwdHostStager,
    seqno: int,
    rows: int,
    offset: int = 0,
    prompt_rows: int = 8,
    req_id: str = "r0",
    refill: bool = False,
    prefix_nonce: int = 0,
) -> list[tuple[FakeBuffer, int, int]]:
    return stager.plan_recv_segments(
        seqno=seqno,
        numel=rows * HIDDEN,
        hidden_size=HIDDEN,
        req_ids=[req_id],
        num_tokens_per_req=[rows],
        prompt_lens=[prompt_rows],
        row_base=[offset],
        refills=[refill],
        prefix_nonces=[prefix_nonce],
    )


def test_multi_chunk_arrival_and_read_coordinates() -> None:
    stager = make_stager()
    _, slot = stager.allocate(1, 2 * HIDDEN)
    first = plan_chunk(stager, 1, rows=2, prompt_rows=6)
    assert [(view.start, source, size) for view, source, size in first] == [(0, 0, 8)]
    req = stager.take_req("r0")
    assert req.host.numel() == 6 * HIDDEN
    original_host = req.host

    assert stager.publish_received(1) == [("r0", 2, 0)]
    assert stager.busy_slots() == []
    assert stager.recv_rows("r0") == 2
    assert stager.segment_for("r0", 2) == (0, 8, -1)
    stager.consume("r0", 2)

    stager.allocate(2, 3 * HIDDEN)
    second = plan_chunk(stager, 2, rows=3, offset=2, prompt_rows=6)
    assert [(view.start, source, size) for view, source, size in second] == [(8, 0, 12)]
    assert stager.take_req("r0").host is original_host
    assert stager.publish_received(2) == [("r0", 3, 2)]
    assert stager.recv_rows("r0") == 5
    assert stager.consumed_rows("r0") == 2
    assert stager.segment_for("r0", 3) == (8, 12, -1)
    assert slot == 0


def test_duplicate_arrival_is_idempotent() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, prompt_rows=6)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=3, offset=2, prompt_rows=6)
    stager.publish_received(2)
    assert stager.recv_rows("r0") == 5

    duplicate = plan_chunk(stager, 3, rows=3, offset=2, prompt_rows=6)
    assert [(view.start, source, size) for view, source, size in duplicate] == [(8, 0, 12)]
    assert stager.publish_received(3) == [("r0", 3, 2)]
    assert stager.recv_rows("r0") == 5
    assert stager.take_req("r0").available() == 5


def test_nonzero_first_chunk_seeds_prefix_watermarks() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, offset=2, prompt_rows=6)
    req = stager.take_req("r0")
    assert (req._stage_floor, req.recv_rows, req._written_hi, req.consumed) == (2, 2, 2, 2)
    assert req.prefix_nonce == 0
    assert stager.stage_floor("r0") == 2
    with pytest.raises(LwdHostStageGeometryError, match="after arrivals"):
        req.seed_prefix(2)

    assert stager.publish_received(1) == [("r0", 2, 2)]
    assert stager.recv_rows("r0") == 4
    assert stager.segment_for("r0", 2) == (8, 8, -1)
    with pytest.raises(LwdHostStageGeometryError, match="before drain"):
        plan_chunk(stager, 2, rows=2, offset=4, prompt_rows=6, prefix_nonce=1)
    assert stager.take_req("r0") is req


def test_refill_completion_hole_and_second_refill() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=6, offset=4, prompt_rows=10)
    stager.publish_received(1)
    assert stager.take_refill_done() == []

    assert stager.rewind_consumed("r0", 2)
    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=10, refill=True)
    stager.publish_received(2)
    assert stager.take_refill_done() == [("r0", 0)]
    assert stager.take_refill_done() == []
    assert stager.stage_floor("r0") == 2
    assert stager.recv_rows("r0") == 10
    assert stager.segment_for("r0", 2) == (8, 8, -1)
    stager.consume("r0", 2)

    assert stager.rewind_consumed("r0", 0)
    plan_chunk(stager, 3, rows=1, offset=0, prompt_rows=10, refill=True)
    stager.publish_received(3)
    assert stager.take_refill_done() == []
    with pytest.raises(LwdHostStageGeometryError, match="unfilled refill hole"):
        stager.segment_for("r0", 2)

    plan_chunk(stager, 4, rows=1, offset=1, prompt_rows=10, refill=True)
    stager.publish_received(4)
    assert stager.take_refill_done() == [("r0", 0)]
    assert stager.stage_floor("r0") == 0
    assert stager.segment_for("r0", 2) == (0, 8, -1)


def test_refill_done_preserves_prefix_nonce() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=6, offset=4, prompt_rows=10, prefix_nonce=7)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=10, refill=True, prefix_nonce=7)

    stager.publish_received(2)
    assert stager.take_refill_done() == [("r0", 7)]


def test_refill_gap_overrun_and_flag_mismatch_fail_fast() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=6, offset=4, prompt_rows=10)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=1, offset=1, prompt_rows=10, refill=True)
    stager.publish_received(2)

    plan_chunk(stager, 3, rows=1, offset=3, prompt_rows=10, refill=True)
    with pytest.raises(LwdHostStageGeometryError, match="refill gap"):
        stager.publish_received(3)
    assert stager.take_refill_done() == []
    with pytest.raises(LwdHostStageGeometryError, match="overrun stage floor"):
        plan_chunk(stager, 4, rows=2, offset=3, prompt_rows=10, refill=True)
    with pytest.raises(LwdHostStageGeometryError, match="refill flag"):
        plan_chunk(stager, 5, rows=1, offset=1, prompt_rows=10)


def test_aborted_refill_is_not_reported() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=6, offset=4, prompt_rows=10)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=10, refill=True)
    assert stager.abort_request("r0")

    assert stager.publish_received(2) == []
    assert stager.take_refill_done() == []
    assert stager.stage_floor("r0") == 4


def test_gap_and_out_of_bounds_fail_fast() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, prompt_rows=6)
    stager.publish_received(1)

    plan_chunk(stager, 2, rows=1, offset=4, prompt_rows=6)
    with pytest.raises(LwdHostStageGeometryError, match="arrival gap"):
        stager.publish_received(2)
    assert stager.recv_rows("r0") == 2

    with pytest.raises(LwdHostStageGeometryError, match="exceed prompt_rows"):
        plan_chunk(stager, 3, rows=5, offset=2, prompt_rows=6)
    with pytest.raises(LwdHostStageGeometryError, match="exceed prompt_rows"):
        plan_chunk(stager, 4, rows=7, prompt_rows=6)


def test_slot_busy_and_abandon() -> None:
    stager = make_stager(slots=1, max_numel=16)
    view, slot = stager.allocate(1, 8)
    assert (view.numel(), slot) == (8, 0)
    assert stager.busy_slots() == [0]
    with pytest.raises(LwdHostStageSlotBusy):
        stager.allocate(2, 8)

    stager.abandon(1)
    stager.abandon(1)
    assert stager.busy_slots() == []
    assert stager.allocate(2, 8)[1] == 0
    with pytest.raises(LwdHostStageGeometryError, match="slot capacity"):
        stager.allocate(3, 17)


def test_abort_is_idempotent_and_release_is_at_request_end() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, prompt_rows=6)
    stager.publish_received(1)
    assert stager.host_bytes == 6 * HIDDEN * 2
    assert stager.batches_holding("r0") == [1]
    assert not stager.release_if_drained(1)

    assert stager.abort_request("r0")
    assert stager.abort_request("r0")
    assert stager.aborted_requests == 1
    assert stager.take_req("r0").done()
    assert stager.request_count == 1
    assert stager.host_bytes == 48
    assert not stager.abort_request("missing")

    assert stager.release_request("r0") == 48
    assert stager.release_request("r0") == 0
    assert stager.host_bytes == 0
    assert stager.released_requests == 1
    assert stager.release_if_drained(1)
    assert stager.staged_count == 0


def test_full_consumption_finishes_without_early_release() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=4, prompt_rows=4)
    stager.publish_received(1)
    stager.consume("r0", 4)

    assert stager.take_req("r0").done()
    assert stager.request_count == 1
    assert stager.host_bytes == 32
    assert stager.release_request("r0") == 32
    assert stager.request_count == 0


def test_rewind_consumed_preserves_arrival_and_rereads() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=4, prompt_rows=6)
    stager.publish_received(1)
    stager.consume("r0", 4)
    req = stager.take_req("r0")

    assert stager.rewind_consumed("r0", 2)
    assert req.consumed == 2
    assert req.recv_rows == 4
    assert req._written_hi == 4
    assert stager.segment_for("r0", 2) == (8, 8, -1)

    assert stager.rewind_consumed("r0", 0)
    assert req.consumed == 0
    assert req.recv_rows == 4
    assert req._written_hi == 4
    assert stager.segment_for("r0", 4) == (0, 16, -1)
    for invalid in (-1, 1):
        with pytest.raises(LwdHostStageGeometryError, match="rewind target"):
            stager.rewind_consumed("r0", invalid)
        assert req.consumed == 0
    assert not stager.rewind_consumed("missing", 0)


def test_skip_consumed_for_self_hit_stays_within_recv() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=4, prompt_rows=6)
    stager.publish_received(1)
    stager.consume("r0", 2)

    assert stager.skip_consumed("r0", 4)
    assert stager.consumed_rows("r0") == 4
    assert stager.recv_rows("r0") == 4
    with pytest.raises(LwdHostStageGeometryError, match="skip target 5"):
        stager.skip_consumed("r0", 5)
    assert stager.consumed_rows("r0") == 4
    assert stager.rewind_consumed("r0", 2)
    assert not stager.skip_consumed("missing", 2)


def test_dsv4_embed_and_tail_source_offsets() -> None:
    stager = make_stager()
    num_hash_layers = 2
    num_experts_per_tok = 3
    tail_per_row = 4 * num_hash_layers * num_experts_per_tok
    segments = stager.plan_recv_segments(
        seqno=1,
        numel=3 * (HIDDEN + tail_per_row),
        hidden_size=HIDDEN,
        req_ids=["r0", "r1"],
        num_tokens_per_req=[2, 1],
        prompt_lens=[2, 3],
        num_hash_layers=num_hash_layers,
        num_experts_per_tok=num_experts_per_tok,
    )
    assert [(view.start, source, size) for view, source, size in segments] == [
        (0, 0, 8),
        (8, 12, 48),
        (0, 8, 4),
        (12, 60, 24),
    ]
    batch = stager.batch(1)
    assert batch.src_offsets == [0, 8]
    assert batch.src_tail_offsets == [12, 60]
    assert stager.take_req("r0").host.numel() == 56
    assert stager.take_req("r1").host.numel() == 84
    stager.publish_received(1)
    assert stager.segment_for("r0", 2) == (0, 8, 8)
    assert stager.segment_for("r1", 1) == (0, 4, 12)

    stager.consume("r1", 1)
    assert stager.rewind_consumed("r1", 0)
    assert stager.segment_for("r1", 1) == (0, 4, 12)
    stager.consume("r1", 1)
    continuation = stager.plan_recv_segments(
        seqno=2,
        numel=2 * (HIDDEN + tail_per_row),
        hidden_size=HIDDEN,
        req_ids=["r1"],
        num_tokens_per_req=[2],
        prompt_lens=[3],
        row_base=[1],
        num_hash_layers=num_hash_layers,
        num_experts_per_tok=num_experts_per_tok,
    )
    assert [(view.start, source, size) for view, source, size in continuation] == [
        (4, 0, 8),
        (36, 8, 48),
    ]
    stager.publish_received(2)
    assert stager.segment_for("r1", 2) == (4, 8, 36)


def test_dsv4_refill_reuses_embed_and_tail_layout() -> None:
    stager = make_stager()
    num_hash_layers = 2
    num_experts_per_tok = 3
    tail_per_row = 4 * num_hash_layers * num_experts_per_tok
    common = {
        "hidden_size": HIDDEN,
        "req_ids": ["r0"],
        "num_tokens_per_req": [2],
        "prompt_lens": [4],
        "num_hash_layers": num_hash_layers,
        "num_experts_per_tok": num_experts_per_tok,
    }
    stager.plan_recv_segments(
        seqno=1,
        numel=2 * (HIDDEN + tail_per_row),
        row_base=[2],
        **common,
    )
    stager.publish_received(1)
    refill = stager.plan_recv_segments(
        seqno=2,
        numel=2 * (HIDDEN + tail_per_row),
        row_base=[0],
        refills=[True],
        **common,
    )

    assert [(view.start, source, size) for view, source, size in refill] == [(0, 0, 8), (16, 8, 48)]
    stager.publish_received(2)
    assert stager.take_refill_done() == [("r0", 0)]
    assert stager.rewind_consumed("r0", 0)
    assert stager.segment_for("r0", 2) == (0, 8, 16)


def test_read_consume_and_table_geometry_errors() -> None:
    stager = make_stager()
    with pytest.raises(LwdHostStageGeometryError, match="equal length"):
        stager.plan_recv_segments(1, 8, HIDDEN, ["r0"], [2], [])
    with pytest.raises(LwdHostStageGeometryError, match="payload"):
        stager.plan_recv_segments(1, 9, HIDDEN, ["r0"], [2], [2])

    plan_chunk(stager, 1, rows=2, prompt_rows=4)
    stager.publish_received(1)
    with pytest.raises(LwdHostStageGeometryError, match="received rows"):
        stager.segment_for("r0", 3)
    with pytest.raises(LwdHostStageGeometryError, match="consumed end"):
        stager.consume("r0", 3)
    with pytest.raises(LwdHostStageGeometryError, match="n_rows >= 1"):
        stager.consume("r0", 0)
    with pytest.raises(LwdHostStageGeometryError, match="not staged"):
        stager.segment_for("missing", 1)


def test_abandon_rolls_back_plan_and_only_new_hosts() -> None:
    stager = make_stager(slots=1)
    stager.allocate(1, 2 * HIDDEN)
    plan_chunk(stager, 1, rows=2, prompt_rows=6)
    stager.publish_received(1)
    original = stager.take_req("r0")
    original_bytes = stager.host_bytes
    original_staged_count = stager.staged_count

    stager.allocate(2, 3 * HIDDEN)
    stager.plan_recv_segments(
        seqno=2,
        numel=3 * HIDDEN,
        hidden_size=HIDDEN,
        req_ids=["r0", "r1"],
        num_tokens_per_req=[2, 1],
        prompt_lens=[6, 4],
        row_base=[2, 0],
    )
    assert stager.take_req("r1") is not None
    assert stager.busy_slots() == [0]
    stager.abandon(2)

    assert stager.busy_slots() == []
    assert stager.host_bytes == original_bytes
    assert stager.staged_count == original_staged_count
    assert stager.take_req("r0") is original
    assert stager.recv_rows("r0") == 2
    assert stager.take_req("r1") is None
    assert stager.batch(2) is None
    assert stager.batches_holding("r0") == [1]
    with pytest.raises(LwdHostStageGeometryError, match="no arrival plan"):
        stager.publish_received(2)


def test_abandon_rolls_back_seeded_fresh_host() -> None:
    stager = make_stager(slots=1)
    stager.allocate(1, 2 * HIDDEN)
    plan_chunk(stager, 1, rows=2, offset=2, prompt_rows=6)
    seeded = stager.take_req("r0")
    assert seeded.recv_rows == 2
    assert stager.host_bytes == 48

    stager.abandon(1)
    assert stager.take_req("r0") is None
    assert stager.host_bytes == 0
    assert stager.request_count == 0
    assert stager.busy_slots() == []

    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=6)
    published = stager.take_req("r0")
    stager.publish_received(2)
    stager.abandon(2)
    assert stager.take_req("r0") is published
    assert stager.recv_rows("r0") == 4
    assert stager.host_bytes == 48


def test_abort_drains_all_old_plans_before_request_id_reuse() -> None:
    stager = make_stager(slots=2)
    plan_chunk(stager, 1, rows=2, prompt_rows=8)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=8)
    plan_chunk(stager, 3, rows=2, offset=4, prompt_rows=8)
    assert stager.abort_request("r0")
    assert stager.release_request("r0") == 64

    assert plan_chunk(stager, 4, rows=2, offset=4, prompt_rows=8) == []
    assert not stager.release_if_drained(2)
    assert stager.publish_received(2) == []
    assert plan_chunk(stager, 5, rows=2, offset=4, prompt_rows=8, prefix_nonce=2) == []
    assert stager.publish_received(3) == []

    reused = plan_chunk(stager, 6, rows=2, prompt_rows=4, prefix_nonce=2)
    assert [(view.start, source, size) for view, source, size in reused] == [(0, 0, 8)]
    assert stager.take_req("r0").prefix_nonce == 2
    assert stager.publish_received(4) == []
    assert stager.publish_received(5) == []
    assert stager.publish_received(6) == [("r0", 2, 0)]
    assert stager.recv_rows("r0") == 2
    assert stager.host_bytes == 32
    for seqno in (2, 3, 4, 5):
        assert stager.release_if_drained(seqno)
    assert not stager.release_if_drained(6)
    assert stager.release_request("r0") == 32
    assert stager.release_if_drained(6)


def test_finish_drains_old_plan_and_allows_request_id_reuse() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, prompt_rows=4)
    stager.publish_received(1)
    plan_chunk(stager, 2, rows=2, offset=2, prompt_rows=4)
    assert stager.release_request("r0") == 32

    assert plan_chunk(stager, 3, rows=2, offset=2, prompt_rows=4) == []
    assert not stager.release_if_drained(2)
    assert stager.publish_received(2) == []
    reused = plan_chunk(stager, 4, rows=1, prompt_rows=2)
    assert [(view.start, source, size) for view, source, size in reused] == [(0, 0, 4)]
    assert stager.publish_received(3) == []
    assert stager.publish_received(4) == [("r0", 1, 0)]
    assert stager.recv_rows("r0") == 1
    assert stager.release_if_drained(2)
    assert stager.release_if_drained(3)


def test_finish_without_pending_allows_immediate_request_id_reuse() -> None:
    stager = make_stager()
    plan_chunk(stager, 1, rows=2, prompt_rows=2)
    stager.publish_received(1)
    stager.consume("r0", 2)
    assert stager.release_request("r0") == 16
    assert stager.release_if_drained(1)

    reused = plan_chunk(stager, 2, rows=2, prompt_rows=2)
    assert [(view.start, source, size) for view, source, size in reused] == [(0, 0, 8)]
    assert stager.publish_received(2) == [("r0", 2, 0)]
    assert stager.request_count == 1
    assert stager.host_bytes == 16
    assert stager.release_request("r0") == 16
    assert stager.release_if_drained(2)


def test_mixed_batch_skips_old_generation_without_shifting_sources() -> None:
    stager = make_stager(slots=1)
    assert not stager.abort_request("done")
    stager.allocate(1, 5 * HIDDEN)
    segments = stager.plan_recv_segments(
        seqno=1,
        numel=5 * HIDDEN,
        hidden_size=HIDDEN,
        req_ids=["done", "live"],
        num_tokens_per_req=[2, 3],
        prompt_lens=[2, 3],
    )
    assert [(view.start, source, size) for view, source, size in segments] == [(0, 8, 12)]
    assert stager.take_req("done") is None
    assert stager.batch(1).src_offsets == [0, 8]
    assert stager.batch(1).src_tail_offsets == [-1, -1]
    assert stager.batch(1).total_rows == 5
    assert plan_chunk(stager, 2, rows=2, prompt_rows=2, req_id="done") == []

    assert stager.publish_received(1) == [("live", 3, 0)]
    reused = plan_chunk(stager, 3, rows=2, prompt_rows=2, req_id="done")
    assert [(view.start, source, size) for view, source, size in reused] == [(0, 0, 8)]
    assert stager.publish_received(2) == []
    assert stager.publish_received(3) == [("done", 2, 0)]
    assert stager.busy_slots() == []
    assert stager.recv_rows("done") == 2
    assert stager.recv_rows("live") == 3
    assert stager.release_if_drained(2)
    assert stager.release_request("done") == 16
    assert stager.release_if_drained(3)
    assert stager.release_request("live") == 24
    assert stager.release_if_drained(1)
