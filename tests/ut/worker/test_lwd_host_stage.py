# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdReqEmbed`` / ``LwdHostStager`` tests.

Covers the preemption-recompute read-pointer rewind (cloud-side KV
preemption zeroes ``num_computed_tokens``, and the retained pinned
payload makes a local re-prefill legal once the read pointer is rolled
back), the prefix-cache seeding (``seed_prefix``: 首个 hint 的 prompt
offset 即暂存地板 S,设计 §3.2)与 refill 补发跟踪(``_refill_hi``
推进/补齐事件/``take_refill_done``,设计 §5.3)。

torch-free: ``lwd_host_stage`` must stay importable without torch/NPU (its
module docstring's hard constraint), so this file stubs ``vllm.logger`` and
loads the module from its file location - it runs on a plain CPU box.
"""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_host_stage.py"

# Importing vllm.logger pulls in the whole vllm package (torch-dependent);
# the module under test only needs ``init_logger``.
if "vllm.logger" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")
    fake_logger_mod = types.ModuleType("vllm.logger")
    fake_logger_mod.init_logger = logging.getLogger
    fake_vllm.logger = fake_logger_mod
    sys.modules.setdefault("vllm", fake_vllm)
    sys.modules["vllm.logger"] = fake_logger_mod

SPEC = importlib.util.spec_from_file_location("lwd_host_stage", MODULE)
module = importlib.util.module_from_spec(SPEC)
sys.modules["lwd_host_stage"] = module  # dataclass processing looks this up
SPEC.loader.exec_module(module)

LwdHostStageGeometryError = module.LwdHostStageGeometryError
LwdHostStager = module.LwdHostStager
LwdReqEmbed = module.LwdReqEmbed

HIDDEN = 4


def make_req(prompt_rows=10, recv=10, consumed=0) -> "LwdReqEmbed":
    req = LwdReqEmbed(
        req_id="r0",
        host=None,
        numel=prompt_rows * HIDDEN,
        prompt_rows=prompt_rows,
        hidden_size=HIDDEN,
    )
    if recv:
        req.note_arrival(0, recv)
    if consumed:
        req.note_consumed(consumed)
    return req


class TestRewindConsumed(unittest.TestCase):
    def test_rewind_to_zero_allows_full_reread(self):
        req = make_req(consumed=6)
        req.rewind_consumed(0)
        self.assertEqual(req.consumed, 0)
        # Arrival high-water mark is untouched: the whole prompt is
        # dispatchable again.
        self.assertEqual(req.recv_rows, 10)
        self.assertEqual(req.available(), 10)
        # Re-reading from row 0 is legal and hits the retained payload.
        self.assertEqual(req.read_range(4), (0, 4 * HIDDEN))

    def test_partial_rewind(self):
        req = make_req(consumed=6)
        req.rewind_consumed(3)
        self.assertEqual(req.consumed, 3)
        self.assertEqual(req.read_range(2), (3 * HIDDEN, 5 * HIDDEN))

    def test_rewind_to_current_pointer_is_noop(self):
        req = make_req(consumed=6)
        req.rewind_consumed(6)
        self.assertEqual(req.consumed, 6)

    def test_rewind_beyond_consumed_raises(self):
        req = make_req(consumed=6)
        with self.assertRaises(LwdHostStageGeometryError):
            req.rewind_consumed(7)
        with self.assertRaises(LwdHostStageGeometryError):
            req.rewind_consumed(-1)
        # Failed rewinds must not move the pointer.
        self.assertEqual(req.consumed, 6)

    def test_rewind_preserves_arrival_bookkeeping(self):
        req = make_req(consumed=6)
        written_hi = req._written_hi
        req.rewind_consumed(0)
        self.assertEqual(req.recv_rows, 10)
        self.assertEqual(req._written_hi, written_hi)

    def test_preemption_recompute_scenario(self):
        # Prompt fully consumed (decode phase), then preempted: the rewind
        # reopens the whole prompt and read_range's W2 bound still passes
        # because recv_rows kept its high-water mark.
        req = make_req(consumed=10)
        self.assertTrue(req.done())
        req.rewind_consumed(0)
        self.assertFalse(req.done())
        self.assertEqual(req.read_range(10), (0, 10 * HIDDEN))


class TestStagerRewind(unittest.TestCase):
    def make_stager(self, req) -> "LwdHostStager":
        stager = LwdHostStager(device="cpu", slots=1, empty_factory=lambda n: None)
        stager._reqs[req.req_id] = req
        return stager

    def test_rewind_moves_pointer_and_returns_true(self):
        req = make_req(consumed=6)
        stager = self.make_stager(req)
        self.assertTrue(stager.rewind_consumed("r0", 0))
        self.assertEqual(stager.consumed_rows("r0"), 0)

    def test_rewind_unknown_request_returns_false(self):
        stager = self.make_stager(make_req())
        self.assertFalse(stager.rewind_consumed("gone", 0))

    def test_rewind_out_of_range_propagates(self):
        req = make_req(consumed=6)
        stager = self.make_stager(req)
        with self.assertRaises(LwdHostStageGeometryError):
            stager.rewind_consumed("r0", 8)
        self.assertEqual(stager.consumed_rows("r0"), 6)


class TestSkipConsumed(unittest.TestCase):
    """读指针前跳(self-hit,设计 §5.3):重准入命中越过暂存进度。"""

    def test_skip_forward_moves_pointer(self):
        req = make_req(consumed=4)
        req.skip_consumed(7)
        self.assertEqual(req.consumed, 7)
        # 前跳后从新起点正常读
        self.assertEqual(req.read_range(2), (7 * HIDDEN, 9 * HIDDEN))
        # 到达水位不动,不变量 consumed <= recv_rows 保持
        self.assertEqual(req.recv_rows, 10)
        self.assertEqual(req.available(), 3)

    def test_skip_to_current_pointer_is_noop(self):
        req = make_req(consumed=4)
        req.skip_consumed(4)
        self.assertEqual(req.consumed, 4)

    def test_skip_to_recv_high_water(self):
        req = make_req(consumed=4)
        req.skip_consumed(10)
        self.assertEqual(req.consumed, 10)
        self.assertTrue(req.done())  # prompt 全长 = 10,读指针到尾

    def test_skip_below_consumed_raises(self):
        req = make_req(consumed=4)
        with self.assertRaises(LwdHostStageGeometryError):
            req.skip_consumed(3)  # 回退请走 rewind_consumed
        self.assertEqual(req.consumed, 4)

    def test_skip_beyond_recv_rows_raises(self):
        req = make_req(consumed=4)
        with self.assertRaises(LwdHostStageGeometryError):
            req.skip_consumed(11)  # recv_rows=10,越过会读到未写入行
        self.assertEqual(req.consumed, 4)

    def test_skip_beyond_prompt_rows_raises(self):
        # prompt 上界是防呆分支:recv_rows <= prompt_rows 恒成立,正常
        # 不可达;白盒抬高 recv_rows 以触达该校验。
        req = make_req(consumed=4)
        req.recv_rows = 12  # 仅测试构造,现实中不可能
        with self.assertRaises(LwdHostStageGeometryError):
            req.skip_consumed(11)  # prompt_rows=10
        self.assertEqual(req.consumed, 4)

    def test_skip_then_rewind_is_legal(self):
        # 前跳后再被抢占:回退目标只需 <= consumed,与 rewind 语义正交
        req = make_req(consumed=4)
        req.skip_consumed(8)
        req.rewind_consumed(4)
        self.assertEqual(req.consumed, 4)


class TestStagerSkip(unittest.TestCase):
    def make_stager(self, req) -> "LwdHostStager":
        stager = LwdHostStager(device="cpu", slots=1, empty_factory=lambda n: None)
        stager._reqs[req.req_id] = req
        return stager

    def test_skip_moves_pointer_and_returns_true(self):
        req = make_req(consumed=4)
        stager = self.make_stager(req)
        self.assertTrue(stager.skip_consumed("r0", 7))
        self.assertEqual(stager.consumed_rows("r0"), 7)

    def test_skip_unknown_request_returns_false(self):
        stager = self.make_stager(make_req())
        self.assertFalse(stager.skip_consumed("gone", 5))

    def test_skip_out_of_range_propagates(self):
        req = make_req(consumed=4)
        stager = self.make_stager(req)
        with self.assertRaises(LwdHostStageGeometryError):
            stager.skip_consumed("r0", 11)
        self.assertEqual(stager.consumed_rows("r0"), 4)


# ---------------------------------------------------------------------- #
# Prefix-cache seeding + refill tracking (design §3.2 / §5.3)             #
# ---------------------------------------------------------------------- #
def make_seeded_req(prompt_rows=10, floor=4) -> "LwdReqEmbed":
    req = LwdReqEmbed(
        req_id="r0",
        host=None,
        numel=prompt_rows * HIDDEN,
        prompt_rows=prompt_rows,
        hidden_size=HIDDEN,
    )
    req.seed_prefix(floor)
    return req


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture_logs():
    """版本无关的日志捕获(assertNoLogs 需要 py3.10+)。"""
    log = logging.getLogger("vllm.lwd.host_stage")
    handler = CaptureHandler()
    log.addHandler(handler)
    return log, handler


class TestSeedPrefix(unittest.TestCase):
    def tearDown(self):
        log = logging.getLogger("vllm.lwd.host_stage")
        for h in list(log.handlers):
            if isinstance(h, CaptureHandler):
                log.removeHandler(h)

    def test_seed_sets_all_watermarks(self):
        req = make_seeded_req(floor=4)
        self.assertEqual(req.recv_rows, 4)
        self.assertEqual(req.consumed, 4)
        self.assertEqual(req._written_hi, 4)
        self.assertEqual(req._stage_floor, 4)
        # buffer 行号回到 prompt 绝对坐标(命中段 [0,S) 永不写入)
        self.assertEqual(req._row_base, 0)
        # 首 chunk 到达前 recv_rows − num_computed = 0,调度不可见
        self.assertEqual(req.available(), 0)
        self.assertFalse(req.done())

    def test_unseeded_request_has_zero_floor(self):
        req = make_req()
        self.assertEqual(req._stage_floor, 0)
        self.assertIsNone(req._refill_hi)

    def test_first_chunk_at_floor_no_gap_warning(self):
        req = make_seeded_req(floor=4)
        log, handler = capture_logs()
        req.note_arrival(4, 3)
        log.removeHandler(handler)
        warnings = [
            r for r in handler.records if r.levelno >= logging.WARNING
        ]
        self.assertEqual(warnings, [])
        self.assertEqual(req.recv_rows, 7)
        self.assertEqual(req.available(), 3)

    def test_available_and_read_start_after_seed(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 3)
        self.assertEqual(req.available(), 3)
        # 读起点与首个调度窗口起点(num_computed = S)对齐
        self.assertEqual(req.read_range(2), (4 * HIDDEN, 6 * HIDDEN))
        req.note_consumed(2)
        self.assertEqual(req.read_range(1), (6 * HIDDEN, 7 * HIDDEN))

    def test_seed_rejects_invalid_args(self):
        req = LwdReqEmbed(
            req_id="r0", host=None, numel=10 * HIDDEN, prompt_rows=10,
            hidden_size=HIDDEN,
        )
        with self.assertRaises(LwdHostStageGeometryError):
            req.seed_prefix(0)
        with self.assertRaises(LwdHostStageGeometryError):
            req.seed_prefix(10)  # 命中不得吞掉整个 prompt(尾部 >= 1)
        req.seed_prefix(4)
        with self.assertRaises(LwdHostStageGeometryError):
            req.seed_prefix(4)  # 重复播种 = 记账已分叉
        req2 = LwdReqEmbed(
            req_id="r1", host=None, numel=10 * HIDDEN, prompt_rows=10,
            hidden_size=HIDDEN,
        )
        req2.note_arrival(0, 2)
        with self.assertRaises(LwdHostStageGeometryError):
            req2.seed_prefix(2)  # 已有到达后禁止播种

    def test_rewind_below_floor_is_legal(self):
        # refill 场景:重准入 rewind 目标 H' < S,合法,不做下界拦截
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_consumed(6)
        req.rewind_consumed(2)
        self.assertEqual(req.consumed, 2)
        self.assertEqual(req.recv_rows, 10)


class TestRefillTracking(unittest.TestCase):
    def test_refill_rows_do_not_move_arrival_watermarks(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_arrival(2, 2)  # refill [2,4)
        self.assertEqual(req.recv_rows, 10)
        self.assertEqual(req._written_hi, 10)
        self.assertEqual(req._refill_hi, 4)
        self.assertTrue(req._refill_completed)

    def test_refill_progresses_fifo_from_h_prime(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_arrival(1, 2)  # 首个 refill 行起点即 H'
        self.assertEqual(req._refill_hi, 3)
        self.assertFalse(req._refill_completed)
        req.note_arrival(3, 1)
        self.assertEqual(req._refill_hi, 4)
        self.assertTrue(req._refill_completed)

    def test_refill_replay_is_idempotent(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_arrival(2, 2)
        req.note_arrival(2, 2)  # 重放:不推进、不报错
        req.note_arrival(1, 1)  # 落在已补区间:幂等
        self.assertEqual(req._refill_hi, 4)
        self.assertEqual(req.recv_rows, 10)

    def test_refill_lo_records_hole_start(self):
        # 洞起点 _refill_lo = 首个 refill 行的 offset(= H'),供补齐时
        # 推进暂存地板(B1:二次 refill 依赖地板推进与状态复位)
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        self.assertIsNone(req._refill_lo)
        req.note_arrival(2, 1)
        self.assertEqual(req._refill_lo, 2)

    def test_read_below_floor_before_any_refill_fails(self):
        # M1:refill 一行未达(_refill_hi is None)时,consumed < S 的读
        # 同样命中未填的洞,必须拦
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_consumed(6)
        req.rewind_consumed(2)  # 重准入 rewind 到 H'=2,补发还在路上
        with self.assertRaises(LwdHostStageGeometryError):
            req.read_range(1)
        req.note_arrival(2, 1)  # 补到 3:[2,3) 可读
        self.assertEqual(req.read_range(1), (2 * HIDDEN, 3 * HIDDEN))
        with self.assertRaises(LwdHostStageGeometryError):
            req.read_range(2)  # [2,4) 越过 refill_hi=3,仍拦

    def test_refill_interleaved_with_normal_rows(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 4)
        req.note_arrival(1, 2)   # refill 部分到
        req.note_arrival(8, 2)   # 在飞正常行照常推进高水位
        self.assertEqual(req.recv_rows, 10)
        self.assertEqual(req._refill_hi, 3)
        self.assertFalse(req._refill_completed)
        req.note_arrival(3, 1)   # refill 补齐
        self.assertTrue(req._refill_completed)
        self.assertEqual(req.recv_rows, 10)

    def test_refill_overrun_beyond_floor_raises(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        with self.assertRaises(LwdHostStageGeometryError):
            req.note_arrival(3, 2)  # end=5 越过 S=4,会覆写在飞正常行

    def test_refill_gap_fails_before_completion(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_arrival(1, 1)
        with self.assertRaisesRegex(LwdHostStageGeometryError, "refill gap"):
            req.note_arrival(3, 1)  # [2,3) 未补,不得越过连续水位
        self.assertEqual(req._refill_hi, 2)
        self.assertFalse(req._refill_completed)
        self.assertEqual(req._stage_floor, 4)

    def test_read_into_unfilled_hole_fails_fast(self):
        req = make_seeded_req(floor=4)
        req.note_arrival(4, 6)
        req.note_consumed(6)
        req.rewind_consumed(2)   # 重准入 rewind 到 H'=2
        req.note_arrival(2, 1)   # refill 只补到 3,洞 [3,4) 未填
        # 已补区间可读
        self.assertEqual(req.read_range(1), (2 * HIDDEN, 3 * HIDDEN))
        # 越过 refill 低水位读洞:闸门失效,fail-fast
        with self.assertRaises(LwdHostStageGeometryError):
            req.read_range(2)
        req.note_arrival(3, 1)   # 补齐后读合法
        self.assertEqual(req.read_range(2), (2 * HIDDEN, 4 * HIDDEN))


class FakeBuf:
    """plan_recv_segments 的 duck-typed host buffer(只切视图、不落数)。"""

    def __init__(self, numel, start=0):
        self._numel = numel
        self.start = start

    def numel(self):
        return self._numel

    def __getitem__(self, s):
        return FakeBuf(s.stop - s.start, s.start)


class TestStagerSeedAndRefill(unittest.TestCase):
    def make_stager(self) -> "LwdHostStager":
        return LwdHostStager(
            device="cpu", slots=2, empty_factory=lambda n: FakeBuf(n)
        )

    def plan(self, stager, seqno, rid="r1", rows=6, offset=4, plen=10,
             refills=None):
        return stager.plan_recv_segments(
            seqno=seqno,
            numel=rows * HIDDEN,
            hidden_size=HIDDEN,
            req_ids=[rid],
            num_tokens_per_req=[rows],
            prompt_lens=[plen],
            row_base=[offset],
            refills=refills,
        )

    def test_first_hint_offset_seeds_floor(self):
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        self.assertEqual(stager.stage_floor("r1"), 4)
        self.assertEqual(stager.recv_rows("r1"), 4)
        self.assertEqual(stager.consumed_rows("r1"), 4)

    def test_first_hint_offset_zero_keeps_legacy_behavior(self):
        stager = self.make_stager()
        log, handler = capture_logs()
        self.plan(stager, seqno=1, rows=6, offset=0)
        self.assertEqual(stager.stage_floor("r1"), 0)
        self.assertEqual(stager.recv_rows("r1"), 0)
        out = stager.publish_received(1)
        log.removeHandler(handler)
        self.assertEqual(out, [("r1", 6, 0)])
        self.assertEqual(stager.recv_rows("r1"), 6)
        warnings = [
            r for r in handler.records if r.levelno >= logging.WARNING
        ]
        self.assertEqual(warnings, [])

    def test_publish_first_chunk_at_floor_no_gap_warning(self):
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        log, handler = capture_logs()
        out = stager.publish_received(1)
        log.removeHandler(handler)
        self.assertEqual(out, [("r1", 6, 4)])
        self.assertEqual(stager.recv_rows("r1"), 10)
        warnings = [
            r for r in handler.records if r.levelno >= logging.WARNING
        ]
        self.assertEqual(warnings, [])

    def test_refill_publish_reports_completion(self):
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        stager.publish_received(1)
        self.assertEqual(stager.take_refill_done(), [])
        segs = self.plan(stager, seqno=2, rows=2, offset=2, refills=[True])
        # refill 段的 D2H 落点即洞 [2,4) 对应的 buffer 行
        self.assertEqual((segs[0][0].start, segs[0][2]), (2 * HIDDEN, 2 * HIDDEN))
        stager.publish_received(2)
        self.assertEqual(stager.take_refill_done(), ["r1"])
        # 取走即清空,重复取幂等
        self.assertEqual(stager.take_refill_done(), [])
        # 高水位不被 refill 推动
        self.assertEqual(stager.recv_rows("r1"), 10)
        # B1:补齐后暂存地板推进到洞起点 H'=2,refill 状态复位
        self.assertEqual(stager.stage_floor("r1"), 2)
        req = stager.take_req("r1")
        self.assertIsNone(req._refill_hi)
        self.assertIsNone(req._refill_lo)

    def test_second_refill_after_floor_advance(self):
        # B1:二次抢占更深逐出(H''=0 < 新地板 2),refill 必须对新地板
        # 重新走完整 FIFO 链路 —— 旧实现 _refill_hi 停在旧 S 不复位,
        # 二次 refill 首个 chunk 即误判补齐、提前上报 refill_done。
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        stager.publish_received(1)
        self.plan(stager, seqno=2, rows=2, offset=2, refills=[True])
        stager.publish_received(2)
        self.assertEqual(stager.take_refill_done(), ["r1"])
        self.assertEqual(stager.stage_floor("r1"), 2)
        # 二次 refill:洞 [0,2),分两个 chunk 到达
        self.plan(stager, seqno=3, rows=1, offset=0, refills=[True])
        stager.publish_received(3)
        # 只补到 1,不得上报(旧实现在此处即误报补齐)
        self.assertEqual(stager.take_refill_done(), [])
        req = stager.take_req("r1")
        self.assertEqual((req._refill_lo, req._refill_hi), (0, 1))
        self.plan(stager, seqno=4, rows=1, offset=1, refills=[True])
        stager.publish_received(4)
        self.assertEqual(stager.take_refill_done(), ["r1"])
        # 地板再次推进到 H''=0:全覆盖,此后任何 refill 标志都是协议错误
        self.assertEqual(stager.stage_floor("r1"), 0)
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=5, rows=1, offset=0, refills=[True])

    def test_refill_overrun_rejected_at_plan_time(self):
        # M2:refill 越过暂存地板在 plan 期(D2H 挂出前)即 fail-fast,
        # 而不是等 publish 期数据已写进正常行区域才发现
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        stager.publish_received(1)
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=2, rows=3, offset=2, refills=[True])
        # plan 期拦截 ⇒ arrival plan 未登记,publish 无计划可发
        with self.assertRaises(LwdHostStageGeometryError):
            stager.publish_received(2)

    def test_refill_first_hint_rejected(self):
        stager = self.make_stager()
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=1, rows=2, offset=2, refills=[True])

    def test_refill_flag_geometry_mismatch_rejected(self):
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        # offset < S 而标志为 False
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=2, rows=2, offset=2, refills=[False])
        # offset >= S 而标志为 True
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=3, rows=2, offset=8, refills=[True])

    def test_refill_flag_default_and_short_table(self):
        stager = self.make_stager()
        self.plan(stager, seqno=1, rows=6, offset=4)
        # 缺省(None)= 全 False:正常续传不受影响
        self.plan(stager, seqno=2, rows=4, offset=4, refills=None)
        # 长表 = 协议错误
        with self.assertRaises(LwdHostStageGeometryError):
            self.plan(stager, seqno=3, rows=2, offset=8,
                      refills=[False, True])


if __name__ == "__main__":
    unittest.main()
