# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdReqEmbed.rewind_consumed`` / ``LwdHostStager.rewind_consumed`` tests.

Covers the preemption-recompute read-pointer rewind: cloud-side KV
preemption zeroes ``num_computed_tokens``, and the retained pinned payload
makes a local re-prefill legal once the read pointer is rolled back.

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


if __name__ == "__main__":
    unittest.main()
