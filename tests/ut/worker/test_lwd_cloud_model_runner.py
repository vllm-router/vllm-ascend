# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdCloudModelRunner._lwd_rewind_staged_reads`` regression tests.

The runner rewinds a request's staged read pointer when the scheduler's
watermark regresses below it (cloud-side KV preemption zeroes
``num_computed_tokens``; the next schedule re-dispatches the prompt from
row 0 against the retained host payload).

torch-free: the method is extracted from the runner source by AST and
executed with a stub ``logger`` - the runner module itself needs
torch/torch_npu and is never imported.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py"


def load_rewind():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_lwd_rewind_staged_reads"
    )
    ns = {"logger": logging.getLogger("test.lwd_rewind")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(RUNNER), "exec"), ns)
    return ns["_lwd_rewind_staged_reads"]


REWIND = load_rewind()


class FakeStager:
    def __init__(self, consumed):
        self._consumed = dict(consumed)
        self.rewinds = []

    def consumed_rows(self, req_id):
        return self._consumed.get(req_id, 0)

    def rewind_consumed(self, req_id, to_rows):
        self.rewinds.append((req_id, to_rows))
        self._consumed[req_id] = to_rows
        return True


def scheduler_output(computed):
    return NS(
        scheduled_cached_reqs=NS(
            req_ids=list(computed),
            num_computed_tokens=list(computed.values()),
        )
    )


def runner_with(stager):
    return NS(worker=NS(_lwd_stager=stager) if stager is not None else NS())


class TestRewindStagedReads(unittest.TestCase):
    def test_preemption_rewind_to_zero(self):
        stager = FakeStager({"A": 100})
        REWIND(runner_with(stager), scheduler_output({"A": 0}))
        self.assertEqual(stager.rewinds, [("A", 0)])

    def test_no_regression_no_rewind(self):
        stager = FakeStager({"A": 100})
        REWIND(runner_with(stager), scheduler_output({"A": 100}))
        self.assertEqual(stager.rewinds, [])

    def test_forward_progress_no_rewind(self):
        stager = FakeStager({"A": 100})
        REWIND(runner_with(stager), scheduler_output({"A": 200}))
        self.assertEqual(stager.rewinds, [])

    def test_decode_phase_rollback_does_not_touch_prompt_pointer(self):
        # Spec-decode rejection rolls the watermark back inside the decode
        # range; the prompt read pointer (== prompt_rows) stays below it,
        # so no rewind may fire.
        stager = FakeStager({"A": 100})  # prompt_rows == 100, fully consumed
        REWIND(runner_with(stager), scheduler_output({"A": 103}))
        self.assertEqual(stager.rewinds, [])

    def test_only_regressed_requests_rewind(self):
        stager = FakeStager({"A": 100, "B": 50})
        REWIND(runner_with(stager), scheduler_output({"A": 0, "B": 60}))
        self.assertEqual(stager.rewinds, [("A", 0)])

    def test_unknown_request_is_ignored(self):
        stager = FakeStager({})
        REWIND(runner_with(stager), scheduler_output({"A": 0}))
        self.assertEqual(stager.rewinds, [])

    def test_no_stager_is_noop(self):
        # Non-endpoint ranks have no host pool.
        REWIND(runner_with(None), scheduler_output({"A": 0}))
        REWIND(NS(worker=None), scheduler_output({"A": 0}))


if __name__ == "__main__":
    unittest.main()
