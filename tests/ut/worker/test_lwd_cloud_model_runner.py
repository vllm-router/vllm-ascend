# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdCloudModelRunner._lwd_rewind_staged_reads`` regression tests.

The runner rewinds a request's staged read pointer when the scheduler's
watermark regresses below it (cloud-side KV preemption zeroes
``num_computed_tokens``; the next schedule re-dispatches the prompt from
row 0 against the retained host payload), and skips it forward when a
re-admission prefix hit lands above it (self-hit, 设计 §5.3: the skipped
rows are covered by the prefix cache and never read; decode-phase
watermarks past the prompt are excluded by the prompt-region gate).

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
    """prompt_rows 缺省取 consumed 初值(prompt 已全部暂存,等价于
    decode 期形态);self-hit 用例显式给更大的 prompt_rows。"""

    def __init__(self, consumed, prompt_rows=None):
        self._consumed = dict(consumed)
        self._prompt_rows = prompt_rows or dict(consumed)
        self.rewinds = []
        self.skips = []

    def consumed_rows(self, req_id):
        return self._consumed.get(req_id, 0)

    def take_req(self, req_id):
        if req_id not in self._consumed:
            return None
        return NS(prompt_rows=self._prompt_rows[req_id])

    def rewind_consumed(self, req_id, to_rows):
        self.rewinds.append((req_id, to_rows))
        self._consumed[req_id] = to_rows
        return True

    def skip_consumed(self, req_id, to_rows):
        self.skips.append((req_id, to_rows))
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
        # 水位越过读指针但读指针已在 prompt 末尾(prompt_rows == 100,
        # decode 期形态):前跳分支被 prompt 区闸门排除,不动指针。
        stager = FakeStager({"A": 100})
        REWIND(runner_with(stager), scheduler_output({"A": 200}))
        self.assertEqual(stager.rewinds, [])
        self.assertEqual(stager.skips, [])

    def test_decode_phase_rollback_does_not_touch_prompt_pointer(self):
        # Spec-decode rejection rolls the watermark back inside the decode
        # range; the prompt read pointer (== prompt_rows) stays below it,
        # so no rewind may fire.
        stager = FakeStager({"A": 100})  # prompt_rows == 100, fully consumed
        REWIND(runner_with(stager), scheduler_output({"A": 103}))
        self.assertEqual(stager.rewinds, [])
        self.assertEqual(stager.skips, [])

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

    def test_self_hit_forward_skip(self):
        # 抢占重准入的重查命中越过本请求抢占前的暂存进度(H' > P,
        # 共享前缀的旁请求补长了缓存链):读指针前跳,跳过的行由前缀
        # 缓存覆盖、永不读取。
        stager = FakeStager({"A": 100}, prompt_rows={"A": 400})
        REWIND(runner_with(stager), scheduler_output({"A": 200}))
        self.assertEqual(stager.rewinds, [])
        self.assertEqual(stager.skips, [("A", 200)])
        self.assertEqual(stager.consumed_rows("A"), 200)

    def test_unknown_request_forward_is_ignored(self):
        # 前跳目标请求已不在暂存池(async 调度下调度-执行间终结):
        # 与 rewind 同款容错,不炸不跳。
        stager = FakeStager({})
        REWIND(runner_with(stager), scheduler_output({"A": 5}))
        self.assertEqual(stager.skips, [])


if __name__ == "__main__":
    unittest.main()
