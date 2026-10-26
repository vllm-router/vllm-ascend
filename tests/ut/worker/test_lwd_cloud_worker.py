# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdCloudWorker._lwd_commit_received`` 的 refill_done 上报测试(设计 §5.3)。

refill 补齐时向 done MQ 上报 ``("refill_done", request_id)``(引擎侧
``_lwd_drain_irecv_completions`` 解析后调 ``scheduler.lwd_refill_done``
摘闸门);逐请求**累计**已收行数上报语义不回归。

torch-free:被测方法从 ``lwd_cloud_worker.py`` 源码按 AST 抽出执行(worker
模块本体依赖 torch/torch_npu,永不导入;与
tests/ut/worker/test_lwd_cloud_model_runner.py 同款手法)。
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py"


def load_commit():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdCloudWorker"
    )
    method = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_lwd_commit_received"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ns = {
        "logger": logging.getLogger("test.lwd_commit"),
        "hash_layer_count": lambda hf: 0,
        "LwdCommFuture": object,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), ns)
    return ns["_lwd_commit_received"]


COMMIT = load_commit()


class FakeStager:
    def __init__(self, plan_rows, refill_done=(), recv=10):
        self._plan_rows = plan_rows
        self._refill_done = list(refill_done)
        self._recv = recv
        self.host_bytes = 0
        self.published = []

    def publish_received(self, seqno):
        self.published.append(seqno)
        return self._plan_rows

    def recv_rows(self, req_id):
        return self._recv

    def take_refill_done(self):
        out, self._refill_done = self._refill_done, []
        return out


class FakeMQ:
    def __init__(self):
        self.queued = []

    def enqueue(self, item):
        self.queued.append(item)


def make_worker(stager, mq):
    return NS(
        _lwd_stager=stager,
        _irecv_done_mq=mq,
        _lwd_last_total={},
        _lwd_req_meta={},
        model_config=NS(hf_config=NS(num_experts_per_tok=0)),
    )


def make_future(seqno=7):
    return NS(request=NS(seqno=seqno), release_device_tensor=lambda: None)


class TestCommitReceivedRefillDone(unittest.TestCase):
    def test_refill_completion_reported_on_done_mq(self):
        stager = FakeStager([("r1", 6, 4)], refill_done=["r2"])
        mq = FakeMQ()
        arrived = COMMIT(make_worker(stager, mq), make_future())
        # 累计行数上报语义不变(recv_rows=10 > 已报 0 → 报累计 10)
        self.assertEqual(arrived, [("r1", 10)])
        self.assertEqual(mq.queued, [("refill_done", "r2")])
        self.assertEqual(stager.published, [7])
        # 名单取走即清空
        self.assertEqual(stager.take_refill_done(), [])

    def test_no_refill_no_report(self):
        stager = FakeStager([("r1", 6, 4)])
        mq = FakeMQ()
        COMMIT(make_worker(stager, mq), make_future())
        self.assertEqual(mq.queued, [])

    def test_no_stager_returns_empty(self):
        mq = FakeMQ()
        worker = make_worker(None, mq)
        self.assertEqual(COMMIT(worker, make_future()), [])
        self.assertEqual(mq.queued, [])

    def test_stalled_rows_still_report_refill(self):
        # 已收行数未前进(refill 行不推高水位)→ arrived 为空,
        # 但 refill 补齐仍须上报
        stager = FakeStager([("r1", 2, 2)], refill_done=["r1"], recv=10)
        mq = FakeMQ()
        worker = make_worker(stager, mq)
        worker._lwd_last_total = {"r1": 10}
        arrived = COMMIT(worker, make_future())
        self.assertEqual(arrived, [])
        self.assertEqual(mq.queued, [("refill_done", "r1")])


if __name__ == "__main__":
    unittest.main()
