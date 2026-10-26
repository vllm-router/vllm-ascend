# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdEdgeWorker._execute_lwd_embed`` 的 refill 批 mm 特征摘除测试(§5.3)。

refill 批(带外补发)的 ``scheduled_new_reqs`` 会重新登记带 data 的
mm_features;该请求的 finished_req_ids 早已随嵌入完结送达,兜底释放
不再触发,不摘即常驻泄漏。修复:refill 批(``LwdEmbedBatch.is_refill``,
跨仓契约)执行后只摘本批重登记的请求。

torch-free:被测方法从 ``lwd_edge_worker.py`` 源码按 AST 抽出、包进合成类
执行(模块本体依赖 torch/torch_npu,永不导入;与
tests/ut/worker/test_lwd_cloud_model_runner.py 同款手法)。
"""

import ast
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "vllm_ascend/worker/lwd_edge_worker.py"


class _FakeTorch:
    """@torch.inference_mode() 装饰器的恒等桩。"""

    @staticmethod
    def inference_mode():
        return lambda fn: fn


def load_worker_class():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdEdgeWorker"
    )
    methods = [
        n for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in (
            "_execute_lwd_embed", "_lwd_mm_release_request",
            "_lwd_up_send_done_watermark", "_lwd_up_send_done_watermarks",
        )
    ]
    assert len(methods) == 4
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="EdgeEmbedWorker",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ns = {
        "torch": _FakeTorch,
        "time": time,
        "logger": logging.getLogger("test.lwd_edge_embed"),
        "LwdCommRequest": lambda **kw: NS(**kw),
        "LwdChannelType": NS(UP="lwd_up"),
        "LwdEmbedBatch": object,
        "SchedulerOutput": object,
        "ModelRunnerOutput": lambda **kw: NS(**kw),
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), ns)
    return ns["EdgeEmbedWorker"]


WORKER_CLS = load_worker_class()


class FakeTensor:
    def numel(self):
        return 8


def make_worker(gather_result=([], None)):
    worker = WORKER_CLS()
    worker.model_runner = NS(
        get_model=lambda: NS(embed_input_ids=lambda *a, **k: FakeTensor())
    )
    worker.tid2eid_tables = []
    worker._lwd_mm_features_dict = {}
    worker.comm_service = NS(submit_send=lambda req: NS(done=lambda: False))
    worker._lwd_embed_ids_h2d = lambda ids: None
    worker._lwd_gather_chunk_mm = lambda meta: gather_result
    worker._lwd_up_send_done_watermark = lambda engine: -1
    return worker


def make_inputs(is_refill, req_id="r1", with_feature=True):
    feats = [NS(modality="image", data=object())] if with_feature else []
    scheduler_output = NS(
        scheduled_new_reqs=[NS(req_id=req_id, mm_features=feats)]
    )
    batch_meta = NS(token_ids=[[10, 11]], req_ids=[req_id])
    if is_refill is not None:
        batch_meta.is_refill = is_refill
    return batch_meta, scheduler_output


class TestRefillMmFeatureRelease(unittest.TestCase):
    def test_refill_batch_drops_reregistered_features(self):
        worker = make_worker()
        batch_meta, so = make_inputs(is_refill=True)
        worker._execute_lwd_embed(7, batch_meta, so)
        # refill 批执行后,本批重登记的特征被摘除,不留常驻条目
        self.assertEqual(worker._lwd_mm_features_dict, {})

    def test_normal_batch_keeps_features(self):
        # 正常批:特征留给后续 chunk 复用(越过全部 mm 窗口时由
        # _lwd_gather_chunk_mm 摘除,或 finished_req_ids 兜底)
        worker = make_worker()
        batch_meta, so = make_inputs(is_refill=False)
        worker._execute_lwd_embed(7, batch_meta, so)
        self.assertIn("r1", worker._lwd_mm_features_dict)

    def test_missing_is_refill_attr_keeps_legacy_behavior(self):
        # 旧版 vllm 的 LwdEmbedBatch 没有 is_refill:getattr 缺省 False,
        # 退化为既有行为(不摘),不炸
        worker = make_worker()
        batch_meta, so = make_inputs(is_refill=None)
        worker._execute_lwd_embed(7, batch_meta, so)
        self.assertIn("r1", worker._lwd_mm_features_dict)

    def test_refill_batch_without_mm_registers_nothing(self):
        # 纯文本 refill 批:无特征可登记,摘除路径空转不炸
        worker = make_worker()
        batch_meta, so = make_inputs(is_refill=True, with_feature=False)
        worker._execute_lwd_embed(7, batch_meta, so)
        self.assertEqual(worker._lwd_mm_features_dict, {})

    def test_refill_drop_does_not_touch_other_requests(self):
        # 只摘本批重登记的:同 worker 上其他请求的特征不受影响
        worker = make_worker()
        worker._lwd_mm_features_dict["other"] = [NS(modality="image")]
        batch_meta, so = make_inputs(is_refill=True)
        worker._execute_lwd_embed(7, batch_meta, so)
        self.assertIn("other", worker._lwd_mm_features_dict)
        self.assertNotIn("r1", worker._lwd_mm_features_dict)

    def test_refill_restores_features_for_live_normal_chunks(self):
        worker = make_worker()
        original = [NS(modality="image", data=object())]
        worker._lwd_mm_features_dict["r1"] = original
        batch_meta, so = make_inputs(is_refill=True)
        worker._execute_lwd_embed(7, batch_meta, so)
        self.assertIs(worker._lwd_mm_features_dict["r1"], original)
        normal_meta, normal_so = make_inputs(is_refill=False)
        normal_so.scheduled_new_reqs = []
        worker._execute_lwd_embed(8, normal_meta, normal_so)
        self.assertIs(worker._lwd_mm_features_dict["r1"], original)


class _Future:
    def __init__(self, done):
        self._done = done

    def done(self):
        return self._done


class TestPerEngineUpWatermark(unittest.TestCase):
    def test_same_seqno_on_two_engines_is_isolated(self):
        worker = WORKER_CLS()
        worker._lwd_up_send_futures = {
            0: [(0, _Future(False))],
            1: [(0, _Future(True))],
        }
        self.assertEqual(worker._lwd_up_send_done_watermark(1), 0)
        self.assertEqual(worker._lwd_up_send_done_watermark(0), -1)
        self.assertEqual(len(worker._lwd_up_send_futures[0]), 1)
        self.assertEqual(worker._lwd_up_send_futures[1], [])

    def test_watermarks_returns_each_engine_independently(self):
        worker = WORKER_CLS()
        worker._lwd_up_send_futures = {
            0: [(3, _Future(True)), (4, _Future(False))],
            1: [(0, _Future(True)), (1, _Future(True))],
        }
        self.assertEqual(worker._lwd_up_send_done_watermarks(), {0: 3, 1: 1})
        self.assertEqual(worker._lwd_up_send_futures[0][0][0], 4)
        self.assertEqual(worker._lwd_up_send_futures[1], [])

    def test_single_dp_keeps_engine_zero_semantics(self):
        worker = WORKER_CLS()
        worker._lwd_up_send_futures = {
            0: [(7, _Future(True)), (8, _Future(True))],
        }
        self.assertEqual(worker._lwd_up_send_done_watermark(0), 8)
        self.assertEqual(worker._lwd_up_send_done_watermarks(), {0: -1})


if __name__ == "__main__":
    unittest.main()
