# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""云侧唯一排名与实际边侧恢复函数的 CPU 回归测试。"""

import ast
import importlib.util
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
SPEC = importlib.util.spec_from_file_location("lwd_token_rank", ROOT / "worker/lwd_token_rank.py")
rank_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rank_module)


def load_edge_functions():
    path = ROOT / "worker/lwd_edge_worker.py"
    tree = ast.parse(path.read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("select_token_batch", "compute_top_id_th")
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TestLwdTokenRank(unittest.TestCase):
    def setUp(self):
        self.encoder = rank_module.LwdTokenRankEncoder()
        self.edge = load_edge_functions()

    def assert_round_trip(self, logits, selected):
        original_logits, original_selected = logits.clone(), selected.clone()
        ranks = self.encoder.encode(logits, selected)
        self.assertEqual(self.edge["select_token_batch"](logits, ranks.tolist()), selected.tolist())
        self.assertTrue(torch.equal(logits, original_logits))
        self.assertTrue(torch.equal(selected, original_selected))
        return ranks

    def test_every_candidate_in_tied_and_non_max_groups(self):
        # 穷举所有候选，覆盖最高分并列、非最高分并列和最低分并列。
        values = torch.tensor([12.0, 10.0, 10.0, 10.0, -2.0, -2.0], dtype=torch.bfloat16)
        selected = torch.arange(len(values))
        ranks = self.assert_round_trip(values.repeat(len(values), 1), selected)
        self.assertEqual(ranks.tolist(), list(range(len(values))))
        for token in selected.tolist():
            self.assertEqual(self.edge["compute_top_id_th"](values, token), int(ranks[token]))

    def test_round_trip_many_quantized_ties(self):
        generator = torch.Generator().manual_seed(7)
        logits = torch.randint(-4, 5, (31, 257), generator=generator).to(torch.bfloat16)
        selected = torch.randint(0, 257, (31,), generator=generator)
        self.assert_round_trip(logits, selected)

    def test_production_vocab_and_reported_tied_ids(self):
        logits = torch.zeros(1, 129280, dtype=torch.bfloat16)
        logits[0, 9213] = logits[0, 9828] = 42.75
        ranks = self.assert_round_trip(logits, torch.tensor([9828]))
        self.assertEqual(ranks.tolist(), [1])

    def test_masked_logits_and_all_equal_scores(self):
        logits = torch.tensor([[float("-inf"), 1.0, 1.0, float("-inf")], [0.0, 0.0, 0.0, 0.0]])
        ranks = self.assert_round_trip(logits, torch.tensor([3, 2]))
        self.assertEqual(ranks.tolist(), [3, 2])

    def test_rejected_slots_are_safe_and_vocab_index_is_reused(self):
        logits = torch.tensor([[2.0, 2.0, 1.0]]).repeat(3, 1)
        self.assertEqual(self.encoder.encode(logits, torch.tensor([1, -1, 2])).tolist(), [1, 0, 2])
        cached = self.encoder.vocab_ids
        self.encoder.encode(logits[:1], torch.tensor([0]))
        self.assertIs(self.encoder.vocab_ids, cached)
        self.assertEqual(self.encoder.encode(torch.ones(1, 5), torch.tensor([4])).tolist(), [4])
        self.assertEqual(self.encoder.vocab_ids.numel(), 5)

    def test_collector_filters_discarded_requests_and_preserves_payload(self):
        path = ROOT / "worker/lwd_cloud/lwd_cloud_model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_lwd_collect_down_payload")
        namespace = dict(torch=torch, np=np, time=time, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        runner = NS(
            _lwd_token_rank_encoder=self.encoder,
            worker=NS(),
            _lwd_meta_pinned=lambda n: torch.empty(n, dtype=torch.int32),
        )
        logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(3, 1)
        hidden = torch.arange(6).reshape(3, 2).float()
        sampler = NS(sampled_token_ids=torch.tensor([[0], [2], [1]]))
        with patch.object(torch, "npu", NS(Event=Mock), create=True):
            result = namespace[method.name](
                runner, hidden, logits, None, NS(), ["a", "discard", "b"], np.array([False, True, False]), sampler
            )
        self.assertEqual(result[3], ["a", "b"])
        self.assertTrue(torch.equal(result[0], hidden[[0, 2]]))
        self.assertEqual(result[1].tolist(), [0, 1, 1, 1, 1, 1])

    def test_speculative_collector_keeps_segments_and_skips_rejected_tail(self):
        path = ROOT / "worker/lwd_cloud/lwd_cloud_model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_lwd_collect_down_payload")
        namespace = dict(torch=torch, np=np, time=time, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        runner = NS(
            _lwd_token_rank_encoder=self.encoder,
            worker=NS(),
            _lwd_h2d_stage_i64=lambda arr: torch.from_numpy(arr),
            _lwd_meta_pinned=lambda n: torch.empty(n, dtype=torch.int32),
        )
        # a: 接受一枚草稿后输出替代 token，最后一行 -1；discard 整段不发送。
        # b: 两枚草稿全部接受，再输出一枚 bonus token。
        logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(7, 1)
        hidden = torch.arange(14).reshape(7, 2).float()
        sampled = torch.tensor([[1, 2, -1], [0, -1, -1], [1, 0, 2]])
        with patch.object(torch, "npu", NS(Event=Mock), create=True):
            result = namespace[method.name](
                runner,
                hidden,
                logits,
                NS(num_draft_tokens=[2, 0, 2]),
                NS(),
                ["a", "discard", "b"],
                np.array([False, True, False]),
                NS(sampled_token_ids=sampled),
            )
        self.assertEqual(result[3], ["a", "b"])
        self.assertTrue(torch.equal(result[0], hidden[[0, 1, 2, 4, 5, 6]]))
        metadata = result[1].tolist()
        self.assertEqual(metadata, [1, 2, 0, 1, 0, 2, 2, 3, 3, 3])
        recovered = self.edge["select_token_batch"](logits[[0, 1, 2, 4, 5, 6]], metadata[:6])
        self.assertEqual(recovered[:2], [1, 2])
        self.assertEqual(recovered[3:6], [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
