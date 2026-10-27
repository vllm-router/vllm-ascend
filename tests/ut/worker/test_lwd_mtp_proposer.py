# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``LwdMTPProposer._lwd_build_first_pass_embeds`` 的 PC 守卫测试(设计 §5.1)。

前缀缓存开启时检测续算请求(``computed + 1 < prompt``):
  * 与 PCP 并存 → RuntimeError(all-gather 布局使右移对齐假设不成立);
  * 基类返回 None(会回退到占位 token-id 查表)→ RuntimeError fail-fast。

torch-free:被测方法从 ``lwd_mtp_proposer.py`` 源码按 AST 抽出、包进
一个以 FakeBase 为基类的合成类后执行(方法模块本体依赖
torch/torch_npu,永不导入;与 tests/ut/worker/test_lwd_cloud_model_runner.py
同款手法)。
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
PROPOSER = ROOT / "vllm_ascend/worker/lwd_cloud/lwd_mtp_proposer.py"


class FakeBase:
    """基类桩:返回值逐用例可控(None = 基类拒绝,调用方回退占位查表)。"""

    def _lwd_build_first_pass_embeds(self, *args):
        return self._fake_result


def load_guard():
    tree = ast.parse(PROPOSER.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdMTPProposer"
    )
    method = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_lwd_build_first_pass_embeds"
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="GuardedProposer",
                bases=[ast.Name(id="FakeBase", ctx=ast.Load())],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ns = {"FakeBase": FakeBase, "logger": logging.getLogger("test.lwd_mtp")}
    exec(compile(ast.fix_missing_locations(module), str(PROPOSER), "exec"), ns)
    return ns["GuardedProposer"]


GUARDED = load_guard()


def make_proposer(pc_enabled=True, computed=(3, 10), prompts=(10, 10),
                  pcp_manager=None, base_result=None):
    proposer = GUARDED()
    proposer.vllm_config = NS(
        cache_config=NS(enable_prefix_caching=pc_enabled)
    )
    proposer.runner = NS(
        input_batch=NS(
            req_ids=["a", "b"],
            num_computed_tokens_cpu=list(computed),
            num_prompt_tokens=list(prompts),
        ),
        pcp_manager=pcp_manager,
    )
    proposer._fake_result = base_result
    return proposer


ARGS = (0, 0, None, None, None)  # 守卫不读实参,基类桩也不读


class TestPrefixMTPGuard(unittest.TestCase):
    def test_continued_request_with_pcp_fails(self):
        # computed=3 + 1 < prompt=10 → 续算请求;与 PCP 并存 → fail-fast
        proposer = make_proposer(pcp_manager=object())
        with self.assertRaises(RuntimeError) as ctx:
            proposer._lwd_build_first_pass_embeds(*ARGS)
        self.assertIn("PCP", str(ctx.exception))

    def test_continued_request_base_none_fails(self):
        # 续算请求存在而基类返回 None(会回退占位 token-id 查表)
        proposer = make_proposer(base_result=None)
        with self.assertRaises(RuntimeError) as ctx:
            proposer._lwd_build_first_pass_embeds(*ARGS)
        self.assertIn("authoritative prompt embeddings", str(ctx.exception))

    def test_continued_request_base_result_passes(self):
        sentinel = object()
        proposer = make_proposer(base_result=sentinel)
        self.assertIs(proposer._lwd_build_first_pass_embeds(*ARGS), sentinel)

    def test_pure_decode_batch_passes_none(self):
        # computed+1 >= prompt:无续算请求,基类 None 是合法降级,不拦
        proposer = make_proposer(computed=(9, 10), base_result=None)
        self.assertIsNone(proposer._lwd_build_first_pass_embeds(*ARGS))

    def test_pc_disabled_passes_none(self):
        # PC 关闭:守卫整体不激活(即使 PCP 在场、基类 None)
        proposer = make_proposer(
            pc_enabled=False, pcp_manager=object(), base_result=None
        )
        self.assertIsNone(proposer._lwd_build_first_pass_embeds(*ARGS))

    def test_missing_computed_offsets_fails(self):
        proposer = make_proposer()
        proposer.runner.input_batch.num_computed_tokens_cpu = [3]
        with self.assertRaises(RuntimeError) as ctx:
            proposer._lwd_build_first_pass_embeds(*ARGS)
        self.assertIn("computed-token offsets", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
