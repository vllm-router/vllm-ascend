"""CPU-only tests for prefix-resumed MTP embeddings; run directly with Python."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]


def isolated_method(relative, class_name, method, base=object):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), cls], type_ignores=[]
    )
    scope = {"Base": base}
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), scope)
    return scope[class_name]


class Rows(list):
    def tolist(self):
        return list(self)


class TestPrefixMtpEmbeddings(unittest.TestCase):
    def proposer(self):
        native = isolated_method(
            "vllm_ascend/spec_decode/llm_base_proposer.py",
            "AscendSpecDecodeBaseProposer",
            "_lwd_build_first_pass_embeds",
        )
        cls = isolated_method(
            "vllm_ascend/worker/lwd_cloud/lwd_mtp_proposer.py",
            "LwdMTPProposer",
            "_lwd_build_first_pass_embeds",
            native,
        )
        proposer = cls()
        proposer.vllm_config = NS(cache_config=NS(enable_prefix_caching=True))
        proposer._lwd_accepts_inputs_embeds = True
        # 本分支基类比参考多一道 provider 判活门:测试置 dummy provider
        # 放行(覆写路径直读 inputs_embeds.gpu,不实际调用 provider)
        proposer._lwd_prompt_embeds_provider = staticmethod(lambda req_id: None)
        proposer.runner = NS(
            input_batch=NS(
                req_ids=["decode", "resumed"], num_computed_tokens_cpu=[100, 32768], num_prompt_tokens=[20, 32772]
            ),
            inputs_embeds=NS(gpu=Rows([100, 101, 200, 201, 202, 203])),
        )
        proposer.model = NS(embed_input_ids=lambda ids: [i * 10 for i in ids])
        proposer.input_ids = Rows([11, 12, 0, 0, 0, 13])
        proposer.inputs_embeds = Rows([0] * 6)
        return proposer, NS(query_start_loc_cpu=Rows([0, 2, 6]))

    def test_resumed_prompt_uses_current_gpu_segment_not_absolute_offset(self):
        proposer, cad = self.proposer()
        result = proposer._lwd_build_first_pass_embeds(6, 6, None, None, cad)
        # Draft shift reads only the 4 received suffix rows. Decode and the
        # sampled next-token embedding retain the ordinary token-ID path.
        self.assertEqual(result, [110, 120, 201, 202, 203, 130])

    def test_missing_embedding_support_cannot_use_placeholder_token_ids(self):
        proposer, cad = self.proposer()
        proposer._lwd_accepts_inputs_embeds = False
        with self.assertRaisesRegex(RuntimeError, "authoritative prompt embeddings"):
            proposer._lwd_build_first_pass_embeds(6, 6, None, None, cad)

    def test_decode_only_allows_normal_token_path(self):
        proposer, cad = self.proposer()
        proposer.runner.input_batch.num_computed_tokens_cpu = [100, 32772]
        self.assertIsNone(proposer._lwd_build_first_pass_embeds(6, 6, None, None, cad))

    def test_pcp_layout_is_rejected(self):
        proposer, cad = self.proposer()
        proposer.runner.pcp_manager = object()
        with self.assertRaisesRegex(RuntimeError, "PCP"):
            proposer._lwd_build_first_pass_embeds(6, 6, None, None, cad)


if __name__ == "__main__":
    unittest.main()
