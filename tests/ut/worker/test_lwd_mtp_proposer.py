# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Prefix-caching guards for LWD MTP first-pass embeddings."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PROPOSER = ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_mtp_proposer.py"


class _FakeBase:
    def _lwd_build_first_pass_embeds(self, *args: object) -> object:
        return self._fake_result


def _load_guard() -> type:
    tree = ast.parse(PROPOSER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LwdMTPProposer")
    cls.bases = [ast.Name(id="_FakeBase", ctx=ast.Load())]
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_lwd_build_first_pass_embeds"
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    namespace = {"_FakeBase": _FakeBase}
    exec(compile(module, str(PROPOSER), "exec"), namespace)
    return namespace[cls.name]


_GUARDED = _load_guard()
_ARGS = (0, 0, None, None, None)


def _make_proposer(
    prefix_enabled: bool = True,
    computed: tuple[int, ...] = (3, 10),
    prompts: tuple[int, ...] = (10, 10),
    pcp_manager: object | None = None,
    base_result: object | None = None,
) -> object:
    proposer = _GUARDED()
    proposer.vllm_config = NS(cache_config=NS(enable_prefix_caching=prefix_enabled))
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


def test_continued_request_with_pcp_fails() -> None:
    proposer = _make_proposer(pcp_manager=object())
    with pytest.raises(RuntimeError, match="PCP"):
        proposer._lwd_build_first_pass_embeds(*_ARGS)


def test_continued_request_base_none_fails() -> None:
    proposer = _make_proposer(base_result=None)
    with pytest.raises(RuntimeError, match="authoritative prompt embeddings"):
        proposer._lwd_build_first_pass_embeds(*_ARGS)


def test_continued_request_tensor_passes() -> None:
    embeds = torch.arange(6).reshape(3, 2)
    proposer = _make_proposer(computed=(9, 3), base_result=embeds)
    assert proposer._lwd_build_first_pass_embeds(*_ARGS) is embeds


def test_pure_decode_batch_passes_none() -> None:
    proposer = _make_proposer(computed=(9, 10), base_result=None)
    assert proposer._lwd_build_first_pass_embeds(*_ARGS) is None


def test_prefix_disabled_passes_none() -> None:
    proposer = _make_proposer(
        prefix_enabled=False,
        pcp_manager=object(),
        base_result=None,
    )
    assert proposer._lwd_build_first_pass_embeds(*_ARGS) is None


def test_missing_computed_offsets_fails() -> None:
    proposer = _make_proposer(computed=(3,))
    with pytest.raises(RuntimeError, match="computed-token offsets"):
        proposer._lwd_build_first_pass_embeds(*_ARGS)
