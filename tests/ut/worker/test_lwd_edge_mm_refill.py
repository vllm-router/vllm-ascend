# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MM chunk-window filtering and refill positional-feature restoration."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"
CACHE_PATH = ROOT / "worker/lwd_edge_model/lwd_edge_mm_cache.py"
RUNNER_PATH = ROOT / "worker/lwd_edge_model/lwd_edge_model_runner.py"


def _load_class(path: Path, class_name: str, methods: set[str] | None = None) -> type:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    if methods is not None:
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *constants, cls], type_ignores=[]))
    namespace = {
        "torch": torch,
        "time": time,
        "lwd_logger": NS(info=lambda *args: None),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


_CACHE_CLS = _load_class(CACHE_PATH, "LwdEdgeMmCache")
_RUNNER_CLS = _load_class(RUNNER_PATH, "LwdEdgeModelRunner", {"_lwd_gather_mm"})


def _feature(identifier: str, offset: int, length: int, data: object = None) -> object:
    return NS(
        identifier=identifier,
        modality="image",
        data=data,
        mm_position=NS(offset=offset, length=length, is_embed=None),
    )


def _runner(cache: object) -> object:
    runner = _RUNNER_CLS()
    runner.mm_cache = cache
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    return runner


def _batch(offset: int, is_refill: bool, mrope_positions: object = None) -> object:
    return NS(
        req_ids=["r"],
        token_ids=[[10, 11]],
        prompt_offsets=[offset],
        is_refill=is_refill,
        mrope_positions=mrope_positions,
    )


def _new_req(feature: object) -> object:
    return NS(req_id="r", mm_features=[feature])


def test_refill_restores_live_features_and_preserves_normal_chunks() -> None:
    cache = _CACHE_CLS()
    original = _feature("shared", offset=3, length=4)
    cache.register("r", [original])
    encoded = torch.arange(8).reshape(4, 2)
    cache._hash_embeds["shared"] = encoded
    runner = _runner(cache)
    mrope_positions = [b"unchanged"]
    batch = _batch(1, True, mrope_positions)

    mm_embeds, is_mm = runner._lwd_gather_mm(
        batch,
        NS(scheduled_new_reqs=[_new_req(_feature("shared", 1, 2))]),
        NS(),
    )

    assert torch.equal(mm_embeds[0], encoded[:2])
    assert is_mm.tolist() == [True, True]
    assert cache._features["r"][0] is original
    assert cache._hash_embeds["shared"] is encoded
    assert batch.mrope_positions is mrope_positions

    normal_embeds, normal_mask = runner._lwd_gather_mm(_batch(3, False), NS(scheduled_new_reqs=[]), NS())
    assert torch.equal(normal_embeds[0], encoded[:2])
    assert normal_mask.tolist() == [True, True]
    assert cache._features["r"][0] is original


def test_refill_removes_temporary_features_without_clearing_hash_cache() -> None:
    cache = _CACHE_CLS()
    encoded = torch.arange(4).reshape(2, 2)
    cache._hash_embeds["shared"] = encoded
    runner = _runner(cache)

    runner._lwd_gather_mm(
        _batch(1, True),
        NS(scheduled_new_reqs=[_new_req(_feature("shared", 1, 2))]),
        NS(),
    )

    assert "r" not in cache._features
    assert cache._hash_embeds["shared"] is encoded


def test_refill_restores_features_when_encoding_fails() -> None:
    cache = _CACHE_CLS()
    original = _feature("shared", 3, 4)
    cache.register("r", [original])
    encoded = torch.arange(8).reshape(4, 2)
    cache._hash_embeds["shared"] = encoded
    runner = _runner(cache)

    with pytest.raises(RuntimeError, match="mm feature data missing"):
        runner._lwd_gather_mm(
            _batch(1, True),
            NS(scheduled_new_reqs=[_new_req(_feature("missing", 1, 2))]),
            NS(),
        )

    assert cache._features["r"][0] is original
    assert cache._hash_embeds == {"shared": encoded}


def test_normal_chunk_keeps_new_registration() -> None:
    cache = _CACHE_CLS()
    cache._hash_embeds["shared"] = torch.arange(8).reshape(4, 2)
    feature = _feature("shared", 1, 4)
    runner = _runner(cache)

    runner._lwd_gather_mm(_batch(0, False), NS(scheduled_new_reqs=[_new_req(feature)]), NS())

    assert cache._features["r"] == [feature]


def test_prefix_covered_feature_needs_no_data_or_hash() -> None:
    cache = _CACHE_CLS()
    cache.register("r", [_feature("covered", 0, 2)])
    mrope_positions = [b"unchanged"]
    batch = _batch(2, False, mrope_positions)

    mm_embeds, is_mm = cache.gather_chunk(batch, NS(), torch.device("cpu"), False)

    assert mm_embeds == []
    assert is_mm.tolist() == [False, False]
    assert cache._hash_embeds == {}
    assert batch.mrope_positions is mrope_positions


def test_future_feature_keeps_position_metadata_without_data() -> None:
    cache = _CACHE_CLS()
    feature = _feature("future", 3, 2)
    cache.register("r", [feature])

    mm_embeds, is_mm = cache.gather_chunk(_batch(0, False), NS(), torch.device("cpu"), False)

    assert mm_embeds == []
    assert is_mm.tolist() == [False, False]
    assert cache._features["r"] == [feature]
    assert cache._hash_embeds == {}


def test_intersecting_feature_without_data_fails_on_cold_cache() -> None:
    cache = _CACHE_CLS()
    cache.register("r", [_feature("missing", 1, 2)])

    with pytest.raises(RuntimeError, match="mm feature data missing"):
        cache.gather_chunk(_batch(1, False), NS(), torch.device("cpu"), False)


def test_feature_spanning_chunks_preserves_slices_and_hash_sharing() -> None:
    cache = _CACHE_CLS()
    feature = _feature("shared", 1, 4)
    cache.register("r", [feature])
    encoded = torch.arange(8).reshape(4, 2)
    cache._hash_embeds["shared"] = encoded

    first, first_mask = cache.gather_chunk(_batch(0, False), NS(), torch.device("cpu"), False)
    second, second_mask = cache.gather_chunk(_batch(2, False), NS(), torch.device("cpu"), False)
    third, third_mask = cache.gather_chunk(_batch(4, False), NS(), torch.device("cpu"), False)

    assert torch.equal(first[0], encoded[:1])
    assert torch.equal(second[0], encoded[1:3])
    assert torch.equal(third[0], encoded[3:4])
    assert first_mask.tolist() == [False, True]
    assert second_mask.tolist() == [True, True]
    assert third_mask.tolist() == [True, False]
    assert "r" not in cache._features
    assert cache._hash_embeds["shared"] is encoded
