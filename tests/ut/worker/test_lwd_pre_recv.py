# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests for LWD early-receive hint geometry."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path, modules: dict[str, types.ModuleType] | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with mock.patch.dict(sys.modules, modules or {}):
        spec.loader.exec_module(module)
    return module


lwd_types = load_module(
    "lwd_types_for_pre_recv_test",
    ROOT / "vllm_ascend/distributed/lwd_data_plane/lwd_types.py",
)
parents = {
    name: types.ModuleType(name)
    for name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.lwd_data_plane",
    )
}
pre_recv = load_module(
    "lwd_pre_recv_under_test",
    ROOT / "vllm_ascend/worker/lwd_cloud_model/lwd_pre_recv.py",
    {
        **parents,
        "vllm_ascend.distributed.lwd_data_plane.lwd_types": lwd_types,
    },
)


def valid_hint() -> dict:
    return pre_recv.make_recv_hint(
        3,
        4,
        1,
        req_ids=["a", "b"],
        rows=[2, 2],
        prompt_offsets=[1, 0],
        prompt_lens=[5, 4],
    )


def test_make_hint_preserves_complete_row_tables() -> None:
    hint = valid_hint()
    assert hint == {
        "channel": lwd_types.LwdChannelType.UP,
        "seqno": 3,
        "num_tokens": 4,
        "aux_rows": 1,
        "req_ids": ["a", "b"],
        "rows": [2, 2],
        "prompt_offsets": [1, 0],
        "prompt_lens": [5, 4],
        "refills": [],
        "prefix_nonces": [],
    }
    assert pre_recv.hint_fields(hint) == (
        3,
        4,
        1,
        ["a", "b"],
        [2, 2],
        [1, 0],
        [5, 4],
        [False, False],
        [0, 0],
    )


def test_prefix_nonces_are_a_parallel_table_with_zero_defaults() -> None:
    hint = valid_hint()
    hint["prefix_nonces"] = [7]
    assert pre_recv.hint_fields(hint).prefix_nonces == [7, 0]

    del hint["prefix_nonces"]
    assert pre_recv.hint_fields(hint).prefix_nonces == [0, 0]
    for invalid in ([1, 2, 3], True, ["7", 0], [True, 0], [-1, 0]):
        hint["prefix_nonces"] = invalid
        assert pre_recv.hint_fields(hint) is None


def test_refills_are_a_parallel_table_with_false_defaults() -> None:
    hint = valid_hint()
    hint["refills"] = [True]
    assert pre_recv.hint_fields(hint).refills == [True, False]

    del hint["refills"]
    assert pre_recv.hint_fields(hint).refills == [False, False]
    hint["refills"] = [True, False, True]
    assert pre_recv.hint_fields(hint) is None
    hint["refills"] = True
    assert pre_recv.hint_fields(hint) is None


def test_sideband_slot_is_four_kib() -> None:
    assert pre_recv._SIDEBAND_MAX_CHUNK_BYTES == 4096


@pytest.mark.parametrize(
    "field", ["seqno", "num_tokens", "aux_rows", "req_ids", "rows", "prompt_offsets", "prompt_lens"]
)
def test_hint_requires_every_field_without_refill(field: str) -> None:
    hint = valid_hint()
    del hint[field]
    assert pre_recv.hint_fields(hint) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seqno", -1),
        ("num_tokens", 0),
        ("aux_rows", -1),
        ("aux_rows", 5),
        ("req_ids", []),
        ("req_ids", ["a", "a"]),
        ("req_ids", "ab"),
        ("rows", [2]),
        ("rows", [1, 2]),
        ("rows", [0, 4]),
        ("prompt_offsets", [-1, 0]),
        ("prompt_offsets", [4, 0]),
        ("prompt_offsets", [1]),
        ("prompt_lens", [5, 0]),
        ("prompt_lens", [5]),
    ],
)
def test_hint_rejects_invalid_row_table_geometry(field: str, value: object) -> None:
    hint = valid_hint()
    hint[field] = value
    assert pre_recv.hint_fields(hint) is None
