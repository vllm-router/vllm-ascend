# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""CPU-only tests for the payload scramble / descramble module.

The module is a pure leaf utility, so everything here runs without NPU and
without an edge-cloud deployment.  The cases mirror the design doc
(``blog-repo/vllm/加扰与解扰-设计.md``).

Note what these do *not* cover: the NPU implementation of ``view(torch.int16)``
+ in-place ``bitwise_xor_``.  On CPU this exercises the semantics, not the
Ascend kernel.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_ascend.distributed.lwd_comm import payload_scramble as ps

_FACTORS = {
    "device_id": "ECPAIR-SN-8A3F91",
    "custom_factor": "biz-2024-q3",
    "channel": "lwd_up",
    "seqno": 7,
    "num_elements": 2048,
}

_MODULE_PATH = Path(ps.__file__)


def _scrambler(device_id=None, custom_factor=None):
    return ps.PayloadScrambler(
        device_id=_FACTORS["device_id"] if device_id is None else device_id,
        custom_factor=(
            _FACTORS["custom_factor"] if custom_factor is None else custom_factor
        ),
    )


def _seed(**overrides):
    return _scrambler().seed_for(
        channel=overrides.get("channel", _FACTORS["channel"]),
        seqno=overrides.get("seqno", _FACTORS["seqno"]),
        num_elements=overrides.get("num_elements", _FACTORS["num_elements"]),
    )


# --------------------------------------------------------------------------
# seed derivation
# --------------------------------------------------------------------------


def test_seed_is_deterministic():
    assert _seed() == _seed()


def test_seed_survives_hash_randomisation():
    """``hash()`` is salted per process; the KDF must not use it."""
    code = (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('ps', r'{_MODULE_PATH}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['ps'] = m\n"
        "spec.loader.exec_module(m)\n"
        "s = m.PayloadScrambler(device_id='ECPAIR-SN-8A3F91',"
        "custom_factor='biz-2024-q3')\n"
        "print(s.seed_for(channel='lwd_up', seqno=7, num_elements=2048))\n"
    )
    seen = set()
    for hashseed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
        )
        assert result.returncode == 0, result.stderr
        seen.add(result.stdout.strip())
    assert seen == {str(_seed())}


def test_seed_is_64_bit():
    assert 0 <= _seed() < 2**64


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_id": "other"},
        {"custom_factor": "other"},
        {"channel": "lwd_down"},
        {"seqno": 8},
        {"num_elements": 4096},
    ],
)
def test_each_factor_contributes(overrides):
    if "device_id" in overrides or "custom_factor" in overrides:
        # static factors -> build a different instance
        base = _scrambler(
            device_id=overrides.pop("device_id", None),
            custom_factor=overrides.pop("custom_factor", None),
        )
        dynamic = {
            "channel": _FACTORS["channel"],
            "seqno": _FACTORS["seqno"],
            "num_elements": _FACTORS["num_elements"],
        }
        assert base.seed_for(**dynamic) != _seed()
    else:
        assert _seed(**overrides) != _seed()


def test_whitespace_is_normalised():
    base = _seed()
    assert (
        _scrambler(device_id=" ECPAIR-SN-8A3F91 ", custom_factor="\tbiz-2024-q3\n")
        .seed_for(
            channel="  lwd_up  ",
            seqno=7,
            num_elements=2048,
        )
        == base
    )


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        (0, 0),
        (None, -1),
        (-3, -1),
        ("7", 7),
        ("x", -1),
        (2.9, 2),
        (2**63, 2**63),
        (float("inf"), -1),
        (float("nan"), -1),
    ],
)
def test_canonical_nonneg_int(raw, expected):
    assert ps._canonical_nonneg_int(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, ""), ("", ""), ("  a \n", "a")]
)
def test_canonical_str(raw, expected):
    assert ps._canonical_str(raw) == expected


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0x0001, 0x3D0F, 0x55AA, 0x7FFF, 0x8000, 0xFFFF])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exhaustive_16_bit_round_trip(seed, dtype):
    """All 65536 bit patterns survive a round trip, bit for bit."""
    patterns = torch.arange(-32768, 32768, dtype=torch.int16)
    tensor = patterns.view(dtype).clone()
    scr = _scrambler()
    scr.scramble(tensor, seed=seed)
    assert not torch.equal(tensor.view(torch.int16), patterns)
    scr.descramble(tensor, seed=seed)
    assert torch.equal(tensor.view(torch.int16), patterns)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_special_values_round_trip(dtype):
    tensor = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 1e-30, 65504.0],
        dtype=dtype,
    )
    reference = tensor.view(torch.int16).clone()
    scr = _scrambler()
    scr.scramble(tensor, seed=0xBEEF)
    scr.descramble(tensor, seed=0xBEEF)
    assert torch.equal(tensor.view(torch.int16), reference)


@pytest.mark.parametrize("shape", [(3, 8), (2, 4, 8)])
def test_2d_and_3d_tensors(shape):
    tensor = torch.randn(*shape, dtype=torch.bfloat16)
    reference = tensor.view(torch.int16).clone()
    scr = _scrambler()
    scr.scramble(tensor, seed=0x1234)
    scr.descramble(tensor, seed=0x1234)
    assert torch.equal(tensor.view(torch.int16), reference)


def test_scramble_is_in_place_and_returns_the_same_tensor():
    tensor = torch.randn(4, 8, dtype=torch.bfloat16)
    returned = _scrambler().scramble(tensor, seed=1)
    assert returned is tensor


def test_two_scrambles_need_two_descrambles():
    tensor = torch.randn(4, 8, dtype=torch.bfloat16)
    reference = tensor.view(torch.int16).clone()
    scr = _scrambler()
    scr.scramble(tensor, seed=0x1234)
    scr.scramble(tensor, seed=0x1234)
    scr.descramble(tensor, seed=0x1234)
    assert not torch.equal(tensor.view(torch.int16), reference)
    scr.descramble(tensor, seed=0x1234)
    assert torch.equal(tensor.view(torch.int16), reference)


# --------------------------------------------------------------------------
# validation (fail fast)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_id", "custom_factor"),
    [(None, "b"), ("", "b"), ("   ", "b"), ("a", None), ("a", ""), ("a", "  ")],
)
def test_empty_static_factor_is_rejected(device_id, custom_factor):
    with pytest.raises(ValueError):
        ps.PayloadScrambler(device_id=device_id, custom_factor=custom_factor)


def test_static_factors_are_keyword_only():
    with pytest.raises(TypeError):
        ps.PayloadScrambler("a", "b")


def test_dynamic_factors_and_seed_are_keyword_only():
    scr = _scrambler()
    with pytest.raises(TypeError):
        scr.seed_for("lwd_up", 1, 2)
    with pytest.raises(TypeError):
        scr.scramble(torch.zeros(4, dtype=torch.bfloat16), 1)


@pytest.mark.parametrize("bad", [None, "1", 1.5, True])
def test_non_int_seed_is_rejected(bad):
    with pytest.raises(ValueError, match="seed"):
        _scrambler().scramble(torch.zeros(4, dtype=torch.bfloat16), seed=bad)


@pytest.mark.parametrize("bad", [torch.randn(4, 8), torch.arange(4)])
def test_non_16_bit_dtype_is_rejected(bad):
    with pytest.raises(ValueError, match="16-bit"):
        _scrambler().scramble(bad, seed=1)


def test_non_contiguous_tensor_is_rejected():
    non_contiguous = torch.randn(8, 4, dtype=torch.bfloat16).t()
    with pytest.raises(ValueError, match=r"\.contiguous\(\)"):
        _scrambler().scramble(non_contiguous, seed=1)


# --------------------------------------------------------------------------
# instance semantics and purity
# --------------------------------------------------------------------------


def test_instance_is_immutable():
    scr = _scrambler()
    with pytest.raises(Exception):
        scr.device_id = "other"
    assert hash(scr) == hash(_scrambler())


def test_repr_hides_factor_values():
    text = repr(_scrambler())
    assert _FACTORS["device_id"] not in text
    assert _FACTORS["custom_factor"] not in text
    assert "set" in text


def test_seed_is_a_pure_function_of_the_runtime_factors():
    """Same instance, four different factor sets -> four different seeds."""
    scr = _scrambler()
    seeds = {
        _seed(),
        scr.seed_for(channel="lwd_down", seqno=7, num_elements=2048),
        scr.seed_for(channel="lwd_up", seqno=8, num_elements=2048),
        scr.seed_for(channel="lwd_up", seqno=9, num_elements=2048),
    }
    assert len(seeds) == 4


def test_seed_for_works_without_torch():
    """Only the operators need torch."""
    code = (
        "import sys, importlib.util\n"
        "sys.modules['torch'] = None\n"
        f"spec = importlib.util.spec_from_file_location('ps', r'{_MODULE_PATH}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['ps'] = m\n"
        "spec.loader.exec_module(m)\n"
        "s = m.PayloadScrambler(device_id='a', custom_factor='b')\n"
        "print(s.seed_for(channel='lwd_up', seqno=1, num_elements=2))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != ""


# --------------------------------------------------------------------------
# public surface and purity
# --------------------------------------------------------------------------


def test_public_surface_is_the_class_plus_the_process_factory():
    """One class with three methods, plus the process-wide accessor."""
    assert ps.__all__ == [
        "PayloadScrambler",
        "get_payload_scrambler",
        "is_payload_scrambler_initialized",
    ]
    public_methods = {
        name for name in dir(ps.PayloadScrambler) if not name.startswith("_")
    }
    assert {"seed_for", "scramble", "descramble"} <= public_methods
    operators = {name for name in public_methods if "scramble" in name}
    assert operators == {"scramble", "descramble"}


def test_module_is_a_leaf():
    """stdlib only at module scope; torch only as a TYPE_CHECKING annotation."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "hashlib",
        "dataclasses",
        "typing",
        # Annotation-only: `from torch import Tensor` sits under
        # `if TYPE_CHECKING:` and is never executed at import time, so the
        # module does not actually import torch.
        "torch",
    }, imported


def test_no_nondeterminism_sources():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    banned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in {"hash", "id", "random", "time", "monotonic"}:
                banned.append(name)
    assert not banned, banned
