# SPDX-License-Identifier: Apache-2.0
"""Exercise the runner's actual batch-remapping branch without an NPU."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


def load_remap():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs")
    branch = next(
        n for n in ast.walk(prepare)
        if isinstance(n, ast.If)
        and ast.unparse(n.test).startswith("self.use_async_scheduling and")
        and "_lwd_spec_persist_enabled" in ast.unparse(n.test)
    )
    return compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("previous_count", [2, 4])
@pytest.mark.parametrize("persist", [False, True])
def test_new_prefix_hit_and_returning_request_do_not_inherit_row_zero(previous_count, persist):
    # Previous row 0 belongs to an unrelated request; rows 1 and 0 reorder.
    # The new request has a cached prefix, but no prior speculative state.
    batch = NS(
        req_ids=["continuing", "prefix-hit", "returning", "unknown", "invalid"],
        num_accepted_tokens_cpu=np.array([previous_count, 1, 1, 1, 1]),
    )
    runner = NS(
        use_async_scheduling=True,
        _lwd_spec_persist_enabled=persist,
        prev_positions=NS(np=np.array([1, -1, -1, -1, -1])),
        num_accepted_tokens=NS(np=np.zeros(5, dtype=int)),
        input_batch=batch,
        requests={
            "prefix-hit": NS(num_computed_tokens=256, num_prompt_tokens=3510, num_accepted_tokens=0),
            "returning": NS(num_computed_tokens=3600, num_prompt_tokens=3510, num_accepted_tokens=2),
            "invalid": NS(num_computed_tokens=3600, num_prompt_tokens=3510, num_accepted_tokens=0),
        },
    )
    exec(load_remap(), {"self": runner, "np": np, "num_reqs": 5, "prev_req_id_to_index": {"old": 0, "continuing": 1}})
    assert batch.num_accepted_tokens_cpu.tolist() == [1, 1, 2 if persist else 1, 1, 1]
    # Mamba prefix restoration uses accepted - 1 as its state-copy offset.
    assert batch.num_accepted_tokens_cpu[1] - 1 == 0


@pytest.mark.parametrize("previous_map", [{}, None])
@pytest.mark.parametrize("saved_count", [1, 2, 4])
def test_return_after_prefill_only_batch_restores_own_state(previous_map, saved_count):
    batch = NS(
        req_ids=["returning", "prefill", "new", "invalid"],
        num_accepted_tokens_cpu=np.array([1, 4, 4, 4]),
    )
    runner = NS(
        use_async_scheduling=True,
        _lwd_spec_persist_enabled=True,
        prev_positions=NS(np=np.full(4, -1)),
        num_accepted_tokens=NS(np=np.zeros(4, dtype=int)),
        input_batch=batch,
        requests={
            "returning": NS(num_computed_tokens=3600, num_prompt_tokens=3510, num_accepted_tokens=saved_count),
            "prefill": NS(num_computed_tokens=256, num_prompt_tokens=3510, num_accepted_tokens=4),
            "new": NS(num_computed_tokens=0, num_prompt_tokens=3510, num_accepted_tokens=0),
            "invalid": NS(num_computed_tokens=3600, num_prompt_tokens=3510, num_accepted_tokens=0),
        },
    )
    exec(load_remap(), {"self": runner, "np": np, "num_reqs": 4, "prev_req_id_to_index": previous_map})
    assert runner.num_accepted_tokens.np.tolist() == [saved_count, 1, 1, 1]
    assert batch.num_accepted_tokens_cpu.tolist() == [saved_count, 1, 1, 1]


@pytest.mark.parametrize("async_mode,persist", [(False, True), (False, False), (True, False)])
def test_direct_count_path_without_async_persistence_is_unchanged(async_mode, persist):
    batch = NS(num_accepted_tokens_cpu=np.array([2, 4]))
    runner = NS(
        use_async_scheduling=async_mode,
        _lwd_spec_persist_enabled=persist,
        input_batch=batch,
        num_accepted_tokens=NS(np=np.zeros(2, dtype=int)),
    )
    exec(load_remap(), {"self": runner, "np": np, "num_reqs": 2, "prev_req_id_to_index": {}})
    assert runner.num_accepted_tokens.np.tolist() == [2, 4]
