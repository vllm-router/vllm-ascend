# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""What a cloud worker step spends, inside the engine's exec_rpc.

The engine times its round trip to the workers as one number, because
from the step thread that is all there is: it hands a batch over and
waits. A long wait there could be input staging, the forward pass, a
collective waiting on another rank, or the DOWN send -- and those have
nothing to do with each other.

This splits the worker's half of that round trip at the calls it
already makes:

  - worker_pre: before the model. Abort notices, releasing finished
    requests, reaping the keepalive queue.
  - forward_call: the native execute_model. Host time around the forward
    pass, not NPU time: the device is asynchronous, so some of this
    step's queue drains into the next call. No synchronize is added to
    sharpen it, because that would change the pipeline being measured.
  - native_sample: the native sample_tokens.
  - down_collect: gathering this step's hidden rows and metadata.
  - down_submit: handing the DOWN packet to the data plane.

Reported per interval like the other meters, with any step over the
threshold named on its own line. The engine's `down` seqno appears on
both sides, so a slow step there can be matched to its step here.

Timing is kept here rather than in the worker so the call sites stay
one line each, and so the per-step `lwd_logger.debug` lines these
replace cannot come back: those are silent under the log level a
performance run uses, which is the one setting where they would matter.
"""

from __future__ import annotations

import time

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
)

logger = init_logger(__name__)

# A worker step worth naming individually. Matched to the engine's own
# slow-step threshold so the two logs mark the same steps.
LWD_SLOW_WORKER_STEP_S = 1.0


class LwdCloudWorkerMeter(LwdIntervalMeter):
    """Per-step breakdown of the cloud worker's share of a step.

    Only the rank that owns the DOWN channel collects and sends, so the
    last two segments are zero on every other rank; the rank is on the
    line because a step that is slow on one rank only is a different
    fault from one that is slow on all of them.
    """

    def __init__(self, rank: int, dp_idx: int) -> None:
        self._rank = rank
        self._dp_idx = dp_idx
        # The execute half of the step in progress, held until the sample
        # half closes it. Not a window counter: a step straddling a report
        # would otherwise lose the half that came first.
        self._execute_s = 0.0
        super().__init__()

    def _reset(self) -> None:
        self._steps = 0
        self._slow = 0
        self._total = LwdSpan()
        self._pre = LwdSpan()
        self._forward = LwdSpan()
        self._sample = LwdSpan()
        self._collect = LwdSpan()
        self._submit = LwdSpan()

    def note_execute(self, pre_s: float, forward_s: float) -> None:
        """The execute_model half: preparation, then the forward call."""
        self._pre.add(pre_s)
        self._forward.add(forward_s)
        self._execute_s = pre_s + forward_s

    def note_sample(
        self,
        native_s: float,
        collect_s: float,
        submit_s: float,
        seqno: int,
        rows: int,
    ) -> None:
        """The sample_tokens half, which closes the step: sampling, the
        DOWN collect, and the submit."""
        execute_s = self._execute_s
        self._execute_s = 0.0
        self._steps += 1
        self._sample.add(native_s)
        self._collect.add(collect_s)
        self._submit.add(submit_s)
        total = execute_s + native_s + collect_s + submit_s
        self._total.add(total)
        if total >= LWD_SLOW_WORKER_STEP_S:
            self._slow += 1
            logger.info(
                "[lwd][cloud-worker-slow] dp%d rank%d %.0fms: "
                "worker_pre+forward=%.0f native_sample=%.0f "
                "down_collect=%.0f down_submit=%.0f (ms); "
                "down=%d rows=%d; ts=%.3f",
                self._dp_idx,
                self._rank,
                total * 1000,
                execute_s * 1000,
                native_s * 1000,
                collect_s * 1000,
                submit_s * 1000,
                seqno,
                rows,
                # Wall clock, to line this up against the engine's log and
                # the other ranks'. Every duration above is monotonic.
                time.time(),
            )
        self._roll()

    def _report(self) -> None:
        if not self._steps:
            return
        logger.info(
            "[lwd][cloud-worker] dp%d rank%d %d steps (%d slow): total=%s; "
            "worker_pre=%s forward_call=%s native_sample=%s "
            "down_collect=%s down_submit=%s (avg/max ms)",
            self._dp_idx,
            self._rank,
            self._steps,
            self._slow,
            self._total,
            self._pre,
            self._forward,
            self._sample,
            self._collect,
            self._submit,
        )
