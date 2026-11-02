# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` is set (LWD deployment).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

# ops must initialize before the model_runner chain, otherwise device_op
# and the ops package circularly import (mirrors worker.py's import order)
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import lwd_get_data_plane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType, LwdTransferRequest
from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import (
    LwdHostStager,
    LwdHostStageSlotBusy,
)
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel

if TYPE_CHECKING:
    from typing import Any

    from vllm.distributed.device_communicators.shm_broadcast import Handle
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

    from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import LwdRecvHintFields

# Early-recv thread loop tuning: MQ dequeue timeout, backoff sleep after a
# dequeue error, and the per-iteration sleep that caps CPU while the MQ
# reader window spins.
_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S = 0.0005
_LWD_EARLY_RECV_ERROR_SLEEP_S = 0.01
_LWD_EARLY_RECV_LOOP_SLEEP_S = 0.002
_LWD_EARLY_RECV_MAX_INFLIGHT = 2
_LWD_HOST_STAGE_SLOTS = 2


class LwdCloudWorker(LwdWorkerBase):
    """LwdWorkerBase + LWD data-plane wiring (cloud side only).

    Selected by ``platform.py`` only when LWD is enabled on a cloud
    process; the data-plane wiring below runs unconditionally."""

    def init_device(self) -> None:
        super().init_device()
        self._lwd_down_next_seqno = 0
        self._init_lwd_early_recv_state()
        if self.use_v2_model_runner:
            raise RuntimeError("LWD data plane requires the V1 model runner; use_v2_model_runner is not supported")
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(self.vllm_config, self.device, worker=self)
        self._init_lwd_data_plane()
        import torch.distributed as dist

        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        self._lwd_dp_idx = self.vllm_config.parallel_config.data_parallel_index
        endpoint_rank = lwd_config.instance.dp[self._lwd_dp_idx].ranks[0]
        global_rank = dist.get_rank()
        self._lwd_early_recv_endpoint = global_rank == endpoint_rank
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        self._lwd_up_num_hash_layers = hash_layer_count(self.model_config.hf_config)
        self._lwd_up_num_experts_per_tok = getattr(self.model_config.hf_config, "num_experts_per_tok", 0)
        if self._lwd_early_recv_endpoint:
            self._lwd_init_lwd_host_stager()
            self._start_lwd_early_recv_thread()
        lwd_logger.info(
            "[Lwd][pre-recv] worker local_rank=%d global_rank=%d dp_idx=%d endpoint=%s hidden=%d",
            self.rank,
            global_rank,
            self._lwd_dp_idx,
            self._lwd_early_recv_endpoint,
            self._lwd_up_hidden_size,
        )
        self._register_lwd_prompt_embeds_provider()

    def _lwd_init_lwd_host_stager(self) -> None:
        """Create endpoint host staging with fixed limits."""
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        max_numel = hash_payload_numel(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            self._lwd_up_hidden_size,
            self._lwd_up_num_hash_layers,
            self._lwd_up_num_experts_per_tok,
        )
        self._lwd_stager = LwdHostStager(device=str(self.device), slots=_LWD_HOST_STAGE_SLOTS, max_numel=max_numel)

    def load_model(self) -> None:
        """Log the real layer split after loading: layer count / first-last
        layer names / pp slicing args, so a half-model suspicion is decided
        at startup (full coverage from layer 0 = normal; halved or nonzero
        start = wrong pp slicing)."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the cloud install point; edge and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.model_runner.get_model())
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(backbone, "decoder_layers", None)
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules() if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        lwd_logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — full load should cover every layer from 0",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size,
            pc.tensor_parallel_size,
            self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # [Lwd][perf] Worker start time (subtract the engine's rpc-enqueue
        # ts = dispatch-to-start latency; spread across ranks = collective
        # alignment cost)
        lwd_logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank,
            time.monotonic(),
        )
        self._lwd_up_post_recvs(scheduler_output)

        # finished_req_ids releases runner and host-stage request state.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        lwd_get_data_plane().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] Per-step exec_model timing on the cloud: total = the
        # whole forward pass (sample/collect are itemized separately in
        # sample_tokens)
        lwd_logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output: SchedulerOutput) -> None:
        """Apply abort notices carried by normal or abort-only scheduler batches."""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or self._lwd_stager is None:
            return
        for req_id in meta.aborted_req_ids or ():
            self._lwd_stager.abort_request(req_id)

    # ------------------------------------------------------------------ #
    # Early receive: comm thread (hint -> post recv -> report watermark) #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """Initialize queued, submitted, and delivered UP receive state."""
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        self._early_recv_futures: dict[tuple[LwdChannelType, int], LwdTransferFuture] = {}
        self._early_recv_pending: dict[tuple[LwdChannelType, int], dict[str, Any]] = {}
        self._early_recv_last_delivered = -1
        self._early_recv_current_seqno = -1
        self._early_recv_lock = threading.Lock()
        self._lwd_stager: LwdHostStager | None = None
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        self._lwd_early_recv_thread: threading.Thread | None = None

    def _lwd_build_up_recv_request(
        self,
        fields: LwdRecvHintFields,
        recv_buffer: Any,
        stage_segments: list[tuple[Any, int, int]],
    ) -> LwdTransferRequest:
        """Build a staged UP receive with the exact hash-payload geometry."""
        return LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="recv",
            dp_idx=self._lwd_dp_idx,
            num_elements=hash_payload_numel(
                fields.num_tokens,
                self._lwd_up_hidden_size,
                self._lwd_up_num_hash_layers,
                self._lwd_up_num_experts_per_tok,
            ),
            seqno=fields.seqno,
            aux_num_elements=fields.aux_rows * 3,
            recv_buffer=recv_buffer,
            stage_segments=stage_segments,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """Rebuild the hint MQ reader, create the done MQ writer, and start
        the early-recv comm thread.

        One log line before and after the thread start: "starting" before
        (proof this code ran) and "up" from inside the thread body after
        setting the device (proof the thread is alive). Only the first
        line = the thread died during device setup (traceback on stderr);
        both lines but no SUBMIT = no hints arrived (hint MQ not attached;
        the log will say attached/none).
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        self._irecv_hint_mq = attach_hint_mq()
        self._irecv_done_mq = create_report_mq()
        # The engine fetches this MQ's handle through
        # lwd_early_recv_done_handle over collective_rpc once the executor
        # is up, so the vllm executor stays free of LWD plumbing.
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def lwd_early_recv_done_handle(self) -> Handle | None:
        """Handle of the early-recv done MQ, for the engine to attach its
        reader; None on ranks that do not report completions (the
        non-endpoint ranks hold no cross-node channel)."""
        if self._irecv_done_mq is None:
            return None
        return self._irecv_done_mq.export_handle()

    def _lwd_early_recv_loop(self) -> None:
        """Comm thread body: take hints -> pre-post recvs; poll futures ->
        report completions.

        It only does four featherweight things: ``dequeue`` /
        ``submit_recv`` / ``future.done()`` / ``enqueue`` — it never
        ``wait()``s any future and never touches the model, so it cannot
        conflict with the busy loop's HCCL usage (an irecv on one side of
        a channel plus a cross-thread wait on the other is disallowed by
        HCCL).
        """
        # The current device is thread-local state; the comm thread must
        # set it once itself (or the irecv lands on card 0)
        torch.npu.set_device(self.device)
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank,
            self.device,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mq = self._irecv_hint_mq
        while True:
            try:
                if hint_mq is not None:
                    try:
                        hint = hint_mq.dequeue(timeout=_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S)
                    except TimeoutError:
                        pass
                    except Exception:
                        lwd_logger.exception("[Lwd][pre-recv] hint MQ dequeue failed")
                        time.sleep(_LWD_EARLY_RECV_ERROR_SLEEP_S)
                    else:
                        self.start_early_irecv(hint)
                self._report_irecv_completions()
            except Exception as exc:
                self._lwd_notify_early_recv_error(exc)
                return
            # The ring's read side sched_yield-spins within the
            # just-read window (busy_loop_s=1s), so a 0.5ms timeout alone
            # does not cap the CPU — this sleep does; the cost is ~2ms of
            # added visibility latency on hints and watermarks each.
            time.sleep(_LWD_EARLY_RECV_LOOP_SLEEP_S)

    def _lwd_notify_early_recv_error(self, exc: Exception) -> None:
        """Report a terminal comm failure once; the loop returns afterwards."""
        seqno = self._early_recv_current_seqno
        lwd_logger.exception("[Lwd][pre-recv] comm thread failed seqno=%d", seqno)
        done_mq = self._irecv_done_mq
        if done_mq is None:
            return
        try:
            done_mq.enqueue(("error", seqno, repr(exc)))
        except Exception:
            lwd_logger.exception("[Lwd][pre-recv] failed to report comm error seqno=%d", seqno)

    def start_early_irecv(self, hint: dict[str, Any]) -> None:
        """Queue one validated UP hint and submit it when capacity allows."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None or hint.get("channel") is not LwdChannelType.UP:
            lwd_logger.warning("[Lwd][pre-recv] malformed hint %s, skipped", hint)
            return
        key = (LwdChannelType.UP, fields.seqno)
        with self._early_recv_lock:
            if (
                fields.seqno <= self._early_recv_last_delivered
                or key in self._early_recv_futures
                or key in self._early_recv_pending
            ):
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _lwd_submit_early_recv(self, fields: LwdRecvHintFields) -> LwdTransferFuture:
        """Plan host destinations, allocate a device slot, and submit one receive."""
        stager = self._lwd_stager
        assert stager is not None
        numel = hash_payload_numel(
            fields.num_tokens,
            self._lwd_up_hidden_size,
            self._lwd_up_num_hash_layers,
            self._lwd_up_num_experts_per_tok,
        )
        recv_buffer, _slot = stager.allocate(fields.seqno, numel)
        segments = stager.plan_recv_segments(
            seqno=fields.seqno,
            numel=numel,
            hidden_size=self._lwd_up_hidden_size,
            req_ids=fields.req_ids,
            num_tokens_per_req=fields.rows,
            prompt_lens=fields.prompt_lens,
            row_base=fields.prompt_offsets,
            num_hash_layers=self._lwd_up_num_hash_layers,
            num_experts_per_tok=self._lwd_up_num_experts_per_tok,
            refills=fields.refills,
            prefix_nonces=fields.prefix_nonces,
        )
        request = self._lwd_build_up_recv_request(fields, recv_buffer, segments)
        return lwd_get_data_plane().submit_recv(request)

    def _early_recv_pump(self) -> None:
        """Submit queued hints in order, leaving a busy hint at the head."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        while True:
            with self._early_recv_lock:
                if not self._early_recv_pending or len(self._early_recv_futures) >= self._early_recv_max_inflight:
                    return
                key, hint = next(iter(self._early_recv_pending.items()))
                fields = hint_fields(hint)
                assert fields is not None
                self._early_recv_current_seqno = fields.seqno
                try:
                    future = self._lwd_submit_early_recv(fields)
                except LwdHostStageSlotBusy:
                    self._early_recv_current_seqno = -1
                    return
                except Exception:
                    self._early_recv_pending.pop(key, None)
                    assert self._lwd_stager is not None
                    self._lwd_stager.abandon(fields.seqno)
                    raise
                self._early_recv_pending.pop(key)
                self._early_recv_futures[key] = future
                self._early_recv_current_seqno = -1
                inflight = len(self._early_recv_futures)
                queued = len(self._early_recv_pending)
            lwd_logger.info(
                "[Lwd][pre-recv] SUBMIT channel=%s seqno=%d num_tokens=%d aux_rows=%d inflight=%d queued=%d",
                key[0].value,
                fields.seqno,
                fields.num_tokens,
                fields.aux_rows,
                inflight,
                queued,
            )

    def _report_irecv_completions(self) -> None:
        """Publish staged arrivals in seqno order and report cumulative rows."""
        done_mq = self._irecv_done_mq
        stager = self._lwd_stager
        if done_mq is None or stager is None:
            return
        with self._early_recv_lock:
            pending = list(self._early_recv_futures.items())
        for key, future in pending:
            if not future.done():
                break
            self._early_recv_current_seqno = key[1]
            future.result()
            if not future.staged_done():
                self._early_recv_current_seqno = -1
                break
            future.result()
            try:
                published = stager.publish_received(key[1])
                stager.release_if_drained(key[1])
                arrived = [(req_id, offset + rows) for req_id, rows, offset in published]
                future.release_device_tensor()
                for req_id, prefix_nonce in stager.take_refill_done():
                    done_mq.enqueue(("refill_done", req_id, prefix_nonce))
                done_mq.enqueue((key, arrived))
            except Exception:
                stager.abandon(key[1])
                raise
            with self._early_recv_lock:
                self._early_recv_futures.pop(key, None)
                self._early_recv_last_delivered = key[1]
            self._early_recv_current_seqno = -1
            lwd_logger.info(
                "[Lwd][pre-recv] REPORT channel=%s seqno=%d arrivals=%s",
                key[0].value,
                key[1],
                arrived,
            )
            self._early_recv_pump()

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN: send the hidden packet (channel-async,
        # in-stream ordered); the step meta stays on device until the
        # fenced harvest below brings it to the host with a device-side
        # stream dependency (dedicated stream wait_event on the ready
        # event, completion event recorded on the copy stream itself).
        payload = self.model_runner.take_lwd_pending_down_payload()
        if payload is not None and output is not None:
            (hidden, meta_dev, seg_lens_host, ready_event, req_ids) = payload
            seqno = self._lwd_next_down_seqno()
            lwd_logger.info(
                "[Lwd][cloud-worker] DOWN send seqno=%d numel=%d",
                seqno,
                hidden.numel(),
            )
            lwd_get_data_plane().submit_send(
                LwdTransferRequest(
                    channel=LwdChannelType.DOWN,
                    op="send",
                    num_elements=hidden.numel(),
                    dp_idx=self._lwd_dp_idx,
                    tensor=hidden,
                    seqno=seqno,
                )
            )
            # Fenced harvest: device-side dependency guarantees the copy
            # starts only after the main-stream work that produced
            # meta_dev; done.synchronize() waits for this copy only
            # (microseconds), not the whole GPU. This replaces the retired
            # pinned D2H + shared-event scheme, which could not fence the
            # side-stream copy on this platform (the engine could read a
            # previous generation's meta image — wrong accepted counts).
            _t_harvest = time.monotonic()
            stream = self._lwd_get_harvest_stream()
            n_dev = meta_dev.numel()
            n_seg = len(seg_lens_host) if seg_lens_host is not None else 0
            buf = self._lwd_get_harvest_pinned(max(n_dev + n_seg, 4096))
            with torch.npu.stream(stream):
                stream.wait_event(ready_event)
                buf[:n_dev].copy_(meta_dev, non_blocking=True)
                done = torch.npu.Event()
                done.record()
            done.synchronize()
            if n_seg:
                buf.numpy()[n_dev : n_dev + n_seg] = seg_lens_host
            # Attach the finished 4-tuple (wire layout [meta|seg_lens]
            # unchanged); the engine parse path needs no changes.
            target = getattr(output, "_model_runner_output", output)
            target.lwd_down_carrier = (
                buf[: n_dev + n_seg],
                req_ids,
                hidden.numel(),
                seqno,
            )
            # [Lwd][perf] Per-step cloud timing: sample = sampling +
            # collection (rank computation included); send = the DOWN
            # submit (snapshot clone + isend + bridge wait); harvest =
            # the fenced meta copy; total = the whole step
            lwd_logger.info(
                "[Lwd][perf] cloud-step seqno=%d sample=%.2f send=%.2f harvest=%.2f total=%.2fms rows=%d",
                seqno,
                (_t_sample - _t0) * 1000,
                (time.monotonic() - _t_sample) * 1000,
                (time.monotonic() - _t_harvest) * 1000,
                (time.monotonic() - _t0) * 1000,
                hidden.shape[0] if hidden is not None else 0,
            )
        return output

    def _lwd_get_harvest_stream(self) -> torch.npu.Stream:
        """Lazily create the dedicated harvest stream (one per worker)."""
        if not hasattr(self, "_lwd_harvest_stream"):
            self._lwd_harvest_stream = torch.npu.Stream()
        return self._lwd_harvest_stream

    def _lwd_get_harvest_pinned(self, min_capacity: int) -> torch.Tensor:
        """Lazily allocate (or grow) the pinned harvest buffer."""
        buf = getattr(self, "_lwd_harvest_pinned", None)
        if buf is None or buf.numel() < min_capacity:
            buf = self._lwd_harvest_pinned = torch.empty(
                min_capacity,
                dtype=torch.int32,
                pin_memory=True,
            )
        return buf

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def _lwd_cloud_flush_finished(self, finished_req_ids: list[str]) -> None:
        """Release runner bookkeeping and all host-stage request storage."""
        lwd_logger.info("[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids))
        self.model_runner.lwd_release_finished(finished_req_ids)
        if self._lwd_stager is None:
            return
        for req_id in finished_req_ids:
            seqnos = self._lwd_stager.batches_holding(req_id)
            self._lwd_stager.release_request(req_id)
            for seqno in seqnos:
                self._lwd_stager.release_if_drained(seqno)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_remote_embeds`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str) -> torch.Tensor | None:
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(_lwd_prompt_embeds_provider)

    def _lwd_next_down_seqno(self) -> int:
        seqno = self._lwd_down_next_seqno
        self._lwd_down_next_seqno += 1
        return seqno
