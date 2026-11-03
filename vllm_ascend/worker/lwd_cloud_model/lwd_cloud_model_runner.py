# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` is set.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.lwd_logger import lwd_logger
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_cloud_model.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
    hash_payload_numel,
    unpack_hash_payload,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from typing import Any

    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import (
        GrammarOutput,
        LwdEmbedBatch,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ECConnectorOutput, SamplerOutput
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker


# Endpoint host-gate timeout for the UP harvest wait: exceeding it means the
# matching isend never arrived (edge never posted or the wire stalled), so
# fail loudly instead of waiting forever.
_LWD_UP_HOST_GATE_TIMEOUT_S = 60.0


class LwdUpStagedBatch(NamedTuple):
    """Per-batch harvest record flowing stage -> tensor phase.

    ``t_wait`` carries the probe start so the tensor phase's
    up-recv-wait window matches the pre-split single loop; the inject
    pass receives ``rows`` from the window check instead of recomputing
    indices (the duplicate existence check used to run twice per
    request).
    """

    seqno: int
    meta: LwdEmbedBatch
    future: LwdTransferFuture | None
    num_tokens: int
    t_wait: float


class LwdDownSelection(NamedTuple):
    """Row/rank selection product of one DOWN collect step."""

    hidden_packet: torch.Tensor
    ranks: list[torch.Tensor]
    counts_dev: torch.Tensor | None
    seg_lens_host: list[int] | None


class LwdCloudModelRunner(LwdBaseModelRunner):
    """NPUModelRunner + LWD cloud-side runner logic."""

    # Config-derived constants, computed once at first use; lwd_config and
    # the HF config are immutable at runtime. Class-level defaults keep
    # instances built via __new__ (tests) from tripping on missing attrs.
    _lwd_hash_dims_cached: tuple[int, int] | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdCloudWorker | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self._lwd_pending_down_payload = None
        # Row-slot residue cleanup: flush reused rows as soon as a request
        # is admitted (2026-10-21 MTE root-cause closure)
        self._lwd_reset_reused_rows_on_admission()
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        if num_hash_layers and vllm_config.lwd_config is not None:
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (
                parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers,
                config.num_experts_per_tok,
                config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens,
                num_hash_layers,
                config.num_experts_per_tok,
                dtype=torch.int32,
                device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
            # Retain the zero placeholder so graph replay uses the captured storage.
            self._lwd_hash_placeholder_ids = torch.zeros_like(self.input_ids.gpu)

    def _lwd_reset_reused_row(
        self,
        request: CachedRequestState,
        req_index: int,
    ) -> None:
        """Clear tokens left in a reused batch row by the previous request.

        LWD cloud requests carry no token ids (the wire ships embeddings
        only), so the native add_request skips writing the row's prompt
        region for them. Stale cells from the row's previous occupant -
        including -1 placeholders that async scheduling leaves behind and
        nothing repairs for penalty-free requests - survive row reuse and
        ride the prompt-token tensor into the penalty bincount kernel,
        where negative ids become negative-offset writes (the aivec
        fault). Native requests carry real ids and overwrite the region
        on every admission, which is exactly the behavior missing here.

        Fill [0, num_prompt) with vocab_size - the "no token" padding
        value every downstream consumer already ignores - once, at
        admission, before any reader can observe the row.
        """
        if request.prompt_token_ids is not None:
            return  # rows with real ids are already overwritten natively
        batch = self.input_batch
        num_prompt = int(batch.num_prompt_tokens[req_index])
        batch.token_ids_cpu[req_index, :num_prompt] = batch.vocab_size

    def _lwd_reset_reused_rows_on_admission(self) -> None:
        """Hook the row reset onto request admission (instance-level).

        The admission call site lives inside the shared update_states
        with no LWD-owned seam; overriding it would mean copying shared
        logic that breaks on every upstream change. Wrap add_request on
        this process's InputBatch instance instead: native bookkeeping
        first, then _lwd_reset_reused_row. Migrate to an upstream hook
        if one is ever provided.

        Idempotent per InputBatch instance: initialize_kv_cache
        REBUILDS the InputBatch when the final KV geometry differs from
        the pre-load guess (always true for the hybrid groups here,
        model_runner_v1 re-init), so the hook must be re-anchored on
        the live instance - see initialize_kv_cache below. The first
        acceptance run failed exactly this way: the wrapper stayed on
        the discarded instance and the residue poison survived.
        """
        batch = self.input_batch
        if getattr(batch, "_lwd_row_reset_hooked", False):
            return
        batch._lwd_row_reset_hooked = True
        orig_add = batch.add_request

        def add_request_then_reset_row(request):
            req_index = orig_add(request)
            self._lwd_reset_reused_row(request, req_index)
            return req_index

        batch.add_request = add_request_then_reset_row

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        """Re-anchor the row-reset hook after the InputBatch rebuild.

        The base may replace self.input_batch here (hybrid KV geometry);
        the per-instance hook flag on the fresh instance is absent, so
        the installer below (re)hooks the live one."""
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context)
        self._lwd_reset_reused_rows_on_admission()

    def _get_drafter(self) -> SpecDecodeBaseProposer | None:
        """Use the LWD MTP embedding adapter only in LWD mode."""
        if (
            self.vllm_config.lwd_config is not None
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        return super()._get_drafter()

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        """UP harvest and mrope writes happen before super(); embeds
        overwrite happens after it.

        mrope timing contract (fixes a source-branch defect found during
        the multimodal migration review): _calc_mrope_positions inside
        super() copies the req_state.mrope_positions window rows to the
        GPU within the same step (model_runner_v1.py:1033-1045). If the
        mrope write landed after super() (the original shape), the
        current chunk's mrope rows would never be seen by its own
        forward (every chunk uses the ids=None arange initialization,
        so image rows would be wrong). Hence the harvest (endpoint host
        gate + in-place TP-group broadcast) and the mrope/delta writes
        are hoisted before super().

        The embeds overwrite stays after super(): the base fill loop and
        copy_to_gpu only see zero buffers, then the harvested UP embeds
        are written straight into the inputs_embeds.gpu scheduling
        window of the current step (last write wins), ordered on-stream
        via the future completion event — non-blocking; the recv wait is
        visible separately via the [Lwd][perf] up-recv-wait timer.
        """
        harvested = self._lwd_harvest_up_batches(num_scheduled_tokens)
        # Skip the base fill loop's per-row copy for LWD placeholder rows
        # (gpu_model_runner.py:952-996): it host-memcpys 84MB from the CPU
        # assembly buffer (all zeros at this point) plus an H2D into
        # inputs_embeds, which the inject then overwrites on the same
        # window (last write wins) — pure waste, measured ~17-20ms per
        # chunk in the super() segment. Clearing the map skips it
        # wholesale; the inject/release logic restores the reference
        # after super() returns. Every cloud-side LWD request is a
        # placeholder row, so no real prompt-embeds consumer is hurt.
        embeds_map = self.input_batch.req_prompt_embeds
        self.input_batch.req_prompt_embeds = {}
        try:
            out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            self.input_batch.req_prompt_embeds = embeds_map
        self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
        self._lwd_release_consumed_prompt_embeds()
        self._lwd_inject_remote_embeds(harvested, num_scheduled_tokens)
        if self.lwd_hash_state is not None:
            self._lwd_prepare_hash_batch(num_scheduled_tokens)
        return out

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens: np.ndarray) -> None:
        """Skip remote expert staging for decode, preserving graph buffers."""
        num_reqs = self.input_batch.num_reqs
        counts = num_scheduled_tokens[:num_reqs]
        computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_reqs]
        self._lwd_hash_step_tokens = sum(int(n) for n in counts)
        has_prompt = any(
            int(count) > 0 and int(start) < int(prompt_len)
            for count, start, prompt_len in zip(counts, computed, prompt_lengths)
        )
        # Clear the full mask, including graph padding and previous larger
        # batches. Keep its address and the captured merge graph unchanged.
        self._lwd_hash_prompt_mask.zero_()
        if not has_prompt:
            # The merge selects local lookup for every row. Stale expert IDs
            # are only unselected values, never indices into an expert table.
            return
        batch, mask = self.lwd_hash_state.build_batch(
            self.input_batch.req_ids,
            counts,
            computed,
            prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[: batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[: mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens: np.ndarray) -> None:
        """Rebuild the flattened is_token_ids mask for the prompt-embeds branch.

        Root cause (async spec decode optimism): the runner's
        num_computed_tokens_cpu assumes every previous draft was accepted
        (r tokens past the real commit position), while the native
        flattening (token_indices = positions + idx*M in
        model_runner_v1.py) gathers from the 2D mask with those
        optimistic positions — decode rows shift right by r and the last
        r rows of each segment fall past the True-marked region, reading
        False/stale values. The prompt-embeds branch then treats them as
        embeds rows, skipping local embedding and leaving stale garbage
        in inputs_embeds.gpu; those rows are exactly the sampled rows
        (logits_indices takes the last 1+k rows of each segment), so
        garbage logits pollute committed tokens directly. Native/phase/
        single-request shapes are unaffected (the mask is only consumed
        on the prompt-embeds branch; those shapes take the MM branch).

        Under LWD semantics only injected rows (prompt segments) may be
        embeds rows (False); decode/draft rows are always token ids
        (True). This rebuild is deterministic from scheduler state and
        needs no optimistic-value correction: a decode request's
        optimistic computed only over-estimates, so prefill_rows is
        always 0 — inherently safe. The CPU-view repair is kept for the
        parent path (.np and .cpu share storage); the async path
        regenerates the GPU mask in _preprocess from corrected device
        positions and consumes the device embedding directly.
        """
        if not self.input_batch.req_prompt_embeds:
            # No embeds buffers in the batch (MM/text branch; nobody reads the mask)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(n, max(int(num_prompt[i]) - int(computed[i]), 0))
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Refresh the decode mask after the async speculative-position fix, then prepare model inputs."""
        if (
            self.use_async_spec_decode
            and self.enable_prompt_embeds
            and intermediate_tensors is None
            and (self.vllm_config.parallel_config.prefill_context_parallel_size == 1)
        ):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (
                get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
                and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)
            ):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    # ------------------------------------------------------------------ #
    # Zero-base-weave seams: all LWD behavior surfaces via subclass    #
    # overrides; the base classes carry no seams.                       #
    # ------------------------------------------------------------------ #

    def _update_states(self, scheduler_output: SchedulerOutput) -> Callable[[], None] | None:
        """LWD: materialize meta placeholder embeds in place before they
        reach the base (zero base seams).

        The engine attaches prompt embeds as meta tensors (shape only,
        empty storage), so the placeholder rides the MQ inline pickle
        path instead of the overflow slow channel; the base
        CachedRequestState construction passes new_req_data.prompt_embeds
        straight through — swap meta tensors for local all-zeros
        allocations here, native path behavior unchanged.
        NOTE: a meta tensor still reports full nbytes (numel * element_size);
        do not infer transport cost from nbytes."""
        for new_req in scheduler_output.scheduled_new_reqs:
            embeds = new_req.prompt_embeds
            if embeds is not None and embeds.is_meta:
                new_req.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_states(scheduler_output)

    def _update_streaming_request(self, req_id: str, new_req_data: NewRequestData) -> CachedRequestState:
        """LWD: meta placeholder materialization for streaming updates
        (same as _update_states, zero base seams)."""
        embeds = new_req_data.prompt_embeds
        if embeds is not None and embeds.is_meta:
            new_req_data.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_streaming_request(req_id, new_req_data)

    def _init_mrope_positions(self, req_state: CachedRequestState) -> None:
        """LWD prompt-embeds-only: text-only M-RoPE positions do not depend
        on the ids content (the scan only honors vision markers), so build
        them directly from the prompt length; with ids present, fall back
        to the native scan (zero base seams)."""
        if req_state.prompt_token_ids is None:
            n = req_state.num_prompt_tokens
            req_state.mrope_positions = torch.arange(n).unsqueeze(0).expand(3, n).clone()
            req_state.mrope_position_delta = 0
            return
        super()._init_mrope_positions(req_state)

    @property
    def supports_mm_inputs(self) -> bool:
        """LWD: present injected batches as "no MM this step" so the native
        MM gate skips the encoder naturally (recomputed embeddings would
        wholesale-overwrite the remotely injected values); decode and
        text-only steps return the model's true value. Zero base seams."""
        input_batch = getattr(self, "input_batch", None)
        if input_batch is not None and input_batch.req_prompt_embeds:
            return False
        return getattr(self, "_model_supports_mm_inputs", False)

    @supports_mm_inputs.setter
    def supports_mm_inputs(self, value: bool) -> None:
        # The base __init__ assigns directly; keep the backing for the dynamic getter.
        self._model_supports_mm_inputs = value

    def _lwd_preprocess_prompt_embeds(
        self, scheduler_output: SchedulerOutput, num_input_tokens: int
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Consume the corrected device mask on the LWD cloud decoder path."""
        count = scheduler_output.total_num_scheduled_tokens
        is_token_ids = self.is_token_ids.gpu[:count]
        # Remote prompt rows do not carry meaningful token IDs.
        token_ids = self.input_ids.gpu[:count].masked_fill(~is_token_ids, 0)
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        embeds = self.inputs_embeds.gpu[:count]
        embeds.copy_(torch.where(is_token_ids.unsqueeze(-1), tokens_to_embeds, embeds))

        # Match the parent's position layout and clear padding for graph replay.
        # The retired base exposed an xdrope-dimension attribute here; this
        # base splits positions two-way (mrope/plain) exactly like its own
        # dummy run, so xdrope models follow whichever layout the base uses.
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > count:
                self.positions[count:num_input_tokens].zero_()

        # DSV4-only: the upstream model-layer fail-fast requires a non-None ids
        # tensor on hash MoE layers, but LWD keeps real token IDs off the cloud
        # entirely. Nothing true exists here, so the placeholder is
        # information-free by construction (all zeros): the gate below and the
        # remote expert-table publication in _model_forward key off the same
        # lwd_hash_state, so the router always swaps the tensor for row
        # indices before any read -- the values are dead on arrival. Other
        # models keep the native None unchanged.
        model_input_ids: torch.Tensor | None = None
        if self.lwd_hash_state is not None:
            model_input_ids = self._lwd_hash_placeholder_ids[:num_input_tokens]
        return (
            model_input_ids,
            self.inputs_embeds.gpu[:num_input_tokens],
            positions,
            None,
            self._init_model_kwargs(),
            None,
        )

    def _lwd_refresh_decode_embedding_mask(self, scheduler_output: SchedulerOutput) -> None:
        """Rebuild the mask from actual device positions so decode rows never
        reuse stale embeddings.

        Upload per-request prompt lengths asynchronously and expand them
        to token rows via the on-device request indices. The mask is
        written straight into the device buffer and consumed on-device
        by the LWD prompt-embeds path; positions are not round-tripped
        and the CPU mask is not refreshed.
        """
        num_reqs = len(self.input_batch.req_ids)
        count = scheduler_output.total_num_scheduled_tokens
        prompt_lens_gpu = self._lwd_prompt_lens_gpu
        assert prompt_lens_gpu is not None
        prompt_lens_gpu[:num_reqs].copy_(
            self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        prompt_lens_rows = prompt_lens_gpu[self.req_indices.gpu[:count]]
        self.is_token_ids.gpu[:count] = self.positions[:count] >= prompt_lens_rows

    # Clear the real-batch marker before cloud-side dummy forwards, so the
    # previous step's routing is never reused.
    def _dummy_run(self, *args, **kwargs) -> None:
        """Clear the real-batch Hash routing marker, then run the dummy forward."""
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Shard embeddings per the SP layout and publish the full Hash routing data into the forward context."""
        context = get_forward_context()
        # flash_comm_v1_enabled is only set by the SP fast path; without SP
        # (the LWD 1+8 TP layout) it is absent and the sharding is skipped.
        if (
            inputs_embeds is not None
            and intermediate_tensors is None
            and getattr(context, "flash_comm_v1_enabled", False)
            and not self.supports_mm_inputs
        ):
            # Passing embeddings directly bypasses the native embedding layer; re-apply the sequence split here.
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                if inputs_embeds.shape[0] != num_tokens_padded:
                    raise ValueError("LWD SP expects full-token inputs_embeds before slicing")
                if context.pad_size:
                    inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, context.pad_size))
                inputs_embeds = inputs_embeds.chunk(tp_group.world_size, dim=0)[tp_group.rank_in_group]
            # Positions, attention metadata and the Hash routing tables keep the full-token layout.
        if self.lwd_hash_state is not None:
            # Initialize the routing index cache first so padded rows use the legal zero index.
            self._lwd_hash_token_ids.zero_()
            if self._lwd_hash_step_tokens is None:
                # Synthetic forwards have no remote prompt; route dummy token 0.
                self._lwd_hash_prompt_mask.zero_()
            else:
                count = self._lwd_hash_step_tokens
                if count > num_tokens_padded:
                    raise ValueError("LWD Hash routing rows exceed model input rows")
                # Native input preparation retains real decode IDs even when
                # model input_ids=None because the model consumes embeddings.
                self._lwd_hash_token_ids[:count].copy_(self.input_ids.gpu[:count])
            context.input_ids = self._lwd_hash_token_ids[:num_tokens_padded]
            context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens_padded]
            context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens_padded]
            if input_ids is None:
                # Covers the dummy/profile run: with enable_prompt_embeds it
                # also takes the inputs_embeds branch (input_ids=None), and
                # the model-layer hash fail-fast rejects None before the
                # router's dummy-token-0 path can run. Same contract as the
                # request-side placeholder: all zeros, never read (the
                # router swaps in row indices).
                input_ids = self._lwd_hash_placeholder_ids[:num_tokens_padded]
        return super()._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    def _lwd_hash_dims(self) -> tuple[int, int]:
        """(num_hash_layers, experts_per_tok), cached at first use."""
        dims = self._lwd_hash_dims_cached
        if dims is None:
            config = self.model_config.hf_config
            dims = (
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            self._lwd_hash_dims_cached = dims
        return dims

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        if self.lwd_hash_state is not None:
            for req_id in list(self.lwd_hash_state.requests):
                request = self.requests.get(req_id)
                if request is None or request.num_computed_tokens >= request.num_prompt_tokens:
                    self.lwd_hash_state.discard(req_id)
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        """Release collection bookkeeping for finished requests (sole owner of
        the release path): the Hash chunk table plus prompt-embeds assembly
        buffers. Called by the worker on the engine's finished_req_ids signal
        (backstop for prefill aborted mid-way; normal prefill completion is
        already released in _lwd_release_consumed_prompt_embeds)."""
        embeds_map = self.input_batch.req_prompt_embeds
        req_id_to_index = self.input_batch.req_id_to_index
        for req_id in finished_req_ids:
            if self.lwd_hash_state is not None:
                self.lwd_hash_state.discard(req_id)
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _lwd_harvest_up_batches(
        self, num_scheduled_tokens: np.ndarray
    ) -> list[tuple[int, LwdEmbedBatch, torch.Tensor, list[tuple[str, int, int]]]]:
        """UP batch harvest (called before super()._prepare_inputs; timing
        contract in _prepare_inputs). Three stages interleave per seqno with
        an event order that matches the pre-split single loop point for
        point: length check (_lwd_stage_up_batch) -> endpoint host gate +
        TP broadcast (_lwd_harvest_tensors_one) -> window check + mrope
        write (_lwd_check_up_windows).

        Returns [(seqno, meta, up_flat, rows)] so the post-super() inject
        reuses the validated batch positions. Moving the tensor stage
        wholesale after super() to overlap host work is deferred item D1
        (see docs/refactor/lwd_worker_packages_refactor_plan.md); at that
        point only the _lwd_harvest_tensors_one call site leaves this
        loop."""
        worker = self.worker
        if worker is None:
            return []
        posted = getattr(worker, "_lwd_up_recv_futures", None)
        if not posted:
            return []
        harvested = []
        for batch_seqno in list(posted.keys()):
            item = worker._lwd_up_recv_futures.pop(batch_seqno, None)
            if item is None:
                continue
            future, meta = item
            staged = self._lwd_stage_up_batch(batch_seqno, future, meta)
            up_flat = self._lwd_harvest_tensors_one(staged)
            if up_flat is None:
                continue
            rows = self._lwd_check_up_windows(
                batch_seqno,
                meta,
                num_scheduled_tokens,
            )
            harvested.append((batch_seqno, meta, up_flat, rows))
        return harvested

    def _lwd_stage_up_batch(
        self,
        batch_seqno: int,
        future: LwdTransferFuture | None,
        meta: LwdEmbedBatch,
    ) -> LwdUpStagedBatch:
        """Harvest staging: compute the batch token count and check the meta
        length contract (sat before the host gate pre-split; order kept).
        The probe start rides on the record so the tensor stage's
        up-recv-wait window matches the pre-split one."""
        t_wait = time.monotonic()
        num_tokens = sum(len(t) for t in meta.token_ids)
        if not (
            len(meta.req_ids)
            == len(meta.token_ids)
            == len(meta.prompt_offsets)
            == len(meta.has_mrope)
            == len(meta.mrope_positions)
        ):
            raise ValueError("LWD requires one prompt offset and mrope flag per request")
        return LwdUpStagedBatch(
            seqno=batch_seqno,
            meta=meta,
            future=future,
            num_tokens=num_tokens,
            t_wait=t_wait,
        )

    def _lwd_check_up_windows(
        self,
        batch_seqno: int,
        meta: LwdEmbedBatch,
        num_scheduled_tokens: np.ndarray,
    ) -> list[tuple[str, int, int]]:
        """Per-request window check + mrope row write; returns one
        (req_id, batch index, row count) per row.

        The three fail-fasts match the pre-split code verbatim; proceeding
        would decode on uninjected dirty embeds (silent corruption), so
        each is exposed on the spot. Since wire_version 2, mrope
        positions arrive via the control plane with the meta (per-request
        [3,n] int64 bytes, frombuffer written straight into req_state
        with zero transposition), no longer through data-plane aux frames
        and a second TP broadcast; text-only requests carry empty bytes."""
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        mrope_rows = 0
        _mrope_cpu_ms = _mrope_delta_ms = 0.0
        rows: list[tuple[str, int, int]] = []
        for k, (req_id, token_ids) in enumerate(zip(meta.req_ids, meta.token_ids)):
            n = len(token_ids)
            idx = self.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                # Fail-fast: the embeds arrived but the request is not in
                # this step's batch — proceeding would decode it on
                # uninjected dirty embeds (silent corruption); expose on
                # the spot instead of dropping. Seeing this means the
                # scheduler/data-plane timing contract is broken; the fix
                # belongs upstream, not here.
                raise RuntimeError(
                    f"[Lwd][cloud-runner] INJECT req={req_id} seqno="
                    f"{batch_seqno} rows={n}: req not in input_batch "
                    f"(batch={self.input_batch.req_ids})"
                )
            if n > 0:
                # Fail-fast: the chunk row count must equal exactly the
                # request's scheduled token count this step. When it does
                # not (scheduled window != edge chunk, e.g. budget-squeeze
                # truncation), injection would overrun into neighbor rows
                # or leave an uninjected tail — a silent-corruption source;
                # expose directly.
                scheduled = int(num_scheduled_tokens[idx]) if idx < len(num_scheduled_tokens) else 0
                if n != scheduled:
                    raise RuntimeError(
                        f"[Lwd][cloud-runner] INJECT window mismatch "
                        f"req={req_id} seqno={batch_seqno}: chunk rows="
                        f"{n} != scheduled={scheduled} (computed="
                        f"{int(computed[idx])} prompt="
                        f"{int(num_prompt[idx])})"
                    )
                start = int(meta.prompt_offsets[k])
                prompt_len = int(num_prompt[idx])
                if start != int(computed[idx]) or not 0 <= start <= start + n <= prompt_len:
                    raise ValueError(
                        f"LWD prompt chunk offset mismatch for {req_id}: "
                        f"start={start}, computed={int(computed[idx])}, rows={n}, prompt={prompt_len}"
                    )
                if meta.has_mrope[k]:
                    # Positions bytes arrive via the control plane with
                    # the meta ([3,n] int64); frombuffer builds the CPU
                    # tensor and writes req_state with zero transposition;
                    # empty bytes = edge/cloud version mismatch, fail-fast
                    payload = meta.mrope_positions[k]
                    if not payload:
                        raise RuntimeError(
                            f"[Lwd][cloud-runner] mrope payload missing "
                            f"req={req_id} seqno={batch_seqno}: "
                            f"has_mrope=True but control-plane positions "
                            f"are empty (edge/cloud version mismatch; deploy matching versions)"
                        )
                    cpu_ms, delta_ms = self._lwd_inject_mrope_positions(
                        req_id,
                        start,
                        n,
                        torch.frombuffer(bytearray(payload), dtype=torch.int64).view(3, n),
                        int(num_prompt[idx]),
                    )
                    _mrope_cpu_ms += cpu_ms
                    _mrope_delta_ms += delta_ms
                    mrope_rows += n
            rows.append((req_id, idx, n))
        if mrope_rows > 0:
            # [Lwd][perf] mrope inject probe: cpu = positions-byte
            # frombuffer + direct CPU write into req_state (control-plane
            # delivery, no device round-trip); delta = last-chunk max()
            # self-derivation, accumulated per request
            lwd_logger.info(
                "[Lwd][perf] cloud mrope-inject seqno=%s rows=%d cpu=%.2f delta=%.2fms",
                batch_seqno,
                mrope_rows,
                _mrope_cpu_ms,
                _mrope_delta_ms,
            )
        return rows

    def _lwd_harvest_tensors_one(
        self,
        staged: LwdUpStagedBatch,
    ) -> torch.Tensor | None:
        """Harvest tensor stage (single batch): endpoint passes the host gate,
        then an in-place TP-group broadcast lands the main embeds frame
        (each frame is its own launch+wait atomic pair on the current
        stream — the demo_br discipline; unreliable
        cross-stream ordering was the root cause of truncated broadcast
        deliveries before). Returns None when the endpoint res.tensor is
        None (the pre-split continue semantics)."""
        # get_tp_group stays imported inside the function: the v4 migration
        # tests stub parallel_state per call via sys.modules, so the
        # module-level name is invisible in their trimmed namespace.
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        is_endpoint = self.worker._lwd_early_recv_endpoint
        hidden_size = self.model_config.get_hidden_size()
        num_hash_layers, top_k = self._lwd_hash_dims()
        numel = hash_payload_numel(
            staged.num_tokens,
            hidden_size,
            num_hash_layers,
            top_k,
        )
        if is_endpoint:
            # Endpoint: pass the host readiness gate first (the demo's
            # wait gate): once the done_event poll succeeds the P2P data
            # has fully landed in the buffer; a timeout raises explicitly
            # instead of silent corruption. Only after the gate does the
            # broadcast read — guaranteed complete data.
            res = staged.future.wait(timeout=_LWD_UP_HOST_GATE_TIMEOUT_S)
            up_flat = res.tensor
            if up_flat is None:
                return None
        else:
            # Non-endpoint: allocate the receive buffer on the spot and wait for the endpoint's broadcast.
            up_flat = torch.empty(numel, dtype=torch.bfloat16, device="npu")
        if tp.world_size > 1:
            # Same as the demo: reuse the framework TP communication domain
            # (group construction order guaranteed by the framework, verified
            # on the PD-disaggregation path); no self-managed
            # _LWD_EMBED_BCAST_GROUP.
            work = dist.broadcast(
                up_flat,
                src=tp.ranks[0],
                group=tp.device_group,
                async_op=True,
            )
            work.wait()  # Bridge broadcast completion onto the current (compute) stream so later copies are ordered
        lwd_logger.info(
            "[Lwd][perf] cloud up-recv-wait seqno=%s dur=%.2fms",
            staged.seqno,
            (time.monotonic() - staged.t_wait) * 1000,
        )
        lwd_logger.info(
            "[Lwd][cloud-runner] UP embeds harvested seqno=%s rows=%d reqs=%s mrope_reqs=%s",
            staged.seqno,
            staged.num_tokens,
            staged.meta.req_ids,
            sum(staged.meta.has_mrope),
        )
        return up_flat

    def _lwd_inject_remote_embeds(
        self,
        harvested: list[tuple[int, LwdEmbedBatch, torch.Tensor, list[tuple[str, int, int]]]] | None,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        """Device-side inject (after super()): write the harvested UP embeds
        straight into each request's current scheduling window in
        ``inputs_embeds.gpu``.

        The flattened output offsets reproduce the native fill loop's
        accumulation (per-request scheduled segment start), so rows land
        exactly where the prompt-embeds branch expects them.  The CPU
        draft first-pass provider reads the SAME window straight from
        ``inputs_embeds.gpu`` (llm_base_proposer), so the CPU assembly
        buffer is no longer written at all — the per-chunk 84MB D2H is
        gone.  The base fill loop is skipped for LWD placeholder rows
        (see ``_prepare_inputs``); our device write is authoritative.

        Batch positions reuse the rows carried by the harvest stage (the
        duplicated "req not in batch" defense has been consolidated into
        the harvest stage alone; indices here are pre-validated).
        """
        if not harvested:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        hidden_size = self.model_config.get_hidden_size()
        gpu_embeds = self.inputs_embeds.gpu

        # Flattened output offset per request (same accumulation as the
        # native fill loop): each request's scheduling-segment start in the
        # flattened token sequence, indexed by batch position.
        offsets: list[int] = []
        off = 0
        for i in range(len(self.input_batch.req_ids)):
            offsets.append(off)
            off += int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0

        num_hash_layers, top_k = self._lwd_hash_dims()
        for batch_seqno, meta, up_flat, rows in harvested:
            num_tokens = sum(len(t) for t in meta.token_ids)
            expert_ids = None
            if num_hash_layers:
                embeds, expert_ids = unpack_hash_payload(
                    up_flat,
                    num_tokens,
                    hidden_size,
                    num_hash_layers,
                    top_k,
                )
            else:
                embeds = up_flat.view(num_tokens, hidden_size)
            lwd_logger.info(
                "[Lwd][cloud-runner] UP embeds injected seqno=%s rows=%s reqs=%s",
                batch_seqno,
                embeds.shape[0],
                meta.req_ids,
            )
            row = 0
            for meta_idx, (req_id, idx, n) in enumerate(rows):
                assert idx >= 0, f"LWD inject index missing for {req_id}"
                if n > 0:
                    start = int(meta.prompt_offsets[meta_idx])
                    if self.lwd_hash_state is not None:
                        if expert_ids is None:
                            raise ValueError(f"Missing LWD V4 expert payload for {req_id}")
                        self.lwd_hash_state.add_chunk(
                            req_id,
                            int(num_prompt[idx]),
                            start,
                            expert_ids[row : row + n],
                        )
                    out = offsets[idx]
                    # Write the scheduling window of inputs_embeds.gpu directly on-stream
                    gpu_embeds[out : out + n].copy_(embeds[row : row + n], non_blocking=True)
                    # No D2H write into the CPU assembly buffer anymore:
                    # the draft provider reads the same window straight from
                    # inputs_embeds.gpu (this write is the authoritative
                    # copy), eliminating the per-chunk 84MB synchronous
                    # D2H tail
                    self.input_batch.is_token_ids[idx, start : start + n] = False
                row += n
            # Cross-stream lifetime registration: the endpoint recv buffer
            # is allocated and recycled by the channel stream (with
            # same-size chunks the allocator nearly always hands back the
            # same block), while this step's copy runs on the compute
            # stream. Without record_stream the allocator could hand the
            # block to the next seqno's irecv before this batch's copy has
            # executed — under deep async queues this shows up as
            # cross-seqno data bleeding (multi-request corruption).
            # Registration orders the reuse correctly.
            up_flat.record_stream(torch.npu.current_stream())

    # ------------------------------------------------------------------ #
    # Collect probe (measurement only; output discarded, protocol        #
    # still returns token_ids directly)                                   #
    # ------------------------------------------------------------------ #

    def _lwd_inject_mrope_positions(
        self,
        req_id: str,
        start: int,
        n: int,
        chunk_positions: torch.Tensor,
        prompt_len: int,
    ) -> tuple[float, float]:
        """Write one chunk's mrope positions ([3,n] int64, CPU — frombuffer
        of the control-plane RangeItem.mrope_positions bytes) into the
        request's cached [3, prompt] window; when the last chunk lands,
        self-derive delta (= positions.max()+1-prompt_len, value-identical
        to the edge side) for the native _calc_mrope_positions to compute
        completion-segment positions during decode.

        req_state.mrope_positions is initialized pass-through with
        ids=None (arange text-only positions) — multimodal requests have
        their window overwritten chunk by chunk; text-only requests never
        take this path (the edge sends them no positions).

        Returns (cpu_ms, delta_ms): timings of the direct CPU write and
        the last-chunk delta derivation, aggregated per seqno by the
        caller."""
        req_state = self.requests.get(req_id)
        assert chunk_positions.shape == (3, n), (
            f"mrope payload shape {tuple(chunk_positions.shape)} != (3, {n}) (req={req_id})"
        )
        _t_cpu = time.monotonic()
        req_state.mrope_positions[:, start : start + n] = chunk_positions
        cpu_ms = (time.monotonic() - _t_cpu) * 1000
        delta_ms = 0.0
        if start + n >= prompt_len:
            positions = req_state.mrope_positions[:, :prompt_len]
            req_state.mrope_position_delta = int(positions.max().item()) + 1 - prompt_len
            lwd_logger.info(
                "[Lwd][cloud-runner] mrope complete: req=%s prompt=%d delta=%d",
                req_id,
                prompt_len,
                req_state.mrope_position_delta,
            )
        return cpu_ms, delta_ms

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput:
        from vllm.distributed.parallel_state import get_tp_group

        # Only the cloud TP-group first rank (= the DOWN channel endpoint
        # rank) collects/sends; the other ranks have no channel peer, so
        # their collect-and-discard is skipped too (8-card redundancy).
        wire_endpoint = get_tp_group().is_first_rank
        captured = None
        if wire_endpoint and self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # Snapshot the batch order/mask as well: with overlapping
            # batch queues, the next step's _prepare_inputs may reorder
            # input_batch before the collect; reading live state would
            # slice batch-A tensors with batch-B order (the root cause of
            # high-concurrency misalignment)
            captured = (
                state[5],
                state[1],
                state[2],
                state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        if captured is not None and self._lwd_captured_sampler_output is not None:
            self._lwd_pending_down_payload = self._lwd_collect_down_payload(
                captured[0],
                captured[1],
                captured[2],
                captured[3],
                captured[4],
                captured[5],
                self._lwd_captured_sampler_output,
            )
        return output

    def take_lwd_pending_down_payload(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, None, list[str], None] | None:
        """The worker layer takes away this step's DOWN payload (single-slot overwrite, consumed every step)."""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        scheduler_output: SchedulerOutput,
        batch_req_ids: list[str],
        discard_mask_np: np.ndarray,
        sampler_output: SamplerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int] | None, torch.npu.Event, list[str]] | None:
        """Production DOWN collection (rank-replay): hidden packaging + global
        ranks + num_accepted.

        Orchestration: row selection / rank computation branches on spec
        vs non-spec (_lwd_pack_plain_down_rows / _lwd_pack_spec_down_rows).
        The step meta stays on device (no pinned D2H here — the side-stream
        copy was not fenceable on this platform); a per-step ready event is
        recorded on the main stream, and the fenced harvest (dedicated
        stream + wait_event + same-stream done event) runs in the worker's
        sample_tokens, off this collect. Zero added synchronization on the
        compute critical path; batch order/mask come from the capture-time
        snapshot (guards against batch-queue overlap reordering).
        Returns:
          (hidden_packet, meta_dev, seg_lens_host, ready_event, req_ids)
        or None (no in-flight requests this step).
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return None
        # batch_req_ids/discard_mask come from the capture-time snapshot
        # (same batch order as logits/hidden); the live input_batch is
        # never read here
        valid = ~discard_mask_np[: len(batch_req_ids)]
        if spec_decode_metadata is None:
            packed = self._lwd_pack_plain_down_rows(
                sample_hidden_states,
                logits,
                sampled,
                valid,
                batch_req_ids,
            )
        else:
            packed = self._lwd_pack_spec_down_rows(
                sample_hidden_states,
                logits,
                sampled,
                spec_decode_metadata,
                valid,
                batch_req_ids,
            )
        if packed is None:
            return None
        hidden_packet, ranks_list, counts_dev, seg_lens_host = packed
        if seg_lens_host is not None:
            meta_dev = torch.cat(ranks_list + [counts_dev])
            # seg_lens stay off-device (host writes the pinned tail
            # segment): disjoint from the D2H prefix region, and the
            # engine reads the pinned buffer only after the ready_event
            # sync (later still) — no race.
        else:
            n_req = sum(1 for i in range(len(batch_req_ids)) if valid[i])
            counts_dev = torch.ones(n_req, dtype=torch.int32, device=logits.device)
            meta_dev = torch.cat(ranks_list + [counts_dev, counts_dev])
        selection_done = time.monotonic()
        # Per-step ready event: recorded on the main stream before the
        # payload leaves the collect. The fenced harvest (dedicated stream
        # + wait_event + same-stream done event) lives in the worker's
        # sample_tokens, NOT here — keeping the compute critical path free
        # of any synchronization. The device tensor travels with the
        # payload; the pinned D2H + shared-event scheme is retired (it
        # could not fence the side-stream copy on this platform).
        ready_event = torch.npu.Event()
        ready_event.record()
        lwd_logger.info(
            "[Lwd][perf] collect rows=%d rank=%.2f total=%.2fms",
            hidden_packet.shape[0],
            (selection_done - _t0) * 1000,
            (time.monotonic() - _t0) * 1000,
        )
        return (
            hidden_packet,
            meta_dev,
            seg_lens_host,
            ready_event,
            [r for i, r in enumerate(batch_req_ids) if valid[i]],
        )

    def _lwd_pack_plain_down_rows(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """Non-spec row selection and rank computation (correspondence:
        equivalent to the not-is_spec branch of the original
        _lwd_collect_down_payload).

        Cheap computation: direct bf16 comparison (logits are natively
        bf16, bit-identical to a cast) and full-batch rows computed
        wholesale when fully covered (no fancy-index gathers)."""
        idx = [i for i in range(len(batch_req_ids)) if valid[i]]
        if not idx:
            return None
        full_cover = all(valid[: len(batch_req_ids)])
        lg_sel = logits if full_cover else logits[idx]
        sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
        thresh = lg_sel.gather(1, sm_sel.long().unsqueeze(1))
        ranks = [(lg_sel > thresh).sum(dim=1).to(torch.int32)]
        hidden_packet = torch.cat([sample_hidden_states[i : i + 1] for i in idx])
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=None,
            seg_lens_host=None,
        )

    def _lwd_pack_spec_down_rows(
        self,
        sample_hidden_states: torch.Tensor,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """All-zero synchronous spec row selection and rank computation
        (vectorized; correspondence: equivalent to the is_spec branch of
        the original _lwd_collect_down_payload).

        Segment lengths derive from the host-side
        scheduled_spec_decode_tokens (1+draft_len); rows are packed as
        whole segments (rejected rows included; the edge takes the valid
        prefix by num_accepted). Lengths come from
        spec_decode_metadata.num_draft_tokens (host list, the real
        runtime layout, matching the sample_hidden_states segment
        structure), NOT from
        scheduler_output.scheduled_spec_decode_tokens (scheduler input
        that may disagree with what actually ran). counts derive from
        the same sampled tensor on-device (same source as ranks/rows,
        naturally batch-position aligned); seg_lens/counts only include
        valid requests so [ranks|counts|seg_lens] stay length-aligned.
        One host pass builds the indices, replacing the per-request
        device loop (the original issued ~6 small kernels per request
        and the rank segment grew past 20ms with concurrency; after
        vectorization the launch count is independent of N). Index
        semantics are bitwise-equivalent to the original loop:
        vidx/vlen = valid request batch position / actual row count
        (capped by total segment length, = the original rows_i);
        req_idx/pos_idx = per-row request position / in-segment offset
        (the original sampled[i, :rows_i]); row_sel = global row number
        of each valid row."""
        seg_lens = [d + 1 for d in spec_decode_metadata.num_draft_tokens]
        counts_all = (sampled != -1).sum(dim=1)
        seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
        seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
        total_rows = sample_hidden_states.shape[0]
        rows_per_req = np.minimum(seg_lens_np, np.maximum(total_rows - seg_starts, 0))
        vidx_np = np.nonzero(valid[: len(batch_req_ids)] & (rows_per_req >= 1))[0].astype(np.int64)
        if vidx_np.size == 0:
            return None
        vlen_np = rows_per_req[vidx_np].astype(np.int64)
        starts_v = seg_starts[vidx_np].astype(np.int64)
        n_rows = int(vlen_np.sum())
        req_idx_np = np.repeat(vidx_np, vlen_np)
        pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(np.cumsum(vlen_np) - vlen_np, vlen_np)
        row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
        # All indices go through one pinned async H2D (a pageable
        # synchronous H2D would stall the main stream waiting for the
        # device to drain — the same lesson as seg_lens)
        buf = self._lwd_h2d_stage_i64(np.concatenate([req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]))
        n_valid = vidx_np.size
        req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
        sel_d = buf[2 * n_rows : 3 * n_rows]
        vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
        vlen_d = buf[3 * n_rows + n_valid :]
        sampled_rows = sampled[req_d, pos_d]
        if n_rows == total_rows:
            # All valid with no capping: rows are a contiguous prefix; skip the gather
            lg_rows = logits[:n_rows]
            hidden_packet = sample_hidden_states[:n_rows]
        else:
            lg_rows = logits[sel_d]
            hidden_packet = sample_hidden_states[sel_d]
        thresh = lg_rows.gather(1, sampled_rows.long().unsqueeze(1))
        ranks = [(lg_rows > thresh).sum(dim=1).to(torch.int32)]
        # Stale sampled slots (requests the spec step skipped may retain
        # the previous spec step's tokens): accepted/ranks/seg_lens are
        # all capped by the actual hidden row count — the row count is
        # the ground truth of real sampling positions
        counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(torch.int32)
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=counts_dev,
            seg_lens_host=vlen_np.tolist(),
        )

    def _lwd_h2d_stage_i64(self, arr: np.ndarray) -> torch.Tensor:
        """Async H2D staging for int64 host index arrays (pinned 4-slot
        rotation + non_blocking): replaces torch.tensor(arr, device=...)
        pageable synchronous copies — those stall the main stream waiting
        for the device to drain (under MTP, the draft-propose tail).
        Ring depth rationale mirrors the meta ring: cover batch_queue
        depth + engine lag so a later host overwrite cannot land while
        the previous copy is still queued.
        Correspondence: LwdPinnedStaging.stage_to_device ≍ the original
        hand-rolled pinned+device 4-slot pair rotation (capacity floor
        1024 unchanged)."""
        return self._lwd_staging(
            "down_index",
            dtype=torch.int64,
            min_capacity=1024,
        ).stage_to_device(arr, arr.size)
