# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Edge-side multimodal cache (moved verbatim from LwdEdgeWorker).

Two registries with different lifecycles: per-request feature
bookkeeping (position info; released when the request's chunks pass all
MM windows or the request finishes) and content-hash shared encoder
outputs (FIFO eviction, NOT request-scoped — the first reference-count
design evicted shared entries before later requests arrived and the
same image was re-encoded 30-way under concurrency).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.core.sched.output import LwdEmbedBatch

# mm_hash encode cache capacity (entries; one 1080p item is about 26MB HBM)
_LWD_MM_HASH_CACHE_MAX_ITEMS = 32


class LwdEdgeMmCache:
    """Per-request MM feature bookkeeping + hash-shared encoder outputs."""

    def __init__(self) -> None:
        self._features: dict[str, list[MultiModalFeatureSpec]] = {}
        self._hash_embeds: dict[str, object] = {}

    def register(self, req_id: str, feats: list[object]) -> None:
        """Record a new request's MM features (first chunk carries data)."""
        self._features[req_id] = feats

    def snapshot_features(self, req_ids: list[str]) -> dict[str, list[MultiModalFeatureSpec] | None]:
        return {req_id: self._features.get(req_id) for req_id in req_ids}

    def restore_features(self, snapshot: dict[str, list[MultiModalFeatureSpec] | None]) -> None:
        for req_id, features in snapshot.items():
            if features is None:
                self._features.pop(req_id, None)
            else:
                self._features[req_id] = features

    def release_finished(self, finished_req_ids: list[str]) -> None:
        """Completion backstop: when a request finishes (abort included),
        drop its mm feature/position registration (the healthy path already
        removed it early once the chunk window passed all mm segments)."""
        for req_id in finished_req_ids:
            self.release_request(req_id)

    def _features_in_window(self, req_id: str, p_offset: int, n: int) -> list[MultiModalFeatureSpec]:
        chunk_end = p_offset + n
        return [
            feature
            for feature in self._features.get(req_id) or []
            if p_offset < feature.mm_position.offset + feature.mm_position.length
            and feature.mm_position.offset < chunk_end
        ]

    def encode_missing(
        self,
        batch_meta: LwdEmbedBatch,
        model: torch.nn.Module,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        """Encode cache misses only for features intersecting this chunk.

        Cross-request batching mirrors the centralized _execute_mm_encoder
        (one forward for the whole batch instead of N per-request calls):
        under eager ViT the fixed per-call overhead (per-op dispatch, H2D,
        syncs) dominates, and serializing per request would pay it N times
        — those serial calls also sit inside the worker's embed execution
        and block later unembed deliveries. Cache key (identifier) and
        capacity-eviction semantics are unchanged; backfill order =
        collection order (req_ids order × per-request feature order), and
        the batched outputs are consumed in the same order — any
        misordering misattributes embeddings, with a total-count assert as
        the backstop."""
        missing_features = []
        missing_data = []
        for req_id, token_ids, p_offset in zip(batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets):
            for f in self._features_in_window(req_id, p_offset, len(token_ids)):
                if f.identifier not in self._hash_embeds:
                    if f.data is None:
                        # Without data the image embeds cannot be computed —
                        # silently skipping would embed image tokens as text;
                        # fail fast.
                        missing_data.append((req_id, f.identifier))
                    else:
                        missing_features.append(f)
        if missing_data:
            raise RuntimeError(
                f"[LWD] mm feature data missing: {missing_data}; "
                "cannot compute image embeddings; "
                "check mm processor/receiver cache chain"
            )
        if not missing_features:
            return
        _t_enc = time.monotonic()
        from vllm.multimodal.utils import group_and_batch_mm_kwargs

        mm_kwargs = [(f.modality, f.data) for f in missing_features]
        outputs_by_modality: dict[str, list[torch.Tensor]] = {}
        num_outputs = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=device,
            pin_memory=pin_memory,
        ):
            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
            assert len(batch_outputs) == num_items, (
                f"encoder outputs {len(batch_outputs)} != items {num_items} (modality={modality})"
            )
            outputs_by_modality.setdefault(modality, []).extend(batch_outputs)
            num_outputs += len(batch_outputs)
        assert num_outputs == len(missing_features), (
            f"encoder outputs {num_outputs} != missing features {len(missing_features)}"
        )
        for feature in missing_features:
            self._hash_embeds[feature.identifier] = outputs_by_modality[feature.modality].pop(0)
        # Capacity eviction (FIFO): drop the oldest entry past capacity; an evicted item is at most re-encoded once
        while len(self._hash_embeds) > _LWD_MM_HASH_CACHE_MAX_ITEMS:
            self._hash_embeds.pop(next(iter(self._hash_embeds)))
        # [Lwd][perf] Vision tower duration (batch level, cache misses only; same-image reuse never re-encodes)
        lwd_logger.info(
            "[Lwd][perf] mm-encoder reqs=%d items=%d dur=%.2fms",
            len(batch_meta.req_ids),
            len(missing_features),
            (time.monotonic() - _t_enc) * 1000,
        )

    def encoder_outputs(self, req_id: str, p_offset: int, n: int) -> list[tuple[PlaceholderRange, torch.Tensor]]:
        """Read cached encoder outputs for features intersecting this chunk.

        Encoding moved up to the batch level (encode_missing, one grouped
        batch); this reads the cache only — a miss here means the batch
        encode skipped it, so fail fast. Encoded outputs are shared across
        requests by content hash (mm_hash/identifier) — the same
        granularity as the centralized encoder cache, so same-image reuse
        never re-encodes; position info (mm_position) stays per-request,
        only the encoded outputs are shared; capacity eviction is FIFO and
        does not follow request lifetime. release_request drops the
        request's positional features when the chunk window passes all mm
        segments or on the finished_req_ids backstop (the encode cache is
        untouched)."""
        features = self._features_in_window(req_id, p_offset, n)
        if not features:
            return []
        missing = [f.identifier for f in features if f.identifier not in self._hash_embeds]
        if missing:
            raise RuntimeError(f"[LWD] mm embeds missing from batch encode: req={req_id} (mm_hashes={missing})")
        return [(f.mm_position, self._hash_embeds[f.identifier]) for f in features]

    def release_request(self, req_id: str) -> None:
        """Release the request's mm positional features; hash-encoded outputs
        are not released with the request (capacity eviction, see
        encoder_outputs)."""
        self._features.pop(req_id, None)

    def gather_chunk(
        self,
        batch_meta: LwdEmbedBatch,
        model: torch.nn.Module,
        device: torch.device,
        pin_memory: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Collect mm embeds rows for this chunk window + build the
        is_multimodal mask.

        Encoding completes at the batch level before slicing (one grouped
        batch, see encode_missing); afterwards each request only slices
        the read-only cache. Window semantics match the native
        _gather_mm_embeddings verbatim: intersect the feature's
        placeholder span [offset, offset+length) with the chunk window
        [p_offset, p_offset+n), slice embeds rows and set mask bits over
        the intersection; images spanning chunks are collected segment by
        segment. Requests whose window passed all mm segments have their
        features removed along the way."""
        total = sum(len(t) for t in batch_meta.token_ids)
        is_mm = torch.zeros(total, dtype=torch.bool)
        mm_embeds: list[torch.Tensor] = []
        if len(batch_meta.prompt_offsets) != len(batch_meta.req_ids):
            return mm_embeds, is_mm
        self.encode_missing(batch_meta, model, device, pin_memory)
        req_start = 0
        for req_id, token_ids, p_offset in zip(batch_meta.req_ids, batch_meta.token_ids, batch_meta.prompt_offsets):
            n = len(token_ids)
            features = self._features.get(req_id) or []
            max_end = max((f.mm_position.offset + f.mm_position.length for f in features), default=0)
            cached = self.encoder_outputs(req_id, p_offset, n)
            for pos_info, item_embeds in cached:
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length
                start_idx = max(p_offset - start_pos, 0)
                end_idx = min(p_offset - start_pos + n, num_encoder_tokens)
                if start_idx >= end_idx:
                    continue
                req_start_pos = req_start + start_pos - p_offset
                if (is_embed := pos_info.is_embed) is not None:
                    s, e = pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                    if s == e:
                        continue
                    rows = item_embeds[s:e]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] |= is_embed[start_idx:end_idx]
                else:
                    rows = item_embeds[start_idx:end_idx]
                    is_mm[req_start_pos + start_idx : req_start_pos + end_idx] = True
                mm_embeds.append(rows)
            if features and p_offset + n >= max_end:
                self.release_request(req_id)
            req_start += n
        return mm_embeds, is_mm
