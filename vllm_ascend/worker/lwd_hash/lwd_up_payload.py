# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Segmented layout of the LWD UP payload.

One prefill batch travels UP as a single 1-D bf16 tensor:

    [embeds | hash-MoE expert-id bytes | Engram bucket-index bytes]

Embeddings are `[T, hidden_size]` bf16. The integer segments are int32
tensors whose bytes are carried one per bf16 element: every byte value
0..255 is exactly representable in bf16, so the round trip is lossless.
A numeric bf16 cast would round ids above 256, and reinterpreting the int32
bits as bf16 would produce NaN patterns that later float ops may canonicalize.
A segment is present iff its dimensions are positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.transformers_utils.config import get_hf_text_config

from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_LWD_INT32_BYTES = torch.tensor([], dtype=torch.int32).element_size()


def lwd_engram_payload_dims(hf_text_config: PretrainedConfig) -> tuple[int, int]:
    """(num_engram_layers, num_hash_cols) of the Engram UP segment.

    `(0, 0)` for a config without Engram layers. The column count is
    `(max_ngram_size - 1) * n_heads`, the n-gram sizes 2..max times heads.
    """
    layer_ids = getattr(hf_text_config, "engram_layer_ids", None)
    if not layer_ids:
        return 0, 0
    num_cols = (hf_text_config.engram_max_ngram_size - 1) * hf_text_config.engram_n_heads
    return len(layer_ids), num_cols


def _lwd_encode_int32(ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """int32 ids -> one exactly representable float element per byte.

    The uint8 view moves to `device` before widening, so a CPU source
    crosses H2D at one byte per element rather than two.
    """
    return ids.to(torch.int32).contiguous().view(torch.uint8).flatten().to(device).to(dtype)


def _lwd_decode_int32(segment: torch.Tensor) -> torch.Tensor:
    """Inverse of `_lwd_encode_int32` on the segment's device."""
    return segment.to(torch.uint8).contiguous().view(torch.int32)


def _lwd_check_int_segment(name: str, ids: torch.Tensor | None, shape: tuple[int, ...]) -> None:
    present = shape[-1] > 0 and shape[-2] > 0
    if not present:
        if ids is not None:
            raise ValueError(f"[LWD] UP payload layout has no {name} segment, got {name} {tuple(ids.shape)}")
        return
    if ids is None:
        raise ValueError(f"[LWD] UP payload layout requires a {name} segment of shape {shape}")
    if tuple(ids.shape) != shape:
        raise ValueError(f"[LWD] UP {name} shape {tuple(ids.shape)} != expected {shape}")


@dataclass(frozen=True)
class LwdUpPayloadLayout:
    """Segment dimensions of one model's UP payload; both sides build the same one.

    The edge builds it from its embedding size, its tid2eid tables and its
    `LwdEngramHasher`; the cloud from its own config. HCCL P2P needs both ends
    to agree on `numel`, so a mismatch shows up as a size error at unpack.
    """

    hidden_size: int
    num_hash_layers: int = 0
    """DeepSeek V4 hash-MoE layers."""
    hash_top_k: int = 0
    num_engram_layers: int = 0
    """DeepSeek V4.1 Engram layers."""
    num_engram_cols: int = 0

    def __post_init__(self) -> None:
        dims = (self.num_hash_layers, self.hash_top_k, self.num_engram_layers, self.num_engram_cols)
        if self.hidden_size <= 0 or any(dim < 0 for dim in dims):
            raise ValueError(f"[LWD] invalid UP payload layout {self}")
        if (self.num_hash_layers > 0) != (self.hash_top_k > 0):
            raise ValueError(f"[LWD] UP hash-MoE segment needs both layers and top_k, got {self}")
        if (self.num_engram_layers > 0) != (self.num_engram_cols > 0):
            raise ValueError(f"[LWD] UP Engram segment needs both layers and columns, got {self}")

    @classmethod
    def from_hf_config(cls, hidden_size: int, hf_config: PretrainedConfig) -> LwdUpPayloadLayout:
        """The layout the cloud expects, derived from its own config.

        Hash MoE is DeepSeek V4 only (`hash_layer_count`); Engram comes from
        the text config's `engram_layer_ids`.
        """
        num_hash_layers = hash_layer_count(hf_config)
        num_engram_layers, num_engram_cols = lwd_engram_payload_dims(get_hf_text_config(hf_config))
        return cls(
            hidden_size=hidden_size,
            num_hash_layers=num_hash_layers,
            hash_top_k=hf_config.num_experts_per_tok if num_hash_layers else 0,
            num_engram_layers=num_engram_layers,
            num_engram_cols=num_engram_cols,
        )

    @property
    def _hash_bytes_per_token(self) -> int:
        return _LWD_INT32_BYTES * self.num_hash_layers * self.hash_top_k

    @property
    def _engram_bytes_per_token(self) -> int:
        return _LWD_INT32_BYTES * self.num_engram_layers * self.num_engram_cols

    def numel(self, num_tokens: int) -> int:
        """bf16 element count of a `num_tokens`-row payload."""
        return num_tokens * (self.hidden_size + self._hash_bytes_per_token + self._engram_bytes_per_token)

    def pack(
        self,
        embeds: torch.Tensor,
        expert_ids: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack one batch into a 1-D bf16 payload on the embeddings' device.

        Args:
            embeds: `[T, hidden_size]` bf16.
            expert_ids: int `[T, num_hash_layers, hash_top_k]`, iff that segment exists.
            engram_hashes: int `[T, num_engram_layers, num_engram_cols]`, iff that segment exists.

        Raises:
            ValueError: A tensor is missing, unexpected, or of the wrong shape or dtype.
        """
        if embeds.dtype != torch.bfloat16:
            raise ValueError(f"[LWD] UP payload requires bfloat16 embeddings, got {embeds.dtype}")
        if embeds.ndim != 2 or embeds.shape[1] != self.hidden_size:
            raise ValueError(f"[LWD] UP embeddings shape {tuple(embeds.shape)} != [T, {self.hidden_size}]")
        num_tokens = embeds.shape[0]
        _lwd_check_int_segment("hash-MoE expert ids", expert_ids, (num_tokens, self.num_hash_layers, self.hash_top_k))
        _lwd_check_int_segment(
            "Engram hashes", engram_hashes, (num_tokens, self.num_engram_layers, self.num_engram_cols)
        )
        segments = [embeds.flatten()]
        for ids in (expert_ids, engram_hashes):
            if ids is not None:
                segments.append(_lwd_encode_int32(ids, embeds.dtype, embeds.device))
        return torch.cat(segments) if len(segments) > 1 else segments[0]

    def unpack(
        self, payload: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Split a received payload.

        Returns:
            `(embeds, expert_ids, engram_hashes)`: embeds is a `[T, hidden_size]`
            view on the payload device; expert_ids is CPU int32
            `[T, num_hash_layers, hash_top_k]` (request bookkeeping lives on
            CPU); engram_hashes is int32 `[T, num_engram_layers,
            num_engram_cols]` on the payload device. Absent segments are None.

        Raises:
            ValueError: `payload.numel()` does not match the layout.
        """
        expected = self.numel(num_tokens)
        if payload.numel() != expected:
            raise ValueError(
                f"[LWD] UP payload size mismatch: got {payload.numel()}, expected {expected} for {num_tokens} tokens"
            )
        flat = payload.flatten()
        embed_end = num_tokens * self.hidden_size
        hash_end = embed_end + num_tokens * self._hash_bytes_per_token
        embeds = flat[:embed_end].view(num_tokens, self.hidden_size)
        expert_ids = None
        if self.num_hash_layers:
            expert_ids = _lwd_decode_int32(flat[embed_end:hash_end].cpu()).view(
                num_tokens, self.num_hash_layers, self.hash_top_k
            )
        engram_hashes = None
        if self.num_engram_layers:
            engram_hashes = _lwd_decode_int32(flat[hash_end:]).view(
                num_tokens, self.num_engram_layers, self.num_engram_cols
            )
        return embeds, expert_ids, engram_hashes
