# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD Hash MoE expert-ID merging and request-position bookkeeping.

The expert ids travel UP as a segment of `lwd_up_payload.LwdUpPayloadLayout`.
"""

import torch


def hash_layer_count(config: "PretrainedConfig") -> int:
    """Return the Hash MoE layer count for DeepSeek V4; zero for other models."""
    return config.num_hash_layers if getattr(config, "model_type", None) == "deepseek_v4" else 0


def merge_hash_expert_ids(
    tid2eid: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_experts: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    """Use remote IDs for prompt rows and checkpoint lookup for decode rows."""
    # The token buffer's prompt slots are intentionally unspecified. Never use
    # them as table indices, even in the unselected branch of torch.where.
    safe_token_ids = torch.where(prompt_mask | (token_ids == -1), 0, token_ids).to(torch.int64)
    decode_experts = tid2eid[safe_token_ids]
    return torch.where(prompt_mask.unsqueeze(-1), prompt_experts, decode_experts).to(torch.int32)


class LwdHashRoutingState:
    """CPU prompt tables keyed by request ID, with explicit received-row masks."""

    def __init__(self, num_layers: int, top_k: int, num_experts: int) -> None:
        """Store the Hash routing dimensions and initialize the per-request prompt expert-ID cache."""
        self.num_layers = num_layers
        self.top_k = top_k
        self.num_experts = num_experts
        self.requests: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def discard(self, req_id: str) -> None:
        """Release a request's expert IDs and validity mask; a missing request is a no-op."""
        self.requests.pop(req_id, None)

    def add_chunk(
        self, req_id: str, prompt_len: int, start: int, expert_ids: torch.Tensor
    ) -> None:
        """Validate a remote expert-ID chunk, write it into the request's absolute prompt position and mark it valid."""
        ids = expert_ids.to(device="cpu", dtype=torch.int32)
        if ids.ndim != 3 or tuple(ids.shape[1:]) != (self.num_layers, self.top_k):
            raise ValueError(f"Invalid LWD expert shape for {req_id}: {tuple(ids.shape)}")
        end = start + ids.shape[0]
        if not 0 <= start <= end <= prompt_len:
            raise ValueError(f"Invalid LWD expert chunk for {req_id}: [{start}, {end})/{prompt_len}")
        if torch.any(ids < 0) or torch.any(ids >= self.num_experts):
            raise ValueError(f"LWD expert IDs out of range for {req_id}")
        entry = self.requests.get(req_id)
        if entry is None or entry[0].shape[0] != prompt_len:
            entry = (
                torch.zeros(prompt_len, self.num_layers, self.top_k, dtype=torch.int32),
                torch.zeros(prompt_len, dtype=torch.bool),
            )
            self.requests[req_id] = entry
        entry[0][start:end].copy_(ids)
        entry[1][start:end] = True

    def build_batch(
        self,
        req_ids: "Sequence[str]",
        counts: "Sequence[int]",
        computed: "Sequence[int]",
        prompt_lengths: "Sequence[int]",
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Assemble CPU expert IDs and the prompt mask by request order and computed positions; raise when prompt data is missing."""
        if not len(req_ids) == len(counts) == len(computed) == len(prompt_lengths):
            raise ValueError("LWD Hash routing batch metadata lengths do not match")
        total = sum(int(n) for n in counts)
        batch = torch.zeros(total, self.num_layers, self.top_k, dtype=torch.int32)
        mask = torch.zeros(total, dtype=torch.bool)
        row = 0
        for req_id, count, start, prompt_len in zip(req_ids, counts, computed, prompt_lengths):
            count, start, prompt_len = int(count), int(start), int(prompt_len)
            prompt_count = max(0, min(count, prompt_len - start))
            if prompt_count:
                entry = self.requests.get(req_id)
                end = start + prompt_count
                if entry is None or end > entry[0].shape[0] or not entry[1][start:end].all():
                    raise ValueError(f"Missing LWD Hash MoE expert IDs for {req_id} positions [{start}, {end})")
                batch[row : row + prompt_count].copy_(entry[0][start:end])
                mask[row : row + prompt_count] = True
            row += count
        return batch, mask
