# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Reverse lookup of a prompt embedding into the raw id that produced it.

The cloud holds no prompt token ids, yet the first decode steps hash n-grams
reaching back into the prompt tail, whose ids the native lookback window
wants. The edge sends those embeddings already: the tail is made of prompt
positions, so it rides the UP payload with every other prompt row.

A row leaves the edge as ``embed_tokens.weight[id]`` and arrives bit-for-bit
(the UP transfer copies, it does not reduce), and both sides serve the same
checkpoint through the same quantization. The row is therefore in the cloud's
own table and the lookup is an exact match, not a nearest neighbour: a row
matching nothing means the two sides disagree on the table, and reporting
that beats hashing a guess.

Comparing against every table row per request would read the whole table, so
rows are digested once at load time and looked up by digest. The digest only
narrows the candidates -- every hit is confirmed bitwise -- so a collision
costs one extra comparison and never a wrong id.
"""

from __future__ import annotations

import random

import torch
import torch.distributed as dist

from vllm.lwd_logger import lwd_logger

LWD_NO_EMBED_ROW = -1
"""Id reported for an embedding row no table row matches."""

_LWD_MISS = torch.iinfo(torch.int64).max
"""Reduction-neutral stand-in for "this shard holds no such row"."""
_LWD_DIGEST_SEED = 0x5DEECE66D
"""Seed of the per-element digest weights; any fixed value will do."""
_LWD_DIGEST_CHUNK = 1024
"""Table rows digested per pass, bounding the peak of the load-time scan."""
_LWD_INT_VIEWS = {1: torch.int8, 2: torch.int16, 4: torch.int32}
"""Integer type of the same width as a float element, to read its bits."""


class LwdEmbedLookup:
    """Raw ids of embedding rows, by digest of the cloud's own embedding table.

    One instance per cloud worker, built once the model has loaded. Tensor
    parallelism splits the table over the vocabulary, so each rank answers for
    its own shard and the ranks reduce their findings.
    """

    def __init__(self, weight: torch.Tensor, vocab_start: int, vocab_size: int) -> None:
        """Digest this rank's shard of the embedding table.

        Args:
            weight: This rank's ``[rows, hidden]`` shard of ``embed_tokens``.
            vocab_start: Id of the shard's first row.
            vocab_size: Ids of the whole vocabulary, bounding the shard's own;
                the shard's padding rows past it hold no id and are left out.

        Raises:
            ValueError: The table's elements are of a width with no integer
                view, leaving their bits unreadable.
        """
        self._weight = weight
        self._hidden_size = weight.shape[1]
        self._vocab_start = vocab_start
        element_size = weight.element_size()
        if element_size not in _LWD_INT_VIEWS:
            raise ValueError(f"[LWD] embedding table of {element_size}-byte elements has no digest")
        self._int_view = _LWD_INT_VIEWS[element_size]
        # Masked to the element's own width: sign extension would spread one
        # element's sign bit over the high bytes of its product, where it
        # would cancel other elements instead of mixing with them.
        self._element_mask = (1 << (8 * element_size)) - 1
        # One odd weight per hidden element; the digest is their wrapping inner
        # product with the row's bits. Odd keeps every element contributing to
        # the low bit, so no element can be silently dropped.
        rng = random.Random(_LWD_DIGEST_SEED)
        self._weights = torch.tensor(
            [rng.getrandbits(63) | 1 for _ in range(self._hidden_size)],
            dtype=torch.int64,
            device=weight.device,
        )
        num_local = min(weight.shape[0], max(0, vocab_size - vocab_start))
        # The search runs on the host: it walks a few candidates per row and
        # reads them one at a time, which on device would be a sync apiece.
        digests = self._table_digests(num_local).cpu()
        # Stable, so rows sharing a digest stay in ascending row order and the
        # first match of a walk is the lowest id: ranks and runs agree on which
        # id a row bitwise equal to several table rows resolves to.
        self._digests, order = torch.sort(digests, stable=True)
        self._rows = order.to(torch.int64)
        shared = int((self._digests[1:] == self._digests[:-1]).sum()) if num_local > 1 else 0
        lwd_logger.info(
            "[engram] embed lookup: %d local rows from id %d, hidden %d, %d sharing a digest",
            num_local,
            vocab_start,
            self._hidden_size,
            shared,
        )

    def _table_digests(self, num_local: int) -> torch.Tensor:
        """Digest of each of the shard's first ``num_local`` rows."""
        if not num_local:
            return torch.zeros(0, dtype=torch.int64, device=self._weight.device)
        return torch.cat(
            [
                self._row_digests(self._weight[start : min(start + _LWD_DIGEST_CHUNK, num_local)])
                for start in range(0, num_local, _LWD_DIGEST_CHUNK)
            ]
        )

    def _row_digests(self, rows: torch.Tensor) -> torch.Tensor:
        """Wrapping inner product of each row's bits with the digest weights.

        Elements enter as their raw bit pattern, so rows digest equal exactly
        when they are bitwise equal: the digest reads -0.0 and 0.0 as the
        distinct patterns they are, which is what an exact-match lookup wants.

        Args:
            rows: ``[rows, hidden]`` of the table's dtype.

        Returns:
            int64 ``[rows]``.
        """
        bits = rows.view(self._int_view).to(torch.int64).bitwise_and(self._element_mask)
        return (bits * self._weights).sum(dim=1)

    def lookup(self, embeds: torch.Tensor) -> list[int]:
        """Raw ids of ``embeds``, ``LWD_NO_EMBED_ROW`` where no table row matches.

        A row equal to several table rows takes the lowest matching id, so
        every rank answers alike for a row its shard holds more than once.

        Args:
            embeds: ``[rows, hidden]`` in the table's own dtype.

        Returns:
            One id per row of ``embeds``.

        Raises:
            ValueError: The rows are of another width or dtype than the table;
                converting them would change the bits an exact match needs.
        """
        if embeds.shape[1] != self._hidden_size:
            raise ValueError(f"[LWD] embedding rows of width {embeds.shape[1]} != table width {self._hidden_size}")
        if embeds.dtype != self._weight.dtype:
            raise ValueError(
                f"[LWD] embedding rows of dtype {embeds.dtype} != table dtype {self._weight.dtype}; "
                "an exact match reads the bits, which a conversion would change"
            )
        found = self._local_ids(embeds)
        # A row belongs to exactly one shard, so the lowest id over the ranks
        # is the finding of whichever rank holds it. A rank holding none sends
        # the miss sentinel, which no real id can lose to.
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group() if dist.is_initialized() else None
        if tp_group is not None and tp_group.world_size > 1:
            device_found = found.to(self._weight.device)
            dist.all_reduce(device_found, op=dist.ReduceOp.MIN, group=tp_group.device_group)
            found = device_found.cpu()
        return [LWD_NO_EMBED_ROW if id_ == _LWD_MISS else int(id_) for id_ in found.tolist()]

    def _local_ids(self, embeds: torch.Tensor) -> torch.Tensor:
        """Host int64 ``[rows]``: this shard's id of each row, else ``_LWD_MISS``."""
        found = torch.full((embeds.shape[0],), _LWD_MISS, dtype=torch.int64)
        if not self._digests.numel():
            return found
        digests = self._row_digests(embeds).cpu()
        # Equal digests are adjacent after the sort, so the first row of a
        # digest plus a walk forward visits exactly its candidates, in
        # ascending row order: the first bitwise match is the lowest id.
        first = torch.searchsorted(self._digests, digests)
        for row, (position, digest) in enumerate(zip(first.tolist(), digests.tolist())):
            for candidate in range(int(position), self._digests.numel()):
                if int(self._digests[candidate]) != digest:
                    break
                table_row = int(self._rows[candidate])
                if torch.equal(self._weight[table_row], embeds[row]):
                    found[row] = table_row + self._vocab_start
                    break
        return found
