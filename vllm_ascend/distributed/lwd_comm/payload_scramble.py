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
"""Scramble / descramble of the edge-cloud hidden-state payload.

The intermediate activations a split model ships between the edge and the
cloud are the model's *real* hidden states -- not statistics, indices or
metadata -- so they must not sit on the wire as directly readable bf16/fp16.
Both peers derive the same seed from the same five factors and apply the
inverse transform, so the seed itself never crosses the node boundary.

This module is deliberately a **leaf**: it reads no configuration, imports
nothing from this project, and only ``hashlib`` from the stdlib (``torch`` is
imported lazily, so the seed path works without it).  There is exactly one
public class with three methods, used at four call sites by the caller:

    1. construction        -- the two static factors (read from config by the
                              caller) are bound here;
    2. ``seed_for``        -- the three per-flight factors -> one 64-bit seed;
    3. ``scramble``        -- before the payload is sent;
    4. ``descramble``      -- after it is received, before it is consumed.

Scope note: the factors are largely public and travel on the control plane, so
this is transport-state obfuscation, **not** cryptography -- anyone who knows
the factors can recompute the seed, and the operator only uses the low 16 bits
of it.  See ``blog-repo/vllm/加扰与解扰-设计.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import weight at zero
    from torch import Tensor

__all__ = [
    "PayloadScrambler",
    "get_payload_scrambler",
    "is_payload_scrambler_initialized",
]

# Domain separator: bump the version suffix on any factor-set / canonical-form
# / order change.  A bump makes two peers running different versions derive
# different seeds loudly instead of silently mis-deriving.
_KDF_DOMAIN = b"vllm-ascend/payload-scramble/v3"
_US = b"\x1f"  # unit separator between factor fields
_SEED_PERSON = b"ec-scr01"
_SEED_BYTES = 8  # seed width (64 bit)
_MASK16 = 0xFFFF  # the operator only uses the low 16 bits of the seed

_FALLBACK_NONNEG_INT = -1


# ---------------------------------------------------------------------------
# Factor canonicalisation.  Module-level privates: the KDF below needs them,
# so making them class members would invert the dependency (module-level
# helpers depending on the class defined further down).
# ---------------------------------------------------------------------------


def _canonical_str(v: str | None) -> str:
    """String factor: strip surrounding whitespace, keep everything else.

    The static factors usually come from a config or a mounted file, and a
    trailing newline / CRLF / stray spaces would otherwise differ byte-wise
    between peers.  No case folding: contents must match byte for byte.

    ``strip()`` does **not** remove a BOM (U+FEFF is not whitespace), so read
    factor files with ``encoding="utf-8-sig"``.
    """
    return (v or "").strip()


def _canonical_nonneg_int(v: int | None) -> int:
    """Non-negative int factor (seqno / num_elements): invalid/negative -> -1.

    Degrading instead of raising is deliberate: both peers must map the same
    malformed input to the same fallback or their seeds would desynchronise.
    ``OverflowError`` is caught because ``int(float("inf"))`` raises it rather
    than ``ValueError``.
    """
    try:
        n = int(v)
    except (TypeError, ValueError, OverflowError):
        return _FALLBACK_NONNEG_INT
    return n if n >= 0 else _FALLBACK_NONNEG_INT


# ---------------------------------------------------------------------------
# Derivation and operator core (module-level privates)
# ---------------------------------------------------------------------------


def _derive_seed(
    *,
    device_id: str,
    custom_factor: str,
    channel: str,
    seqno: int,
    num_elements: int,
) -> int:
    """Derive the 64-bit seed from the five canonical factors.

    Uses ``hashlib`` only: ``hash()`` is salted per process by
    ``PYTHONHASHSEED`` and would silently break peer alignment.  The factor
    order is frozen and each field is ``\\x1f``-separated so contents cannot be
    mistaken for a field boundary; reordering is a breaking change on purpose.
    """
    h = hashlib.blake2b(digest_size=_SEED_BYTES, person=_SEED_PERSON)
    h.update(_KDF_DOMAIN)
    for part in (
        _canonical_str(device_id),
        _canonical_str(custom_factor),
        _canonical_str(channel),
        str(_canonical_nonneg_int(seqno)),
        str(_canonical_nonneg_int(num_elements)),
    ):
        h.update(_US)
        h.update(part.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def _require_torch() -> Any:
    """Import torch on first use so the seed method stays torch-free."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dep in prod
        raise RuntimeError(
            "PayloadScrambler: the operators need torch; seed_for does not and "
            "remains usable without it"
        ) from exc
    return torch


def _require_seed(seed: int) -> int:
    """Validate ``seed`` and return the 16-bit offset ``k``.

    ``bool`` is rejected explicitly: ``True`` would silently mean ``k=1``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"scramble/descramble: seed must be an int, got {type(seed).__name__}"
        )
    return seed & _MASK16


def _signed_k(k: int) -> int:
    """Express the 16-bit offset as a value int16 can hold.

    The bit pattern is unchanged (two's complement) and ``bitwise_xor_``
    operates bit-wise on int16 regardless of sign, so this is purely a range
    fix-up for the scalar operand.
    """
    return k if k <= 0x7FFF else k - 0x10000


def _xor_in_place(t: "Tensor", k: int, torch: Any) -> None:
    """XOR ``k`` into every element of ``t``, in place.

    The transform XORs the raw 16-bit float bit patterns: per-bit independent,
    so no carry, no overflow, no platform-dependent wrap semantics, and the
    inverse is the identical call.  ``fadd`` is deliberately absent -- float
    addition in bf16/fp16 is not invertible and could not satisfy "restore the
    original matrix".

    ``bitwise_xor_`` on the int16 view is required rather than the out-of-place
    ``view ^ k``: the latter allocates a full-size buffer and leaves the input
    untouched, which would break the in-place contract and the zero-extra-
    memory property.
    """
    t.view(torch.int16).bitwise_xor_(_signed_k(k))


def _require_scrambleable(t: "Tensor", torch: Any) -> None:
    """Reject anything that cannot take a 16-bit bit view."""
    if not isinstance(t, torch.Tensor):
        raise ValueError(
            f"scramble/descramble: expected a torch.Tensor, got {type(t).__name__}"
        )
    if t.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "scramble/descramble: only 16-bit floats can be reinterpreted as "
            f"int16, got dtype={t.dtype}"
        )
    if not t.is_contiguous():
        raise ValueError(
            "scramble/descramble: tensor must be contiguous to take a bit "
            "view, call .contiguous() on it first"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PayloadScrambler:
    """Scramble / descramble of one hidden-state matrix with one seed.

    The two deployment-time factors are bound once at construction (the caller
    reads them from config and passes the strings); the three per-flight
    factors are passed per call to :meth:`seed_for`.  The instance is immutable
    and the seed path is a pure function of the five factors, so a single
    instance is safe to share between threads and between the virtual workers
    of one process.
    """

    #: Shared fingerprint of the **device pair** -- identical on both peers.
    #: Not a per-machine ``machine-id`` / NPU serial, which would necessarily
    #: differ between the two sides.
    device_id: str
    #: Business / licensing factor -- identical on both peers.  Prefer a
    #: mounted file over a command line or config blob so it cannot leak into
    #: startup logs or version control.
    custom_factor: str

    # ---------------------------------------------------------------- 构造
    def __post_init__(self) -> None:
        # Canonicalise once, at construction, so a whitespace-only factor fails
        # here rather than silently producing a different seed than the peer.
        object.__setattr__(self, "device_id", _canonical_str(self.device_id))
        object.__setattr__(
            self, "custom_factor", _canonical_str(self.custom_factor)
        )
        if not self.device_id:
            raise ValueError("PayloadScrambler: device_id must not be empty")
        if not self.custom_factor:
            raise ValueError("PayloadScrambler: custom_factor must not be empty")

    # ------------------------------------------------------------ 派生
    def seed_for(self, *, channel: str, seqno: int, num_elements: int) -> int:
        """Derive the 64-bit seed of one flight.

        The three arguments belong to the LWD data plane and are identical on
        both peers for a given payload, so the seed is too:

        * ``channel`` -- the physical direction (``"lwd_up"`` / ``"lwd_down"``).
        * ``seqno`` -- that channel's request-level sequence number, which
          increments by one per flight on both peers.
        * ``num_elements`` -- the payload size in bf16 elements; HCCL P2P
          requires both ends to match it exactly.

        Recomputing is cheap and deterministic: a pure function of the five
        factors.
        """
        return _derive_seed(
            device_id=self.device_id,
            custom_factor=self.custom_factor,
            channel=channel,
            seqno=seqno,
            num_elements=num_elements,
        )

    # ------------------------------------------------------------ 算子
    def scramble(self, t: "Tensor", *, seed: int) -> "Tensor":
        """Scramble one matrix in place; returns ``t``.

        Raises:
            ValueError: ``seed`` is not an int, ``t`` is not a 16-bit float
                tensor, or ``t`` is not contiguous.
        """
        return self._apply(t, seed=seed)

    def descramble(self, t: "Tensor", *, seed: int) -> "Tensor":
        """Exact inverse of :meth:`scramble`; ``seed`` must match.

        Same body on purpose: the transform is its own inverse, so a separate
        implementation could only ever be a place for the two to drift apart.
        """
        return self._apply(t, seed=seed)

    def _apply(self, t: "Tensor", *, seed: int) -> "Tensor":
        torch = _require_torch()
        k = _require_seed(seed)
        _require_scrambleable(t, torch)
        _xor_in_place(t, k, torch)
        return t

    # ------------------------------------------------------------ 日志
    def __repr__(self) -> str:
        """Report only whether each factor is set -- never the values."""
        return (
            f"PayloadScrambler(device_id={'set' if self.device_id else 'unset'}, "
            f"custom_factor={'set' if self.custom_factor else 'unset'})"
        )


_PROCESS_SCRAMBLER: PayloadScrambler | None = None


def get_payload_scrambler(
    *, device_id: str | None = None, custom_factor: str | None = None
) -> PayloadScrambler:
    """Return the process-wide instance, mirroring ``get_comm_service()``.

    The worker builds it once at startup (place 1 in the design doc) with the
    two static factors; the send/recv funnels (places 3 and 4) fetch it with
    no arguments.  Bound once per process, so a multi-worker deployment reads
    the factor source once rather than per worker.

    Only ``device_id`` / ``custom_factor`` live here because they are the only
    factors fixed for the whole process.  Do not stash ``channel`` / ``seqno``
    here: they are per flight.
    """
    global _PROCESS_SCRAMBLER
    if _PROCESS_SCRAMBLER is None:
        if device_id is None or custom_factor is None:
            raise ValueError(
                "get_payload_scrambler: the first call must pass both "
                "device_id and custom_factor"
            )
        _PROCESS_SCRAMBLER = PayloadScrambler(
            device_id=device_id, custom_factor=custom_factor
        )
    elif device_id is not None or custom_factor is not None:
        candidate = PayloadScrambler(
            device_id=(
                device_id if device_id is not None else _PROCESS_SCRAMBLER.device_id
            ),
            custom_factor=(
                custom_factor
                if custom_factor is not None
                else _PROCESS_SCRAMBLER.custom_factor
            ),
        )
        if candidate != _PROCESS_SCRAMBLER:
            raise ValueError(
                "get_payload_scrambler: already initialised with different "
                "factors; the process-wide instance is bound once"
            )
    return _PROCESS_SCRAMBLER


def is_payload_scrambler_initialized() -> bool:
    """True once the process-wide instance has been built (scramble is on)."""
    return _PROCESS_SCRAMBLER is not None
