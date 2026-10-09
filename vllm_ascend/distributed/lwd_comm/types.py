# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Type definitions for the prefill_only duplex comm package."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Literal


class LwdChannelType(enum.Enum):
    """The two physical data-plane channels, direction-only.

    Each channel maps 1:1 to a dedicated HCCL communicator + NPU stream
    (see ``vllm_ascend.distributed.lwd_wire``).  HCCL P2P matching order
    per (communicator, peer) is exactly the per-channel ``seqno``
    submission order — HCCL does not support tags.
    """

    UP = "lwd_up"      # edge -> cloud: embeddings + optional V4 expert-ID bytes
    DOWN = "lwd_down"  # cloud -> edge: combined c2e packet


class LwdCommStatus(enum.Enum):
    PENDING = "pending"
    OK = "ok"
    ERROR = "error"


@dataclass
class LwdCommRequest:
    """One communication task submitted to the service.

    Everything the comm layer needs travels with the request — it never
    reaches back into scheduler/model state.
    """

    channel: LwdChannelType
    op: Literal["send", "recv"]
    # Number of bf16 elements of the payload (UP: N*H, or N*(H+4*L*K)
    # for V4 with L Hash layers and K experts per token; DOWN:
    # 32 + R*(H+3K)).  recv: used to allocate the exact-size buffer
    # (HCCL P2P requires matching numel on both ends).
    num_elements: int
    # send: payload tensor (snapshotted into a communication-owned buffer
    # at submit time).  recv: None.
    tensor: Any | None = None
    # Per-channel request-level sequence number.  Contract: each
    # channel's counter starts at 0 and increments by one per request on
    # both peers; ops are posted to HCCL only once all lower seqnos have
    # been submitted, so send order and the peer's recv-post order always
    # agree.  Do not mix sequenced and unsequenced requests on one
    # channel.
    seqno: int | None = None
    # Explicit global peer rank; None -> the channel's configured peer.
    src_dst: int | None = None
    # Cloud instance this traffic belongs to (design doc §4.3/§4.4).
    # ``src_dst`` is an in-group rank inside the instance's channel pair,
    # so the same in-group peer address is valid for every instance; the
    # instance selects WHICH physical (communicator, stream, FIFO) the
    # op runs on.  Stamped by the worker: the edge reads it from
    # ``LwdBatch.instance_id`` (0 = single-instance / unstamped legacy
    # traffic), a cloud worker stamps its own
    # ``instance_parallel_start_rank``.  The scheduler's per-instance
    # seqno counters and the service's per-instance channels are two
    # halves of the same contract: without this field every instance's
    # traffic collapses onto one channel whose seqno space then rejects
    # the second instance's first batch ("duplicate or replayed
    # seqno") — treat that error as a missing-stamp tripwire.
    instance_id: int = 0
    # 目标/来源云 engine(= dp_rank),多云 DP 下选择 per-engine 通道组
    # 与该 engine 的 seqno 流;单云恒 0。seqno 契约是 per-(instance,
    # engine, channel, op) 的。
    engine_index: int = 0
    # Optional second payload of the SAME logical request (currently
    # UP-only: mrope positions, int64 [n,3], accompanying an embeds chunk
    # when the batch has mm data).  Physically a second P2P issued right
    # after the main one on the same channel, but sequencing stays at
    # logical-request granularity: one seqno covers both frames, so
    # skip/drain semantics are unchanged.  send: tensor; recv: None and
    # ``aux_num_elements`` sizes the exact buffer.
    #
    # 扩展预留:再增加一种载荷时,把本组字段(aux_tensor/
    # aux_num_elements/aux_dtype)扩成 list[TensorSpec] 即可——快照、
    # keepalive、handles 桥接、recv 缓冲分配在 channel/future 里已是
    # 按"主+aux 两条 op"写成的机械扩展;seqno 纪律与 sizing 预告
    # (notify 携带逐帧 numel/dtype)同步扩展。
    aux_tensor: Any | None = None
    aux_num_elements: int = 0
    # aux 帧的元素 dtype(torch.dtype;None = int64,mrope positions)。
    # 收端据此分配精确缓冲;后续其他载荷(如 bf16 的 deepstack 帧)
    # 显式指定即可,无需改通道结构。
    aux_dtype: Any | None = None


@dataclass
class LwdCommResult:
    """Outcome of a completed request (cross-node P2P finished on device)."""

    status: LwdCommStatus
    tensor: Any | None = None
    error: BaseException | None = None
    # Received aux payload (mrope positions int64), None when the
    # request carried no aux frame.
    aux_tensor: Any | None = None
