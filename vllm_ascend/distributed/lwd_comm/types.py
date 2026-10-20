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
    # 多边多云数据面选路维度:指定该次通信属于哪个 (edge, cloud, engine)
    # 通信域。None -> 本进程身份回填(与旧行为逐位一致);上层按路由
    # 结果/来源边回填。
    edge_id: int | None = None
    cloud_id: int | None = None
    # 目标/来源云 engine(云内 DP 引擎序号,= 云进程的 dp_rank);多云 DP
    # 下选择 per-engine 通道组与该 engine 的 seqno 流;单引擎恒 0。
    # None -> 本进程身份回填。seqno 契约是 per-(edge, cloud, engine,
    # channel, op) 的。
    engine_index: int | None = None
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
