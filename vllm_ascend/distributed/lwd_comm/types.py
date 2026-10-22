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
    # 目标/来源云 engine(= dp_rank),多云 DP 下选择 per-engine 通道组
    # 与该 engine 的 seqno 流;单云恒 0。seqno 契约是 per-(engine,
    # channel, op) 的。
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

    # ------------------------------------------------------------------ #
    # recv 侧可选落点(host 暂存方案,见 docs/design/lwd_host_staging.md)  #
    # ------------------------------------------------------------------ #
    # recv:调用方预提供的**主帧接收 buffer**(NPU 上的乒乓槽视图)。
    # 给了就用它、不再自己 torch.empty —— 这是"接收缓冲随 D2H 完成即
    # 归还并被下一批复用"的落点。numel 必须与 num_elements 严格相等
    # (HCCL P2P 要求两端等长),本层会断言。
    recv_buffer: Any | None = None
    # recv:分段 D2H 的逐段描述 ``[(dst_pinned_view, src_elem_offset,
    # numel)]``。非 None 即表示"收完后在通道流上把主帧按段异步拷进各
    # 请求自己的 pinned buffer"(design v7 §4.3 改版:提交期建行表,
    # 通信线程零 CPU 拷贝)。src 偏移以主帧 flat 元素计。
    stage_segments: Any | None = None
    # recv:aux 帧的 host 落点(预留;mrope 自 wire_version 2 起走控制面,
    # 当前恒为 None)。字段先落位,避免将来扩 aux 暂存时再动 dataclass。
    aux_stage_buffer: Any | None = None
    # recv:该 recv 落在哪个设备槽(-1 = 非池化分配)。纯诊断:池的账
    # (占用/归还)由调用方的 stager 维护,通道层只把它透传到 future 上,
    # 让日志与泄漏排查能对上号。
    recv_slot: int = -1


@dataclass
class LwdCommResult:
    """Outcome of a completed request (cross-node P2P finished on device)."""

    status: LwdCommStatus
    tensor: Any | None = None
    error: BaseException | None = None
    # Received aux payload (mrope positions int64), None when the
    # request carried no aux frame.
    aux_tensor: Any | None = None
    # Reserved legacy field (always None now): the staged payload used to
    # be exposed as one host tensor; segmented D2H (``stage_segments``)
    # lands directly into per-request buffers, so there is no single
    # staged tensor any more.  ``staged_event`` still signals completion.
    staged_tensor: Any | None = None
