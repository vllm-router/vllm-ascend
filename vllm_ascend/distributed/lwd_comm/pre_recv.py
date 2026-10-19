# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 提前收(early recv,多边多云版):引擎⇄worker 旁路链路 + 提示/水位协议。

数据路径(与单边版同构,全部键控按 edge_id 分域)::

    云引擎 IO 线程                        云端点 worker 通信线程
    ───────────────                       ──────────────────────
    RangeNotify 到达(边侧预告,信封带 edge_id)
      └─ post_irecv_hint ──hint MQ(下行,每边一条)──► start_early_irecv
                                                       └─ submit_recv(UP, edge_id, seqno, numel)
                                                             │  irecv 已提前挂出
                                                       future.done() 轮询
                                                             │
    record_irecv_completions ◄──done MQ(上行,每边一条)── _report_irecv_completions
      └─ 水位(按 (edge_id, channel))→ is_irecv_complete(edge_id, UP, seqno)
           └─ PD_mix 调度闸门:收完才下发这个 prefill 批

两条 MQ 都是旁路,都不能借 ``rpc_broadcast_mq``:那条队列由
``worker_busy_loop`` 单线程消费,它在 ``execute_model`` 里为一个 prefill 批
阻塞几百毫秒 —— 提示排在那条队列上,就等不到"提前"挂 recv 的机会。

多边多云形态:

* hint MQ:每边一条。引擎(executor)在 worker 创建之前按 registry 的
  edge_ids 逐边建 writer,handle 写进 ``..._E{edge_id}`` 环境变量;云 rank
  只对"本 rank 是其 UP 端点"的边 attach reader(端点随边轮转);
* done MQ:每边一条。每边的云端点 rank 建 writer,handle 随 READY 握手
  回传引擎,引擎挂 N 个 reader 逐步排空;
* 配对键一律 (edge_id, seqno):各边 seqno 流独立从 0 起,裸 seqno 多边
  互撞;cloud_id 是云进程本地身份,不进键。

水位语义:``is_irecv_complete(edge_id, channel, seqno)`` ==
``seqno <= watermark[(edge_id, channel)]``。每 (edge, cloud, channel) 通信域
完成序 = seqno 序(``channel.py`` 的 FIFO:队首没完成,后面不可能完成),
"已完成集合"恒为前缀,故一个最大水位即可精确表示。水位只是**就绪判据**,
不参与任务优先级。

提示字段:``num_tokens`` = 该批(本 chunk 批)flat token 数 —— UP 主帧
numel 由 hash 载荷公式在 **worker 侧**按本机配置算(见
``lwd_cloud_worker._lwd_build_up_recv_request``),提示只带"事实"(token
数),不带尺寸公式,避免两侧各算一遍算歪。**不带 aux_rows**:mrope
positions 已走控制面(wire_version>=2,LwdRangeItem.mrope_positions),
数据面 aux 帧退役,UP recv 只挂主帧(aux 恒 0)。

非端点 rank(云端其余 TP rank)没有跨机通道:它们的数据在注入时刻由 TP 组内
广播取到(见 ``lwd_cloud_model_runner._lwd_inject_remote_embeds``),既不挂
recv 也不上报 —— 本模块的提示/上报只走端点 rank 这一条线。
"""

from __future__ import annotations

import base64
import os
import pickle
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_ascend.distributed.lwd_comm.types import LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

# 日志名必须落在 **vllm.** 命名空间内:vLLM 只给名为 "vllm" 的 logger 挂
# handler(DEFAULT_LOGGING_CONFIG,propagate=False),子 logger 靠向上传播拿到
# 那个 handler。vllm_ascend.* 的 logger 传不到 "vllm"(根 logger 无 handler)。
logger = init_logger("vllm.lwd.pre_recv")

#: 下行提示 MQ 的 handle 环境变量前缀(实际键 = 前缀 + str(edge_id))。
#: 执行器在创建 worker 之前逐边导出,worker 在 ``init_device`` 里重建
#: reader —— 两侧只认这一处常量名。
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE_E"

# 旁路 MQ 规格:每槽 1KB、共 64 槽。一批一条提示/上报;边侧 pending 闸门
# (LwdEdgeScheduler _LWD_MAX_PENDING_CHUNKS,按 cloud_id 分域默认 2)把
# 每边未消费 chunk 数压在个位数,最坏在飞提示 ≈ cap+1 条/边,64 槽绰绰
# 有余,enqueue 永不阻塞 IO 线程;提示是正确性依赖(丢一条 = 该 recv 不
# 提前挂、水位不前进、闸门后的批永不下发),所以留足余量、不做流控。
_SIDEBAND_MAX_CHUNK_BYTES = 1024
_SIDEBAND_MAX_CHUNKS = 64


# ---------------------------------------------------------------------- #
# 提示(hint):调度侧生产,worker 通信线程消费                              #
# ---------------------------------------------------------------------- #
def make_recv_hint(edge_id: int, seqno: int, num_tokens: int) -> dict:
    """构造一条 UP 提前收提示。

    * ``edge_id``:来源边(dispatch 信封带来;多边分派与键控的第一维度)
    * ``seqno``:该边 UP 链配对号(边侧按 cloud_id 分域分配)
    * ``num_tokens``:本批 flat token 数(主帧尺寸由 worker 按 hash 载荷公式推)
    """
    return {
        "edge_id": int(edge_id),
        "channel": LwdChannelType.UP,
        "seqno": int(seqno),
        "num_tokens": int(num_tokens),
    }


def hint_fields(hint: dict) -> tuple[int, int, int] | None:
    """提示自检:返回 (edge_id, seqno, num_tokens);坏提示返回 None。

    缺 edge_id = 旧线/错线提示,不与之一致静默兼容,fail-visible 丢弃。"""
    try:
        edge_id = int(hint["edge_id"])
        seqno = int(hint["seqno"])
        num_tokens = int(hint["num_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    if edge_id < 0 or seqno < 0 or num_tokens <= 0:
        return None
    return edge_id, seqno, num_tokens


# ---------------------------------------------------------------------- #
# 旁路 MQ 生命周期(按边)                                                 #
# ---------------------------------------------------------------------- #
def _new_sideband_mq() -> "MessageQueue":
    """1 读 1 写的旁路 MQ(提示/上报同一规格;每条都只有该边端点 rank 参与)。"""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(
        1, 1,
        max_chunk_bytes=_SIDEBAND_MAX_CHUNK_BYTES,
        max_chunks=_SIDEBAND_MAX_CHUNKS,
    )


def create_hint_mq(edge_id: int) -> "MessageQueue":
    """引擎侧:建该边的 hint MQ(writer),并把 handle 导出到环境变量。

    必须在 worker 被创建之前调用(worker 启动时按环境变量重建 reader);
    fork/spawn 两种启动方式都靠环境变量继承。
    """
    mq = _new_sideband_mq()
    os.environ[LWD_RECV_HINT_MQ_ENV + str(edge_id)] = base64.b64encode(
        pickle.dumps(mq.export_handle())
    ).decode("ascii")
    return mq


def attach_hint_mq(edge_id: int) -> "MessageQueue | None":
    """worker 侧:按环境变量里的 handle 重建该边 hint MQ reader。

    没有 handle(本 rank 不是该边端点/该角色不接提示)返回 None:通信线程
    照旧跑上报,于是"提示没来"只影响提前挂 recv 的时机,不影响水位前进
    (消费点会兜底补挂)。
    """
    raw = os.environ.get(LWD_RECV_HINT_MQ_ENV + str(edge_id))
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(
        pickle.loads(base64.b64decode(raw)), 0
    )


def create_report_mq() -> "MessageQueue":
    """worker 侧:建该边的 done MQ(writer),handle 随后随 READY 握手回传引擎。"""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# 调度进程侧:提示下发 + 完成水位                                          #
# ---------------------------------------------------------------------- #
# edge_id -> 下发闭包(引擎按边注册;相位逃生分支不注册 = 正常形态)
_hint_senders: dict[int, Callable[[dict], None]] = {}
# (edge_id, channel) -> 已确认收完的最大 seqno
_watermarks: dict[tuple[int, LwdChannelType], int] = {}
_lock = threading.Lock()


def register_hint_sender(edge_id: int, sender: Callable[[dict], None]) -> None:
    """装上下发某边提示用的传输(每个引擎进程每边装一次)。

    传进来的闭包必须**可靠投递**(把提示 enqueue 进 hint MQ):闸门开启时提示
    是正确性依赖 —— 丢一条,那条 recv 就不提前挂,水位不前进,被闸门拦下的批
    永不下发。
    """
    _hint_senders[int(edge_id)] = sender


def post_irecv_hint(hint: dict) -> None:
    """把一条 recv 提示发给本地 worker 的通信线程(按 hint 的 edge_id 路由)。

    成功路径不打日志(由调用方 ``lwd_cloud_engine`` 打 ``HINT ...``):本模块只留
    "本该不发生"的诊断,而且它必须能从日志里看到。
    """
    sender = _hint_senders.get(hint.get("edge_id", -1))
    if sender is None:
        # 没有传输 = 本进程不参与提前收(如相位调度逃生通道):丢一条提示不影响
        # 正确性 —— 该分支没有闸门在等它,recv 由消费点自己挂。
        logger.warning(
            "[Lwd][pre-recv] HINT DROPPED: no sender registered edge=%s seqno=%s",
            hint.get("edge_id"), hint.get("seqno"),
        )
        return
    sender(hint)


def record_irecv_completions(
    items: Iterable[tuple[int, LwdChannelType, int]],
) -> list[tuple[int, LwdChannelType, int]]:
    """把 worker 上报的完成项并进按边水位(引擎每步排空 done MQ 后调用)。

    返回**本次真正推进水位**的项,由调用方(``lwd_cloud_engine``)打日志:
    水位是否在动是定位"卡在哪一段"的第一手证据,日志放在已证明能打出来的
    模块里。
    """
    advanced: list[tuple[int, LwdChannelType, int]] = []
    with _lock:
        for edge_id, channel, seqno in items:
            key = (int(edge_id), channel)
            if seqno > _watermarks.get(key, -1):
                _watermarks[key] = seqno
                advanced.append((edge_id, channel, seqno))
    return advanced


def watermark(edge_id: int, channel: LwdChannelType) -> int:
    """当前水位:该边该通道上"已确认收完"的最大 seqno(-1 = 一条都没有)。"""
    with _lock:
        return _watermarks.get((int(edge_id), channel), -1)


def is_irecv_complete(edge_id: int, channel: LwdChannelType, seqno: int) -> bool:
    """该边该通道 seqno 这条 recv 收完了没 —— 只作就绪判据,不打日志。

    "在等谁"由闸门那一行给出(``lwd_cloud_mixed_scheduler`` 每步打
    ``gate edge=.. seqno=.. watermark=.. ready=..``,DEBUG 级),这里保持纯判据。
    """
    return seqno <= watermark(edge_id, channel)


def reset_edge_watermark(edge_id: int) -> None:
    """边失联清理挂点:摘除该边全部水位条目。

    本轮无调用方(当前线尚无 edge-loss 检测点);供后续检测落地时接线。
    """
    with _lock:
        for key in [k for k in _watermarks if k[0] == int(edge_id)]:
            del _watermarks[key]
