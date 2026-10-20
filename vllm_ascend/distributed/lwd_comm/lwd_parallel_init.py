#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Lwd 边云模式的 ascend 侧并行组初始化。

原生 ``init_ascend_model_parallel`` 按 (dp, pp, pcp, tp) 均匀网格切分
全局 rank,表达不了"边 edge_npu_count + 云 cloud_npu_count"的非对称
拓扑。本模块按 Lwd 布局构建 ascend 侧依赖的通信组(MC2 等)并注入
``vllm_ascend.distributed.parallel_state``,由 LwdCloudWorker /
LwdEdgeWorker 覆写的 ``_init_worker_distributed_environment`` 调用,
替代原生入口。
"""

import torch
from vllm.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

import vllm_ascend.distributed.parallel_state as ascend_parallel_state
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import flashcomm2_enable

logger = init_logger(__name__)


def init_lwd_ascend_model_parallel(parallel_config) -> None:
    """按 Lwd 布局构建 ascend 侧通信组。

    对齐原生 ``init_ascend_model_parallel`` 的无特性开启路径:仅 MC2,
    及与其同分组的 dynamic_eplb / multistream_overlap_gate 变体。
    """
    if ascend_parallel_state.model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    lwd = parallel_config.lwd_config
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # 云侧复用(registry):world 校验与 MC2 分组直接按声明的实例布局
    # (每边实例一组 + 每云实例一组,云组 = 该云全部引擎的 rank,对齐
    # EP=dp×tp 分片域——EP 开启时专家按 ep_size=dp×tp 跨引擎分片,
    # all2all 组必须罩住本云全部 rank,否则 token 只能路由到部分专家,
    # MoE 输出乱码),与 vllm 侧 parallel_state 的 registry 分支分组
    # 语义对齐;layout 为空走 1E1C 连续 edge-first 公式,老路径不变。
    edge_layout = getattr(lwd, "edge_ranks_layout", None) or {}
    cloud_layout = getattr(lwd, "cloud_ranks_layout", None) or {}
    if edge_layout or cloud_layout:
        expected_world = sum(len(r) for r in edge_layout.values()) + sum(
            len(r) for r in cloud_layout.values())
        assert world_size == expected_world, (
            f"Lwd world size mismatch: got {world_size}, registry "
            f"declares {expected_world}")
        mc2_groups: list[list[int]] = (
            [list(edge_layout[e]) for e in sorted(edge_layout)]
            + [list(cloud_layout[c]) for c in sorted(cloud_layout)]
        )
    else:
        edge_count = lwd.edge_npu_count
        cloud_count = lwd.cloud_npu_count
        # legacy 公式分支(共享边布局):world = E + dp*C,engine i 占
        # [E+i*C, E+(i+1)*C)。cloud_dp_size/cloud_engine_rank_base 由
        # vllm 侧 glm 版公式提供;老版本 vllm 无此属性时退化 dp=1。
        cloud_dp = getattr(lwd, "cloud_dp_size", 1) or 1
        engine_base = getattr(lwd, "cloud_engine_rank_base", None)
        if not callable(engine_base):
            engine_base = lambda i: edge_count + i * cloud_count  # noqa: E731
        assert world_size == edge_count + cloud_count * cloud_dp, (
            f"Lwd world size mismatch: got {world_size}, expected "
            f"{edge_count} + {cloud_count} * {cloud_dp}")
        # MC2 组(EP/MoE all2all 的载体,fused_moe 经 get_mc2_group()
        # 取):边 rank 一组(不跑 MoE);全部云 rank 一组(跨引擎,
        # 对齐 EP=dp×tp 分片域;锁步由 DPEngineCoreProc wave 保证)。
        mc2_groups = [[r for r in range(edge_count)]]
        cloud_ranks_all: list[int] = []
        for dp_idx in range(cloud_dp):
            base = engine_base(dp_idx)
            cloud_ranks_all.extend(base + r for r in range(cloud_count))
        mc2_groups.append(cloud_ranks_all)
    ascend_parallel_state._MC2 = init_model_parallel_group(
        mc2_groups,
        get_world_group().local_rank,
        backend,
        group_name="mc2",
    )
    # 就地强制初始化 MC2 的 HCCL comm(init_device 的锁步窗口内做
    # 一次小组集合通信):懒初始化会把 comm init 推迟到模型加载期
    # (16 个 rank 在两个引擎进程里各自撞进 FusedMoE 第一层),跨进程
    # 时机错开时 rootInfo 协商可能失败(HCCL_E_UNAVAIL);提前到全 rank
    # 对齐的初始化窗口内完成,失败也能当场报出真实原因
    try:
        _probe = torch.zeros(1, device="npu")
        torch.distributed.all_reduce(
            _probe, group=ascend_parallel_state._MC2.device_group
        )
        logger.info(
            "Lwd ascend MC2 comm eagerly initialized (world_size=%s)",
            world_size,
        )
    except Exception:
        logger.exception(
            "Lwd ascend MC2 comm eager init failed (world_size=%s, "
            "mc2_groups=%s)", world_size, mc2_groups,
        )
        raise

    ascend_config = get_ascend_config()
    if ascend_config.eplb_config.dynamic_eplb:
        ascend_parallel_state._DYNAMIC_EPLB = init_model_parallel_group(
            mc2_groups,
            get_world_group().local_rank,
            backend,
            group_name="dynamic_eplb",
        )
    if ascend_config.multistream_overlap_gate:
        ascend_parallel_state._FC3_QUANT_X = init_model_parallel_group(
            mc2_groups,
            get_world_group().local_rank,
            backend,
            group_name="fc3_quant_x",
        )

    # 以下特性依赖均匀网格推导的通信组,尚未适配 Lwd 非对称布局;
    # 开启时显式失败,避免缺组问题拖到运行期才暴露。
    finegrained = ascend_config.finegrained_tp_config
    enabled_finegrained = [
        name
        for name, size in (
            ("oproj", finegrained.oproj_tensor_parallel_size),
            ("lmhead", finegrained.lmhead_tensor_parallel_size),
            ("embedding", finegrained.embedding_tensor_parallel_size),
            ("mlp", finegrained.mlp_tensor_parallel_size),
        )
        if size and size > 0
    ]
    unsupported: list[str] = []
    if enabled_finegrained:
        unsupported.append(f"finegrained tp ({', '.join(enabled_finegrained)})")
    if flashcomm2_enable():
        unsupported.append("flashcomm2")
    if ascend_config.layer_sharding is not None:
        unsupported.append("layer_sharding")
    if unsupported:
        raise NotImplementedError(
            "features not yet supported in Lwd edge-cloud mode: "
            + ", ".join(unsupported)
        )

    logger.info(
        "Lwd ascend model parallel initialized: world_size=%s, MC2 groups=%s",
        world_size,
        mc2_groups,
    )
