#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Lwd 边云模式的 ascend 侧并行组初始化。

原生 ``init_ascend_model_parallel`` 按 (dp, pp, pcp, tp) 均匀网格切分
全局 rank,表达不了"边共享 edge_npu_count + 每云 engine
cloud_npu_count(×cloud_dp_size)"的非对称拓扑。本模块按 Lwd 布局构建 ascend 侧依赖的通信组(MC2 等)并注入
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
    edge_count = lwd.edge_npu_count
    cloud_count = lwd.cloud_npu_count
    cloud_dp = lwd.cloud_dp_size
    world_size = torch.distributed.get_world_size()
    assert world_size == lwd.lwd_world_size, (
        f"Lwd world size mismatch: got {world_size}, expected "
        f"{edge_count} + {cloud_count} * {cloud_dp}")
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # MC2 组(EP/MoE all2all 的载体,fused_moe 经 get_mc2_group() 取):
    # 对齐原生 init_ascend_model_parallel 的 dp×tp 建组语义——EP 开启时
    # 专家按 ep_size=dp×tp 分片(跨 engine),all2all 组必须罩住全部云
    # rank,否则每个 token 只能路由到半数专家,MoE 输出错(乱码)。
    # 边 rank 一组(不跑 MoE);锁步由 DPEngineCoreProc 的 wave 机制保证。
    mc2_groups: list[list[int]] = [[r for r in range(edge_count)]]
    cloud_ranks_all: list[int] = []
    for dp_idx in range(cloud_dp):
        base = lwd.cloud_engine_rank_base(dp_idx)
        cloud_ranks_all.extend(base + r for r in range(cloud_count))
    mc2_groups.append(cloud_ranks_all)
    ascend_parallel_state._MC2 = init_model_parallel_group(
        mc2_groups,
        get_world_group().local_rank,
        backend,
        group_name="mc2",
    )

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
