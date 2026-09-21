#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""LWD (layerwise disaggregated) prefill_only mode edge worker."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm.v1.core.sched.output import (
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm_ascend.batch_invariant import init_batch_invariance
from vllm_ascend.distributed.lwd_comm.lwd_parallel_init import (
    init_lwd_ascend_model_parallel,
)
from vllm_ascend.distributed.lwd_comm.service import get_lwd_comm_service
from vllm_ascend.distributed.lwd_comm.types import LwdChannelType, LwdCommRequest
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import pack_hash_payload
from vllm_ascend.worker.lwd_hash.lwd_hash_tables import load_tid2eid_tables
from vllm_ascend.worker.worker import NPUWorker

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec


# ---- token recovery / logit-rank lookup ----
def select_token_batch(
    logits: torch.Tensor, ranks: list[int]
) -> list[int]:
    """Batched rank→token lookup: one sort, one D2H sync for all rows.

    logits [R,V] rows sorted together (NPU sorts rows in parallel);
    sorted_idx[i, ranks[i]] is the token id at rank ranks[i] for row i.
    """
    _, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    row = torch.arange(len(ranks), device=logits.device)
    return sorted_idx[row, torch.tensor(ranks, device=logits.device)].tolist()


def compute_top_id_th(logits: torch.Tensor, token_id: int) -> int:
    """Look up the descending-logits ordinal of ``token_id`` (cloud side)."""
    logits = logits.reshape(-1)
    order = torch.argsort(logits, descending=True)
    return int((order == token_id).nonzero()[0].item())


# ---- edge worker ----
class LwdEdgeWorker(NPUWorker):
    """LWD edge worker: embed (prefill) + unembed (token recovery) only."""

    def load_model(self) -> None:
        """加载边侧模型，并为 DeepSeek V4 从权重文件读取 CPU Hash 路由表。"""
        super().load_model()
        self.tid2eid_tables: list[torch.Tensor] = []
        config = self.model_config.hf_config
        if getattr(config, "model_type", None) != "deepseek_v4":
            return
        num_hash_layers = config.num_hash_layers
        if not 0 <= num_hash_layers <= config.num_hidden_layers:
            raise ValueError(f"Invalid DeepSeek V4 num_hash_layers: {num_hash_layers}")
        if num_hash_layers == 0:
            logger.info("[lwd-edge] tid2eid loading skipped: num_hash_layers=0")
            return
        logger.info("[lwd-edge] loading tid2eid tables for %d Hash MoE layers", num_hash_layers)
        # Reuse vLLM's checkpoint resolution (revision, cache and shard filtering).
        loader = DefaultModelLoader(self.vllm_config.load_config)
        _, weight_files, use_safetensors = loader._prepare_weights(
            self.model_config.model,
            subfolder=None,
            revision=self.model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
        )
        if not use_safetensors:
            raise ValueError("LWD DeepSeek V4 tid2eid loading requires safetensors weights")
        self.tid2eid_tables = load_tid2eid_tables(
            weight_files, num_hash_layers, config.vocab_size,
            config.num_experts_per_tok, config.n_routed_experts,
        )

    def _init_worker_distributed_environment(self) -> None:
        """覆写原生入口(worker.py):ascend 侧并行组按 Lwd 布局构建。

        vllm 侧分组由 parallel_state.initialize_model_parallel 的 Lwd
        分支完成;ascend 侧原生 init_ascend_model_parallel 按 (dp, pp,
        pcp, tp) 均匀网格切分,表达不了非对称边云拓扑,故以
        init_lwd_ascend_model_parallel 替代(构建 MC2 等组后注入)。
        其余步骤与原生 worker.py 保持一致。"""
        init_batch_invariance()
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            "hccl",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )
        init_lwd_ascend_model_parallel(self.parallel_config)
        ensure_ec_transfer_initialized(self.vllm_config)

    def init_device(self):
        # The duplex channels MUST be built here, never in ``__init__``.
        # ``init_lwd_duplex_channels`` creates the two HCCL process groups
        # (``dist.new_group``) and then warms them up with a two-sided P2P
        # exchange (world barrier plus the edge/cloud isend/irecv pair), so it
        # needs the distributed environment -- world group and PP group --
        # which ``NPUWorker._init_worker_distributed_environment`` only builds
        # inside ``init_device``.  The executor constructs the worker first and
        # calls ``init_device`` afterwards, so running it from ``__init__``
        # would trip the "group is not initialized" assertions; the two-sided
        # warmup additionally requires both peers to reach it in the same fixed
        # order, which only holds once every rank is past its distributed init.
        super().init_device()
        self.comm_service = get_lwd_comm_service()

        from vllm_ascend.distributed import lwd_wire
        lwd_wire.init_lwd_duplex_channels()
        logger.info(
            "[lwd-edge] worker ready: duplex channels (UP/DOWN) initialized "
            "on global rank=%d",
            self.rank,
        )

    def compile_or_warm_up_model(self):
        # LWD 边侧运行期 forward 被 LWD 流程劫持(execute_model 只走
        # embed/unembed 直调),模型级 warmup 与 cudagraph 捕获均用不到;
        # 且 0 层拓扑下 full-forward 会在 final norm 解包失败。整体跳过。
        from vllm.v1.worker.worker_base import CompilationTimes

        logger.info("[lwd-edge] skip model warmup/capture (forward is hijacked by LWD)")
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_kv_cache_spec(self) -> dict[str, "KVCacheSpec"]:
        """The LWD edge runs no attention/transformer, so it needs no KV cache."""
        return {}

    def execute_model(self, scheduler_output: "SchedulerOutput"):
        lwd_batch = scheduler_output.lwd_batch
        if lwd_batch is None:
            logger.debug("[lwd-edge] step carries no LWD batch; nothing to do")
            return None

        batch_meta = lwd_batch.batch_meta
        # 多实例路由 stamp(§4.3/§4.4):控制面(实例派发)落地前默认 0,
        # 即全部流量走实例 0 的通道对——单实例行为与打标前逐比特一致。
        instance_id = int(getattr(lwd_batch, "instance_id", 0) or 0)
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            logger.info(
                "[Lwd][edge-worker] EMBED seqno=%d instance=%d reqs=%d "
                "tokens=%d",
                lwd_batch.seqno,
                instance_id,
                len(batch_meta.req_ids),
                sum(len(token_ids) for token_ids in batch_meta.token_ids),
            )
            return self._execute_lwd_embed(
                lwd_batch.seqno, batch_meta, instance_id
            )
        if lwd_batch.batch_type == LwdBatchType.LWD_UNEMBED:
            logger.info(
                "[Lwd][edge-worker] UNEMBED seqno=%d instance=%d reqs=%d "
                "accepted=%d num_elements=%d",
                lwd_batch.seqno,
                instance_id,
                len(batch_meta.req_ids),
                sum(batch_meta.num_accept_tokens),
                batch_meta.recv_num_elements,
            )
            return self._execute_lwd_unembed(
                lwd_batch.seqno, batch_meta, instance_id
            )

        logger.debug(
            "[lwd-edge] unknown LWD batch type %r; nothing to do",
            lwd_batch.batch_type,
        )
        return None

    def _execute_lwd_embed(
        self, seqno: int, batch_meta: LwdEmbedBatch, instance_id: int = 0
    ) -> None:
        model = self.model_runner.get_model()
        device = self.model_runner.device
        if not batch_meta.token_ids:
            return

        # Flatten all requests' prompt tokens into one batch (order = req_ids).
        flat_token_ids = [tid for token_ids in batch_meta.token_ids for tid in token_ids]
        logger.debug(
            "[Lwd][edge-worker] embed token_ids=%s", flat_token_ids
        )
        token_ids_tensor = torch.tensor(flat_token_ids, dtype=torch.long, device=device)
        _t0 = time.monotonic()
        embeds = model.embed_input_ids(token_ids_tensor)      # (total_N, H)
        _t_fwd = time.monotonic()
        payload = embeds
        if self.tid2eid_tables:
            # The original prompt IDs stay on the edge; transmit only lookups.
            cpu_ids = torch.tensor(flat_token_ids, dtype=torch.int64)
            expert_ids = torch.stack([table[cpu_ids] for table in self.tid2eid_tables], dim=1)
            payload = pack_hash_payload(embeds, expert_ids)
            logger.info(
                "[lwd-edge] UP Hash MoE seqno=%d expert_shape=%s payload_numel=%d",
                seqno, tuple(expert_ids.shape), payload.numel(),
            )
        request = LwdCommRequest(
            channel=LwdChannelType.UP,
            op="send",
            num_elements=payload.numel(),
            tensor=payload,
            seqno=seqno,
            instance_id=instance_id,
        )
        self.comm_service.submit_send(request)
        # [Lwd][perf] TTFT 探针:forward=embed 前向;submit_send=快照clone+
        # 广播提交(含 bridge 的 handle.wait)——若 UP 世界广播在等云侧
        # 入队,此段会显著变大(chunk 级锁步的直接证据)
        logger.info(
            "[Lwd][perf] embed seqno=%s forward=%.2f submit_send=%.2f "
            "total=%.2fms tokens=%d",
            seqno, (_t_fwd - _t0) * 1000,
            (time.monotonic() - _t_fwd) * 1000,
            (time.monotonic() - _t0) * 1000,
            len(flat_token_ids),
        )

    def _execute_lwd_unembed(
        self, seqno: int, batch_meta: LwdUnembedBatch, instance_id: int = 0
    ) -> ModelRunnerOutput:
        model = self.model_runner.get_model()
        if not batch_meta.req_ids:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        _t0 = time.monotonic()
        recv_future = self.comm_service.submit_recv(
            LwdCommRequest(
                channel=LwdChannelType.DOWN,
                op="recv",
                num_elements=batch_meta.recv_num_elements,  # int = rows_total * hidden_size
                seqno=seqno,
                instance_id=instance_id,
            )
        )
        _t_post = time.monotonic()
        # wait_for_comm:device 序等待,主机不阻塞
        recv_future.wait_for_comm()
        result = recv_future.result()
        _t_tensor = time.monotonic()
        hidden_size = self.model_config.get_hidden_size()
        hidden_states = result.tensor.view(-1, hidden_size)  # (rows_total, H)
        # lm_head 不允许直接调用(ParallelLMHead.forward 强制经 sampler);
        # compute_logits 是标准接口,包装层/单体模型都有。
        logits = model.compute_logits(hidden_states)       # (rows_total, V)
        _t_lm = time.monotonic()
        # L5 对拍:首行 top-20,与云侧 sampler 入口的 [layer-trace]
        # top20 逐位对照——排名换位即重放漂移的直接视图。
        try:
            from vllm_ascend.worker.lwd_layer_trace import (
                lwd_layer_trace_enabled,
            )

            if lwd_layer_trace_enabled() and logits.dim() == 2:
                row = logits[0].detach().float()
                vals, tids = torch.topk(row, min(20, row.numel()))
                logger.info(
                    "[layer-trace] edge lm_head logits shape=%s row0 l2=%.4f "
                    "top20=%s",
                    tuple(logits.shape), row.norm().item(),
                    list(zip(tids.tolist(), [round(v, 3) for v in vals.tolist()])),
                )
        except Exception:  # noqa: BLE001
            pass

        flat_ids = select_token_batch(
            logits, [r for ths in batch_meta.top_id_ths for r in ths]
        )
        sampled_token_ids: list[list[int]] = []
        off = 0
        for accept, ths in zip(
            batch_meta.num_accept_tokens, batch_meta.top_id_ths
        ):
            sampled_token_ids.append(flat_ids[off : off + accept])
            off += len(ths)
        _t_sel = time.monotonic()
        # [Lwd][perf] 临时探针:单请求慢的归因分段——
        # post_recv=挂 irecv;wait_tensor=等张量落卡(网络);lm_head=词表
        # 前向(权重带宽);select=argsort 取名(.item 同步);total=边尾段全长
        logger.info(
            "[Lwd][perf] unembed seqno=%s post_recv=%.2f wait_tensor=%.2f "
            "lm_head=%.2f select=%.2f total=%.2fms reqs=%d",
            seqno,
            (_t_post - _t0) * 1000,
            (_t_tensor - _t_post) * 1000,
            (_t_lm - _t_tensor) * 1000,
            (_t_sel - _t_lm) * 1000,
            (_t_sel - _t0) * 1000,
            len(batch_meta.req_ids),
        )
        req_id_to_index = {
            req_id: index for index, req_id in enumerate(batch_meta.req_ids)
        }
        return ModelRunnerOutput(
            req_ids=batch_meta.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
        )
