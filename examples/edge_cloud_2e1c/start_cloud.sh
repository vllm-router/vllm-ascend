#!/usr/bin/env bash
# =============================================================================
# prefill-only 云侧复用（2E1C）启动脚本 —— 云 0（CLOUD_ID=0，TP=4，headless）
#
# 云侧无对外推理 API（--headless），只跑云引擎核 + 控制面 ROUTER（bind
# registry 里 clouds[0].zmq_port，边主动来连）。
# rendezvous 地址/端口取自 registry yaml 的 world 段（优先级 yaml > CLI），
# 本脚本不再传 --master-addr/--master-port。
# 用法：填好下方参数区后直接 bash start_cloud.sh（建议第二个启动，
# 详见同目录 README_vllm-ascend.md 的启动顺序说明）。
# =============================================================================
set -euo pipefail

# ------------------------------ 参数区（必填） ------------------------------
MODEL="<MODEL_PATH>"          # 模型权重路径（三机同款模型）
REGISTRY="<REGISTRY_PATH>"    # registry_2e1c.yaml 的绝对路径（三机同一份，
                              # 其 world 段指定 rendezvous 地址/端口）
# ------------------------------ 参数区（按需调整） --------------------------
PHYSICAL_NPUS=0,1,2,3         # 本机物理卡列表：卡数须等于 clouds[0].ranks 数量
EDGE_NPU_COUNT=2              # 全部边的卡数总和（= edges[].ranks 总数）
CLOUD_NPU_COUNT=4             # 云的卡数（= clouds[0].ranks 数量）
# ----------------------------------------------------------------------------

export ASCEND_RT_VISIBLE_DEVICES=$PHYSICAL_NPUS

vllm serve "$MODEL" \
  --headless \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --async-scheduling \
  --enable-prefix-caching \
  --role-registry "$REGISTRY" \
  --cloud-id 0 \
  --edge-npu-count "$EDGE_NPU_COUNT" \
  --cloud-npu-count "$CLOUD_NPU_COUNT" \
  --nnodes 2 --node-rank 0 \
  --additional-config '{"lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
  2>&1 | tee /tmp/lwd_2e1c_cloud0.log
