# vLLM‑Ascend 边云典型配置参数
本文档汇总 Qwen3.6‑27B、Kimi‑K2.5、DeepSeek‑V4‑Flash‑w8a8‑mtp 三款模型的边云部署配置参数。

## 边云模式说明
边云部署（Edge‑Cloud）将模型推理拆分到边侧和云侧两个节点，支持以下两种模式：

|模式|说明|edge_head_tail_layers 取值|
|---|---|---|
|embedding_only|边侧仅做 Embedding，云侧做全部 Transformer 层|0|
|head_tail|边侧承载首 N 层和尾 M 层，云侧承载中间层|整数或数组|

其中 head_tail 模式按 edge_head_tail_layers 的取值格式分为两种方式：

|方式|edge_head_tail_layers 格式|含义|适用模型|
|---|---|---|---|
|首一尾一|整数 1|边侧承载首 1 层和尾 1 层|Qwen3.6‑27B、Kimi‑K2.5|
|首三尾一|数组 [3, 1]|边侧承载首 3 层和尾 1 层|DeepSeek‑V4‑Flash|

# 一、Qwen3.6‑27B
## 1.1 A2 边云 — 首一尾一模式（head_tail）
边侧 2 NPU + 云侧 8 NPU，边侧承载首 1 层和尾 1 层，云侧承载中间层。

### 边侧环境变量
```bash
export VLLM_HOST_IP=<边侧IP>
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0
export ASCEND_RT_VISIBLE_DEVICES=4,5
```

### 边侧启动命令
```bash
vllm serve /weight/Qwen3.6-27B \
    --served-model-name "qwen3.6-27B" \
    --host 0.0.0.0 \
    --port 8314 \
    --master-addr <边侧IP> \
    --master-port 29501 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --trust-remote-code \
    --nnodes 2 --node-rank 0 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "edge_cloud_config":{"enabled":true,"role":"edge","mode":"head_tail","enable_decode_graph":true,"edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]}'
```

### 云侧环境变量
```bash
export VLLM_HOST_IP=<云侧IP>
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 云侧启动命令
```bash
vllm serve /weight/Qwen3.6-27B \
    --served-model-name "qwen3.6-27B" \
    --headless \
    --master-addr <边侧IP> \
    --master-port 29501 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --trust-remote-code \
    --nnodes 2 --node-rank 1 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "edge_cloud_config":{"enabled":true,"role":"cloud","mode":"head_tail","enable_decode_graph":true,"edge_head_tail_layers":1}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]}'
```

## 1.2 A2 边云 — Embedding Only 模式
边侧 1 NPU（仅 Embedding）+ 云侧 8 NPU（全部 Transformer 层）。

### 边侧环境变量
```bash
export VLLM_HOST_IP=<边侧IP>
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_GDN_FAST_PATH=1
export VLLM_ASCEND_GDN_MAX_PADDING_RATIO=2.0
export VLLM_ASCEND_GDN_MAX_H_OVERALLOC_RATIO=2.0
export ASCEND_RT_VISIBLE_DEVICES=4
```

### 边侧启动命令
```bash
vllm serve /weight/Qwen3.6-27B \
    --served-model-name "qwen3.6-27B" \
    --host 0.0.0.0 \
    --port 8314 \
    --master-addr <边侧IP> \
    --master-port 29501 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --trust-remote-code \
    --nnodes 2 --node-rank 0 --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "edge_cloud_config":{"enabled":true,"role":"edge","mode":"embedding_only","enable_decode_graph":true,"edge_head_tail_layers":0}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]}'
```

### 云侧环境变量
同 1.1 云侧环境变量。

### 云侧启动命令
```bash
vllm serve /weight/Qwen3.6-27B \
    --served-model-name "qwen3.6-27B" \
    --headless \
    --master-addr <边侧IP> \
    --master-port 29501 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --trust-remote-code \
    --nnodes 2 --node-rank 1 --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "edge_cloud_config":{"enabled":true,"role":"cloud","mode":"embedding_only","enable_decode_graph":true,"edge_head_tail_layers":0}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]}'
```

## 1.3 Qwen3.6‑27B 配置对比速查表
|场景|边云模式|边侧 NPU|云侧 NPU|edge_head_tail_layers|max‑model‑len|max‑num‑seqs|
|---|---|---|---|---|---|---|
|A2|head_tail（首一尾一）|2|8|1|262144|128|
|A2|embedding_only|1|8|0|262144|128|

## 1.4 Qwen3.6‑27B 特性开关
|特性|参数|说明|
|---|---|---|
|关闭 Prefix Cache|`--no-enable-prefix-caching`|混合 KV cache 开启 prefix cache 时 block_size 过大，短前缀无法缓存，建议关闭|
|Function Call|`--enable-auto-tool-choice --tool-call-parser qwen3_coder`|启用自动工具调用|
|Reasoning Content|`--reasoning-parser qwen3`|启用思维链输出|

> 注意：边云场景暂不支持投机解码（`--speculative-config`），拉边云时需删除此参数。

# 二、Kimi‑K2.5
## 2.1 A3 边云 — 纯文本场景
### 2.1.1 Embedding Only 模式
边侧 1 NPU（仅 Embedding）+ 云侧 16 NPU（Atlas 800 A3），max‑model‑len=133120。

#### 边侧环境变量
```bash
nic_name="enp189s0f0"
local_ip="<边侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export ASCEND_RT_VISIBLE_DEVICES=4
```

#### 边侧启动命令
```bash
vllm serve /home/extra/kimi25_w4a8_static_m6 \
    --host 0.0.0.0 \
    --port 9234 \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29234 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 96 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"embedding_only","edge_head_tail_layers":0,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0
```

#### 云侧环境变量
```bash
nic_name="enp196s0f0"
local_ip="<云侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

#### 云侧启动命令
```bash
vllm serve /weight/kimi25_w4a8_static_m6 \
    --headless \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29234 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 96 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","mode":"embedding_only","edge_head_tail_layers":0,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0
```

### 2.1.2 首一尾一模式（head_tail）
边侧 2 NPU（首 1 层+尾 1 层）+ 云侧 16 NPU（Atlas 800 A3），max‑model‑len=133120。

#### 边侧环境变量
```bash
nic_name="enp189s0f0"
local_ip="<边侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

#### 边侧启动命令
```bash
vllm serve /home/extra/kimi25_w4a8_static_m6 \
    --host 0.0.0.0 \
    --port 9233 \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29233 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 96 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"head_tail","edge_head_tail_layers":1,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0
```

#### 云侧环境变量
同 2.1.1 云侧环境变量。

#### 云侧启动命令
```bash
vllm serve /weight/kimi25_w4a8_static_m6 \
    --headless \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29233 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 96 \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","mode":"head_tail","edge_head_tail_layers":1,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0
```

## 2.2 A3 边云 — 多模态场景
### 2.2.1 Embedding Only 模式（多模态）
边侧 1 NPU + 云侧 16 NPU（Atlas 800 A3），max‑model‑len=16384。

#### 边侧环境变量
```bash
nic_name="enp189s0f0"
local_ip="<边侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

#### 边侧启动命令
```bash
vllm serve /home/extra/kimi25_w4a8_static_m6 \
    --host 0.0.0.0 \
    --port 9090 \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29531 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"embedding_only","edge_head_tail_layers":0,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data
```

#### 云侧环境变量
同 2.1.1 云侧环境变量。

#### 云侧启动命令
```bash
vllm serve /mnt/mnt1/kimi25_w4a8_static_m6 \
    --headless \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29531 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud --edge-npu-count 1 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","mode":"embedding_only","edge_head_tail_layers":0,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data
```

### 2.2.2 首一尾一模式（多模态）
边侧 2 NPU + 云侧 16 NPU（Atlas 800 A3），max‑model‑len=16384。

#### 边侧环境变量
同 2.1.2 边侧环境变量。

#### 边侧启动命令
```bash
vllm serve /home/extra/kimi25_w4a8_static_m6 \
    --host 0.0.0.0 \
    --port 9090 \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29531 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 0 \
    --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"edge","mode":"head_tail","edge_head_tail_layers":1,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data
```

#### 云侧环境变量
同 2.1.1 云侧环境变量。

#### 云侧启动命令
```bash
vllm serve /weight/kimi25_w4a8_static_m6 \
    --headless \
    --quantization ascend \
    --master-addr <边侧IP> \
    --master-port 29531 \
    --served-model-name kimi2.5 \
    --no-enable-prefix-caching \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --nnodes 2 \
    --node-rank 1 \
    --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 16 \
    --enable-expert-parallel \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --async-scheduling \
    --additional-config '{"edge_cloud_config":{"enabled":true,"role":"cloud","mode":"head_tail","edge_head_tail_layers":1,"enable_decode_graph":true}}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data
```

## 2.3 Kimi‑K2.5 配置对比速查表
|场景|边云模式|边侧 NPU|云侧 NPU|edge_head_tail_layers|max‑model‑len|max‑num‑seqs|max‑num‑batched‑tokens|多模态|
|---|---|---|---|---|---|---|---|---|
|A3 纯文本|embedding_only|1|16|0|133120|96|8192|否|
|A3 纯文本|head_tail（首一尾一）|2|16|1|133120|96|8192|否|
|A3 多模态|embedding_only|1|16|0|16384|64|16384|是|
|A3 多模态|head_tail（首一尾一）|2|16|1|16384|64|16384|是|

> 注意：边云场景暂不支持投机解码（`--speculative-config`）和均衡调度（`VLLM_ASCEND_BALANCE_SCHEDULING`），拉边云时需删除这些参数。边侧不需要开启 `VLLM_ASCEND_ENABLE_FLASHCOMM1`，仅云侧开启。

# 三、DeepSeek‑V4‑Flash‑w8a8‑mtp
## 3.1 A2 边云 — 首三尾一模式（head_tail）
边侧 2 NPU + 云侧 8 NPU，边侧承载首 3 层和尾 1 层（edge_head_tail_layers: `[3, 1]`），云侧承载中间层。

### 边侧环境变量
```bash
nic_name="enp189s0f1"
local_ip="<边侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

### 边侧启动命令
```bash
vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 7906 \
  --master-addr <边侧IP> \
  --master-port 29600 \
  --trust-remote-code --nnodes 2 --node-rank 0 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 70000 \
  --max-num-batched-tokens 20480 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"edge_cloud_config":{"enabled":true,"role":"edge","enable_decode_graph":true,"edge_head_tail_layers":[3,1]}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]}'
```

### 云侧环境变量
```bash
nic_name="enp189s0f0"
local_ip="<云侧IP>"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HOST_IP="<云侧IP>"
```

### 云侧启动命令
```bash
vllm serve /mnt1/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --headless \
  --master-addr <边侧IP> \
  --master-port 29600 \
  --trust-remote-code --nnodes 2 --node-rank 1 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 70000 \
  --max-num-batched-tokens 20480 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"edge_cloud_config":{"enabled":true,"role":"cloud","enable_decode_graph":true,"edge_head_tail_layers":[3,1]}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]}'
```

## 3.2 A3 边云 — 首三尾一模式（head_tail）
边侧 2 NPU + 云侧 8 NPU，边侧承载首 3 层和尾 1 层（edge_head_tail_layers: `[3, 1]`），云侧承载中间层。

### 边侧环境变量
```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_USE_V1=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=512
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export ASCEND_RT_VISIBLE_DEVICES=4,5
```

### 边侧启动命令
```bash
vllm serve /home/extra/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 7906 \
  --master-addr <边侧IP> \
  --master-port 29600 \
  --trust-remote-code --nnodes 2 --node-rank 0 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 10240 \
  --max-num-batched-tokens 20480 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"edge_cloud_config":{"enabled":true,"role":"edge","enable_decode_graph":true,"edge_head_tail_layers":[3,1]}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

### 云侧环境变量
```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_USE_V1=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=512
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export CODEBASE_DIR="/vllm-workspace"
export PYTHONPATH="${CODEBASE_DIR}/vllm-ascend:${CODEBASE_DIR}/vllm:${PYTHONPATH}"
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 云侧启动命令
```bash
vllm serve /weight/DeepSeek-V4-Flash-w8a8-mtp \
  --headless \
  --master-addr <边侧IP> \
  --master-port 29600 \
  --trust-remote-code --nnodes 2 --node-rank 1 --enable-edge-cloud --edge-npu-count 2 --cloud-npu-count 8 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 10240 \
  --max-num-batched-tokens 20480 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --block-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"edge_cloud_config":{"enabled":true,"role":"cloud","enable_decode_graph":true,"edge_head_tail_layers":[3,1]}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

## 3.3 DS‑V4‑Flash 配置对比速查表
|场景|边侧 NPU|云侧 NPU|edge_head_tail_layers|max‑model‑len|max‑num‑seqs|max‑num‑batched‑tokens|HCCL_BUFFSIZE|FLASHCOMM1|OMP_NUM_THREADS|
|---|---|---|---|---|---|---|---|---|---|
|A2 边云|2|8|[3, 1]|70000|256|20480|1024|1|1|
|A3 边云|2|8|[3, 1]|10240|128|20480|512|0|10|

> 注意：边云场景暂不支持投机解码（`--speculative-config`），拉边云时需删除此参数。

# 四、通用参数说明
## 4.1 边云公共参数
|参数|说明|
|---|---|
|`--nnodes`|总节点数，边云场景为 2|
|`--node-rank`|节点编号。边侧为 0，云侧为 1|
|`--enable-edge-cloud`|启用边云协同|
|`--edge-npu-count`|边侧 NPU 数量 ，embedding_only（仅 Embedding）通常数量为1， head_tail（首尾层分离）通常数量为2|
|`--cloud-npu-count`|云侧 NPU 数量|
|`--master-addr`|主节点（边侧）IP 地址|
|`--master-port`|主节点端口|
|`--headless`|云侧需加此参数，表示非主节点|
|`--async-scheduling`|异步调度，边云场景必须开启|

## 4.2 边云配置（additional‑config 中 edge_cloud_config）
|字段|说明|
|---|---|
|enabled|边云总开关，设为 true|
|role|角色：边侧设 edge，云侧设 cloud|
|mode|边云模式：embedding_only（仅 Embedding）或 head_tail（首尾层分离）|
|edge_head_tail_layers|边侧承载的层数配置。整数格式表示首一尾一（如 1），数组格式表示首 N 尾 M（如 [3, 1]）|
|enable_decode_graph|边侧是否启用 decode 图，推荐 true|

## 4.3 编译配置参数（compilation‑config）
|字段|说明|
|---|---|
|cudagraph_mode|图模式，推荐 FULL_DECODE_ONLY，仅对 decode 阶段建图|
|cudagraph_capture_sizes|图捕获粒度，根据 max‑num‑seqs 设定|

## 4.4 关键环境变量
|环境变量|说明|
|---|---|