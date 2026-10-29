# 使用标准服务器部署边云协同分布式推理服务(边云都使用标准服务器)

本文档以 Atlas 800I A2 推理服务器和 Qwen3.6‑27B 模型为例，让开发者快速开始使用 VLLM 进行大模型推理流程。

## 前提条件
1. 服务器安装完成


2. 容器部署完成


3. 网络部署

与标准服务器不同，一体机使用SP681网卡作为参数面和云测服务器通信，需要在网卡上配置到云测服务器网卡地址的路由

## 边云协同服务部署
1. 在物理机上插入ko(仅仅一体机场景)
```
insmod /usr/lib/modules/$(uname -r)$/updates/npu_peermem.ko
```
2. 进入一体机容器,在容器内安装cann-hcomm和roce驱动(仅一体机场景)
```commandline
./cann-hcomm_*.run --full
dpkg hiroce3-*.deb
```

3. 准备white_list白名单

创建white_list.json文件，在其中写入部署推理服务所需服务器的业务面ip
```commandline
{
    "host_ip":[
        "x.x.x.x",
        "y.y.y.y"
    ]
}
```

4. 准备ranktable
```commandline
{
    "status":"completed",   // ranktable可用标识，completed为可用
    "version":"1.0"         // ranktable模板版本信息,当前必须为"1.0"
    "server_count":"2",  //参与推理的AI Server数目，此例中，只有两个AI Server
    "server_list":[              
        {      //以下为rank信息，一体机在前，服务器在后
            "server_id":"10.0.0.10",   //AI Server标识,即业务面网卡ip
            "container_ip":"10.0.0.10",
            "host_ip":"10.0.0.10",
            "para_net_position":"host",   //类型一体机参数面使用host网卡，为"host"; 服务器使用npu网卡，为"device"
            "para_net_protocol":"rdma",   //  统一使用rdma 
            "device":[
                {
                    "rank_id":"0",      //文件中从0开始计数
                    "device_id":"1",    // npu-smi info 查询出的npu卡id
                    "device_ip":"192.168.1.1",  // 一体机参数面host网卡ip
                    "device_port":"16666",
                    "host_port":60002
                }
            ]
        },
        {
            "server_id":"11.0.0.10",  
            "container_ip":"11.0.0.10",
            "host_ip":"11.0.0.10",
            "para_net_position":"device",
            "para_net_protocol":"rdma", 
            "device":[
                {
                    "rank_id":"1", 
                    "device_id":"0",
                    "device_ip":"192.168.2.2",  //npu网卡地址
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"2", 
                    "device_id":"1",
                    "device_ip":"192.168.2.3",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"3", 
                    "device_id":"2",
                    "device_ip":"192.168.2.4",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"4", 
                    "device_id":"3",
                    "device_ip":"192.168.2.5",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"5", 
                    "device_id":"4",
                    "device_ip":"192.168.2.6",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"6", 
                    "device_id":"5",
                    "device_ip":"192.168.2.7",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"7", 
                    "device_id":"6",
                    "device_ip":"192.168.2.8",
                    "device_port":"16666",
                    "host_port":60002
                },
                {
                    "rank_id":"8", 
                    "device_id":"7",
                    "device_ip":"192.168.2.9",
                    "device_port":"16666",
                    "host_port":60002
                }
            ]   
        }
    ]
}

```

5. 进入容器后，执行以下命令启动服务。
   
(1)边侧一体机

```
#!/bin/bash

# ========== 环境变量 ==========
export VLLM_HOST_IP=<边侧IP>
export GLOO_SOCKET_IFNAME=<边侧通信网卡名>
export TP_SOCKET_IFNAME=<边侧通信网卡名>
export HCCL_SOCKET_IFNAME=<边侧通信网卡名>
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=<边侧IP>
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="<白名单文件路径>"
export RANK_TABLE_FILE='<rank_table文件路径>'
export HCCL_IF_BASE_PORT=16657
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
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
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_PRE_OUT_PORT=7200
export VLLM_ASCEND_LWD_POST_OUT_PORT=7201
export VLLM_LOGGING_LEVEL=INFO

# ========== 启动命令 ==========
vllm serve /home/weight/Qwen3.6-27B \
    --host 0.0.0.0 \
    --port <port> \
    --master-addr <边侧IP> \
    --master-port <master-port> \
    --max-model-len 262144 \
    --served-model-name <模型识别名称> \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 1 \
    --trust-remote-code --nnodes 2 --node-rank 0 --edge-npu-count 1 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --max-num-seqs 196 \
    --max-num-batched-tokens 8192 \
    --async-scheduling \
    --gpu-memory-utilization 0.9
```

（2）云测服务器

```commandline
#!/bin/bash

# ========== 环境变量 ==========
export VLLM_HOST_IP=<云侧IP>
export GLOO_SOCKET_IFNAME=<云侧通信网卡名>
export TP_SOCKET_IFNAME=<云侧通信网卡名>
export HCCL_SOCKET_IFNAME=<云侧通信网卡名>
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=<边侧IP>
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="<白名单文件路径>"
export RANK_TABLE_FILE='<rank_table文件路径>'
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
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
export VLLM_ASCEND_EDGE_CLOUD_MERGE_PAYLOAD=1
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ASCEND_LWD_PRE_OUT_PORT=7200
export VLLM_ASCEND_LWD_POST_OUT_PORT=7201
export VLLM_ASCEND_LWD_PRE_OUT_HOST=<云侧IP>

# ========== 启动命令 ==========
vllm serve /weight/weight/Qwen3.6-27B \
    --master-addr <边侧IP> \
    --master-port <master-port> \
    --max-model-len 262144 \
    --served-model-name <自定义模型名称> \
    -pp 2 \
    --tensor-parallel-size 8 \
    --trust-remote-code --nnodes 2 --node-rank 1  --edge-npu-count 1 --cloud-npu-count 8 \
    --additional-config '{"enable_cpu_binding":true, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256,260,264,268,272,276,280,284,288,292,296,300,304,308,312,316,320,324,328,332,336,340,344,348,352,356,360,364,368,372,376,380,384,388,392,396,400,404,408,412,416,420,424,428,432,436,440,444,448,452,456,460,464,468,472,476,480,484,488,492,496,500,504,508,512,516,520,524,528,532,536,540,544,548,552,556,560,564,568,572,576,580,584,588,592,596,600,604,608,612,616,620,624,628,632,636,640,644,648,652,656,660,664,668,672,676,680,684,688,692,696,700,704,708,712,716,720]}' \
    --max-num-seqs 196 \
    --max-num-batched-tokens 8192 \
    --async-scheduling \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"num_speculative_tokens": 3, "method":"qwen3_5_mtp", "enforce_eager": true}'
```

根据实际情况修改启动命令中的配置参数，参数说明如下

表1 配置参数说明

| 配置项 | 配置说明 |
| --- | --- |
| ASCEND_RT_VISIBLE_DEVICES | 表示启用哪几张卡。对于每个模型实例分配的 npuIds，使用芯片逻辑 ID 表示。 |
| vllm serve | 模型权重路径。程序会读取该路径下的 config.json 中 torch_dtype 和 vocab_size 字段的值，需保证路径和相关字段存在。必填，默认值："/data/atb_testdata/weights/llama1‑65b‑safetensors"。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。 |
| served‑model‑name | 模型名称。 |
| host | 模型服务 IP，0.0.0.0 表示不指定 IP 地址，该服务器的 IP 皆可访问。 |
| port | 模型服务端口号。 |
| max‑model‑len | 模型服务上下文长度。 |

2. 发送请求。
用户可使用 HTTPS 客户端（Linux curl 命令，Postman 工具等）发送 HTTPS 请求，此处以 Linux curl 命令为例进行说明。
重开一个窗口，使用以下命令发送请求。例如验证服务是否拉起：

```
curl http://<IP>:<Port>/v1/chat/completions \
  -H "Content‑Type: application/json" \
  -d '{
    "model": "qwen3.6‑27b",
    "messages": [
      {
        "role": "user",
        "content": "who are you"
      }
    ],
    "max_tokens": 2048,
    "temperature": 0
  }' | python3 -m json.tool
```

回显如下则表明请求发送成功：

```
{
    "id": "chatcmpl‑984264b9b96eee5d",
    "object": "chat.completion",
    "created": 1787584707,
    "model": "qwen3.6‑27B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's a thinking process:\n\n1.  **Analyze User Input:** The user asks \"who are you\"\n2.  **Identify Core Intent:** The user wants to know my identity, capabilities, and origin.\n3.  **Check Internal Guidelines/Constraints:**\n   - I am Qwen (通义千问), developed by Alibaba Group's Tongyi Lab.\n   - I should respond clearly, concisely, and truthfully about my identity.\n   - I should maintain a helpful and professional tone.\n4.  **Formulate Response:**\n   - Acknowledge the question directly.\n   - State my identity: I'm Qwen, a large language model developed by Alibaba Group's Tongyi Lab.\n   - Briefly mention my purpose/capabilities: I'm designed to assist with a wide range of tasks like answering questions, writing, coding, analysis, and more.\n   - Keep it concise and open to further questions.\n5.  **Draft Response (Mental Refinement):**\n   I'm Qwen, a large language model developed by Alibaba Group's Tongyi Lab. I'm designed to help with a wide variety of tasks—whether that's answering questions, writing, coding, analyzing data, or just having a thoughtful conversation. How can I assist you today?\n6.  **Final Check against Guidelines:** Matches identity, concise, helpful tone, no extra fluff. Ready.✅\n\n\nI'm Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. I'm designed to assist with a wide range of tasks, including answering questions, writing, coding, logical reasoning, data analysis, and more. How can I help you today?",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null,
                "reasoning": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "stop_reason": null,
            "token_ids": null,
            "routed_experts": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": "vllm‑0.23.0‑pp2‑4559800a",
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 372,
        "completion_tokens": 359,
        "prompt_tokens_details": null,
        "completion_tokens_details": null
    },
    "prompt_logprobs": null,
    "prompt_token_ids": null,
    "prompt_text": null,
    "kv_transfer_params": null
}
```