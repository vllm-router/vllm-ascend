# DeepSeek V4-Flash

## 1 简介

DeepSeek-V4 在 DeepSeek-V3 的基础上引入了多项关键升级：

- 流形约束超连接（mHC），用于增强传统的残差连接。

- 一种混合注意力架构，通过 Compress-4-Attention 和 Compress-128-Attention 大幅提升长上下文效率。对于混合专家（MoE）组件，仍采用 DeepSeekMoE 架构，仅做了少量调整。

DeepSeek-V4-Flash 是 DeepSeek-V4 系列的轻量级变体，适用于高吞吐量和低延迟的服务场景。

本文档将演示模型的边云协同特性的部署方式， 我们强烈建议使用`vllm-ascend:v0.23.0rc1`镜像进行部署。

## 2 前提条件

### 2.1 模型权重

DeepSeek-V4-Flash-w8a8-mtp（量化版本）：需要1个Atlas 800 A3（128GB × 8）节点或1个Atlas 800 A2（64GB × 8）节点。[下载模型权重](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)

### 2.2 节点通信

部署边云特性环境，需要按照[安装指南](../install.md)中的***节点通信验证***章节验证多节点通信。

### 2.3 一体机配置
在使用一体机作为边侧设备时，需要以下配置
1. 在物理机上插入ko(仅一体机场景)
```
insmod /usr/lib/modules/$(uname -r)$/updates/npu_peermem.ko
```
2. 准备white_list白名单（一体机和云测服务器都需要）

创建white_list.json文件，在其中写入部署推理服务所需服务器的业务面ip
```commandline
{
    "host_ip":[
        "x.x.x.x",
        "y.y.y.y"
    ]
}
```
3. 准备ranktable（一体机和云测服务器都需要）
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

## 3 环境安装

我们推荐使用[vLLM-ascend镜像](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest)创建容器，然后进行编译安装；此外我们也提供了[编译后版本的镜像](https://quay.io/user/kyoshiki/),使用该镜像的话可跳过3.3节。
### 3.1 容器创建
参考以下命令创建容器。
```commandline
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0.rc1 #可修改
docker run -itd --privileged \
--name=<container-name> \
--ipc=host --net=host\
--shm-size 500g \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device /dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu‑smi:/usr/local/sbin/npu‑smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /home/weight:/home/weight \
-it $IMAGE bash
```

表 1 参数说明

| 参数                                                 | 参数说明                                                                                                                                                                                                                                                                                                                                   |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --name                                               | 设置容器名称。                                                                                                                                                                                                                                                                                                                             |
| --device                                             | 表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinciX：NPU 设备，X 是 ID 号，如：davinci0。/dev/davinci_manager：davinci 相关的管理设备。/dev/hisi_hdc：hdc 相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据以下命令查询 device 个数及名称方式，根据需要绑定设备，修改上面命令中的 "--device=****"。`ll /dev/` |
| -v /usr/local/Ascend/driver:/usr/local/Ascend/driver | 将宿主机目录 “/usr/local/Ascend/driver” 挂载到容器，请根据驱动所在实际路径修改。                                                                                                                                                                                                                                                           |
| -v /path‑to‑weights:/path‑to‑weights                 | 设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。                                                                                                                                                                                                                                                   |

### 3.2 执行以下命令进入容器。

```commandline
docker exec -it <container-name> bash
```
### 3.3 编译安装
下载vllm和vllm-ascend代码，并编译
```commandline
#设置安装的版本，需要与下载的代码版本一致 
export VLLM_VERSION_OVERRIDE=0.23.0 
cd /vllm-workspace/vllm
VLLM_TARGET_DEVICE=empty pip install -v -e . 
cd ../vllm-ascend 
pip install -v -e . --no-build-isolation --no-deps
```

### 3.4 一体机容器内安装cann-hcomm和roce驱动(仅一体机场景)
```commandline
./cann-hcomm_*.run --full
dpkg hiroce3-*.deb
```

## 4 服务部署

根据实际情况修改启动命令中的关键配置参数，参数说明如下

表1 关键配置参数说明

| 配置项                    | 配置说明                                                                                                                                                                                                                                      |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ASCEND_RT_VISIBLE_DEVICES | 表示启用哪几张卡。对于每个模型实例分配的 npuIds，使用芯片逻辑 ID 表示。                                                                                                                                                                       |
| vllm serve                | 模型权重路径。程序会读取该路径下的 config.json 中 torch_dtype 和 vocab_size 字段的值，需保证路径和相关字段存在。必填，默认值："/data/atb_testdata/weights/llama1‑65b‑safetensors"。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。 |
| served‑model‑name         | 模型名称。                                                                                                                                                                                                                                    |
| host                      | 模型服务 IP，0.0.0.0 表示不指定 IP 地址，该服务器的 IP 皆可访问。                                                                                                                                                                             |
| port                      | 模型服务端口号。                                                                                                                                                                                                                              |
| max‑model‑len             | 模型服务上下文长度。                                                                                                                                                                                                                          |
| HCCL_IF_BASE_PORT         | 一体机hccl通信端口，需要根据使用npu卡的物理id（图1中最左侧的NPU号）进行修改。从1号卡开始为16657，卡号每加1，端口号减1。                                                                                                                                                                                     |

图1. npu-smi info回显
![img.png](../figure/npu-smi-info/img.png)

（1）边侧一体机
```bash
#!/usr/bin/bash
export VLLM_HOST_IP=76.76.7.134
export GLOO_SOCKET_IFNAME=enp125s0f1
export TP_SOCKET_IFNAME=enp125s0f1
export HCCL_SOCKET_IFNAME=enp125s0f1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=76.76.7.134  # Node 0 的实际 IP
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export USE_MULTI_BLOCK_POOL=1

export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/z30061553/vllm/white_list.json"
export RANK_TABLE_FILE='/home/z30061553/vllm/rank_table_134_158_prefill_only.json'
export HCCL_IF_BASE_PORT=16655


sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_PRE_OUT_PORT=7853
export VLLM_ASCEND_LWD_POST_OUT_PORT=7854
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8088 \
  --master-addr 76.76.7.134 \
  --master-port 29508 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 135168 \
  --max-num-batched-tokens 20480 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --trust-remote-code --nnodes 2 --node-rank 0 --edge-npu-count 1 --cloud-npu-count 8 \
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
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
  --compilation-config '{"cudagraph_mode":"NONE","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
```
（2）
云测服务器
```bash
#!/usr/bin/bash
nic_name="enp189s0f0"
local_ip="76.76.7.158"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_LWD_PRE_OUT_PORT=7853
export VLLM_ASCEND_LWD_POST_OUT_PORT=7854
export VLLM_ASCEND_LWD_PRE_OUT_HOST=76.76.7.158
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/z30061553/vllm/white_list.json"
export RANK_TABLE_FILE='/home/z30061553/vllm/rank_table_134_158_prefill_only.json'

vllm serve /weight/weight/DeepSeek-V4-Flash-w8a8-mtp \
    --safetensors-load-strategy 'prefetch' \
    --master-addr 76.76.7.134\
    --master-port 29508\
    --max-model-len 136096 \
    --max-num-batched-tokens 20480 \
     -pp 2 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --block-size 128 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --trust-remote-code --nnodes 2 --node-rank 1 --edge-npu-count 1 --cloud-npu-count 8 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256]}' \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
```

### 4.2 多实例
因为边云特性在边侧占用很少算力，所以可在边侧部署多个容器，创建多个服务和不同云侧服务器算卡对接，以下以一体机作为边侧算力，云侧使用不同服务器为例部署两实例：

(1)边测一体机容器1
```commandline
#!/usr/bin/bash
export VLLM_HOST_IP=76.76.26.140
export GLOO_SOCKET_IFNAME=enp125s0f1
export TP_SOCKET_IFNAME=enp125s0f1
export HCCL_SOCKET_IFNAME=enp125s0f1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=76.76.26.140   # Node 0 的实际 IP
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export USE_MULTI_BLOCK_POOL=1

export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/l30027191/vllm/ranktable/white_list.json"
export RANK_TABLE_FILE='/home/l30027191/vllm/ranktable/rank_table_ds.json'
export HCCL_IF_BASE_PORT=16656

export VLLM_ENGINE_READY_TIMEOUT_S=3600
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_PRE_OUT_PORT=6853
export VLLM_ASCEND_LWD_POST_OUT_PORT=6854
export HCCL_NPU_SOCKET_PORT_RANGE=18710-18730
export VLLM_ASCEND_LWD_HELLO_TIMEOUT_S=4800
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8088 \
  --master-addr 76.76.26.140 \
    --api-server-count 2 \
    --api-server-rpc-port  23657 \
  --master-port 33657 \
    --pipeline-parallel-size 1 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 300000 \
  --max-num-batched-tokens 12800 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 128 \
    --trust-remote-code --nnodes 2 --node-rank 0 --edge-npu-count 1 --cloud-npu-count 16  --instance_parallel_size 2 --instance_parallel_start_rank 0 \
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
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
  --compilation-config '{"cudagraph_mode":"NONE","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
```
(2)边侧一体机容器2
```commandline
#!/usr/bin/bash
export VLLM_HOST_IP=76.76.26.140
export GLOO_SOCKET_IFNAME=enp125s0f1
export TP_SOCKET_IFNAME=enp125s0f1
export HCCL_SOCKET_IFNAME=enp125s0f1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=76.76.26.140   # Node 0 的实际 IP
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export USE_MULTI_BLOCK_POOL=1

export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/l30027191/vllm/ranktable/white_list.json"
export RANK_TABLE_FILE='/home/l30027191/vllm/ranktable/rank_table_ds.json'
export HCCL_IF_BASE_PORT=16656

export VLLM_ENGINE_READY_TIMEOUT_S=3600
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_PRE_OUT_PORT=6853
export VLLM_ASCEND_LWD_POST_OUT_PORT=6854
export HCCL_NPU_SOCKET_PORT_RANGE=18710-18730
export VLLM_ASCEND_LWD_HELLO_TIMEOUT_S=4800
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8089 \
  --master-addr 76.76.26.140 \
    --api-server-count 2 \
    --api-server-attach  76.76.26.140:23657 \
  --master-port 33657 \
    --pipeline-parallel-size 1 \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 300000 \
  --max-num-batched-tokens 12800 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 128 \
    --trust-remote-code --nnodes 2 --node-rank 0 --edge-npu-count 1 --cloud-npu-count 16  --instance_parallel_size 2 --instance_parallel_start_rank 1 \
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
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
  --compilation-config '{"cudagraph_mode":"NONE","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
```
(3)云侧服务器1
```commandline
#!/usr/bin/bash
nic_name="enp189s0f0"
local_ip="76.76.26.192"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_LWD_PRE_OUT_PORT=6853
export VLLM_ASCEND_LWD_POST_OUT_PORT=6854
export VLLM_ASCEND_LWD_PRE_OUT_HOST=76.76.26.192
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/l30027191/vllm/ranktable/white_list.json"
export RANK_TABLE_FILE='/home/l30027191/vllm/ranktable/rank_table_ds.json'
export HCCL_NPU_SOCKET_PORT_RANGE=18710-18730
export VLLM_ASCEND_LWD_HELLO_TIMEOUT_S=4800
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve /weight/DeepSeek-V4-Flash-w8a8-mtp \
    --safetensors-load-strategy 'prefetch' \
    --master-addr 76.76.26.140\
    --master-port 33657\
    --max-model-len 300000 \
    --max-num-batched-tokens 12800 \
     -pp 2 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.94 \
    --max-num-seqs 128 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --block-size 128 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --trust-remote-code --nnodes 2 --node-rank 1  --edge-npu-count 1 --cloud-npu-count 16 --instance_parallel_size 2 --instance_parallel_start_rank 0 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256]}' \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
```
(4)云侧服务器2
```commandline
#!/usr/bin/bash
nic_name="enp189s0f0"
local_ip="76.76.26.190"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_LWD_PRE_OUT_PORT=6853
export VLLM_ASCEND_LWD_POST_OUT_PORT=6854
export VLLM_ASCEND_LWD_PRE_OUT_HOST=76.76.26.190
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/l30027191/vllm/ranktable/white_list.json"
export RANK_TABLE_FILE='/home/l30027191/vllm/ranktable/rank_table_ds.json'
export HCCL_NPU_SOCKET_PORT_RANGE=18710-18730
export VLLM_ASCEND_LWD_HELLO_TIMEOUT_S=4800
export VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve /weight/DeepSeek-V4-Flash-w8a8-mtp \
    --safetensors-load-strategy 'prefetch' \
    --master-addr 76.76.26.140\
    --master-port 33657 \
    --max-model-len 300000 \
    --max-num-batched-tokens 12800 \
     -pp 2 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.94 \
    --max-num-seqs 128 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --block-size 128 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --trust-remote-code --nnodes 2 --node-rank 1  --edge-npu-count 1 --cloud-npu-count 16 --instance_parallel_size 2 --instance_parallel_start_rank 1 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256]}' \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}}' \
```
### 4.3 云测复用

部署模型时云侧设备算例如果有空闲，可云测复用部署多个服务，以下以多台一体机作为边测算力，云侧使用一台服务器为例部署：
（1）边测一体机1
```commandline
#!/usr/bin/bash
export VLLM_HOST_IP=76.76.7.134
export GLOO_SOCKET_IFNAME=enp125s0f1
export TP_SOCKET_IFNAME=enp125s0f1
export HCCL_SOCKET_IFNAME=enp125s0f1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=76.76.7.134   # Node 0 的实际 IP
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export USE_MULTI_BLOCK_POOL=1

export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/z30061553/vllm/white_list.json"
export RANK_TABLE_FILE='/home/z30061553/vllm/rank_table_134_158_prefill_only_2e1c.json'
export HCCL_IF_BASE_PORT=16657


sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_LWD_PRE_OUT_PORT=7853
export VLLM_ASCEND_LWD_POST_OUT_PORT=7854
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
REGISTRY=/home/z30061553/vllm/2e1c/registry_2e1c_ds.yaml




vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8094 \
  --master-addr 76.76.7.134 \
  --master-port 29508 \
  --edge-id 0 \
  --role-registry "$REGISTRY" \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 135168 \
  --max-num-batched-tokens 20480 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --trust-remote-code --nnodes 3 --node-rank 0 --edge-npu-count 1 --cloud-npu-count 8 \
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
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}, "lwd_coordination": {"enabled": true, "control_url": "http://76.76.7.158:9133/v1/chat/completions", "tenant_key_file": "/home/z30061553/vllm/2e1c/edge-cloud-tenant-key", "consumer_id": "enterprise-a-edge0", "connect_timeout":5.0}}' \
  --compilation-config '{"cudagraph_mode":"NONE","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
```
（1）边测一体机2
```commandline
#!/usr/bin/bash
export VLLM_HOST_IP=76.76.7.134
export GLOO_SOCKET_IFNAME=enp125s0f1
export TP_SOCKET_IFNAME=enp125s0f1
export HCCL_SOCKET_IFNAME=enp125s0f1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export MASTER_ADDR=76.76.7.134   # Node 0 的实际 IP
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export USE_MULTI_GROUPS_KV_CACHE=1

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

export USE_MULTI_BLOCK_POOL=1

export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/z30061553/vllm/white_list.json"
export RANK_TABLE_FILE='/home/z30061553/vllm/rank_table_134_158_prefill_only_2e1c.json'
export HCCL_IF_BASE_PORT=16656


sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ASCEND_LWD_PRE_OUT_PORT=7853
export VLLM_ASCEND_LWD_POST_OUT_PORT=7854
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
REGISTRY=/home/z30061553/vllm/2e1c/registry_2e1c_ds.yaml




vllm serve /home/weight/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8095 \
  --master-addr 76.76.7.134 \
  --master-port 29508 \
  --edge-id 1 \
  --role-registry "$REGISTRY" \
  --safetensors-load-strategy 'prefetch' \
  --max-model-len 135168 \
  --max-num-batched-tokens 20480 \
  --served-model-name dsv4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --trust-remote-code --nnodes 3 --node-rank 1 --edge-npu-count 1 --cloud-npu-count 8 \
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
    --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"edge","mode":"prefill_only","edge_head_tail_layers":[0,0]}, "lwd_coordination": {"enabled": true, "control_url": "http://76.76.7.158:9133/v1/chat/completions", "tenant_key_file": "/home/z30061553/vllm/2e1c/edge-cloud-tenant-key", "consumer_id": "enterprise-a-edge1", "connect_timeout":5.0}}' \
  --compilation-config '{"cudagraph_mode":"NONE","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
```
（3）云测服务器
```commandline
#!/usr/bin/bash
nic_name="enp189s0f0"
local_ip="76.76.7.158"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export USE_MULTI_BLOCK_POOL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_LWD_PRE_OUT_PORT=7853
export VLLM_ASCEND_LWD_POST_OUT_PORT=7854
export VLLM_ASCEND_LWD_PRE_OUT_HOST=76.76.7.158
export TORCH_DISTRIBUTED_DETAIL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
export HCCL_WHITELIST_DISABLE="0"
export HCCL_WHITELIST_FILE="/home/z30061553/vllm/white_list.json"
export RANK_TABLE_FILE='/home/z30061553/vllm/rank_table_134_158_prefill_only_2e1c.json'
REGISTRY=/home/z30061553/vllm/2e1c/registry_2e1c_ds.yaml
vllm serve /weight/weight/DeepSeek-V4-Flash-w8a8-mtp \
    --safetensors-load-strategy 'prefetch' \
    --master-addr 76.76.7.134\
    --master-port 29508\
    --max-model-len 136096 \
    --max-num-batched-tokens 20480 \
    --role-registry "$REGISTRY" \
    --cloud-id 0 \
     -pp 2 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --block-size 128 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --trust-remote-code --nnodes 3 --node-rank 2 --edge-npu-count 1 --cloud-npu-count 8 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200]}' \
    --async-scheduling \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false, "enable_weight_nz_layout":true, "lwd_config":{"enabled":true,"role":"cloud","mode":"prefill_only","edge_head_tail_layers":[0,0]}, "lwd_coordination": {"enabled": true, "listen_host": "0.0.0.0", "listen_port": 9133, "instance_id": "cloud-0"}}' \
```

## 5 功能验证
服务启动后，可通过发送提示词调用模型。支持两个API接口：completions和chat.completions。使用您配置的`--served-model-name`。

Completions API：
```commandline
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<自定义模型名称>",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

Chat Completions API：
```commandline
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<自定义模型名称>",
        "messages": [
            {"role": "user", "content": "The future of AI is"}
        ],
        "max_completion_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```
预期结果：服务返回HTTP 200 OK。JSON响应包含choices字段及生成的文本。completions API的示例输出（内容已截断以保持简洁）：
```commandline
{
    "id": "cmpl-xxxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1780971952,
    "model": "<自定义模型名称>",
    "choices": [
        {
            "index": 0,
            "text": "The future of AI is a rapidly evolving landscape with breakthroughs in natural language understanding, multimodal reasoning, and autonomous agents. As models grow more capable and efficient...",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 54,
        "completion_tokens": 50
    }
}
```