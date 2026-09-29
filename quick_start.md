# 快速入门

## 环境准备

本文档以 Atlas 800I A2 推理服务器和 Qwen3.6‑27B 模型为例，让开发者快速开始使用 VLLM 进行大模型推理流程。

### 前提条件
1. 服务器安装

（1）Atlas服务器

操作系统安装参考：[Atlas服务器openEuler 22.03 LTS SP4 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100469526/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

NPU驱动和固件参考：[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

（2）一体机服务器

操作系统安装参考：[Atlas服务器openEuler 22.03 LTS SP4 操作系统安装](https://support.huawei.com/enterprise/zh/doc/EDOC1100469526/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

NPU驱动和固件安装参考：[Atlas A2 中心推理和训练硬件 26.0.RCx NPU驱动和固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100568431/426cffd9?idPath=23710424|269761314|269769397|269769917|254184887)

SP681网卡驱动安装参考：[SP220&SP600 标准网卡 用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100573093/426cffd9),
[驱动下载地址](https://support.huawei.com/enterprise/zh/computing-module/in220-pid-253287505/software/268628573?idAbsPath=fixnode01|23710424|269761301|269764423|269766398|253287505)


2. 获取模型权重

下载权重，这里以 Qwen3.6‑27B 为例，将权重文件上传至服务器任意目录（如 /home/weight），修改权重文件权限：

```
chmod -R 755 /home/weight
```

## docker部署
### 使用vllm-ascend预构建镜像

1. 进入[昇腾官方镜像仓库](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest)，根据设备型号选择下载对应的镜像。

该镜像已具备模型运行所需的基础环境，包括：CANN、FrameworkPTAdapter、VLLM 与 VLLM‑Ascend，可实现模型快速上手推理。

表 2 容器内各组件安装路径


| 组件 | 安装路径 |
| --- | --- |
| CANN | /usr/local/Ascend/cann |
| CANN‑NNAL‑ATB | /usr/local/Ascend/nnal/atb |
| VLLM | /vllm‑workspace/vllm |
| VLLM‑Ascend | /vllm‑workspace/vllm‑ascend |


2. 下载并加载镜像后，参考以下命令创建容器。

```
docker run -itd --privileged --name=<container-name> --ipc=host --net=host  --shm-size 500g  \
--device=/dev/davinci0  --device=/dev/davinci1  --device=/dev/davinci2  --device=/dev/davinci3  \
--device=/dev/davinci4  --device=/dev/davinci5  --device=/dev/davinci6  --device=/dev/davinci7  \
--device=/dev/davinci_manager  --device=/dev/hisi_hdc  --device /dev/devmm_svm  \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu‑smi:/usr/local/sbin/npu‑smi  -v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf -v /home/weight:/home/weight  fe1e88748912
```

> 
> [!NOTE] 说明
> 
> 
> - “fe1e88748912” 为镜像 ID，可根据实际情况修改。

表 1 参数说明

表格

| 参数 | 参数说明 |
| --- | --- |
| --name | 设置容器名称。 |
| --device | 表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinciX：NPU 设备，X 是 ID 号，如：davinci0。/dev/davinci_manager：davinci 相关的管理设备。/dev/hisi_hdc：hdc 相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据以下命令查询 device 个数及名称方式，根据需要绑定设备，修改上面命令中的 "--device=****"。`ll /dev/` |
| -v /usr/local/Ascend/driver:/usr/local/Ascend/driver | 将宿主机目录 “/usr/local/Ascend/driver” 挂载到容器，请根据驱动所在实际路径修改。 |
| -v /path‑to‑weights:/path‑to‑weights | 设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。 |

3. 执行以下命令进入容器。

```
docker exec -it <container-name> bash
```
4. 编译安装

下载vllm和vllm-ascend代码，并编译
```
#设置安装的版本，需要与下载的代码版本一致 
export VLLM_VERSION_OVERRIDE=0.23.0 
cd /vllm-workspace/vllm
 VLLM_TARGET_DEVICE=empty pip install -v -e . 
 cd ../vllm-ascend 
 pip install -v -e . --no-build-isolation --no-deps
```
### 使用[vllm-router/vllm-ascend]()预构建镜像直接创建容器无需额外编译


## 边云协同分布式推理服务部署
1. [使用标准服务器](README_server.md)

2. [使用一体机](README_all-in-one.md)

## [更多模型部署参考](examples.md)



