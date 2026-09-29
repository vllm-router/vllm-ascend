<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

---
*最新消息* 🔥

- [2026/09] 我们发布了首个边云协同版本 [v0.23.0rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0rc1)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/)开始在 Ascend 上部署边云协同推理服务。

<details>
<summary>更多内容</summary>

- [2026/09] vLLM社区正式创建了[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)仓库，让vLLM可以无缝运行在Ascend NPU。
- [2026/9] 我们正在与 vLLM 社区合作，以支持 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

</details>

---

## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个由社区维护的让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

支持的模型详细信息，请参考[模型支持列表](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html)。

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas A2 Training系列、Atlas 800I A3 Inference系列、Atlas A3 Training系列、Atlas 300I Duo（实验性支持）
- 操作系统：Linux
- 软件：
    - Python >= 3.10, < 3.13
    - CANN == 9.0.1 (Ascend HDK 版本详见 [版本说明](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/releasenote_0000.html))
    - PyTorch == 2.10.0, torch-npu == 2.10.0.post2
    - vLLM (与vllm-ascend版本一致)

## 开始使用

推荐您使用以下版本快速开始使用：

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.23.0rc1 | 最新RC版本 | 请查看[快速入门](quick_start.md)和[安装指南]()了解更多 |

## 支持模型
图例说明

- ●：充分验证支持
- ●：仅功能支持
- ✅：支持该列的特性
- ❌：不支持该列特性

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek‑R1‑Distill‑Llama‑8B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| DeepSeek‑R1‑Distill‑Llama‑70B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |
| DeepSeek‑R1‑Distill‑Qwen‑1.5B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡（推荐1卡） |
| DeepSeek‑R1‑Distill‑Qwen‑7B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| DeepSeek‑R1‑Distill‑Qwen‑14B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
| DeepSeek‑R1‑Distill‑Qwen‑32B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：1/2/4卡(推荐2卡) |
[更多](model_support_lists.md)

## 贡献

请参考[CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。
- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。

## 社区例会

- vLLM Ascend 每周社区例会: <https://tinyurl.com/vllm-ascend-meeting>
- 每周三下午，15:00 - 16:00 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。
