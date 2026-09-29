# VLLM 支持模型列表

图例说明

- ●：充分验证支持
- ●：仅功能支持
- ✅：支持该列的特性
- ❌：不支持该列特性

## 文本生成模型

### DeepSeek 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek‑R1‑Distill‑Llama‑8B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| DeepSeek‑R1‑Distill‑Llama‑70B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |
| DeepSeek‑R1‑Distill‑Qwen‑1.5B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡（推荐1卡） |
| DeepSeek‑R1‑Distill‑Qwen‑7B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| DeepSeek‑R1‑Distill‑Qwen‑14B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
| DeepSeek‑R1‑Distill‑Qwen‑32B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：1/2/4卡(推荐2卡) |
| DeepSeek‑MoE‑16B‑Chat | ● | 4k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡（推荐4卡）   Atlas 800I A3：不支持   Atlas 300I Duo：不支持 |
| DeepSeek‑V2‑Chat‑236B | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |
| DeepSeek‑V3‑0324 | ● | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |
| DeepSeek‑R1‑0528 | ● | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |
| DeepSeek‑V3.1 | ● | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |
| DeepSeek‑V3.1‑Terminus | ● | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |

### Qwen 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2‑1.5B | ● | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| Qwen2‑7B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| Qwen2‑72B‑Instruct | ● | 128k | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐8卡)   Atlas 800I A3：1/2/4卡(推荐4卡)   Atlas 300I Duo：1/2/4卡(推荐4卡) |
| Qwen2.5‑0.5B‑Instruct | ● | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |
| Qwen2.5‑1.5B‑Instruct | ● | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| Qwen2.5‑3B‑Instruct | ● | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Qwen2.5‑7B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4(推荐2卡)   Atlas 800I A3：1/2卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| Qwen2.5‑14B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：1/2/4卡(推荐2卡) |
| Qwen2.5‑32B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)   Atlas 800I A3：2/4卡(推荐2卡) |
| Qwen2.5‑72B‑Instruct | ● | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |
| Qwen3‑0.6B | ● | 32k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |
| Qwen3‑1.7B | ● | 32k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |
| Qwen3‑4B | ● | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 300I Duo：1卡 |
| Qwen3‑8B | ● | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |
| Qwen3‑14B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：2卡   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |
| Qwen3‑32B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：4卡   Atlas 800I A3：2卡   Atlas 300I Duo：2卡 |
| Qwen3‑30B‑A3B‑Instruct‑2507 | ● | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡（推荐4卡）   Atlas 800I A3：2/4卡（推荐2卡）   Atlas 300I Duo：2卡 |
| Qwen3‑235B‑A22B‑Instruct‑2507 | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：4卡8芯 |
| Qwen3‑Coder‑30B‑A3B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：4卡8芯 |
| Qwen3‑Coder‑480B‑A35B‑Instruct | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：4卡8芯 |

### Llama 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama3‑8B | ● | 8k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Llama3‑70B | ● | 8k | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |
| Llama3.1‑8B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Llama3.1‑70B | ● | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |
| Llama3.1‑405B | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡 |

### ERNIE 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERNIE‑4.5‑300B‑A47B | ● | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：8卡   Atlas 300I Duo：不支持 |

### KIMI 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi‑K2‑Instruct | ● | 32k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：32卡   Atlas 800I A3：16卡   Atlas 300I Duo：不支持 |
| Kimi‑K2‑Thinking | ● | 32k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：32卡   Atlas 800I A3：16卡   Atlas 300I Duo：不支持 |

### GLM 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChatGLM2‑6B | ● | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| ChatGLM3‑6B | ● | 8k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| ChatGLM3‑6B‑32K | ● | 32k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| GLM4‑9B | ● | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |
| GLM‑4.5 | ● | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡   Atlas 800I A3：待测试   Atlas 300I Duo：不支持 |

### Bloom

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bloom‑7B | ● | 4096 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：4卡(推荐1卡) |

### Baichuan

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baichuan2‑7B | ● | 4096 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1卡 |
| Baichuan2‑13B | ● | 4096 | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2卡(推荐1卡) |

## 多模态理解模型

### Qwen 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2‑VL‑2B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| Qwen2‑VL‑7B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| Qwen2‑VL‑72B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)   Atlas 800I A3：2/4卡(推荐4卡) |
| Qwen2.5‑VL‑3B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Qwen2.5‑VL‑7B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Qwen2.5‑VL‑32B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)   Atlas 800I A3：2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
| Qwen2.5‑VL‑72B | ● | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)   Atlas 800I A3：2/4卡(推荐4卡)   Atlas 300I Duo：2/4卡(推荐4卡) |
| QVQ‑72B‑Preview | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)   Atlas 800I A3：2/4卡(推荐4卡) |
| Qwen2‑Audio‑7B | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| Qwen3‑VL‑2B | ● | 256K | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)   Atlas 300I Duo：1/2/4/8卡(推荐1卡) |
| Qwen3‑VL‑4B | ● | 256K | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)   Atlas 300I Duo：1/2/4/8卡(推荐1卡) |
| Qwen3‑VL‑8B | ● | 256K | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 300I Duo：1/2/4/8卡(推荐2卡) |
| Qwen3‑VL‑32B | ● | 256K | ✅ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)   Atlas 300I Duo：2/4/8卡(推荐4卡) |
| Qwen3‑VL‑30B‑A3B | ● | 256k | ✅ | ✅ | ❌ | Atlas 800I A2：2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |

### InternVL 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| InternVL2‑8B | ● | 8k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| InternVL2‑40B | ● | 8k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡) |
| InternVL2.5‑8B | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| InternVL2.5‑78B | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：8卡   Atlas 800I A3：4卡 |

### GLM 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| GLM‑4V‑9B | ● | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| GLM‑4.1V‑9B‑Thinking | ● | 64k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |

### MiniCPM‑V

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM‑V 2.6 | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)   Atlas 800I A3：1卡   Atlas 300I Duo：1卡 |

### VITA

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| VITA‑1.5 | ● | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1卡   Atlas 800I A3：1卡 |

### LLaVa 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVa‑1.5‑7B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| LLaVa‑1.5‑13B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
| LLaVa‑v1.6‑7B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| LLaVa‑v1.6‑13B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)   Atlas 800I A3：1/2/4卡(推荐2卡)   Atlas 300I Duo：1/2/4卡(推荐2卡) |
| LLaVa‑v1.6‑34B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)   Atlas 800I A3：2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
| LLaVa‑NeXT‑Video‑7B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| LLaVa‑NeXT‑Video‑34B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)   Atlas 800I A3：2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |

### Llama 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2‑Vision‑11B | ● | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡) |
| Llama 3.2‑Vision‑90B | ● | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)   Atlas 800I A3：2/4卡(推荐4卡) |

### Yi‑VL 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 |
| --- | --- | --- | --- | --- | --- | --- |
| Yi‑VL‑6B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)   Atlas 800I A3：1/2/4卡(推荐1卡)   Atlas 300I Duo：1/2/4卡(推荐1卡) |
| Yi‑VL‑34B | ● | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)   Atlas 800I A3：2/4卡(推荐2卡)   Atlas 300I Duo：2/4卡(推荐2卡) |
