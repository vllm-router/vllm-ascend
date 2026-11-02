# LWD vllm-ascend NPU 适配仓

> [!NOTE]
> **本文件定位**: 团队隐形知识的载体. 团队隐性知识(代码位置、设计意图、约定、边界)必须落入本文件, 而不是留在成员脑中口口相传.
> 录入重点是隐形知识, 不是具体操作步骤 —— agent 通用具备的技能(如 git 命令用法)不记录.

本仓是 [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) 的团队 fork, 为 LWD 边云分布式推理系统提供昇腾 NPU 后端. 控制面与业务逻辑在 vllm 仓 (vllm-data-comm/vllm); 本仓按规范 7.8 只承载 vllm-ascend 覆写 vllm 的部分 (worker / model runner / 分布式通信 / KV 传输 / 底层接口) 与 NPU 适配.

## 存量代码 (mm 分支)

- **基线**: LWD 改动在远端分支 `prefill_only_chunked_prefill_mm`, main 目前只迁入了数据面 (见下). 该分支基于上游 vllm-ascend v0.23.0 并持续 cherry-pick 上游修复.
- **LWD 新增模块**:

  | 位置 | 职责 |
  | --- | --- |
  | `vllm_ascend/worker/lwd_edge_model/`、`vllm_ascend/worker/lwd_cloud_model/` | 边/云 worker 与 model runner; 云侧 early recv、host staging、MTP proposer |
  | `vllm_ascend/worker/lwd_hash/`、`lwd_token_rank.py`、`lwd_layer_trace.py` | 请求 hash 路由与 hash 表、token 排名、层 trace |
  | `vllm_ascend/distributed/lwd_comm/`、`lwd_wire.py` | 边云数据面通信 (channel / future / pre_recv / payload_scramble) 与线协议 |
  | `tests/ut/worker/test_lwd_*.py` | LWD 单元测试 (draft cache、hash decode、token rank、v4 mixed migration) |

- **数据面已迁入 main**: 在 `vllm_ascend/distributed/lwd_data_plane/`, 由 mm 的 `lwd_comm/`、`lwd_wire.py` 迁移并改名 (`LwdCommService` -> `LwdDataPlane`, `LwdChannel` -> `LwdTransferQueue`, `LwdCommFuture` -> `LwdTransferFuture`, `lwd_wire` -> `lwd_channels`, `PayloadScrambler` -> `LwdPayloadScrambler` 等), 提前收已迁入 `worker/lwd_cloud_model/lwd_pre_recv.py`; 建组从 LWD 拓扑推导, 设计见 vllm 仓 `docs/design/lwd_data_plane.md`. 从 mm 移植调用方时改用新名字. `LwdChannelType` 的值 (`lwd_up`/`lwd_down`) 参与加扰种子推导, 其 repr 与 `[Lwd][perf] bridge-wait` 日志行被 mm 的 `log_analyze_tools/lwd_backlog_probe.py` 解析, 改名或改日志时要保持不变.
- **UP host staging 单路径**: 云 endpoint comm thread 从 hint 预投 UP receive, `lwd_transfer_queue.py` 以 recv/staged events 排序分段 D2H; `lwd_host_stage.py` 持有每请求全 prompt pinned buffer 到请求结束 (无容量 cap), `lwd_cloud_model_runner.py` 按 arrival 窗口经独立 stream H2D 并推进 consumed. 云侧 preemption 最小重算保留 W1, runner 在 native `_update_states` 前按新 computed rewind host consumed 或对 self-hit 前跳, visible-set 将 preempted 置 waiting 队头. 全 prompt M-RoPE 走 RequestNotify, DSV4 embeddings+expert tail 同 buffer 布局, abort/tombstone 只屏蔽该请求的 D2H/arrival、保持 seqno 几何; 默认 pending/inflight/slots 为 10/8/8. 设计见 vllm 仓 `docs/design/lwd_host_staging.md`.
- **Prefix caching/refill**: nonce/READY 协商 H, pinned frozen reservation fail-closed; 重准入 H′ 经 arrival/块对齐后对比 staging floor S, H′<S 时按预算切 refill、dispatch 时公告并以 nonce 上报 refill_done, self-hit 前跳. hash/cache 仅覆盖 prompt; MM 仅处理 chunk 相交 feature, prefix MTP guard 拒绝 continued prefill 的 token-id fallback 并要求权威 embeddings. LWD lookup 的 `shared_prefix_boundary=0`; native Mamba fine-grained junction 机制默认关闭. 详见 vllm 仓 `docs/design/lwd_host_staging.md`.

> [!CAUTION]
> **团队规范区**: 下方由 `TEAM STANDARDS (BEGIN)` 与 `TEAM STANDARDS (END)` 标记圈出的区域是团队规范原文, 仅由 committer 通过评审维护.
> 在日常编码/文档任务中, **严禁修改区内任何内容**, 包括格式化、重排、翻译、增删条目.
> 若任务确实要求变更团队规范, 先停下来向用户确认, 并提示修改需经 committer 评审后才能合入.

<!-- ================ TEAM STANDARDS (BEGIN) ================ -->

## (分布式推理) 代码规范与团队规范

### 1. 基本信息

- **PR 登记表**: [PR登记表](https://onebox.huawei.com/v/5df49d10f60ca9174a522758f023c6c7?type=0&sheet=PR%E7%99%BB%E8%AE%B0%E8%A1%A8)
- **检视会议**: 每周四下午 16:00~17:00
- **会议内容**: 逐条过登记表中待评审的 PR; 有分歧的当场讨论定论; 不能到场的提前在登记表留言

### 2. 合入规范

1. 必须committer评审后再合入。
2. 提pr需要使用pr模板
   - 验证报告
   - 通过CI
   - 至少 1 名 committer Approve

### 3. Commit 风格

示例:

```text
feat(lwd): add support for xxx

feat(npu): add flash attention support for Ascend CANN
perf: avoid CPU-NPU sync in attention computation
```

常用标签: `feat` / `fix` / `perf` / `refactor` / `docs` / `test` / `ci` / `chore`

规则:

- 我们使用`(lwd)`
- ***建议让编写代码的那个agent session起草commit message***
- 用祈使句, 不限制长度, 有意放宽 conventional commits 的 50/72 字符惯例
- commit按照功能划分; 一个 commit 只做一件事
- 分散的commit要合并
- commit 语言: 英文

### 4. PR 标签

```text
[Bugfix][Lwd] fix xxx sync problem

[Bugfix][KV Offload] fix kernel budget
[Perf][ROCm][Kernel] improve MoE
```

**类型标签**:

- `[Bugfix]`、`[Feature]`、`[Perf]`、`[Refactor]`: 代码变更
- `[CI]` 或 `[CI/Build]`、`[Test]`、`[Doc]` 或 `[Docs]`: 项目基础设施、测试和文档
- `[Misc]`: 其他类型标签都不合适时

**范围标签**:

- 代码库主要区域:`[Model]`、`[Frontend]`、`[Rust Frontend]`、`[Core]`、`[Kernel]`
- 高频变更子系统: `[MRV2]`、`[Attention]`、`[Multimodal]`、`[Quantization]`、`[MoE]`、`[Spec Decode]`、`[LoRA]`、`[KV Connector]`、`[KV Offload]`

规则:

- 我们使用`[Lwd]`
- 建议使用gh工具提pr
- **紧急合入且没有跑 CI 的, 必须打 `no-test` label**

### 5. PR 模板 (可以保留中文; commit仍然要英文)

```markdown
# 方案 (复杂的需要方案架构图, 不复杂的不需要)

# 评审截图 (涉及外部的需要评审, 不涉及的不需要)

# 修改影响 (类似代码注释的功能, 建议补充, 方便回忆意图)

# 测试结果

## 命令
<!-- 运行命令 -->

## 修改前
<!-- 截图 -->

## 修改后
<!-- 截图 -->

# Checklist
- [ ] 实现全面, 功能完备 -- 需要人看!
- [ ] 避免 if-else 堆叠: 公共流程抽象为基类, 差异流程用子类继承承载 -- 需要人看!
- [ ] 模块间隔离: 模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块 -- 需要人看!
- [ ] 性能敏感路径要评估性能影响 -- 需要人看!
- [ ] 命名要清晰 -- 需要人看!

- [ ] PR描述说明要清晰, 完备
- [ ] PR要有自验证报告, 和测试截图
- [ ] vllm 和 vllm-ascend 的 PR 描述中要互相贴关联链接

- [ ] bugfix需要补充回归测试
- [ ] CI 已通过, 否则打`no-test`标签

- [ ] 避免侵入式修改, 不要使用patch
- [ ] fail-fast, 避免往下传递异常情况
- [ ] 类型标注要全面
- [ ] 函数职责要单一
- [ ] 注释不要冗余
- [ ] 公共函数要提取
- [ ] 检查魔法数字
- [ ] 没做完的地方是否有TODO注释
- [ ] 使用英文, 不要使用中文
- [ ] 测试是否覆盖失败分支
- [ ] 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.

```

### 6. RFC

大的架构修改一般都有设计文档; 这种修改要新建一个 issue(RFC) , 在issue里附上设计文档, 并说明思路:

- **触发条件**: 架构性修改超过 500 LOC(不含 kernel/data/config/test)
- RFC 内容建议: 背景与动机、方案设计(含图)、备选方案对比、影响面
- 对应 PR 必须关联 RFC issue

参考:
<https://github.com/vllm-project/vllm-ascend/issues/2470>
`[RFC]: Proposal for Optimizing KVCache Transfer via Layer-wise Strategy in Disaggregation
 #2470`

### 7. 代码规范

#### 7.1 工具链

- 统一使用 uv 管理与运行
- 统一使用 ruff 格式化与 lint, 必须安装 VSCode ruff插件
- ruff 配置需要与上游 vllm 保持一致. (效果待测试)
- 提交前需要运行precommit hook
- 测试和分发使用docker

```bash
uv tool install pre-commit
pre-commit install

ruff check
ruff format
```

#### 7.2 类型标注

- 函数签名必须标注参数与返回类型
- 签名需要细化, 避免只写一个Callable, Dict, 过于宽泛的
- 有能力的可以使用@overload增加对于None等边界条件的静态检查

```python
# wrong
def register_failure_callback(self, callback: Callable):

# right
FailureCallback = Callable[[], None]
def register_failure_callback(self, callback: FailureCallback):

# @overload
@overload
def convert_tokens_to_ids(self, tokens: str) -> int: ...

@overload
def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:

```

#### 7.3 import

- import 放文件开头; 例外情况(循环依赖、可选依赖、延迟导入优化)不需要
- 注意: ruff 格式化无法将 import 放到文件开头

#### 7.4 魔法数字

- 避免魔法数字, 提取为常量
- 例外: 魔法数字比变量名更清晰时可以保留 (按照品味来)

#### 7.5 命名 / 文件组织 / 函数设计

- 遵循 Python 官方规范(PEP 8)
- 描述性命名: 名字描述**功能**, 不描述实现细节
- 保持代码好味道. 注意编码风格, 类型注释, 公共函数要提取
- 开发时要同时进行代码微重构
- 命名风格要一致: 类、函数、变量命名的前缀统一: Lwd（layerwise_disaggregated）, LwdEdge、LwdCloud.
- 类名使用lwd开头, 新增文件包在lwd路径下面
- 尽量保持非侵入式修改
- 函数行数尽量不要超100行, 函数职责要单一
- 函数、变量命名要能清晰表达其意思, 函数入参尽量控制在5个; 函数名尽量不要太复杂, 也不要太简单导致无法表达其意思

```python
# Bad
is_deepseek_v3_r1
flag1
tmp_var

# Good
supports_dynamic_temperature
uses_speculative_decoding
```

#### 7.6 注释与 TODO

- 代码能自注释的, 不要写注释.
- 特殊场景, 关键算法、绕坑逻辑必须要有解释意图的注释, 其他简单场景不要有注释 -- 必须让用户人工审查
- 未完成项必须打 TODO 注释:`# TODO(name): 描述,关联 issue #xxx`
- 注释用英文

#### 7.7 默认不打patch

- 我们主要是在自己的代码里修改.
- 例外: 昇腾对vllm打了patch, 我们改动很小, 改动的是patch的内容, 不得不在patch内部改动, 否则没有必要.

#### 7.8 主要在 vllm 改代码, 除非是 vllm-ascend 覆写的部分(比如调度器/底层接口等)

#### 7.9 使用lwd的配置文件; 尽量不要使用环境变量

- 使用lwd的配置文件
- 特殊情况, 配置里无法重载的, 经过评审后可以使用
- vllm已存在可用的环境变量时, 尽量和社区保持一致.
- 环境变量前缀统一 `VLLM_LWD_*`
- **需要合理默认值**
- **严禁在性能关键分支读取**: 初始化时读取一次并缓存, 热路径禁止反复 `os.getenv`
- 修改配置文件时要同步维护全景图. 人工维护/ai维护. 需要示例.

#### 7.10 ***模块高内聚, 用抽象降低流程复杂度***

公共流程抽象为基类, 差异流程用子类继承承载; 主流程避免 if-else 分支堆叠, 降低流程复杂度
模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块

需要补充例子.

#### 7.11 fail-fast编码, 避免将异常情况往下传递

- 提前校验, fail-fast快速失败
- 减少错误抛出运行时异常或error日志, 仅在必要时使用

典型例子:

- lwd的类和函数内部不需要再判断是否是lwd

#### 7.12 关键分支和关键点需要info日志

- 日志使用`[LWD]`
- 后续等日志模块实现后, 要使用自己的日志模块和自己的级别

#### 7.13 要有设计文档并归档到代码仓

- 使用markdown + mermaid.
- 没有markdown要贴链接. 缺少时要提醒用户补充.

### 8. 提交流程

#### 8.1 pre-commit hook

```bash
uv tool install pre-commit
pre-commit install
```

- 运行过 pre-commit 后, commit 会自动带 signoff (DCO,上游合入必需)

#### 8.2 commit 整理

- commit 按功能分割, 同一功能的分散 commit 合并, 不同功能不得混并.
- 不要将没有整理的分散的commit提成pr合入仓库.
- 可以让 AI 辅助整理
- **注意先备份分支**:`git branch backup/<name>` 或 push 到个人远端, 防止整理翻车丢代码

#### 8.3 分支

- fork 后提 PR、不要直接 push
- 建议使用gh的官方命令行工具, 可以减少人工操作

### 9. 测试(pytest)

- 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.
- ai编写的测试大部分都很冗余.
- ***ai编写的测试必须列列表让用户审查.***
- 能不能写一个skill来写符合要求的测试

- **UT**(单元测试): 快速、聚焦单个模块
- **ST**(系统测试): 集成/端到端, 需要真实 NPU 环境
- **Nightly**(夜航): 全量回归 + 长稳

要求:

- 必须覆盖失败分支(异常、边界条件), 不能只测 happy path
- 修复回归问题时, 必须补充能复现该 bug 的测试
- 性能关键代码需要独立的性能测试(附基线对比数据)

运行示例:

```bash
# Run specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs
```

### 10. CI

主要是在线ci. 本地跑test即可.

- 目标: CI 保持在 5min 以内
- CI 如何调用 claude-bot 或其他 bot 进行代码审查？
- CI 不绿不允许合入; 紧急情况打 `no-test` 并事后补跑?

急需补充的功能:

- 检查函数行数是否超过100loc
- 检查类名使用lwd开头
- 检查新增文件在lwd路径下面
- 检查函数入参数量在5个及以下, 不符合的要列出表格写回pr评论里, 让committer确认, 确认ok即可放行
- 检查是否使用了patch
- ai-bot检查Checklist
- 超过500loc检查是否关联 RFC issue和设计文档

### 11. Committer 评审

评审重点是了解代码的修改影响, 评估功能实现是否全面, 审查代码可维护性.

- committer要负责维护pr模板和checklist. checklist要保证语言精练, 可以快速确认.
- committer要负责维护pre-commit hook和ai-bot, 将可自动审查的落地成可重复运行的.
- committer要负责将团队的隐性知识落地成agent可阅读的文档.
- committer要负责维护`.github`里的自动化工具和pr的label
- 评审测试用例对于committer的负担会比较重. 可以将重心放在代码功能上

保持合理性, 不要一刀切

### 12. 维护全景图

方便定位问题, 方便新员工上手, 方便快速定位, 方便ai定位
将隐性知识落地.

skill?
ai维护?

<!-- ================ TEAM STANDARDS (END) ================ -->

---

## LWD CI

- CI 检查与 workflow 集中在 [`vllm-data-comm/ci-infra`](https://github.com/vllm-data-comm/ci-infra)（设计文档 `docs/design/lwd_ci.md`），本仓只放调用文件；改检查规则去 ci-infra 提 PR，两个仓同时生效
- 每项检查两个属性：跑在哪条线 × 必须通过/纯粹检查。必须通过（PR 标题、commit+signoff、pre-commit）卡合入；纯粹检查（入参数、函数长度、类名前缀、新文件路径、patch 使用、RFC 关联、bugfix 无测试）只写进 PR 的 checklist 评论，由 committer 评审时判断
- 本轮两仓都只接快线；慢线（LWD 单测、mypy）等 main 完成 mm 分支迁移、测试能代表代码后再接，跟踪在 [ci-infra#4](https://github.com/vllm-data-comm/ci-infra/issues/4)
- 每次 merge 上游后，重跑 `ci-infra` 仓的 `scripts/disable_inherited_workflows.sh vllm-data-comm/<repo>`，停用上游新带来的 workflow
- 本地跑检查：clone ci-infra 后 `uv run python -m lwd_checks.<module> --help`

---

## vllm-ascend 开发工作流

> 本节按本仓语境由上游 vllm-ascend 的 AGENTS.md 裁剪而来; merge 上游时注意核对上游变化, 勿直接用上游原文覆盖本节.

### 环境搭建与安装

测试与分发统一使用 docker (根目录按硬件提供多个 Dockerfile). 源码安装需先装配套的 vllm (empty target), 再可编辑安装本仓:

```bash
cd /vllm-workspace/vllm
# 按需设置 VLLM_VERSION_OVERRIDE
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ../vllm-ascend
pip install -v -e . --no-build-isolation --no-deps
```

### 环境变量

- 所有环境变量必须集中在 `vllm_ascend/envs.py` 的 `env_variables` 字典中定义, 命名遵循 `VLLM_ASCEND_*` 前缀; 在字典注释中补文档: 默认值与有效范围、是否敏感(凭据/密钥).
- 严禁在代码里散落硬编码环境变量名, 统一 `from vllm_ascend import envs` 引用.
- 团队新增开关优先走 lwd 配置文件 (规范 7.9); 确需环境变量的, 评审按上述条目逐项检查.

### 测试

- **UT**: `tests/ut/`, 快速聚焦单个模块
- **ST**: `tests/e2e/`, 端到端与集成
- **Nightly**: `tests/e2e/nightly/`, 全量回归、长稳与基准

```bash
# 单个测试文件
pytest -sv tests/ut/ops/test_prepare_finalize.py

# 单个用例
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# NPU 用例 (需真机)
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py::test_default_full_and_piecewise_res_consistency
```

NPU 相关改动必须在真机 (Ascend 910B/C) 上验证后再请求评审.

### Lint 与提交前检查

```bash
# 本仓要求所有文件类型过检, 含 markdown; 若 markdownlint 自动改了文件需重新 git add
bash format.sh ci

# pre-commit hooks (signoff hook 会自动补 DCO 签名)
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

首次运行 pre-commit 会联网下载并构建 hook 环境. 外网不通时:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
SKIP=actionlint \
pre-commit run --files <files>
```

值得知道的本地 hook: `check-long-functions` (新函数超 100 行必须有注释)、`check-logger` (vllm_ascend 模块禁用 `init_logger(__name__)`)、`check-forbidden-imports`.

### NPU 性能注意事项

- **`tensor.item()` 同步开销**: 对 device tensor 调用 `item()` 会触发 NPU→CPU 同步拷贝, 热路径 (如 AsyncScheduler) 会被阻塞. 无法避免时 (日志、条件分支) 要注释说明, 并尽量合并为单次同步:

```python
# Bad: 热循环内逐个同步
for tensor in tensors:
    value = tensor.item()

# Better: 批量取, 单次同步
values = [t.item() for t in tensors]

# Good: 尽量留在 device 上
max_value = torch.max(tensor)
if max_value > threshold:  # 比较可留在 device
    ...
```

- 热路径避免 CPU-NPU 内存搬运; 安全场景优先原地操作 (`x.add_()`, `x.mul_()`)
- 长稳场景关注内存碎片; 性能改动需在真机用真实负载验证并附基线对比
