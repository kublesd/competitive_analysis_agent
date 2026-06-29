# Stage 08：LangGraph Workflow

## 1. 本阶段目标

本阶段把已经独立工作的 Planner、Researcher、Extractor、Analyst 和 Verifier
连接成一个可运行的 LangGraph 工作流。

输入：

- 目标产品；
- 竞品；
- 分析维度；
- 搜索配置。

输出：

- 调研任务；
- Evidence；
- ProductProfile；
- CompetitiveAnalysis；
- VerificationResult；
- 阶段执行历史和重试次数。

完整数据流：

```text
START
  -> Planner
  -> Researcher
  -> Extractor
  -> Analyst
  -> Verifier
       ├─ passed -> END
       └─ failed and retry_count < 1 -> Analyst -> Verifier -> END
```

本阶段只负责状态和路由，不实现 Reporter、UI、持久化或长期 Memory。

## 2. 实现结果

- 完成的功能：
  - `WorkflowGraphState` 显式保存输入、各阶段产物、错误、验证结果和重试状态；
  - `WorkflowComponents` 集中注入五个已独立测试的业务组件；
  - 五个普通 node wrapper 从 State 读取输入并返回部分更新；
  - 使用固定边连接 Planner、Researcher、Extractor、Analyst 和 Verifier；
  - 在 Extractor 和 Analyst 之间执行 `validate_product_profiles_for_analysis()`，
    清理明显不在请求范围内的价格、冲突价格和套餐级定位；
  - profile validation 产生的原因会进入 `research_errors`，后续 Reporter 可展示为数据限制；
  - 使用条件边根据 `retry_pending` 选择结束或返回 Analyst；
  - Verifier issues 会转换为 Analyst 的 `revision_feedback`；
  - `MAX_ANALYSIS_RETRIES=1` 保证验证回路最多执行一次；
  - `stage_history` 保存简化节点轨迹，`graph.stream()` 可观察逐节点更新；
  - `live_workflow.py` 使用真实 LLM 节点和固定搜索 provider 完成整图验收。
- 关键文件：
  - `competitive_analysis_agent/workflow.py`
  - `competitive_analysis_agent/live_workflow.py`
  - `tests/test_workflow.py`
  - `tests/test_live_workflow.py`
  - `tests/fixtures/workflow_search_results.json`
- 核心数据流：

```text
PlannerInput
  -> research_tasks
  -> Evidence + research_errors
  -> ProductProfile
  -> Profile validation + research_errors
  -> CompetitiveAnalysis
  -> VerificationResult
  -> passed: END
  -> failed: issues -> Analyst -> Verifier -> END
```

- 验证方式与结果：
  - `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_workflow.py -q`：
    9 个图测试通过；
  - `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py tests\test_workflow.py -q`：
    89 个关键链路测试通过；
  - `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`：
    154 个离线测试通过，7 个真实用例默认排除；
  - `python -m compileall -q competitive_analysis_agent tests`：通过；
  - happy path、一次修订、二次失败终止、独立节点和 stream 更新均有测试覆盖。
- 暂未实现：
  - Reporter 和 Markdown 报告；
  - LangGraph checkpointer、任务恢复和长期持久化；
  - Researcher/Extractor 并行执行；
  - 真实搜索供应商；
  - Streamlit UI 和专业 tracing 平台。

## 3. 设计决策

### 决策 1：为什么先把节点写成普通 Python，再接入 LangGraph？

**问题背景**

如果业务逻辑从一开始就写在图框架回调中，节点错误和路由错误会混在一起，难以测试。

**当前方案**

复用 Stage 3-7 已独立测试的类；LangGraph 节点函数只负责从 State
读取输入、调用组件和返回部分状态更新。

**替代方案**

把 Planner、Extractor 等逻辑直接写进 LangGraph node 函数。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 复用独立组件 | 节点和业务逻辑可分别测试，框架耦合低 | 多一层薄包装 |
| 逻辑直接写入 node | 初始文件较少 | 业务代码依赖图状态，复用和调试困难 |

**什么时候考虑切换**

不建议把核心业务重新塞进 node。只有极简单、只服务单个图的状态转换可以直接写在
节点中。

**面试回答参考**

我先把每个 Agent 做成普通 Python 组件并独立测试，Stage 8 才用薄节点包装接入
LangGraph。这样业务失败和编排失败可以分开定位，也避免核心逻辑被框架锁定。

### 决策 2：为什么使用一个共享 State？

**问题背景**

工作流需要把任务、证据、画像、分析和验证结果依次传递，并让条件路由读取验证状态。

**当前方案**

使用一个显式 `WorkflowGraphState`，保存每阶段的结构化产物、执行历史、
重试次数和下一步路由标记。

**替代方案**

节点直接互相调用并通过函数返回值传递，或把所有内容塞进任意字典。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 显式共享 State | 状态可观察、可持久化、路由可读取 | 需要维护字段契约 |
| 节点互相调用 | 小脚本直观 | 控制流和数据流耦合，难插入检查与重试 |
| 任意字典 | 灵活 | 字段拼写和类型错误难发现 |

**什么时候考虑切换**

一次性的线性脚本可以直接函数调用；需要条件路由、可观察状态或恢复执行时，共享
State 更合适。

**面试回答参考**

我使用共享 State 保存每一步的结构化产物。节点不互相直接调用，而是返回部分更新，
由图负责推进。这样能观察状态变化，也为后续 checkpoint 和 UI 状态展示留下边界。

### 决策 3：为什么节点只返回部分状态更新？

**问题背景**

每个节点只拥有少数字段。如果每次都重建完整 State，容易覆盖其他节点的数据。

**当前方案**

Planner 只返回 tasks 和 history，Researcher 只返回 evidence/errors，
后续节点同理。

**替代方案**

每个节点复制并返回完整 State。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 部分更新 | 节点职责清楚，减少意外覆盖 | 需要理解 LangGraph 合并语义 |
| 完整 State | 单步输出看起来完整 | 重复代码多，容易覆盖并发或后续字段 |

**什么时候考虑切换**

普通函数流水线可以显式返回完整对象；LangGraph 节点更适合声明自己改变的字段。

**面试回答参考**

LangGraph 节点返回 partial update，而不是复制整个 State。每个节点只声明自己产生的
结果，能降低字段覆盖风险，也让节点职责从返回值上直接可见。

### 决策 4：为什么验证失败只允许回到 Analyst 一次？

**问题背景**

模型修复可能仍然失败。如果没有上限，Verifier 和 Analyst 会形成无限循环并持续
消耗 Token。

**当前方案**

第一次失败设置 revision feedback，并把 `retry_count` 增加到 1；
第二次验证后无论结果如何都结束。

**替代方案**

持续循环直到通过，或者验证失败立即结束。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 最多重试一次 | 有恢复能力，成本和延迟有上限 | 第二次失败会保留未通过结果 |
| 不重试 | 最简单、成本最低 | 偶发模型问题无法自动恢复 |
| 无限循环 | 可能最终通过 | 成本不可控，可能永不结束 |

**什么时候考虑切换**

后台异步任务可结合预算、人工审批和失败分类增加有限重试，但必须保留硬上限。

**面试回答参考**

Verifier 失败后图最多回到 Analyst 一次，并把具体 issues 作为 revision feedback。
第二次仍失败就结束并保留问题，防止不可控循环。循环上限属于编排层，而不是节点内部。

### 决策 5：为什么把执行历史放进 State？

**问题背景**

只看最终结果无法判断节点是否按顺序执行，也无法确认是否发生过重试。

**当前方案**

每个节点把名称追加到 `stage_history`，测试和未来 UI 可直接观察轨迹。

**替代方案**

只依赖日志或 LangGraph 内部 trace。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| State 中保存简化历史 | 测试简单、结果自包含 | 不是完整可观测性 trace |
| 只用日志 | 实现常见 | 难作为结构化结果断言 |
| 专业 tracing 平台 | 信息完整 | 当前 MVP 配置和成本更高 |

**什么时候考虑切换**

进入生产可观测性阶段可接入 LangSmith 或其他 trace 平台；简化历史仍可用于 UI 状态。

**面试回答参考**

我在 State 中保存节点级 stage history，用它验证顺序和重试行为。它不是完整 tracing，
但对 MVP 的测试和状态展示足够；生产环境再接专业可观测性工具。

### 决策 6：为什么在 Analyst 前做 profile validation？

**问题背景**

Extractor 已经有第一层范围过滤，但真实系统里 ProductProfile 仍可能因为提示词漂移、
测试替身、后续代码修改或历史数据而混入非 API 价格、冲突价格或套餐级定位。
如果这些污染直接进入 Analyst，后续 Verifier 只能检查最终 claim，很难清洗已经污染的画像。

**当前方案**

`competitive_analysis_agent/workflow.py` 中的 `run_extractor_node()` 在拿到
Extractor 输出后调用 `validate_product_profiles_for_analysis()`。这个函数会：

- 用 Extractor 已有的 scope 判断删除默认 API pricing 范围外的价格；
- 删除同一套餐、同一计费周期但价格冲突的 pricing 项；
- 删除看起来来自 pricing/subscription 语境的套餐级 `positioning`；
- 把删除原因转换成 `ResearchError(code="profile_validation")`，并合并进
  `research_errors`。

**为什么这样选择**

这个位置正好位于 `ProductProfile -> Analyst` 的边界，能保证 Analyst 只比较已经收口的画像。
同时它不新增模型调用，不改变 Extractor 的职责，也能让 Reporter 使用现有“数据限制”机制展示问题。

**替代方案**

只在 Extractor 内过滤，或者新增一个独立 LangGraph `profile_validator` 节点。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 在 `run_extractor_node()` 内做入场校验 | 改动小；不改变图拓扑；Analyst 前一定收口；可复用现有 Reporter 限制展示 | `extractor` stage 内包含一个薄校验步骤，stage history 不单独展示 |
| 只在 Extractor 内过滤 | 边界集中，调用方简单 | 其他来源的历史/测试/缓存画像可能绕过过滤 |
| 独立 profile validator 节点 | 图上可见，职责更显式 | 需要新增节点、状态字段和更多 UI/文档同步 |

**什么时候考虑切换**

如果 profile validation 规则继续增加，或者需要在 UI 中单独展示“画像校验”阶段，就可以把它升级成独立节点。
如果未来 ProductProfile 来自数据库或多轮缓存，也应把 validation 作为正式边界对象保存。

**面试回答参考**

我在 Analyst 前加了一道确定性入场校验，因为 Analyst 不应该负责清洗上游污染。
这层会删除明显越界或冲突的 profile 字段，并把原因记录到数据限制里。
它不调用模型，所以成本低、可测试，也能防止错误价格进入后续分析和报告。

## 4. 异常与边界情况

- 初始输入中的产品或维度重复：`PlannerInput` 在进入图前拒绝。
- Planner 输出覆盖不完整：Planner 自己修复一次，仍失败则图传播异常。
- 单个搜索任务失败：Researcher 写入 `research_errors` 并继续其他任务。
- 全部搜索无结果：Extractor 的非空输入校验会阻止无依据分析。
- Profile 中混入默认 API pricing 范围外的价格：Workflow 在 Analyst 前删除该价格项，
  并写入 `research_errors`。
- Profile 中同一套餐、同一计费周期出现多个价格：Workflow 删除冲突价格项，
  并把原因记录为数据限制。
- Profile 的 positioning 看起来来自套餐、订阅或价格页语境：Workflow 清空该定位，
  避免进入 Analyst 形成产品级定位判断。
- Verifier 在没有 `analysis_result` 时运行：node 明确抛出 `ValueError`。
- 第一次验证失败：issues 进入 Analyst 修订消息，`retry_count` 变为 1。
- 第二次验证仍失败：保留失败的 `VerificationResult` 并结束，不继续循环。
- Verifier 报告完全无直接支持的 claim：允许 issue 使用空
  `evidence_ids`；未知或重复 ID 仍会被拒绝。
- 模型、节点或供应商抛出未处理异常：当前图让异常向调用方传播，不静默吞掉。
- 当前没有 checkpointer：进程中断后不能从中间节点恢复，必须重新运行。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：
  `python -m pytest -o addopts= -m live_llm tests/test_live_workflow.py -q`
- 真实调用的组件：
  - Planner；
  - 两次 Extractor；
  - Analyst；
  - Verifier；
  - 必要时由图再调用一次 Analyst 和 Verifier。
- 固定组件：Researcher 使用 fixture provider，避免外部搜索波动干扰图验收。
- 验证的输出契约：
  - 生成 2 个 ResearchTask、2 条 Evidence 和 2 个 ProductProfile；
  - `analysis_result.products` 顺序正确；
  - 最终 `VerificationResult.passed=True`；
  - `retry_count <= 1`；
  - history 顺序正确，Analyst 和 Verifier 均不超过两次。
- 测试结果：通过，`1 passed`
- 最新复跑结果：通过，`1 passed`
- 最新耗时：56.34 秒
- 历史耗时：61.82 秒
- 验收过程发现并修复：
  - 语义 issue 对“完全没有直接支持”的 claim 应允许空 Evidence ID 列表；
  - Analyst 不应从简短功能摘要推断用户需求、市场偏好或产品优越性；
  - 曾发生一次硅基流动连接中断，按瞬时网络故障重跑后通过。

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1b-agent-architectures.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2b-agent-tools-best-practices.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\3-Day\codelabs\day-3a-agent-sessions.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\3-Day\codelabs\day-3b-agent-memory.zh-CN.ipynb`

### Relevant sections

- `第 3 节：Sequential 工作流 - 装配线`
- `第 5 节：Loop 工作流 - 优化循环`
- `第 6 节：总结 - 选择正确的模式`
- `第 4 节：构建工作流`
- `4.1：在工作流中处理 Events`
- `2.2 什么是 Session？`
- `第 5 节：使用 Session State`
- `什么是 Memory？`
- `为什么需要 Memory？`
- `第 2 节：Memory 工作流`

### Core concept

Sequential 模式用于保证有依赖的步骤按固定顺序执行；Loop 模式用于“生成、评审、
修订”，但必须有明确退出条件和最大迭代次数。State 是节点之间共享的动态数据，
Events 或 stream update 则描述运行过程中发生了什么。

Session State 服务于同一段对话中的持续上下文，Memory 服务于跨会话长期检索。
本阶段的 State 只保存一次竞品分析运行的中间产物，不等于用户长期记忆。

### How it appears in this project

- 固定边实现 `Planner -> Researcher -> Extractor -> Analyst -> Verifier`；
- 条件边实现 `Verifier -> Analyst` 的一次受限修订循环；
- `WorkflowGraphState` 对应节点共享的工作流草稿本；
- node partial update 对应每个阶段只写入自己拥有的字段；
- `validate_product_profiles_for_analysis()` 对应工作流中的质量门：在进入下一个专业节点前，
  用确定性规则检查并收口上一步产物；
- `stage_history` 和 `graph.stream(..., stream_mode="updates")` 提供节点级观察；
- `retry_count` 与 `MAX_ANALYSIS_RETRIES` 是循环的硬退出条件。

### ADK vs LangGraph

- Notebook 用 ADK `SequentialAgent`、`LoopAgent`、Session、Events 和
  `output_key`；
- 本项目用 LangGraph 的 node、edge、conditional edge 和 TypedDict State；
- ADK `LoopAgent(max_iterations=...)` 在容器层限制循环，本项目由 State 中的
  `retry_count` 和条件路由显式限制；
- ADK Events 是更完整的运行事件，本项目当前只保存简化 history，并可读取
  LangGraph stream update；
- 两者表达的是相同工程原则：业务节点专门化，编排层掌握顺序、路由和终止条件。

### Intentionally postponed

- Session Service 和跨对话状态；
- 长期 Memory、语义检索和 memory consolidation；
- LangGraph checkpoint、暂停、人工审批与恢复；
- Parallel 工作流；
- 完整 event/trace 持久化与成本统计。

## 7. 理解问题与参考思路

### 问题 1：为什么节点不直接互相调用？

**参考思路：**

- 节点互调会把业务逻辑和控制流绑在一起；
- 图需要统一掌握顺序、条件路由和循环上限；
- State 让中间结果可以被测试、观察和未来持久化。

### 问题 2：为什么 Verifier 失败后回到 Analyst，而不是 Researcher？

**参考思路：**

- 当前 issue 表示分析 claim 不受现有 Evidence 支持，先尝试删除或收窄 claim；
- 如果问题是 Evidence 本身缺失，未来应增加 issue 分类并路由回 Researcher；
- 当前 Stage 8 只实现最小且可控的修订路径。

### 问题 3：为什么必须设置 `MAX_ANALYSIS_RETRIES`？

**参考思路：**

- LLM 修订不保证成功；
- 无上限循环会造成不可控成本和延迟；
- 第二次失败保留结构化 issues，比继续盲目调用更容易人工处理。

### 问题 4：`WorkflowGraphState`、Session State 和 Memory 有什么区别？

**参考思路：**

- Workflow State：一次图运行中的任务、证据、画像和结果；
- Session State：同一用户会话多轮交互中的短期上下文；
- Memory：跨会话可检索的长期知识；
- 当前只需要 Workflow State，后两者属于后续产品能力。

### 问题 5：为什么 profile validation 放在 Analyst 前？

**参考思路：**

- Analyst 只应该比较 ProductProfile，不应该清洗网页来源或修复画像污染；
- Workflow 正好掌握 `Evidence`、`ProductProfile` 和 `research_errors`，适合做边界校验；
- 被删除的价格或定位会变成数据限制，而不是静默消失或继续污染报告。

## 8. 面试追问清单

- 如果流程中途崩溃，如何从 State 恢复？
- 如果 Researcher 和 Extractor 要并行，State 合并规则如何变化？
- 为什么循环上限属于图而不是 Verifier？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 9：Reporter 将把已验证的结构化结果确定性渲染为 Markdown。本阶段不实现。

## 10. 后续变更记录

- Stage 9 已增加 `Reporter` 终端节点；
- 当前图在 Verifier 不再重试时进入 Reporter，再由 Reporter 连接 `END`；
- Stage 8 的共享 State 新增 `final_report`，其余重试和路由规则不变。
- 2026-06-24 根据 `docs/agent-repair-guide-api-pricing-scope.md` 的 P1 建议，
  在 `run_extractor_node()` 中增加 `validate_product_profiles_for_analysis()`。
  这道校验会在 Analyst 前删除非默认 API pricing 范围的价格、冲突价格和套餐级定位，
  并把原因写入 `ResearchError(code="profile_validation")`。
- 新增回归测试：
  `test_extractor_node_validates_profile_before_analyst` 和
  `test_extractor_node_removes_conflicting_profile_prices`。
- 验证命令：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_workflow.py -q`
  结果 `9 passed`；
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py tests\test_workflow.py -q`
  结果 `89 passed`；
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`
  结果 `154 passed, 7 deselected`；
  真实 Workflow smoke
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests\test_live_workflow.py -q`
  结果 `1 passed`，耗时约 56.34 秒。
- Notebook 对应补充：Day 1b 的 Sequential 工作流强调需要固定顺序保证每一步基于前一步输出；
  Day 2b 的 workflow/events 章节强调工作流代码要检查中间事件和状态。本次 profile validation
  就是一个中间状态质量门：不是新 Agent，而是编排层在交给下一个 Agent 前确认数据契约。
