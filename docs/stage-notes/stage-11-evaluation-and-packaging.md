# Stage 11：Evaluation and Packaging

## 1. 本阶段目标

本阶段为完整竞品分析 Agent 建立可重复评测集，并整理成可展示、可运行、可解释的
实习项目。

输入：

- 三个固定工作流案例；
- 每个案例的预期验证状态和重试次数；
- LangGraph 最终 State。

输出：

- 案例通过率；
- 任务成功率；
- 字段覆盖率；
- 引用有效率；
- 来源覆盖率；
- 实际运行耗时；
- JSON 和 Markdown 评测结果；
- 完整 README、架构图、UI 截图和样例报告。

数据流：

```text
fixed evaluation cases
  -> real LangGraph execution
  -> final State
  -> deterministic metrics
  -> evaluation-results.json / .md
  -> README project evidence
```

本阶段不实现真实搜索、部署平台、向量库、长期 Memory 或生产监控。

## 2. 实现结果

- 在 `evaluation/cases.json` 定义三个固定案例：
  `complete_success`、`retry_recovery`、`verification_warning`。
- 在 `competitive_analysis_agent/evaluation.py` 运行真实 LangGraph，并从最终
  `WorkflowGraphState` 计算行为通过率、任务成功率、字段覆盖率、引用有效率、
  来源覆盖率和耗时。
- 使用 `write_evaluation_results()` 同时输出机器可读 JSON 和人工可读 Markdown。
- 新增 `competitive-analysis-eval` 命令入口。
- 在 `tests/test_evaluation.py` 覆盖固定案例、错误引用、部分来源覆盖、运行异常脱敏
  和结果导出。
- 在 `tests/test_live_evaluation.py` 运行一个真实模型固定案例。
- README 已包含架构、安装、运行命令、截图、样例、限制和实测结果。

离线实测结果：

| 指标 | 结果 |
| --- | ---: |
| 案例通过率 | 100.0% |
| 任务成功率 | 66.7% |
| 平均字段覆盖率 | 80.0% |
| 引用有效率 | 100.0% |
| 来源覆盖率 | 100.0% |
| 平均耗时 | 0.0069 秒 |
| 成本 | 未采集 |

## 3. 设计决策

### 决策 1：为什么同时记录案例通过率和任务成功率？

**问题背景**

故障案例也属于评测集。例如 Verifier 连续失败后，系统正确停止并输出警告，这是正确
行为，但用户任务并没有成功完成。

**当前方案**

`case_pass_rate` 检查行为是否符合案例预期，
`task_success_rate` 检查是否最终得到通过验证的报告。

**为什么这样选择**

避免把“正确处理失败”误报成业务成功，也避免故障案例永远让评测集看起来失败。

**替代方案**

只统计一个 success 布尔值。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 区分案例与任务成功 | 语义清楚，能覆盖负向案例 | 指标数量更多 | Agent 回归评测 |
| 单一成功率 | 最简单 | 无法区分预期失败和系统故障 | 全部案例都应成功的简单任务 |

**什么时候考虑切换**

如果生产评测集全部来自真实用户成功任务，可以重点展示任务成功率，但回归测试仍应
保留负向案例和行为断言。

**面试回答参考**

我区分评测案例是否按预期运行，以及用户任务是否真正成功。Verifier 连续失败并正确
停止属于案例通过，但不是任务成功。这让可靠性测试和业务指标不会互相污染。

### 决策 2：为什么默认评测必须离线运行？

**问题背景**

如果每次 pytest 都调用真实模型，测试会变慢、产生费用，并受到网络和模型波动影响。

**当前方案**

三个固定案例使用 Fake Model 和固定搜索数据运行真实 LangGraph；
另保留一个独立 `live_llm` 案例验证真实供应商路径。

**为什么这样选择**

离线评测定位逻辑回归，真实评测确认端点、Prompt 和结构化输出兼容，两者职责不同。

**替代方案**

所有评测都调用真实模型，或者完全不做真实调用。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 离线集 + 独立真实案例 | 快速稳定，同时验证真实集成 | 需要维护两层测试 | 当前 Agent 项目 |
| 全部真实调用 | 最接近线上 | 贵、慢、不稳定、难定位 | 定时质量基准 |
| 全部离线 | 快速无成本 | 无法证明真实模型兼容 | 纯确定性系统 |

**什么时候考虑切换**

进入生产后可以增加每日或发布前真实评测，但不应替代提交时的离线回归。

**面试回答参考**

默认评测用固定输入和 Fake Model，确保快速、稳定、无费用。真实模型评测单独标记，
在阶段验收或发布前运行，用来验证供应商兼容和 Prompt 行为。

### 决策 3：这些指标分别证明什么？

**问题背景**

单一成功率不能说明报告是否完整、引用是否有效、研究任务是否都有来源。

**当前方案**

分别计算字段覆盖、引用有效、来源覆盖、耗时和任务成功。

**为什么这样选择**

每个指标对应不同失败类型，便于从最终结果定位到数据、引用或性能问题。

**替代方案**

只使用 LLM-as-a-Judge 给一个总分。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 确定性多指标 | 可重复、可解释、无模型成本 | 不能评价文风和深层洞察 | 结构与可靠性评测 |
| LLM 总分 | 可评价语义质量 | 有波动、成本和 Judge 偏差 | 补充主观质量评价 |

**什么时候考虑切换**

需要评价洞察深度、语气和商业价值时，可增加人工或 LLM Judge，但确定性指标仍应保留。

**面试回答参考**

任务成功率回答“是否完成”，字段覆盖回答“输出是否完整”，引用有效率回答“引用是否
存在”，来源覆盖率回答“调研任务是否有证据”，耗时回答“运行成本是否可接受”。这些
指标不能证明事实绝对正确，所以仍需 Verifier 和人工复核。

### 决策 4：为什么不把部署和数据库作为最终阶段必选项？

**问题背景**

部署、数据库和长期 Memory 很吸引人，但它们不会自动提高当前分析质量，还会扩大
项目范围。

**当前方案**

最终阶段优先提供可运行命令、测试、实测指标、UI、样例和限制说明。

**为什么这样选择**

对于实习项目，可解释的完整闭环比堆叠基础设施更能证明工程能力。

**替代方案**

继续加入 Docker、云部署、SQLite、向量库和认证。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 先完成评测与包装 | 项目闭环清楚、证据充分 | 暂无公开部署 URL | 面试和 MVP |
| 继续扩展基础设施 | 展示部署能力 | 分散质量重点，维护成本高 | 明确要求上线的项目 |

**什么时候考虑切换**

当项目需要公开演示、多人使用或保存历史报告时，优先选择真实搜索和 SQLite/部署中的
一个作为下一增强。

**面试回答参考**

最终阶段我先把质量证据补齐，包括评测集、指标、测试、UI 和文档。部署和持久化属于
下一步增强，因为它们不应阻塞核心 Agent 的可靠性闭环。

## 4. 异常与边界情况

- `run_evaluation_case()` 捕获单案例异常，只记录 `error_category`，不把请求文本、
  Prompt 或密钥写入结果。
- `verification_warning` 即使最终验证失败也必须生成带警告的报告，并在一次重试后
  停止，证明有限循环有效。
- `calculate_citation_validity()` 在没有引用时返回 100%，含义是“没有无效 ID”，
  不是“内容事实正确”。
- `calculate_source_coverage()` 按产品与主题组合计算，避免用大量重复 Evidence
  掩盖某个研究任务完全缺失。
- 实际 live 评测首次发现 Verifier 把保守的信息不足声明误判为 unsupported；
  第二次发现 Analyst Prompt 中“interpretation 可不引用”与“conclusion 必须引用”
  存在冲突。最终通过收紧 `ANALYST_SYSTEM_PROMPT`、`VERIFIER_SYSTEM_PROMPT` 和
  `validate_analyst_output()` 修复，而没有降低评测门槛。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：
  `python -m pytest -o addopts='' tests/test_live_evaluation.py -q`
- 真实调用的组件：完整 LangGraph 固定评测案例
- 验证的输出契约：结构化 Planner、Extractor、Analyst、Verifier 输出；结论必须
  带有效 Evidence ID；最终验证通过；Reporter 生成报告；轨迹以 reporter 结束。
- 测试结果：`1 passed`
- 耗时：57.39 秒
- 失败原因：无

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4a-agent-observability.zh-CN.ipynb`
  - `什么是 Agent 可观测性？`
  - `Agent 可观测性的基础支柱`
  - `第 3 节：生产环境中的日志记录`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4b-agent-evaluation.zh-CN.ipynb`
  - `第 4 节：系统化评估`
  - `4.1：创建评估配置`
  - `4.2：创建测试用例`
  - `4.3：运行 CLI 评估`
- `4-Day\whitepaper\Agent Quality 中文解析.md`
  - `Agent 质量的四大支柱`
  - `可观测性：让 Agent 的过程可见`
  - `Agent Quality Flywheel`
- `5-Day\whitepaper\Prototype to Production.md`
  - `Evaluation as a Quality Gate`
  - `生产运行：Observe → Act → Evolve`

### Relevant sections

课程把 Agent 可观测性分为 logs、traces、metrics。系统化评估则分成定义指标、创建
固定测试案例、运行 Agent、比较结果四步，并强调回归测试保护已有行为。

### Core concept

- Log 回答一次事件发生了什么。
- Trace 把一次任务的多个步骤连起来，回答结果为什么发生。
- Metric 汇总多次运行，回答系统整体表现如何。
- Evaluation 用固定输入和明确标准判断版本是否退化，可以成为发布质量门禁。

### How it appears in this project

- `research_errors` 和异常类别接近结构化日志。
- `stage_history`、`retry_count` 和 LangGraph State 是当前轻量轨迹。
- `EvaluationSummary` 是聚合指标。
- `evaluation/cases.json` 对应课程 evalset。
- `competitive-analysis-eval` 对应课程 CLI 评测入口。
- 第一次失败的 live 案例变成 Prompt 和校验规则修复，体现质量飞轮：
  观测失败、定位原因、加入规则和测试、再次评测。

### ADK vs LangGraph

Notebook 使用 ADK Plugin、Callback 和 `adk eval` 收集生命周期事件并执行评测。
本项目使用显式 LangGraph State、pytest 和自定义评测运行器。两者概念相同，但 API
不同；当前实现更直接，适合小型学习项目，也避免引入第二套 Agent 框架。

### Intentionally postponed

当前没有实现生产结构化日志平台、trace/span ID、P95/P99 指标、Token 成本采集、
LLM-as-a-Judge、CI 质量门禁和线上 Observe → Act → Evolve 自动闭环。这些属于生产
AgentOps，而不是本次 MVP 的完成条件。

## 7. 理解问题与参考思路

1. 为什么案例通过率和任务成功率不能合并？
   - 负向案例可能正确处理失败，但用户目标仍未完成；合并会混淆系统可靠性与业务成功。
2. 引用有效率 100% 能证明事实正确吗？
   - 不能，只证明 claim 使用的 Evidence ID 存在。来源真实性、时效和语义支持仍需
     Verifier 或人工复核。
3. 这次 live 评测发现了什么普通 fixture 没发现的问题？
   - 真实模型会在冲突指令中选择不同规则，也会把保守声明误判为 unsupported。
     fixture 能稳定保护已知逻辑，但不能替代真实供应商兼容性测试。

## 8. 面试追问清单

- 为什么引用有效率 100% 仍不能证明事实正确？
- 如果真实评测结果波动，如何建立发布门槛？
- 哪个指标最适合发现搜索覆盖不足？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 0-11 到此完成。下一项最有价值的可选增强是真实搜索 Provider，因为它能把当前
固定 Evidence 演示升级为真实产品研究；数据库、异步搜索和部署继续后置。
