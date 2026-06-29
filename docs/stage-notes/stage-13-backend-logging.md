# Stage 13：后台日志

## 1. 本阶段目标

为 Streamlit 应用增加本地后台日志，让一次分析从开始、节点推进到结束或失败都可以被
定位，同时不记录 API Key、Prompt、Evidence 正文和用户填写的原始域名。

核心数据流：

```text
Streamlit 启动
  -> 配置控制台与轮转文件日志
  -> UI 提交 AnalysisRequest
  -> 生成 analysis_id
  -> 记录各 LangGraph 阶段完成事件
  -> 记录成功统计或脱敏异常类别
```

## 2. 实现结果

- `competitive_analysis_agent/logging_config.py` 使用标准库 `logging` 配置控制台和
  `logs/application.log`。
- 文件达到 5 MB 后轮转，保留 3 个历史文件；Streamlit 脚本重跑不会重复添加同一路径
  的 Handler。
- `competitive_analysis_agent/ui_service.py::run_analysis()` 为每次运行生成 12 位
  `analysis_id`，记录开始、节点完成、成功和失败事件。
- 阶段日志记录累计耗时、任务数、证据数、画像数、研究错误数和重试次数。
- 失败日志只记录异常类型、函数名和行号，不记录异常原文。
- `competitive_analysis_agent/streamlit_app.py::main()` 在页面启动时初始化日志。
- `.gitignore` 忽略 `logs/`，README 说明日志位置、轮转规则和安全边界。
- 聚焦测试：`13 passed in 1.39s`。
- 完整离线测试：`86 passed, 7 deselected in 1.61s`。
- 离线工作流手动检查已实际生成六个节点日志和成功终态日志。
- 暂未实现：LangSmith、OpenTelemetry、集中式日志、告警、指标看板和日志查询 UI。

## 3. 设计决策

### 决策 1：为什么先使用标准库轮转日志？

**问题背景**

当前应用是本地单进程 Streamlit，首先需要解决“运行到哪一步、在哪失败”的问题。

**当前方案**

`configure_application_logging()` 使用 `RotatingFileHandler`，同时输出到双击启动的
命令窗口和本地文件。

**为什么这样选择**

它无需新增依赖和外部账号，离线可用，适合当前 MVP；轮转也避免日志无限增长。

**替代方案**

接入 LangSmith、OpenTelemetry 或云日志服务。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 本地轮转日志 | 简单、离线、易测试 | 不支持跨机器检索和分布式 Trace | 本地单进程 MVP |
| 外部可观测平台 | 可检索、可追踪、可告警 | 配置、成本和数据治理更复杂 | 部署后的多实例应用 |

**什么时候考虑切换**

当应用部署到多实例、需要团队协作排障、P95 延迟或错误率告警时，应接入统一平台。

**面试回答参考**

当前项目是单进程 Streamlit，我先用标准库轮转日志解决最直接的排障需求，不增加外部
依赖。等部署到多实例后，再用 OpenTelemetry 或 LangSmith 串联跨服务 Trace。

### 决策 2：为什么每次运行都生成 analysis_id？

**问题背景**

多个用户连续提交时，只有时间戳和节点名无法判断哪些日志属于同一次分析。

**当前方案**

`run_analysis()` 生成短 UUID，并把同一 `analysis_id` 写入开始、节点和终态事件。

**为什么这样选择**

它不改变 Workflow State，也能把一次运行的离散日志串起来，改动范围小。

**替代方案**

把 Trace ID 加入 LangGraph State，或完全依赖时间顺序。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| UI 边界生成 analysis_id | 简单，不污染业务 Schema | 当前主要覆盖 UI 入口 | 单一应用入口 |
| ID 放入 Workflow State | 任意入口都可传播 | 要修改共享契约和更多测试 | CLI、队列等多入口 |
| 只看时间顺序 | 无需实现 | 并发时无法可靠关联 | 单次手工调试 |

**什么时候考虑切换**

当评测 CLI、后台任务和 API 都需要统一追踪时，把 ID 提升为 Workflow State 或标准
Trace Context 更合适。

**面试回答参考**

我用 `analysis_id` 关联一次执行的全部日志，避免并发任务混在一起。当前只在 UI 边界
生成，保持共享 State 简单；多入口部署后会升级为统一 Trace ID。

### 决策 3：为什么不默认记录完整 Prompt、响应和异常文本？

**问题背景**

模型请求、证据正文和第三方异常可能包含用户数据、网页内容、请求头甚至凭证。

**当前方案**

日志只记录计数、阶段、耗时、验证结果和异常类别。失败位置通过函数名与行号定位，
不写 `str(error)`。

**为什么这样选择**

排障信息与敏感内容分离，降低日志泄密风险，也避免大段模型文本快速撑满日志文件。

**替代方案**

DEBUG 级别记录完整 Prompt、工具输入输出和模型响应。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 元数据日志 | 风险低、体积小 | 语义问题需要其他手段复现 | 默认生产行为 |
| 完整内容日志 | 调试信息丰富 | 泄密、合规和存储风险高 | 隔离开发环境且显式脱敏 |

**什么时候考虑切换**

如果需要调试 Prompt，可增加显式开启、短期保留、字段脱敏的 DEBUG 模式，而不是改变
默认日志策略。

**面试回答参考**

Agent 日志不能无条件保存完整上下文，因为 Prompt、工具结果和异常消息可能含敏感数据。
我默认只记运行元数据，并用测试保证伪密钥不会进入失败日志。

## 4. 异常与边界情况

- 日志目录不存在时自动创建；目录不可写时启动阶段明确失败，不静默丢日志。
- Streamlit 每次交互都会重跑脚本，配置函数按文件绝对路径识别已有 Handler。
- Windows 会锁住正在写入的日志文件，测试会先关闭 Handler 再清理临时目录。
- Workflow 失败时记录 `error_type`、`failure_function` 和 `failure_line`，随后保留原异常
  给 UI 的既有错误处理逻辑。
- 输入校验在 Workflow 前失败时，Streamlit 记录 `analysis_submission_failed` 和异常类型。
- 日志目录已加入 `.gitignore`，但本地日志仍应按敏感运行数据管理。
- 2026-06-23 新增阶段 I/O 摘要日志：`competitive_analysis_agent/ui_service.py`
  在 LangGraph `stream()` 返回每个 State 快照时记录
  `analysis_stage_io analysis_id=... stage=... direction=input/output payload_json=...`。
  payload 记录每个阶段契约中的关键字段，例如 Planner 的产品和维度、Researcher 的任务和
  Evidence、Extractor 的产品画像、Analyst 的 claims、Verifier 的 issues、Reporter 的报告预览。
  Evidence 正文和最终报告只保留短预览及字符数，列表只保留前 8 项并记录省略数量，避免日志
  体积失控。回归测试：`tests/test_ui_service.py::test_fixture_workflow_reports_progress_and_returns_report`。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：否。
- 配置来源：本阶段不读取模型配置。
- 测试命令：不适用。
- 真实调用的组件：无。
- 验证的输出契约：本阶段不改变 Prompt、模型输出 Schema 或 LangGraph 路由。
- 测试结果：不适用；使用完整离线回归确认行为未改变。
- 耗时：不适用。
- 失败原因：无。

## 6. 与课程 Notebook 的对应

- Notebook：
  `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4a-agent-observability.zh-CN.ipynb`
- 相关章节：`什么是 Agent 可观测性？`、`Agent 可观测性的基础支柱`、
  `第 3 节：生产环境中的日志记录`
- 通用知识点：Logs 回答某个时刻发生了什么，Traces 串联完整执行路径，Metrics 汇总
  系统整体表现；三者用途不同。
- 在本项目中的实现：每条 `analysis_stage_completed` 是事件日志，共享 `analysis_id`
  可以还原节点顺序，但目前还不是带父子 Span 的完整分布式 Trace。
- ADK 与 LangGraph/本项目的差异：Notebook 通过 ADK `LoggingPlugin` 和 callbacks
  挂接 Agent、模型和工具生命周期；本项目在 Streamlit 应用边界和 LangGraph
  `stream()` 状态快照处显式记录日志。
- 本阶段有意简化的内容：不记录完整 LLM 请求响应，不实现 Plugin 系统、Span、
  Token 指标或外部可观测平台。
- 2026-06-23 增量说明：阶段 I/O 日志相当于轻量版事件 Trace，记录的是项目节点之间的
  Pydantic 契约摘要，不是 LangChain 底层 raw prompt 或 raw model response。若未来需要完整
  模型调用链，应接入专门 Trace 平台并配置脱敏、采样和保留周期。

## 7. 理解问题与参考思路

### 问题 1：日志与 Trace 有什么区别？

**参考思路：**

- 日志是一条独立事件；Trace 使用统一上下文把多个步骤和耗时组织成调用链。
- 当前 `analysis_id` 能关联事件，但没有父子 Span，因此是轻量关联日志。

### 问题 2：为什么异常日志不写 `str(error)`？

**参考思路：**

- 第三方 SDK 的异常原文不完全受项目控制，可能含请求内容。
- 异常类型和代码位置满足第一层定位，敏感内容应通过受控调试方式获取。

### 问题 3：为什么 Handler 要防重复？

**参考思路：**

- Streamlit 交互会重跑脚本。
- 每次都添加 Handler 会让同一事件重复写入多次，并造成文件句柄持续增加。

## 8. 面试追问清单

- 如果数据量扩大十倍，当前方案哪里最先需要调整？
- 如何把 `analysis_id` 升级成 OpenTelemetry Trace ID？
- 如何设计安全的 DEBUG 日志开关和保留周期？
- 当前日志能计算哪些指标，不能直接证明哪些质量问题？

## 9. 下一阶段衔接

Stage 13 已完成。项目当前优先待办仍是为 Stage 12 配置 Tavily Key，并通过真实搜索与
真实 UI 验收；在有部署需求后，再把本地关联日志升级为集中式 Trace 和指标。
