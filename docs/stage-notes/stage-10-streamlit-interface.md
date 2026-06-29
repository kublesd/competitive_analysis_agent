# Stage 10：Streamlit Interface

## 1. 本阶段目标

本阶段为现有 LangGraph 工作流增加最小可用的 Streamlit 页面。

输入：

- 目标产品；
- 竞品列表；
- 分析维度。

输出：

- 节点级运行状态；
- Markdown 竞品分析报告；
- 来源链接；
- Markdown 下载文件；
- 对输入错误和运行失败的可读提示。

完整数据流：

```text
Streamlit form
  -> AnalysisRequest
  -> UI service
  -> LangGraph stream
  -> Session State
  -> report + sources + download
```

当前没有真实搜索 Provider，因此页面只运行固定 Evidence 支持的演示案例：
`Atlas Notes` 对 `Beacon Docs`，维度为 `features`。模型节点仍使用真实硅基流动配置。

本阶段不实现认证、独立后端、后台任务、长期 Memory 或多用户持久化。

## 2. 实现结果

- 完成的功能：
  - `AnalysisRequest` 校验目标产品、竞品和维度；
  - `parse_competitors()` 支持换行、中英文逗号分隔；
  - `validate_demo_request()` 阻止固定 Evidence 被误用于其他产品；
  - `run_analysis()` 使用 `graph.stream(..., stream_mode="values")`
    运行完整 LangGraph 并报告节点状态；
  - `AnalysisRunResult` 只向页面暴露报告、来源、验证和错误摘要；
  - Streamlit 表单提供目标产品、竞品和维度输入；
  - 分析维度支持“常用维度勾选 + 自定义维度文本框”；
  - 节点事件显示为中文阶段状态；
  - Session State 保存报告、来源、执行历史和错误信息；
  - 页面显示 Markdown 报告、独立来源列表和下载按钮；
  - 部分研究失败和验证失败都显示醒目警告；
  - 运行异常转换成不含 traceback 和密钥的用户提示。
- 关键文件：
  - `competitive_analysis_agent/ui_service.py`
  - `competitive_analysis_agent/streamlit_app.py`
  - `tests/test_ui_service.py`
  - `tests/test_streamlit_app.py`
  - `tests/test_live_ui.py`
- 核心数据流：

```text
Streamlit controls
  -> create_analysis_request()
  -> validate_demo_request()
  -> run_analysis()
  -> LangGraph values stream
  -> AnalysisRunResult
  -> st.session_state
  -> status + Markdown + sources + download
```

- 验证方式与结果：
  - `python -m pytest tests/test_ui_service.py tests/test_streamlit_app.py -q`：
    9 个测试通过；
  - `python -m pytest -q`：68 个离线测试通过，5 个真实用例默认排除；
  - Streamlit `AppTest` 验证表单、错误、Session State、警告和下载控件；
  - 实际启动服务后 `/_stcore/health` 返回 `ok`；
  - `python -m compileall -q competitive_analysis_agent tests`：通过；
  - 使用版本：Streamlit 1.40.2。
- 真实 LLM 测试：
  - UI service 使用真实模型运行完整 LangGraph；
  - 最终生成通过验证并包含来源的 Markdown；
  - `1 passed in 63.36s`。
- 暂未实现：
  - 真实搜索 Provider 和任意产品分析；
  - 异步后台任务、取消按钮和任务队列；
  - LangGraph checkpoint 与中断恢复；
  - 用户认证和多用户隔离；
  - 报告历史数据库；
  - 开发者 trace 页面和生产指标。

## 3. 设计决策

### 决策 1：为什么把 UI service 与 Streamlit 页面分开？

**问题背景**

如果页面直接创建模型、构造 LangGraph 并处理状态，UI 代码会混入业务逻辑，也难以
脱离浏览器测试。

**当前方案**

`competitive_analysis_agent/streamlit_app.py` 只收集输入和展示结果，
运行与校验位于 `competitive_analysis_agent/ui_service.py`。

**为什么这样选择**

保持 `streamlit_app.py` 简单，让输入解析、演示边界、状态回调和图执行可以用 pytest
独立验证。

**替代方案**

把所有逻辑直接写在 Streamlit 脚本顶层。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 页面与 service 分离 | 可测试、可复用、UI 更薄 | 多一个模块 | 有真实业务流程的应用 |
| 全部写在页面 | 初始文件少 | 业务和展示耦合，测试困难 | 一次性演示脚本 |

**什么时候考虑切换**

只有不会继续维护、没有外部调用的极小展示脚本才适合把逻辑全部写在页面中。

**面试回答参考**

我把 Streamlit 当作展示层。页面负责输入、状态和下载，普通 Python service 负责
校验并运行 LangGraph。这样同一业务路径既能被 UI 调用，也能在 pytest 中独立验收。

### 决策 2：为什么当前 UI 明确限制为演示案例？

**问题背景**

项目尚未实现真实 Tavily Provider。允许任意产品输入但继续使用固定 Evidence，会让
界面看起来支持真实搜索，实际却产生错误语义。

**当前方案**

表单可编辑，但 `validate_demo_request()` 会验证是否为当前固定 Evidence 支持的案例。

**为什么这样选择**

诚实暴露能力边界，既能完成端到端 UI 验收，也不会伪造任意产品的搜索结果。

**替代方案**

根据用户输入动态生成虚构 Evidence，或在本阶段顺便接入真实搜索。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 限定演示案例 | 可重复、来源真实可查、范围清晰 | 暂不支持任意产品 | MVP UI 验收 |
| 生成虚构 Evidence | 任意输入都能跑 | 会误导用户，破坏可追溯性 | 不适合研究工具 |
| 同时接真实搜索 | 能分析任意产品 | 把新外部服务和 UI 两个阶段混在一起 | 后续独立扩展 |

**什么时候考虑切换**

真实搜索 Provider 完成并通过独立测试后，删除演示案例限制，复用同一 UI 输入契约。

**面试回答参考**

当前还没有真实搜索服务，所以我没有让任意输入悄悄复用固定证据。UI 明确限制为演示
案例，模型和工作流是真实的，搜索输入是固定的。真实 Provider 接入后再放开产品范围。

### 决策 3：为什么用 Session State 保存报告？

**问题背景**

Streamlit 在按钮、下载等交互后会重新执行脚本。如果结果只存在局部变量中，页面重跑
后报告和错误状态会丢失。

**当前方案**

`initialize_session_state()` 和 `run_submitted_analysis()` 保存最终报告、来源、
阶段记录和错误摘要到 `st.session_state`。

**为什么这样选择**

Session State 正好覆盖当前浏览器会话，不需要提前引入数据库或长期 Memory。

**替代方案**

每次页面重跑都重新调用模型，或立即把结果写入数据库。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Session State | 简单、无额外服务、避免重复调用 | 刷新或会话结束后可能丢失 | 当前单会话 MVP |
| 每次重新运行 | 无状态代码简单 | 重复成本高、体验差 | 极便宜的纯计算 |
| 数据库 | 可长期保存、多用户共享 | 需要记录管理和清理策略 | 报告历史产品化 |

**什么时候考虑切换**

需要历史报告、跨设备访问或多用户协作时，再加入数据库；需要中断恢复时加入
LangGraph checkpointer。

**面试回答参考**

Streamlit 每次交互都会重跑脚本，所以我用 Session State 保存当前报告和执行状态，
避免下载时重新调用模型。它是页面会话状态，不是长期 Memory，也不负责图的中断恢复。

### 决策 4：为什么展示节点状态但不展示模型思维过程？

**问题背景**

用户需要知道任务运行到哪里，但模型内部推理不稳定、冗长，也不适合作为产品级解释。

**当前方案**

`run_analysis()` 从 LangGraph values stream 读取 `stage_history`，页面只展示
`planner`、`researcher`、`extractor` 等可验证阶段名称和结果。

**为什么这样选择**

节点状态来自 LangGraph 的实际执行轨迹，足以解释系统进度，同时避免暴露或依赖内部
推理文本。

**替代方案**

实时展示模型完整 prompt、响应或所谓思维链。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 展示阶段事件 | 稳定、简洁、可核对 | 不含模型内部细节 | 用户产品界面 |
| 展示完整 trace | 调试信息丰富 | 信息过载，可能包含敏感数据 | 受控开发调试工具 |

**什么时候考虑切换**

开发环境可以接入专业 tracing 页面，但面向用户的 UI 仍只展示阶段状态和可验证结果。

**面试回答参考**

我展示的是 LangGraph 节点级事件，而不是模型思维过程。用户能看到系统正在规划、
研究、提取、分析还是验证；详细 prompt 和 trace 留在受控调试工具中。

## 4. 异常与边界情况

- 目标产品、竞品或维度为空：Pydantic 在调用工作流前拒绝。
- 目标产品与竞品重复：`AnalysisRequest` 拒绝。
- 竞品使用换行、中英文逗号混合：统一解析为字符串列表。
- 自定义维度使用换行、中英文逗号或分号混合：统一解析并与常用维度按顺序去重。
- 输入超出演示案例：显示能力边界说明，不调用模型。
- 模型环境配置缺失：显示配置提示，不暴露变量值。
- 模型、网络或工作流失败：显示脱敏错误类别、分析编号、失败阶段、已完成阶段、
  失败位置和组件提供的安全定位信息，不展示 traceback、prompt 或原始模型响应。
- Researcher 部分失败：最终报告仍保留成功部分，页面额外显示警告。
- 最终 Verifier 未通过：页面提示用户查看报告顶部验证警告。
- Streamlit 脚本因下载按钮等交互重跑：从 Session State 恢复报告，不重新运行模型。
- 页面重新提交：先清除旧报告，避免新旧结果混淆。
- 浏览器会话结束或服务重启：当前 Session State 可能丢失；本阶段不做长期持久化。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：
  `python -m pytest -o addopts= -m live_llm tests/test_live_ui.py -q`
- 真实调用的组件：UI service 触发的完整 LangGraph
- 验证的输出契约：
  - 最终 `VerificationResult.passed=True`；
  - Markdown 包含报告标题和资料来源章节；
  - 状态回调最后一个节点为 `reporter`；
  - 回调历史与最终 State 历史一致；
  - 生成两条固定 Evidence。
- 测试结果：通过，`1 passed`
- 耗时：63.36 秒
- 验收过程发现并修复：
  - 首次真实调用中，Extractor 连续省略必填 `description`；
  - 增强嵌套 JSON 字段提示和修复指令后通过；
  - 未记录任何密钥或原始敏感配置。
- 失败原因：最终验收无失败。

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1a-from-prompt-to-action.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4a-agent-observability.zh-CN.ipynb`

### Relevant sections

- `第 3 节：尝试 ADK Web 界面`
- `概览`
- `第 2 节：使用 ADK Web UI 动手调试`
- `Events 标签页 - 详细查看 Traces`
- `在 Events 中检查函数调用`

### Core concept

Agent UI 负责把用户输入转换成一次可执行任务，并在运行过程中提供状态、结果和错误。
用户产品界面和开发调试界面关注点不同：用户需要清楚的阶段、可操作错误和最终产物；
开发者需要 prompt、工具参数、span 和耗时等更详细 trace。

可观测性中的 Logs、Traces 和 Metrics 也应分层。用户页面可以展示稳定的节点事件，
而详细函数调用和模型请求应进入受控调试工具，不能直接当作用户解释。

### How it appears in this project

- Streamlit 表单把控件值转换成 `AnalysisRequest`；
- `ui_service.run_analysis()` 负责调用 LangGraph；
- `graph.stream(..., stream_mode="values")` 提供节点级进度；
- `STAGE_LABELS` 把内部节点名转换成用户可理解的状态；
- Session State 保存当前报告，避免页面交互造成重复模型调用；
- `describe_user_error()` 展示可定位的脱敏上下文，并隐藏 traceback 和敏感细节；
- Markdown 报告和 Evidence URL 是用户最终可复核结果。

### ADK vs LangGraph

- Notebook 的 ADK Web UI 是框架自带的开发与调试界面；
- 本项目的 Streamlit 是面向竞品分析使用者的产品页面；
- ADK Events 和 Trace 能查看细粒度模型、工具调用；
- 本项目当前只从 LangGraph State 提取节点级事件；
- LangGraph 与 Agent 业务组件不依赖 Streamlit，页面可以替换而不改工作流。

### Intentionally postponed

- LangSmith、OpenTelemetry 或完整 trace；
- prompt 和模型原始响应查看器；
- Token、耗时、成本指标；
- 后台任务与取消操作；
- 登录、共享和历史报告页面。

## 7. 理解问题与参考思路

### 问题 1：UI 输入如何进入 LangGraph？

**参考思路：**

- Streamlit 控件产生普通字符串和列表；
- `create_analysis_request()` 转成 Pydantic 请求；
- `run_analysis()` 转成 `PlannerInput` 和初始 State；
- LangGraph 节点继续使用原有业务组件。

### 问题 2：长时间运行状态来自哪里？

**参考思路：**

- 不是模型自由生成的进度文字；
- `graph.stream(stream_mode="values")` 在节点后返回 State；
- 根据新增 `stage_history` 触发页面回调；
- 用户看到的是实际完成的节点。

### 问题 3：为什么页面不包含 Agent 业务逻辑？

**参考思路：**

- 页面重跑机制不适合承载复杂业务；
- 业务 service 可以独立测试和复用；
- UI 以后替换成 API 或其他前端时，无需重写 Agent。

### 问题 4：Session State 与 LangGraph State 有什么区别？

**参考思路：**

- LangGraph State 保存一次图运行的中间产物；
- Session State 保存当前浏览器会话的展示结果；
- Session State 不支持图中断恢复；
- 两者都不是跨会话长期 Memory。

## 8. 面试追问清单

- 同步 UI 何时需要升级为后台任务？
- 如果页面刷新，哪些状态保留，哪些会丢失？
- 为什么 UI 不直接创建 Agent 节点？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 11 将增加固定评测案例、指标和项目打包说明。本阶段不实现。

## 10. 后续变更记录

- 2026-06-22 真实 UI 路径复跑 Notion / Confluence 时，工作流已通过 Extractor，
  但在 Verifier 要求 Analyst 修订后遇到一次模型连接断开。`live_config.py` 新增
  `LIVE_MODEL_MAX_RETRIES = 1`，Planner、Extractor、Analyst、Verifier 的
  `ChatOpenAI` 配置统一改为最多一次供应商级重试。这样不会改变业务 State 和
  LangGraph 路由，也不会无限重试；它只降低临时网络断开导致页面直接失败的概率。
- 2026-06-22 用户再次复跑时遇到 `VerifierError`，旧页面只能显示错误类别。
  本次修复在 `ui_service.py` 中给异常挂载 `analysis_id`、`workflow_failed_stage`、
  `workflow_stage_history`、`failure_function` 和 `failure_line`。
- 页面错误现在会显示分析编号、失败阶段、已完成阶段、失败位置和 `public_detail`。
  例如 Verifier 输出结构错误会说明期望顶层 `{"issues": [...]}`，以及缺失或无效字段。
  日志和页面仍不展示 traceback、prompt、原始模型响应或密钥。
- 离线回归新增 `test_run_analysis_attaches_failure_context` 和
  `test_describe_user_error_includes_safe_details`。完整离线测试结果：
  `103 passed, 7 deselected`。
- 2026-06-22 后续真实端到端复跑发现：当 Verifier 已完成并设置
  `retry_pending=True`，下一步实际会回到 Analyst；旧的失败阶段推断只看最后完成阶段，
  因此会把后续 Analyst 失败误报成 Reporter 失败。`ui_service.py` 新增
  `infer_failed_stage_from_state()`，优先根据 `retry_pending` 返回 `analyst`，
  再按线性阶段推断下一步。
- 同次回归新增 `test_retry_pending_state_reports_analyst_as_next_failed_stage`。
  聚焦测试 `python -m pytest tests/test_workflow.py tests/test_ui_service.py tests/test_verifier.py tests/test_analyst.py -q`
  结果 `44 passed`；完整离线测试结果更新为 `109 passed, 7 deselected`。
- 2026-06-23 用户想对比 ChatGPT、Claude、Gemini 时，原页面只能从固定
  `AVAILABLE_DIMENSIONS` 中选择，无法加入 `coding`、`research`、
  `enterprise_security`、`ecosystem` 等自定义维度。本次保留常用维度作为快捷选择，
  在 `streamlit_app.py` 增加自定义维度文本框，并在 `ui_service.py` 新增
  `parse_custom_dimensions()` 与 `build_analysis_dimensions()`。合并时按首次出现顺序
  去重，避免用户把已勾选的 `features` 又写进自定义框后触发重复维度校验。
- 对应测试新增 `test_custom_dimensions_are_merged_with_selected_dimensions` 和
  `test_custom_dimensions_are_submitted_to_service`。聚焦测试
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_ui_service.py tests/test_streamlit_app.py tests/test_planner.py -q`
  结果 `22 passed`；完整离线测试
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`
  结果 `136 passed, 7 deselected`。
- 2026-06-23 用户提交 ChatGPT、Claude、Gemini 和 8 个分析维度后，Planner 生成
  24 个调研任务，Researcher 收集到 57 条 Evidence，Extractor 模型调用失败。
  本次在 `ui_service.py` 新增 `choose_max_results_per_task()`：4 个以内维度仍每个任务取
  3 条结果；5-6 个维度取 2 条；7 个及以上维度取 1 条。这样自定义维度越多，
  Evidence 输入越克制，符合个人项目“先稳定跑完”的目标。
- 对应测试新增 `test_many_dimensions_reduce_search_results_per_task`。聚焦测试
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_extractor.py tests/test_ui_service.py tests/test_workflow.py -q`
  结果 `33 passed`；完整离线测试更新为 `139 passed, 7 deselected`。
