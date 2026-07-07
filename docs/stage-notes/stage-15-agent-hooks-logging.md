# Stage 15：Agent Hooks 日志系统

## 1. 本阶段目标

把原来集中在 `ui_service.py` 的阶段日志逻辑升级为 Agent Hooks 系统，让工作流节点继续专注
业务逻辑，观测逻辑通过统一生命周期事件接入。

核心数据流：

```text
ui_service.run_analysis()
  -> 创建 AgentRunContext 和 HookManager
  -> create_workflow_graph(..., hook_manager=...)
  -> 每个 LangGraph node wrapper 触发 stage hooks
  -> JsonlLoggingHook 写入 logs/agent-events.jsonl
```

默认目标仍是本地排障，不接入外部平台、数据库或告警系统。

## 2. 实现结果

- 新增 `competitive_analysis_agent/agent_hooks.py`：
  - `AgentRunContext` 保存 `analysis_id`、`entrypoint`、开始时间和脱敏配置摘要。
  - `AgentStageContext` 保存阶段名、尝试次数、重试计数和阶段开始时间。
  - `AgentHook` 定义 run/stage 生命周期接口。
  - `HookManager` 顺序调用 Hook；Hook 自身异常只记录 `hook_failed`，不打断主流程。
- 新增 `competitive_analysis_agent/observability.py`：
  - `JsonlLoggingHook` 写入固定字段 JSONL 事件。
  - `StagePayloadSummarizer` 生成阶段输入、输出和错误摘要。
- `workflow.py` 在节点 wrapper 中触发 `stage_started/stage_completed/stage_failed`。
- `ui_service.py` 默认启用 `JsonlLoggingHook`，并支持外部注入额外 hooks。
- `streamlit_app.py` 使用 `entrypoint="streamlit"`；FastAPI 默认入口使用 `entrypoint="api"`。
- `logging_config.py` 保留 `application.log`，新增 `agent-events.jsonl` 独立轮转 handler。

## 3. 设计决策

### 决策 1：为什么使用 Hooks，而不是继续在 UI service 写日志？

当前方案把 run 级上下文放在 `ui_service.py`，把 stage 生命周期放在 `workflow.py` 的节点
wrapper。这样 Streamlit、FastAPI 和未来 CLI 都能复用同一套观测机制。

替代方案是继续在 `graph.stream()` 后读取 State 并写日志。它改动少，但日志机制会继续依赖
UI service 对 State 的理解，未来多入口时容易分叉。

面试回答参考：我把日志做成 Hook，是为了让业务节点不直接依赖日志实现，也让不同入口复用同一
套生命周期事件。第一版只覆盖 run/stage，后续可以扩展模型调用、搜索调用和 token usage。

### 决策 2：为什么默认写 JSONL 文件？

当前应用仍是本地单进程 MVP，JSONL 比 SQLite 和外部平台更轻：每行一条事件，能被 `grep`、
Python 脚本或测试稳定解析，也不需要 schema 迁移。

替代方案是 SQLite 运行历史或 LangSmith/OpenTelemetry。前者适合做查询 UI，后者适合多实例
监控，但都会显著增加配置和学习成本。

面试回答参考：我没有直接上外部可观测平台，而是先用 JSONL 建立结构化事件契约。等部署到
多实例或需要团队协作排障时，再把 Hook 后端替换成 OpenTelemetry 或 LangSmith。

### 决策 3：为什么默认只记录脱敏摘要？

Agent 的 Prompt、模型响应、Evidence 正文和完整报告都可能包含用户输入、网页内容或供应商
错误细节。默认日志只记录数量、ID、阶段、错误类型和字符数，不记录正文。

替代方案是在 DEBUG 日志中保存完整上下文。它排障更方便，但泄密风险和日志体积都更高；如果
未来需要，应做成显式开关、短期保留和字段脱敏，而不是默认行为。

面试回答参考：Agent 日志和普通后端日志不同，模型上下文可能天然带敏感信息。我默认只保存
脱敏摘要，让日志能回答“跑到哪一步、规模多大、哪里失败”，而不保存原始业务内容。

## 4. 异常与边界情况

- Hook 抛异常时，`HookManager` 只写 `hook_failed hook=... method=... error_type=...`。
- 阶段节点抛异常时，先触发 `stage_failed`，再原样抛出给既有错误处理。
- `run_failed` 在 `ui_service.py` 补充 `analysis_id`、失败函数和行号后触发。
- Verifier 退回 Analyst 时，同名阶段的 `attempt_index` 会递增，便于区分重试。
- JSONL 事件不记录 API Key、Prompt、模型原始响应、Evidence `raw_content`、完整 URL、完整报告或异常原文。
- `logs/` 继续由 `.gitignore` 忽略；本地日志仍按运行数据管理。

## 5. 验证结果

- 编译检查：`conda run -n base python -m compileall -q competitive_analysis_agent tests`：通过。
- 聚焦测试：
  - `tests/test_observability.py`
  - `tests/test_logging_config.py`
  - `tests/test_ui_service.py`
  - `tests/test_workflow.py`
  - `tests/test_api_app.py`
  - `tests/test_streamlit_app.py`
  - 结果：`36 passed in 1.59s`。
- 完整离线回归：`conda run -n base python -m pytest -q`：`163 passed, 7 deselected in 1.80s`。
- 真实 LLM 测试：不适用。本阶段不改变 Prompt、模型输出 Schema、模型调用路径或 LangGraph 路由。

## 6. 与课程 Notebook 的对应

- Notebook：
  `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4a-agent-observability.zh-CN.ipynb`
- 相关概念：日志回答“发生了什么”，Trace 串联一次运行路径，Metrics 汇总整体健康状况。
- 本项目映射：`agent-events.jsonl` 是轻量事件 Trace，使用同一个 `analysis_id` 和 stage 事件还原一次
  Agent 运行；它还不是带父子 span、采样和集中查询的完整可观测平台。
- ADK 与本项目差异：课程里 ADK 可通过插件挂接 Agent 生命周期；本项目用 `AgentHook` Protocol 和
  LangGraph node wrapper 手写生命周期 seam。

## 7. 理解问题与参考思路

### 问题 1：为什么 HookManager 不能让 Hook 异常中断主流程？

**参考思路：** Hook 是观测增强，不是业务依赖。日志失败不应让一次本来可完成的分析失败。

### 问题 2：为什么 stage hook 放在 workflow wrapper，而不是每个节点函数里？

**参考思路：** 节点函数保持普通业务函数，仍能单独测试；wrapper 统一处理生命周期事件，减少重复。

### 问题 3：JSONL 目前能回答什么，不能回答什么？

**参考思路：** 它能回答一次运行的入口、阶段顺序、耗时、规模、失败类型和重试次数；不能直接还原
Prompt、模型原始响应、token usage 或跨机器分布式 trace。

## 8. 下一步

后续若要增强生产可观测性，优先扩展新的 Hook 后端，而不是改业务节点：例如模型调用 Hook、搜索调用
Hook、token usage 统计、OpenTelemetry adapter 或一个只读日志查看页面。

## 9. 2026-07-02 完整模型 I/O 后台日志

用户需要在后台排查 Planner、Extractor、Analyst 和 Verifier 的真实模型输入输出。原有
`agent-events.jsonl` 只保存脱敏摘要，不能还原模型看到的 Prompt 和返回内容，因此本次新增独立的
`logs/model-io.jsonl`。

实现结果：

- `logging_config.py` 新增 `MODEL_IO_LOGGER_NAME`、`get_model_io_log_path()` 和独立轮转
  JSONL handler。
- 新增 `competitive_analysis_agent/model_io.py`：
  - `model_io_context()` 在 LangGraph 节点运行期间保存 `analysis_id`、`entrypoint`、
    `stage`、`attempt_index` 和 `retry_count`；
  - `log_model_request()` 记录完整 messages、消息数量和输入字符数；
  - `log_model_response()` 记录结构化响应、`raw.content` 或 Pydantic 输出；
  - `log_model_error()` 只记录异常类型，不保存供应商错误原文。
- `workflow.py` 在 observed node wrapper 中注入模型日志上下文。
- `LangChainPlannerModel`、`LangChainExtractorModel`、`LangChainAnalystModel`、
  `LangChainVerifierModel` 在真实模型边界写 request/response/error 事件。

重要取舍：

- `agent-events.jsonl` 继续保持脱敏摘要，用于默认运行轨迹。
- `model-io.jsonl` 会保存完整 Prompt、Evidence 文本片段和模型响应，便于本地排障；它不保存
  API Key，因为密钥不在 messages 或模型响应中，但日志本身仍应视为运行数据，不应提交到 Git。
- 错误事件不记录供应商异常原文，避免把远端错误消息、请求详情或潜在敏感文本写入日志。

验证记录：

- 聚焦测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_logging_config.py tests\test_model_io.py tests\test_planner.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py`
  结果 `82 passed`。
- 完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest`
  结果 `169 passed, 7 deselected`。
- 真实 LLM 未在本次强制触发；当前用户环境刚出现过硅基流动 Planner `APITimeoutError`，本次改动通过
  fake structured model 覆盖日志行为，避免为了验证日志再次消耗外部模型请求。
