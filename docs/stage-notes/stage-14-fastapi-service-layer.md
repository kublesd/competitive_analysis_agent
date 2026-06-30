# Stage 14：FastAPI 服务层

## 1. 本阶段目标

本阶段在已完成的本地 Agent MVP 之上增加 HTTP 服务入口，让前端、脚本或其他系统可以
用 JSON 调用竞品分析流程，而不必依赖 Streamlit 页面。

核心输入是一次分析请求：

```text
target_product + competitors + dimensions + official_domains_by_product
```

核心输出是一次分析结果：

```text
final_report + stage_history + verification_passed + evidence 摘要 + research_errors
```

它位于完整流程的最外层：

```text
HTTP JSON 请求
  -> FastAPI 请求模型
  -> ui_service.AnalysisRequest
  -> ui_service.run_analysis()
  -> LangGraph 工作流
  -> FastAPI JSON 响应
```

## 2. 实现结果

- 完成的功能：
  - 新增 `competitive_analysis_agent/api_app.py`。
  - `GET /health` 返回脱敏配置状态。
  - `POST /analyses` 同步运行一次竞品分析。
  - API 响应返回 Markdown 报告、阶段轨迹、验证结果、Evidence 摘要和研究错误。
  - 错误响应按 422、503、500 分类，并复用 UI 已有的脱敏错误描述。
- 关键文件：
  - `competitive_analysis_agent/api_app.py`
  - `tests/test_api_app.py`
  - `pyproject.toml`
  - `README.md`
- 核心数据流：
  - `ApiAnalysisRequest.to_service_request()` 把 HTTP body 转成 `ui_service.AnalysisRequest`。
  - `create_analysis()` 调用注入的 `analysis_runner`，默认是 `ui_service.run_analysis()`。
  - `ApiAnalysisResponse.from_run_result()` 把内部结果转成 API JSON。
- 验证方式与结果：
  - 聚焦测试：`python -m pytest tests\test_api_app.py -q`
  - 相关回归：`python -m pytest tests\test_api_app.py tests\test_ui_service.py tests\test_health.py -q`
  - 完整离线回归：`python -m pytest -q`
  - 临时服务检查：启动 uvicorn 后请求 `GET http://127.0.0.1:8000/health`
  - 当前结果：聚焦测试 `4 passed in 0.51s`；相关回归 `18 passed in 0.57s`；
    完整离线回归 `158 passed, 7 deselected in 1.73s`；健康检查返回 `status: ok`。
- 真实 LLM 测试：
  - 本阶段不新增 Prompt、模型输出 Schema、模型调用路径或 LangGraph 路由。
  - 使用 fake runner 验证 API 边界，不需要真实 LLM。
- 暂未实现：
  - 鉴权、限流、后台任务队列、流式进度、任务历史、数据库、Docker 和云部署。

## 3. 设计决策

### 决策 1：FastAPI 是否直接调用 LangGraph 节点？

**问题背景**

项目已经有 `ui_service.run_analysis()`，它负责输入校验、真实组件装配、运行 ID、日志、
错误脱敏和结果整理。新增 API 时，如果绕过这一层，会产生两套入口逻辑。

**当前方案**

`competitive_analysis_agent/api_app.py::create_analysis()` 只把 HTTP 请求转换成
`AnalysisRequest`，然后调用 `ui_service.run_analysis()`。

**为什么这样选择**

FastAPI 是新的外部入口，不是新的 Agent 内核。复用应用服务可以保证 Streamlit 和 API
使用同一套校验、日志、搜索配置和工作流执行路径。

**替代方案**

在 API 中直接创建 `WorkflowComponents`、`create_workflow_graph()` 并手动处理最终 State。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 复用 `ui_service.run_analysis()` | 改动小、行为一致、测试容易 | API 进度暂时只能在结束后返回 | 本地 MVP 和多入口复用 |
| API 直接操作 LangGraph | 可定制每个节点事件和响应形状 | 与 Streamlit 入口重复，容易分叉 | API 需要完全不同执行策略时 |

**什么时候考虑切换**

如果未来 API 要支持 Server-Sent Events、后台任务、取消任务或分布式 Trace，可以把运行编排
提到一个更通用的 application service，而不是让 FastAPI 直接散落调用节点。

**面试回答参考**

我把 FastAPI 设计成薄服务层，只负责 HTTP 协议转换。真正的业务执行仍复用
`ui_service.run_analysis()`，这样 Streamlit 和 API 的行为不会分叉，也不会出现两套
Agent 编排逻辑。后续如果要做异步任务，可以再抽出更通用的 application service。

### 决策 2：为什么第一版 API 使用同步请求？

**问题背景**

竞品分析可能需要模型和搜索，运行时间比普通 CRUD 更长。API 可以直接同步返回，也可以先
创建任务再轮询结果。

**当前方案**

`POST /analyses` 同步运行并返回完整结果。

**为什么这样选择**

当前目标是让项目有一个可演示、可测试的 HTTP 服务层。同步接口最容易理解，也能直接用
FastAPI `TestClient` 和 fake runner 覆盖边界行为。

**替代方案**

`POST /analyses` 只返回 `job_id`，后台任务运行后由 `GET /analyses/{job_id}` 查询。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 同步接口 | 简单、无数据库、端到端结果直观 | 长任务可能超时，不能中途查询进度 | 本地演示和小规模集成 |
| 后台任务 | 支持长任务、轮询、重试和取消 | 需要队列、存储、状态机和更多测试 | 多用户或生产环境 |

**什么时候考虑切换**

当一次分析经常超过前端或网关超时时间、需要多人并发提交、需要任务历史或取消功能时，应改成
后台任务模型。

**面试回答参考**

我先做同步 API，因为这个项目主要是实习展示和本地集成验证，重点是证明 Agent 工作流可以
通过 HTTP 复用。异步任务会引入队列和持久化，适合部署后再加。

### 决策 3：为什么 API Evidence 只返回摘要？

**问题背景**

`Evidence` 可能包含搜索摘要、价格页裁剪正文和网页来源信息。直接返回内部 Evidence 会让
响应体变大，也可能把不适合展示的原始网页内容暴露给调用方。

**当前方案**

`ApiEvidenceResponse` 返回 `evidence_id`、产品、主题、标题、URL、来源类型、采集时间和
`snippet_preview`，不返回 `raw_content`。

**为什么这样选择**

API 调用方通常需要知道报告引用了哪些来源，而不是读取完整网页正文。摘要足够复核来源，
也保持响应稳定、可控。

**替代方案**

直接返回完整 `Evidence.model_dump()`。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Evidence 摘要 | 响应小、泄露风险低、契约稳定 | 不能通过 API 查看完整裁剪正文 | 默认公开 API |
| 完整 Evidence | 调试信息多 | 响应大，可能暴露网页原文或敏感上下文 | 受控调试接口 |

**什么时候考虑切换**

如果要做内部调试，可以增加受鉴权保护的 debug endpoint，而不是改变默认分析接口。

**面试回答参考**

我没有把内部 Evidence 原样返回，而是做了一个 API 专用摘要模型。这样调用方可以复核引用
来源，同时避免把网页原文和过长片段暴露出去。

### 决策 4：为什么要把内部异常映射成 HTTP 状态码？

**问题背景**

API 调用方需要根据错误类型决定是否修改输入、补配置或稍后重试。直接返回 500 会让所有失败
看起来一样；直接暴露异常原文又可能泄露内部信息。

**当前方案**

`build_http_error()` 将配置错误映射为 503，将输入校验错误映射为 422，将未知运行错误映射为
500。错误消息复用 `ui_service.describe_user_error()`。

**为什么这样选择**

这保留了足够的调用方语义：422 是请求问题，503 是外部配置或服务条件不足，500 是未分类的
内部运行失败。同时沿用已有脱敏策略。

**替代方案**

全部异常统一返回 500，或把 `str(error)` 直接写入响应。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 分类且脱敏 | 便于调用方处理，风险低 | 需要维护错误分类表 | 面向集成的 API |
| 全部 500 | 实现最少 | 调用方无法判断是否该重试或改输入 | 临时原型 |
| 暴露异常原文 | 调试方便 | 可能泄露密钥、供应商响应或内部路径 | 本地一次性排障 |

**什么时候考虑切换**

如果将来错误类型增多，可以引入项目级错误基类和错误码，而不是继续按异常类名集合判断。

**面试回答参考**

我把错误响应分成输入错误、配置不可用和未知运行错误三类。这样前端或调用方知道下一步该改
请求、补配置还是稍后重试，同时避免把内部异常原文暴露给外部。

## 4. 异常与边界情况

- 请求字段缺失、空列表或多余字段由 FastAPI/Pydantic 返回 422。
- `official_domains_by_product` 中出现本次请求以外的产品时，转换到
  `AnalysisRequest` 时失败，并返回脱敏后的 422。
- 缺少 LLM 或 Tavily 配置时返回 503，提示补环境配置。
- 未知运行失败返回 500，并包含脱敏错误类别，不返回 traceback。
- Evidence 响应不返回 `raw_content`，只返回短摘要。
- 当前同步接口不保存任务状态；客户端断开后没有任务查询接口。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：否。
- 配置来源：本阶段不读取 `.env.example` 发起模型调用。
- 测试命令：不适用。
- 真实调用的组件：无。
- 验证的输出契约：HTTP 请求模型、响应模型、Evidence 摘要、错误状态码。
- 测试结果：不适用；聚焦离线测试已通过。
- 耗时：不适用。
- 失败原因：无。

## 6. 与课程 Notebook 的对应

- Notebook：
  - `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1a-from-prompt-to-action.zh-CN.ipynb`
  - `F:\大模型应用开发学习\3.Google_and_Kaggle\5-Day\codelabs\day-5b-agent-deployment.zh-CN.ipynb`
- 相关章节：
  - `第 3 节：尝试 ADK Web 界面`、`概览`
  - `你将学习什么`、`其他部署选项`
- 通用知识点：
  - Agent 的外部入口可以是 notebook runner、Web UI、CLI、API server 或云部署；入口负责收集输入、
    展示输出和连接运行时，不应混入 Agent 的内部业务职责。
  - 部署方式需要跟规模匹配。本地 API 或 Cloud Run 适合 demo 和中小规模工作负载，复杂多 Agent
    系统才更需要 Kubernetes 或托管 Agent 平台。
- 在本项目中的实现：
  - `api_app.py` 对应新的服务入口，`ui_service.run_analysis()` 对应实际 Agent 运行时。
  - 与 Streamlit 一样，FastAPI 不重新实现 Planner、Researcher、Extractor、Analyst、Verifier
    和 Reporter。
- ADK 与 LangGraph/本项目的差异：
  - Notebook 中的 ADK Web UI 和部署工具由 ADK runtime 管理 Agent 运行；本项目用 FastAPI
    暴露 HTTP 入口，用 LangGraph 在应用内部编排状态。
  - ADK `api_server` 是框架提供的服务入口；本项目手写 FastAPI 层，因此需要自己定义请求模型、
    响应模型和错误映射。
- 本阶段有意简化的内容：
  - 不接入云部署、托管 session、Memory Bank、自动扩缩容、生产监控和 tracing。
  - 不实现鉴权和多租户隔离；本地演示时仍应把地址当作只供可信环境访问的服务。

## 7. 理解问题与参考思路

### 问题 1：FastAPI 层的输入和输出是什么？

**参考思路：**

- 输入是产品、竞品、分析维度和可选官方域名。
- 输出是 Markdown 报告、执行阶段、验证状态、来源摘要和研究错误。
- 它不输出内部 LangGraph State 的全部字段。

### 问题 2：为什么 API 不直接调用每个 Agent 节点？

**参考思路：**

- Streamlit 已经通过 `ui_service.run_analysis()` 使用同一套应用边界。
- API 复用它能避免两套校验、日志和异常处理逻辑。
- 入口层应做协议转换，工作流编排仍由应用服务负责。

### 问题 3：为什么同步接口不是最终生产形态？

**参考思路：**

- 模型和搜索可能耗时较长，同步接口容易遇到客户端或网关超时。
- 生产环境通常需要任务 ID、后台队列、状态查询、取消和重试。
- 当前项目先保留最小可演示版本，避免过早引入队列和数据库。

## 8. 面试追问清单

- 如果分析耗时超过 60 秒，你会如何改造成后台任务？
- 如果要给 API 加鉴权，应该在哪一层做？
- 为什么默认响应不返回完整 Evidence 原文？
- FastAPI、Streamlit 和 LangGraph 在这个项目中分别承担什么职责？
- 当前错误分类哪些来自测试，哪些是工程约定？

## 9. 下一阶段衔接

Stage 14 已经让项目具备 HTTP 服务入口。下一步如果继续增强，最自然的方向是 API 运行形态：
增加后台任务、进度查询和鉴权；但这些都应在真实调用延迟和部署需求明确后再做。
