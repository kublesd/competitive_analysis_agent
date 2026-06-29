# Agent / 大模型应用开发简历草稿

> 使用方式：把方括号内容替换成你的真实信息；如果投递英文岗位，可以在定稿阶段再翻译成英文版。

## 基本信息

- 姓名：[你的姓名]
- 手机 / 邮箱：[手机号] / [邮箱]
- GitHub：[GitHub 链接]
- 作品集 / Demo：[项目演示链接，可选]
- 求职意向：Agent 开发工程师 / 大模型应用开发工程师 / Python AI 应用开发工程师

## 个人优势

具备 Python 后端与大模型应用开发能力，熟悉基于 LangGraph、Pydantic、LangChain 和 Streamlit 构建可验证的 Agent 工作流。能够将复杂任务拆解为规划、检索、抽取、分析、验证和报告生成等阶段，并通过结构化 Schema、引用校验、有限重试、离线评测和后台日志提升 Agent 输出的可靠性与可排查性。

## 技术栈

- 大模型应用：LangGraph、LangChain、结构化输出、Agent Workflow、Prompt 约束、引用验证
- 后端与工程化：Python、Pydantic、pytest、模块化设计、配置管理、日志轮转
- 搜索与工具调用：Tavily Search API、Search Provider 抽象、搜索结果规范化、错误隔离
- 前端 / 展示：Streamlit、Markdown 报告、交互式参数输入
- 质量保障：离线 fixtures、真实 LLM 测试、真实搜索测试、评测指标设计、敏感信息脱敏

## 项目经历

### 竞品分析 Agent

项目链接：[填写 GitHub 或本地项目链接]  
技术栈：Python、LangGraph、LangChain、Pydantic、Streamlit、Tavily、pytest

项目描述：  
独立开发一个面向产品竞品分析的 AI Agent，将用户输入的目标产品、竞品和分析维度转化为可追溯的结构化研究流程，自动完成调研规划、网页搜索、产品画像抽取、竞品分析、引用验证和 Markdown 报告生成。

核心工作：

- 设计 Planner、Researcher、Extractor、Analyst、Verifier、Reporter 多阶段 Agent 工作流，使用 LangGraph 管理共享 State、条件路由和一次失败修订机制。
- 使用 Pydantic 定义 ResearchTask、Evidence、ProductProfile、CompetitiveAnalysis、VerificationResult 等核心数据契约，降低模型自由输出带来的格式漂移。
- 实现可替换 SearchProvider 与 SearchAdapter，将 Tavily 搜索结果统一映射为项目内部结构，并完成 URL 规范化、去重、官方来源分类和搜索错误隔离。
- 为定价类任务增加网页正文裁剪逻辑，只提取套餐、价格、计费周期和限制相关片段，减少无关网页内容进入后续模型上下文。
- 构建 Verifier 引用验证机制，先通过确定性代码检查 Evidence ID，再由模型检查语义支持与冲突；验证失败时最多回退 Analyst 一次，避免无限循环和成本失控。
- 实现确定性 Reporter，报告阶段只渲染已有结构化产物，不再调用模型或新增事实，保证报告内容可追溯到 Evidence。
- 搭建 Streamlit 交互界面，支持输入产品、竞品、分析维度和官方域名，并展示最终竞品分析报告。
- 增加本地后台日志，为每次分析生成 `analysis_id`，记录节点顺序、累计耗时、任务数、证据数、重试次数和脱敏异常类别，便于定位 Agent 执行问题。
- 建立离线评测集与测试体系，固定案例覆盖成功、一次重试恢复和验证失败告警场景；评测指标包括案例通过率、任务成功率、字段覆盖率、引用有效率和来源覆盖率。

项目成果：

- 完成从用户输入到可引用 Markdown 报告的端到端 Agent MVP。
- 固定评测集 3 个案例，案例通过率 100%，引用有效率 100%，来源覆盖率 100%。
- 支持离线回归测试与真实 LLM / 真实搜索分层验收，避免每次测试都依赖付费 API。
- 后台日志默认不记录 API Key、Prompt、Evidence 正文和完整异常文本，降低日志泄密风险。

可面试展开点：

- 为什么把业务节点写成普通 Python 组件，再用 LangGraph 薄包装接入图编排？
- 为什么使用共享 State 和 Pydantic Schema，而不是让节点自由传字典？
- 为什么 Verifier 失败只允许一次重试？
- 为什么报告生成阶段不再调用模型？
- 如何把当前本地日志升级为 OpenTelemetry / LangSmith Trace？

## 其他项目

### [可选：补充第二个 AI / Python 项目]

项目链接：[链接]  
技术栈：[技术栈]

- [写 3-5 条与岗位相关的工作内容。]
- [优先写可验证结果，例如测试数量、性能、功能完整度、用户流程。]

## 教育经历

- [学校] - [专业] - [学历] - [时间]

## 附加信息

- 可补充：开源贡献、技术博客、课程项目、比赛、证书。
- 如果投递 Agent 岗，建议把 UE5 项目压缩到“其他项目”或“附加项目”，除非岗位强调游戏 AI、仿真或 3D 交互。
