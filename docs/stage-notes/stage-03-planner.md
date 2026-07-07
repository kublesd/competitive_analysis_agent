# Stage 03：Planner

## 1. 本阶段目标

本阶段把目标产品、竞品和分析维度转换成可执行的 `ResearchTask` 列表。

输入：

- 目标产品；
- 竞品列表；
- 分析维度列表。

输出：

- 每个产品与每个维度各对应一条调研任务。

它位于完整流程的最前面：

```text
用户分析目标 -> Planner -> ResearchTask 列表 -> Researcher
```

## 2. 实现结果

- 完成的功能：
  - 校验 Planner 输入；
  - 构建 Planner 提示消息；
  - 通过 Pydantic 约束结构化模型输出；
  - 检查每个产品和维度是否恰好生成一条任务；
  - 首次输出无效时反馈错误并修复一次；
  - 使用固定模型响应完成无网络测试。
  - 通过 LangChain 接入硅基流动 OpenAI 兼容接口；
  - 提供 `python -m competitive_analysis_agent.live_planner` 手动 smoke test。
- 关键文件：
  - `competitive_analysis_agent/planner.py`
  - `competitive_analysis_agent/live_planner.py`
  - `tests/test_planner.py`
  - `tests/test_live_planner.py`
  - `tests/fixtures/planner_outputs.json`
- 核心数据流：`PlannerInput -> 结构化模型调用 -> PlannerOutput -> 覆盖校验`。
- 验证方式与结果：
  - `python -m pytest -q`：全项目 17 个测试通过；
  - 手动样例生成 2 个产品乘以 2 个维度，共 4 条任务。
  - 真实模型 `Qwen/Qwen3-8B` 成功为 Notion 和飞书生成 4 条任务，
    产品与维度覆盖校验通过。
- 暂未实现：
  - Planner 不调用搜索；
  - 不生成证据；
  - 不接入 LangGraph；
  - 真实模型调用只作为手动 smoke test，不加入普通单元测试。

## 3. 设计决策

### 决策 1：为什么 Planner 只生成任务，不直接调用搜索？

**问题背景**

完整流程同时需要规划查询和执行搜索。如果两个职责放在一个组件里，就难以判断失败来自任务拆分还是搜索服务。

**当前方案**

`Planner.plan()` 只返回 `list[ResearchTask]`。`competitive_analysis_agent/planner.py`
没有导入 `SearchAdapter`，提示词也明确要求“不要搜索网页”。

**为什么这样选择**

规划与执行分开后，Planner 可以完全使用固定模型输出测试。下一阶段也可以独立测试某条任务如何转换成证据，错误边界更清晰。

**替代方案**

让一个 Agent 在生成 query 后立即调用搜索工具，并直接返回搜索结果。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Planner 与搜索分离 | 易测试、易定位错误、搜索服务可替换 | 流程中多一个明确步骤 | 多节点 Agent、需要保留中间状态 |
| Planner 内直接搜索 | 原型代码少、单次调用看起来直接 | 规划与工具失败耦合，难以复用和测试 | 一次性的小型搜索助手 |

**什么时候考虑切换**

如果产品只需要一次临时搜索，不需要展示任务、保存中间结果或处理部分失败，可以使用合并方案。

**面试回答参考**

我的 Planner 只负责把产品和维度拆成 `ResearchTask`，不直接搜索。这样规划质量和搜索服务可以分别测试，搜索超时也不会被误认为规划失败。代价是工作流多了一个节点，但对需要证据追踪和部分失败处理的竞品分析更清晰。

### 决策 2：为什么 Planner 输出结构化任务，而不是自然语言计划？

**问题背景**

Researcher 需要稳定读取产品、主题和查询。如果 Planner 返回一段文字，后续节点还要猜测或解析其格式。

**当前方案**

`PlannerOutput` 使用 Pydantic 定义 `tasks: list[ResearchTask]`。
`LangChainPlannerModel` 通过 `with_structured_output(PlannerOutput)` 绑定 Schema，
`validate_planner_output()` 再执行运行时校验。

**为什么这样选择**

结构化输出能尽早发现缺字段和类型错误。产品与维度覆盖则由
`validate_task_coverage()` 使用普通 Python 集合检查，因为这类规则确定且无需模型判断。

**替代方案**

1. 返回自然语言计划，再用正则表达式解析；
2. 完全不用模型，按产品与维度使用字符串模板生成 query。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 结构化模型输出 + 代码校验 | query 可由模型措辞，边界仍可校验 | 有模型成本和输出不稳定性 | 维度可能复杂、查询需要语义调整 |
| 自然语言 + 正则解析 | 初期写法直观 | 格式易漂移，解析脆弱 | 临时演示且输出不进入后续程序 |
| 确定性模板生成 | 最稳定、便宜、无需 API | 查询表达较机械，难适应复杂需求 | 维度和搜索规则长期固定 |

**什么时候考虑切换**

如果实践证明所有查询都只是 `产品名 + official + 维度`，确定性模板会比 LLM 更合适。如果任务需要地区、语言、时间范围或来源策略，结构化模型更有扩展空间。

**面试回答参考**

我使用 Pydantic 约束 Planner 输出，避免 Researcher 解析自然语言。模型负责生成查询表达，代码负责检查产品和维度覆盖，因为确定性规则不应交给模型。对于当前简单矩阵，模板生成也是合理替代；选择模型主要是为后续更复杂的查询规划保留空间。

### 决策 3：为什么模型输出需要覆盖校验和有限修复？

**问题背景**

结构正确不代表业务正确。模型可能返回合法 JSON，却漏掉某个竞品的价格任务，或者重复生成同一组合。

**当前方案**

`validate_task_coverage()` 计算期望的“产品 × 维度”集合，并检测：

- 缺失组合；
- 输入范围外的组合；
- 重复组合。

`Planner.plan()` 首次校验失败后调用一次修复，第二次仍失败就抛出
`PlannerError`。

**为什么这样选择**

一次修复可以处理常见的模型格式或漏项问题，同时给成本、延迟和失败时间设置明确上限。校验错误会进入修复消息，模型不需要盲猜问题。

**替代方案**

1. 第一次失败立即终止；
2. 持续重试直到成功；
3. 由代码自动补齐缺失任务。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 最多修复一次 | 能恢复常见错误，成本有上限 | 第二次失败仍需终止 | 当前同步 MVP |
| 立即失败 | 最简单、成本最低 | 对偶发格式错误不够宽容 | 高确定性模型或离线批处理 |
| 无限或多次重试 | 成功概率可能更高 | 成本和延迟不可控，可能循环 | 有统一重试平台和严格预算 |
| 代码自动补齐 | 确定性强 | 可能生成模型没有真正设计过的查询 | query 可完全模板化时 |

**什么时候考虑切换**

如果线上统计显示第一次失败率极低，可以直接失败以降低成本；如果任务改为后台异步执行，并有明确预算和退避策略，可以增加有限重试次数。

**面试回答参考**

Pydantic 只能保证字段结构，所以我又用代码检查产品与维度覆盖。校验失败时把具体错误反馈给模型修复一次，避免偶发格式问题直接导致失败。重试上限固定为一，防止无限循环和不可控成本。

### 决策 4：为什么通过模型接口注入，而不是在 Planner 内创建具体模型？

**问题背景**

单元测试不能依赖付费 API、网络和真实密钥，但生产代码又需要保留 LangChain 结构化调用边界。

**当前方案**

`Planner` 只依赖 `PlannerModel` 的 `invoke()` 接口。测试使用
`FakePlannerModel`；LangChain 场景使用 `LangChainPlannerModel`，它负责调用
`chat_model.with_structured_output(PlannerOutput)`。

**为什么这样选择**

业务校验与供应商初始化分离后，测试可重复、速度快，也不会消耗 Token。模型供应商变化时，Planner 主逻辑无需修改。

**替代方案**

在 `Planner.__init__()` 内直接读取环境变量并创建 `ChatOpenAI`。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 注入最小模型接口 | 测试简单、供应商解耦、无网络也可运行 | 调用方需要负责创建模型 | 可测试的应用代码 |
| Planner 内创建模型 | 使用入口少、短脚本方便 | 配置和业务耦合，测试需要 mock 全局依赖 | 一次性脚本或实验 Notebook |

**什么时候考虑切换**

如果项目只是单文件实验，可以直接创建模型。进入应用阶段后，配置、测试和供应商替换通常更适合依赖注入。

**面试回答参考**

我没有在 Planner 内硬编码 ChatOpenAI，而是注入一个最小 `invoke` 接口。测试使用固定响应，生产时再用 LangChain 包装器绑定 Pydantic Schema。这样普通单元测试不需要网络或 API Key，也降低了模型供应商与业务逻辑的耦合。

### 决策 5：为什么硅基流动接入使用 JSON mode，并保留 raw 输出？

**问题背景**

LangChain 当前默认的结构化输出方式偏向 OpenAI `json_schema`，而硅基流动
官方文档明确声明支持的是 `response_format={"type": "json_object"}`。第一次
真实调用还发现：如果 LangChain 在内部解析失败并直接抛错，Planner 原有的一次
修复逻辑无法获得模型原始输出。

**当前方案**

`LangChainPlannerModel` 显式调用：

```python
chat_model.with_structured_output(
    PlannerOutput,
    method="json_mode",
    include_raw=True,
)
```

解析成功时返回 `parsed`；解析失败时返回 `raw.content`，再由
`validate_planner_output()` 统一校验并进入一次修复。`live_planner.py` 使用
硅基流动基址、`Qwen/Qwen3-8B`、512 token 上限和关闭思考模式。

**为什么这样选择**

它遵循供应商公开支持的 JSON 模式，同时保留项目自己的错误处理流程。生成长度
上限和零 SDK 重试让手动 smoke test 能快速暴露问题，不会把一次超时放大成多次
长等待。

**替代方案**

1. 使用 LangChain 默认 `json_schema`；
2. 使用 function calling 传递 Schema；
3. 不使用 LangChain 结构化包装，直接调用 OpenAI SDK 并手动解析 JSON。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| JSON mode + raw 输出 | 符合硅基流动文档，能接入现有修复逻辑 | Schema 主要靠提示词和本地 Pydantic 校验 | 当前 OpenAI 兼容接口 |
| 默认 json_schema | Schema 可由服务端严格约束 | 供应商未明确声明完整兼容，迁移风险更高 | 原生 OpenAI 或明确支持该协议的服务 |
| Function calling | 参数结构通常稳定 | 不同模型的工具调用能力差异较大 | 已验证 function calling 的模型 |
| OpenAI SDK 手动解析 | 控制最直接、依赖更少 | 需要自行维护解析与 LangChain 集成 | 单供应商或轻量脚本 |

**什么时候考虑切换**

如果硅基流动后续明确支持 OpenAI JSON Schema，可以切换到
`method="json_schema"`。如果模型的 JSON 模式长期不稳定，则应比较 function
calling 或直接 SDK 解析的实测成功率。

**面试回答参考**

硅基流动明确支持 OpenAI 兼容的 JSON object 模式，所以我没有直接依赖
LangChain 默认的 JSON Schema。调用时保留 raw 输出，让解析失败仍能进入项目
自己的一次修复流程。真实测试还让我加入了生成长度和重试上限，避免免费模型
输出异常时造成长时间阻塞。

## 4. 异常与边界情况

- 目标产品与竞品重复：`PlannerInput` 在模型调用前拒绝。
- 分析维度重复：`PlannerInput` 在模型调用前拒绝。
- 缺少 `query` 等字段：`PlannerOutput.model_validate()` 失败并进入一次修复。
- 漏掉、重复或添加产品维度组合：覆盖校验失败并进入一次修复。
- 第二次输出仍无效：抛出 `PlannerError`，不会继续调用模型。
- 模型调用本身异常：立即转换成 `PlannerError`，不把服务故障误当成输出格式问题。
- LangChain 内部解析失败：保留原始文本，进入 Planner 的一次修复。
- 真实调用超时：smoke test 使用 45 秒超时且关闭 SDK 自动重试。
- 没有 API Key：固定模型测试仍可执行，真实入口明确报告缺少环境变量。

## 5. 与课程 Notebook 的对应

- Notebook：
  - `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1b-agent-architectures.zh-CN.ipynb`
  - `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1a-from-prompt-to-action.zh-CN.ipynb`
- 相关章节：
  - “第 2 节：为什么需要多 Agent 系统？”
  - “2.1 示例：研究与总结系统”
  - “第 6 节：总结 - 选择正确的模式”
  - “2.2 定义你的 Agent”
  - “2.4 它是如何工作的？”
- 通用知识点：
  - 把复杂任务拆给职责单一的组件；
  - instruction 定义 Agent 的职责和限制；
  - 规划与执行是不同责任；
  - LLM 编排灵活，但关键流程仍需要确定性规则保护。
- 在本项目中的实现：
  - `PLANNER_SYSTEM_PROMPT` 只允许拆分任务；
  - `PlannerOutput` 定义结构化输出；
  - `validate_task_coverage()` 保证完整任务矩阵；
  - `Planner` 将任务交给未来的 Researcher，而不是自己搜索。
- ADK 与 LangGraph/本项目的差异：
  - Notebook 使用 ADK `Agent`、`tools` 和 `output_key`；
  - 本阶段还没有 LangGraph，只实现可独立测试的普通 Python Planner；
  - 本项目使用 LangChain 风格的 `with_structured_output()`，并用 Pydantic
    校验输出。
- 本阶段有意简化的内容：
  - 没有 ADK 根协调器；
  - 没有工具调用；
  - 没有动态选择下一个 Agent；
  - 没有完整执行 trace，真实模型只进行了 Planner smoke test。

## 6. 理解问题与参考思路

### 问题 1：为什么 Planner 不直接搜索？

**参考思路：**

- Planner 负责决定“查什么”，Researcher 负责执行“怎么查”。
- 分开后可以独立测试任务覆盖和搜索失败。

### 问题 2：结构化输出和覆盖校验分别解决什么问题？

**参考思路：**

- Pydantic 检查字段、类型和非空值。
- 覆盖校验检查业务规则：每个产品与维度必须恰好出现一次。

### 问题 3：为什么只修复一次？

**参考思路：**

- 一次修复能处理常见漏项或格式错误。
- 固定上限避免无限循环、延迟和 Token 成本失控。

## 7. 面试追问清单

- 如果产品或维度数量扩大十倍，Planner 的任务数量和成本会如何变化？
- 如果移除结构化输出，Researcher 会承担哪些额外复杂度？
- 模型输出校验失败时，为什么不无限重试？
- 哪些结论来自测试，哪些只是工程判断？

## 8. 2026-07-02 Gemini OpenAI 兼容修复

**问题背景**

页面在 Planner 阶段失败，底层异常为 `BadRequestError`。真实最小调用复现后，
Gemini OpenAI 兼容接口返回的脱敏原因是：
`Unknown name "enable_thinking": Cannot find field.` 这说明失败不是任务矩阵、
额度或搜索服务问题，而是请求体里包含了 Gemini 不接受的旧 thinking 参数。

修复后又遇到 `LengthFinishReasonError`，原因是 `gemini-2.5-flash` 默认 thinking
会消耗输出预算，结构化解析在 `finish_reason=length` 时拒绝解析。

**当前方案**

- 在 `competitive_analysis_agent/live_planner.py`、
  `live_extractor.py`、`live_analyst.py` 和 `live_verifier.py` 中删除
  `extra_body={"enable_thinking": False}`。
- 在 `competitive_analysis_agent/live_config.py` 增加
  `build_provider_request_options()`，只在 Gemini OpenAI endpoint 且模型为非 Pro
  的 `gemini-2.5...` 时注入 `reasoning_effort="none"`。
- 其他 OpenAI 兼容供应商不接收 Gemini 专用参数，避免修复一个供应商时破坏另一个
  供应商。

**为什么这样选择**

Google Gemini OpenAI 兼容文档当前要求使用
`https://generativelanguage.googleapis.com/v1beta/openai/` 作为 OpenAI 兼容入口，
并说明 Gemini 2.5 thinking 应通过 OpenAI 兼容字段 `reasoning_effort` 控制。
因此项目不再使用旧的 `enable_thinking`，而是把供应商差异集中在
`live_config.py` 的小函数里。

**替代方案**

1. 直接移除所有 thinking 控制参数；
2. 给所有供应商都传 `reasoning_effort="none"`；
3. 改用 Google GenAI SDK 而不是 OpenAI 兼容接口。

当前选择比方案 1 更稳定，因为 Gemini 2.5 Flash 的 hidden thinking 可能挤占结构化
输出预算；比方案 2 更保守，因为不同 OpenAI 兼容供应商未必接受
`reasoning_effort`；比方案 3 改动小，因为项目已有 LangChain/OpenAI 兼容模型工厂。

**验证记录**

- `python -m pytest tests\test_planner.py tests\test_live_planner.py tests\test_live_config.py tests\test_live_model_factories.py`
  结果：13 个测试通过。
- `python -m pytest`
  结果：167 个离线测试通过，7 个 live 测试按默认配置跳过。
- 真实 Planner smoke test：使用 `.env` 中配置的 Gemini OpenAI 兼容服务，生成
  4 条任务，字段为 `product_name/query/topic`。
- `python -m pytest -m live_llm tests\test_live_workflow.py`
  结果：1 个固定搜索证据的真实 LangGraph 工作流测试通过，覆盖 Planner、
  Extractor、Analyst、Verifier 和 Reporter。
- 配置来源：`.env`，未记录任何变量值。

**面试回答参考**

这次问题不是业务逻辑错误，而是 OpenAI 兼容接口的供应商参数漂移。我的处理方式是先用
Planner 最小真实调用复现 400，再把不兼容的 `enable_thinking` 移除，并把 Gemini 2.5
需要的 `reasoning_effort="none"` 限定在 Gemini endpoint 和非 Pro 2.5 模型上。这样既
解决当前 Gemini 2.5 Flash 的结构化输出问题，也避免把 Gemini 专用参数传给其他供应商。

## 9. 下一阶段衔接

Stage 4 将消费本阶段输出的 `ResearchTask`，逐条调用搜索适配器。本阶段不实现该行为。
