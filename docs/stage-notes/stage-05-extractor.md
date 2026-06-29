# Stage 05：Extractor

## 1. 本阶段目标

本阶段把 Researcher 收集的 `Evidence` 转换成结构化 `ProductProfile`。

输入：

- 一个或多个产品的 Evidence；
- 每条 Evidence 已包含产品名、主题、摘要和证据 ID。

输出：

- 每个产品一个 `ProductProfile`；
- 每个功能和价格项都引用当前产品真实存在的 Evidence ID。

它位于完整流程中的位置：

```text
Researcher -> Evidence -> Extractor -> ProductProfile -> Analyst
```

本阶段只负责证据约束下的信息提取，不比较产品，不浏览网页，也不生成最终报告。

## 2. 实现结果

- 完成的功能：
  - `ExtractorInput` 校验 Evidence 非空且 ID 唯一；
  - `group_evidence_by_product()` 按产品首次出现顺序分组；
  - `Extractor` 为每个产品生成一个结构化 `ProductProfile`；
  - `EXTRACTOR_SYSTEM_PROMPT` 限制模型只能使用提供的 Evidence；
  - `normalize_extractor_raw_output()` 在 Pydantic 校验前修正
    `pricing.main_limits` 中可安全转换的对象列表；
  - `validate_extractor_output()` 校验产品名和 Evidence ID；
  - 首次结构或引用错误后允许一次修复；
  - `LangChainExtractorModel` 使用 JSON mode 并保留 raw 输出；
  - `live_extractor.py` 从指定环境文件加载配置并运行真实验收。
- 关键文件：
  - `competitive_analysis_agent/extractor.py`
  - `competitive_analysis_agent/live_extractor.py`
  - `competitive_analysis_agent/live_config.py`
  - `tests/test_extractor.py`
  - `tests/test_live_extractor.py`
  - `tests/fixtures/extractor_outputs.json`
- 核心数据流：

```text
Evidence
  -> 按 product_name 分组
  -> 单产品结构化模型调用
  -> ProductProfile Schema 校验
  -> product_name / evidence_ids 确定性校验
  -> 有效 ProductProfile
```

- 验证方式与结果：
  - `python -m pytest -q`：87 个离线测试通过，7 个真实用例默认排除；
  - `python -m pytest tests/test_extractor.py -q`：Extractor 8 个离线测试通过；
  - `python -m compileall -q competitive_analysis_agent tests`：通过。
- 真实 LLM 测试：
  - 硅基流动 OpenAI 兼容模型成功生成 Atlas Notes 和 Beacon Docs 两个画像；
  - 输出通过 Pydantic、产品名、Evidence ID 和缺失字段契约检查。
- 暂未实现：
  - 不比较产品；
  - 不访问搜索服务；
  - 不判断不同证据之间的冲突；
  - 不做 LangGraph 编排；
  - 不提供产品级部分成功返回，任一产品修复失败会明确终止本次提取。

## 3. 设计决策

### 决策 1：为什么每个产品单独调用一次模型？

**问题背景**

一次研究结果可能包含多个产品。如果把全部证据放进一次调用，模型容易混淆产品和证据引用，而且单个产品的错误会导致整个结果失效。

**当前方案**

`competitive_analysis_agent/extractor.py` 中的
`group_evidence_by_product()` 先按 `product_name` 分组，
`Extractor.extract()` 再按分组顺序逐个调用模型。

**替代方案**

一次模型调用同时生成全部产品画像。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 按产品调用 | 上下文小、引用范围清晰、错误容易定位 | 模型调用次数随产品数增加 |
| 一次生成全部画像 | 调用次数少，模型可同时看到全部产品 | 容易串数据，单次输出更长且更难修复 |

**什么时候考虑切换**

当产品数量很少、模型支持稳定的长结构化输出，且批量调用成本明显更低时，可以实测一次生成全部画像。

**面试回答参考**

我先按产品分组再提取，因为 ProductProfile 的事实只能引用该产品证据。这样提示词更短，引用校验也更直接。代价是调用次数增加，但当前 MVP 产品数量少，可靠性比吞吐量更重要。

### 决策 2：为什么模型只允许使用给定 Evidence？

**问题背景**

模型可能凭训练记忆补充看似合理的产品信息，但这些内容无法追溯到本次搜索来源。

**当前方案**

`EXTRACTOR_SYSTEM_PROMPT` 明确禁止外部知识，并规定缺失值使用 `null`
或空列表。`validate_extractor_output()` 再检查功能和价格引用的 Evidence ID。

**替代方案**

允许模型使用常识或训练知识补齐缺失字段。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 仅使用 Evidence | 可追溯、可验证，缺失信息边界清晰 | 输出可能不完整 |
| 允许模型补全 | 输出更丰富 | 无法证明来源，容易产生幻觉或过期信息 |

**什么时候考虑切换**

竞品分析中的事实不应切换为无来源补全。如果需要模型提出假设，应使用独立字段明确标记为“推测”，不能混入事实画像。

**面试回答参考**

Extractor 的目标不是写得丰富，而是把证据转换成可验证事实。模型只能使用传入 Evidence，找不到的信息留空。这样后续 Analyst 和 Verifier 能区分“没有资料”和“模型猜测”。

### 决策 3：为什么 Pydantic 校验后还要检查 Evidence ID？

**问题背景**

Pydantic 能检查 `evidence_ids` 是字符串列表，但不能自动知道 `E99` 是否真的存在，或者是否属于另一个产品。

**当前方案**

`ExtractorOutput.model_validate()` 校验输出形状；
`validate_extractor_output()` 检查产品名；
`collect_profile_evidence_ids()` 收集引用，并与当前产品的允许 ID 集合比较。

**替代方案**

只依赖提示词要求模型正确引用。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Schema + 引用校验 | 能确定性拦截虚构和跨产品引用 | 多一层业务校验代码 |
| 只依赖提示词 | 实现代码少 | 错误引用可能静默进入后续流程 |

**什么时候考虑切换**

不建议移除引用校验。未来即使使用数据库或向量库，也应把引用完整性作为程序规则保留。

**面试回答参考**

结构合法不等于业务合法。Pydantic 只能确认字段类型，代码还要检查每个 Evidence ID 是否来自当前产品输入。这种确定性约束不应交给模型判断。

### 决策 4：为什么只允许一次修复？

**问题背景**

真实模型可能返回格式错误、错误产品名或不存在的证据 ID，需要一定恢复能力，但无限重试会导致成本和延迟失控。

**当前方案**

`Extractor.extract()` 首次捕获 `ExtractorValidationError` 后调用
`build_repair_messages()`，把具体错误反馈给模型。第二次仍失败时抛出
`ExtractorError`。

**替代方案**

首次失败立即终止，或者持续重试直到成功。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 最多修复一次 | 能恢复常见错误，成本有上限 | 第二次失败仍会终止 |
| 立即失败 | 最简单、成本最低 | 对偶发格式问题不够宽容 |
| 多次或无限重试 | 成功概率可能提高 | 延迟、成本和循环风险不可控 |

**什么时候考虑切换**

如果线上有统一的重试预算、异步队列和失败统计，可以改成有限次数退避重试，但必须保留明确上限。

**面试回答参考**

我允许一次有错误反馈的修复，处理常见格式和引用问题，同时把调用上限固定为两次。这样比首次失败立即终止更稳健，也避免无限重试带来的成本和延迟。

## 4. 异常与边界情况

- Evidence 为空：`ExtractorInput` 在模型调用前拒绝。
- Evidence ID 重复：输入校验失败，因为一个 ID 无法唯一定位证据。
- 模型缺少必填字段：进入一次修复。
- 模型返回错误产品名：进入一次修复。
- 模型引用不存在或属于其他产品的 ID：进入一次修复。
- 模型把 `pricing.main_limits` 写成 `{name, description}` 对象：在校验前压平成
  字符串列表，保留原有事实含义，不放宽产品名或 Evidence ID 校验。
- 第二次输出仍无效：抛出 `ExtractorError`，不继续无限调用。
- 模型供应商调用异常：转换成 `ExtractorError`。
- 证据没有提供价格：允许价格方案存在，但 `price` 和
  `billing_cycle` 必须保持 `None`。
- 证据没有提供定位、目标用户、优势或限制：提示词要求对应字段保持空值。
- 当前 Schema 只为 feature 和 pricing 保存 Evidence ID，因此其他字段是否
  受到语义支持主要依赖提示词；Stage 7 Verifier 会进一步检查事实支持情况。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：`python -m pytest -o addopts= -m live_llm tests/test_live_extractor.py -q`
- 真实调用的组件：`create_live_extractor()` 创建的 `Extractor`
- 验证的输出契约：
  - 返回 Atlas Notes、Beacon Docs 两个 `ProductProfile`；
  - 输出通过 Pydantic；
  - 每个引用只来自当前产品输入的 Evidence ID；
  - 未提供的定位、目标用户、优势和限制保持为空；
  - Beacon Docs 未提供的价格和计费周期保持为 `None`。
- 测试结果：通过，`1 passed`
- 耗时：33.37 秒
- 失败原因：无

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2a-agent-tools.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1a-from-prompt-to-action.zh-CN.ipynb`

### Relevant sections

- Day 2：`第 3 节：用代码提升 Agent 可靠性`
- Day 2：`3.1：内置代码执行器`
- Day 2：`3.2：更新 Agent 的 instruction 和 toolset`
- Day 1：`2.4 它是如何工作的？`

### Core concept

模型擅长理解自然语言和归纳语义，但不应承担所有确定性规则。Notebook
把算术交给代码执行器，是因为代码计算比模型心算更可靠。本项目同样把
Evidence ID 集合校验交给普通 Python，只让模型负责把摘要整理成字段。

instruction 还负责限制 Agent 的行为边界。Extractor 的 instruction 明确规定
只能使用给定 Evidence、不得补充常识、缺失信息必须留空。

### How it appears in this project

- `EXTRACTOR_SYSTEM_PROMPT` 定义模型职责和禁止事项；
- `ExtractorOutput` 使用 Pydantic 约束结构；
- `validate_extractor_output()` 用确定性代码检查产品归属和引用完整性；
- `build_repair_messages()` 把可操作的错误反馈给模型修复一次。

### ADK vs LangGraph

- Notebook 使用 ADK `LlmAgent`、`BuiltInCodeExecutor` 和 `AgentTool`；
- 本项目当前使用 LangChain `with_structured_output()` 调用模型；
- 本阶段还没有 LangGraph，Extractor 仍是可独立测试的普通 Python 类；
- 本项目不需要执行模型生成的 Python，因为 Evidence ID 校验直接写成普通函数
  更简单、更安全。

### Intentionally postponed

- LangGraph 节点与共享 State；
- 多产品并发提取；
- 证据冲突检测；
- Analyst 横向比较；
- Verifier 对定位、优势等自由文本字段做进一步支持性检查。

## 7. 理解问题与参考思路

### 问题 1：grounding 与普通生成有什么不同？

**参考思路：**

- 普通生成允许模型根据训练知识组织答案；
- grounding 要求每个事实只能来自当前提供的 Evidence；
- 缺少资料时应明确留空，而不是追求完整叙述。

### 问题 2：功能和价格如何链接回原始来源？

**参考思路：**

- `FeatureItem` 和 `PricingPlan` 保存 `evidence_ids`；
- Extractor 只允许引用当前产品 Evidence 集合中的 ID；
- 后续报告可通过 ID 找回 Evidence 的标题、URL 和摘要。

### 问题 3：来源没有价格时会发生什么？

**参考思路：**

- 如果连方案名也没有，`pricing` 可以是空列表；
- 如果有方案名但没有价格，可以保留方案；
- `price` 和 `billing_cycle` 必须是 `None`，不能猜测。

### 问题 4：为什么既需要真实模型测试，也需要固定输出测试？

**参考思路：**

- 固定输出测试快速检查校验、修复和错误分支；
- 真实测试验证供应商接口、提示词和 JSON mode 是否真的兼容；
- 两者解决的问题不同，不能互相替代。

## 8. 面试追问清单

- 如果 Evidence 数量扩大十倍，当前方案哪里最先需要调整？
- 如果移除引用校验，会造成什么风险？
- 当前错误处理是快速失败还是降级处理？为什么？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 6：Analyst 将消费 `ProductProfile` 做横向比较。本阶段不实现该行为。

## 10. 变更记录

- Stage 6 引入第二个真实模型入口后，环境文件加载逻辑从
  `live_extractor.py` 移到共享的 `live_config.py`。
- Stage 10 真实 UI 路径发现模型连续省略 `FeatureItem.description`；
  `EXTRACTOR_SYSTEM_PROMPT` 和 `build_repair_messages()` 现已明确列出
  feature 与 pricing 的全部嵌套字段。
- Pydantic 输出契约没有放宽，模型必须补齐必填描述字段。
- 2026-06-22 真实 Notion / Confluence 搜索路径发现 Confluence 价格限制会被模型写成
  `main_limits` 对象列表，例如 `{name, description}`。代码新增
  `normalize_extractor_raw_output()`、`normalize_pricing_items()` 和
  `normalize_main_limits()`，只把这类限制对象压平成字符串，再进入原有 Pydantic、
  产品名和 Evidence ID 校验。回归测试：
  `python -m pytest tests/test_extractor.py -q`，结果 `8 passed`。
- 同日复跑真实 UI 输入：
  `Notion=notion.com,notion.so`、`Confluence=www.atlassian.com`。工作流已通过
  `planner -> researcher -> extractor -> analyst -> verifier -> analyst -> verifier -> reporter`，
  生成 15 条 Evidence、0 个搜索错误；最终 Verifier 仍保留一个未支持结论警告，因此
  报告是待复核草稿，但不再因 ExtractorError 中断。
- 2026-06-22 后续报告仍显示定位和目标用户大量缺失。`EXTRACTOR_SYSTEM_PROMPT`
  现已明确：官网标题、产品标语、产品概览中的产品类别可以进入 `positioning`；
  use case、customer、team、enterprise、small business 页面中直接写出的用户群可以进入
  `target_users`，但不能从功能名称反推用户。
- 同日新增 `normalize_profile_summary_fields()`：当模型漏掉定位时，
  `fill_missing_positioning_from_evidence()` 会从 `topic=positioning` Evidence 的官网标题或
  首句中保守摘取短句，例如 “Confluence is a team workspace where knowledge and
  collaboration meet.”。这仍然只使用 Evidence，不引入外部知识。
- 同日新增 `normalize_pricing_defaults()`：当计划名明确为 `Free` 时补齐 `$0`；当价格文本
  包含 `/month`、`per seat/month`、`monthly` 等标记时补齐 `billing_cycle="monthly"`。
  不会为 Enterprise 或没有价格文本的方案猜价格。回归测试：
  `python -m pytest tests/test_extractor.py -q`，结果 `11 passed`。
- 2026-06-22 最新报告显示 pricing 页面已搜到，但搜索摘要没有稳定携带完整价格表。
  `EXTRACTOR_SYSTEM_PROMPT` 现已明确：如果 Evidence 包含 `raw_content`，它是 Researcher
  从价格页正文裁剪出的价格页正文片段；pricing 提取可以同时使用 `snippet` 和
  `raw_content` 中明确出现的 plan、price、billing cycle 和限制，但仍不能使用 Evidence
  外的信息。
- 2026-06-22 最新报告暴露出免费方案口径问题：Evidence 写的是 `Free forever for
  10 users`，模型可能把邻近的 “per month” 限制误保留为 `billing_cycle="monthly"`。
  `normalize_pricing_defaults()` 现在会在价格明确为 `Free` 或 `$0` 时清空计费周期，
  避免后续生成 “Free with monthly billing”。新增回归：
  `test_free_price_clears_model_supplied_billing_cycle`。
- 2026-06-23 最新通过验证的报告继续暴露价格字段噪声：`$0 per seat/month`、
  `Free to try, then $10 per 1,000 monthly Notion credits`、`Beta` 和
  `Custom pricing` 会被混入 `price/billing_cycle`。本次把价格清洗集中到
  `competitive_analysis_agent/pricing_utils.py`，并在 `normalize_pricing_defaults()`
  中应用：`$0 per seat/month` 收窄成 `$0`，`Beta` 不再保留为计费周期，
  `Custom pricing` 不再要求 billing cycle，普通 `monthly credits` 不再被误判为月付。
  新增回归：`test_pricing_normalization_removes_status_and_duplicates`。
- 2026-06-23 对比 ChatGPT、Claude、Gemini 且选择 8 个维度时，Researcher 收集到
  57 条 Evidence，Extractor 在 `_invoke_model` 阶段因模型连接/输入压力失败。
  本次在 `build_extractor_messages()` 前新增 Evidence 压缩：只发送
  `evidence_id`、`product_name`、`topic`、`title`、`url`、`snippet`、
  `source_type` 和必要的 `raw_content`，不再发送 `collected_at` 等提取画像不需要的字段；
  同时裁剪过长标题、摘要和价格正文片段。
- `ExtractorError` 现在支持 `public_detail`，模型调用失败时会显示产品名、证据条数、
  模型输入字符数和底层异常类型，但不显示供应商原始异常文本、prompt 或密钥。
  新增回归：
  `test_extractor_messages_compact_long_evidence_text`、
  `test_model_call_failure_exposes_safe_public_detail`。
- 离线验证：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_extractor.py tests/test_ui_service.py tests/test_workflow.py -q`
  结果 `33 passed`；完整离线测试
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`
  结果 `139 passed, 7 deselected`。
- 真实 Extractor 验收：
  首次调用成功返回了证据支持的 `positioning`，因此同步更新 live test 断言为
  “如果有定位，必须能在该产品 Evidence 文本中找到”；随后两次重跑
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests/test_live_extractor.py -q`
  均在 HTTPS 握手阶段失败，错误类型为 `SSL: UNEXPECTED_EOF_WHILE_READING` /
  `APIConnectionError`。沙箱外重跑仍相同，因此本次真实验收状态为
  `外部网络或供应商连接失败，代码路径未通过 live 验收`。
- 2026-06-23 后续用户复跑显示新的安全错误详情：
  `产品：ChatGPT；证据条数：11；模型输入约 21563 个字符；底层异常类型：APITimeoutError`。
  说明只裁剪单条 Evidence 文本仍不够，单产品 Evidence 数量也必须受控。
  本次新增 `select_evidence_for_extraction()`：按 topic 轮询选择代表证据，优先保留
  `pricing`、`features`、`positioning`、`target_users`，每个 topic 最多 2 条，
  每个产品最多 6 条。这样 4 个维度、每个任务 3 条搜索结果时，Extractor 不会把
  单产品 11 条证据全部塞给模型。
- 同时进一步收紧 Extractor 输入长度：标题最多 160 字符，摘要最多 700 字符，
  价格正文片段最多 1400 字符。`ExtractorError.public_detail` 现在区分
  `原始证据条数` 和 `送入模型证据条数`，方便判断是搜索阶段资料太多，还是模型服务本身异常。
- 新增回归：
  `test_select_evidence_round_robins_topics_before_model_call`，
  并更新模型失败详情测试。验证命令：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_extractor.py -q`
  结果 `16 passed`；
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_extractor.py tests/test_ui_service.py tests/test_workflow.py -q`
  结果 `34 passed`；
  完整离线测试 `140 passed, 7 deselected`。
- 真实 Extractor 验收再次执行：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests/test_live_extractor.py -q`。
  沙箱内和沙箱外均失败在 HTTPS 握手，错误仍为
  `SSL: UNEXPECTED_EOF_WHILE_READING` / `APIConnectionError`。该失败发生在固定
  2 条短 Evidence 的 smoke test 上，因此归类为外部网络或供应商连接失败，不是本次
  Evidence 选择逻辑的离线回归失败。
- 2026-06-24 最新 ChatGPT / Claude / Gemini 报告显示，Extractor 仍会把套餐级页面内容
  提升成产品级画像：Claude 的 Max 计划适用人群被写成 Claude 的整体定位，Gemini 的
  Evidence 中混入 Google Home Premium 价格，进而让 Analyst 输出冲突价格。根因不是
  Verifier 太严格，而是上游画像缺少产品范围过滤。
- 本次在 `normalize_profile_summary_fields()` 后处理链路中新增三道确定性收口：
  `remove_plan_level_positioning()` 会删除来自价格/订阅语境的套餐级定位；
  `filter_pricing_plans_by_product_scope()` 会删除明显属于其他产品线的价格项，例如
  Gemini 画像中的 Google Home Premium；`remove_conflicting_pricing_duplicates()`
  会删除同一套餐、同一计费周期下出现多个价格的冲突项。这样宁可少展示，也不把可疑价格
  写进后续比较。
- 同步收紧 `EXTRACTOR_SYSTEM_PROMPT`：要求不要把价格页里的套餐适用人群、额度说明或
  订阅说明当作 `positioning`；如果 Evidence 页明显属于另一个产品线，不要放入当前产品
  `pricing`。提示词减少模型误抽取，确定性后处理负责兜住真实输出。
- 新增回归：
  `test_plan_level_positioning_from_pricing_page_is_removed`、
  `test_gemini_home_premium_pricing_is_removed`、
  `test_conflicting_duplicate_pricing_plan_is_removed`。聚焦验证：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_extractor.py tests\test_analyst.py -q`
  结果 `43 passed`。
- 完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果
  `147 passed, 7 deselected`。真实 Extractor/Analyst smoke：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests\test_live_extractor.py tests\test_live_analyst.py -q`，
  结果 `2 passed`，耗时约 62.19 秒；配置来源仍为 `.env.example`，未记录任何变量值。
- 2026-06-24 根据 `docs/agent-repair-guide-api-pricing-scope.md` 继续收紧默认模型产品的
  API pricing 范围。前一次只排除了 Gemini Home Premium 这类明显旁支产品，但
  `ChatGPT`、`Claude`、`Gemini` 默认分析对象已经被用户定义为 API 价格：
  ChatGPT 对应 OpenAI / ChatGPT API，Claude 对应 Anthropic Claude API，
  Gemini 对应 Gemini API / Google AI API。因此官方域名和产品名本身不再足以证明
  pricing 属于目标范围。
- Researcher 侧新增 `build_pricing_search_terms()` 和
  `API_PRICING_QUERY_TERMS_BY_PRODUCT`：当 topic 是 `pricing` 且产品是
  `ChatGPT`、`OpenAI`、`Claude`、`Anthropic`、`Gemini` 或 `Google AI` 时，
  查询词改为 `API pricing`、`developer platform`、`console`、`ai.google.dev`、
  `token`、`model pricing`、`input/output` 等 API 语境；普通产品继续沿用原来的
  pricing 查询词。这样先降低搜索阶段混入 Plus/Pro/Team/Workspace 等订阅页的概率，
  但不把语义解释放进 Researcher。
- Extractor 侧新增 `build_api_pricing_scope_rules()`、
  `classify_pricing_source_scope()`、`pricing_plan_matches_requested_scope()` 和
  `build_pricing_plan_scope_text()`。后处理会同时读取 `plan_name`、`price`、
  `billing_cycle`、`main_limits` 以及证据的 `title/url/snippet/raw_content`：
  只有明确命中 API/token/model pricing 语境的价格项才保留；明显命中订阅或旁支产品
  语境的价格项删除；同时命中 API 和非 API 标记、或者无法判断的价格项也删除。这个选择
  会牺牲一些召回率，但能防止订阅价格污染后续 Analyst 和 Reporter。
- 同步更新 `EXTRACTOR_SYSTEM_PROMPT`，明确默认范围：ChatGPT 只提取
  OpenAI / ChatGPT API 价格；Claude 只提取 Anthropic Claude API 价格；Gemini
  只提取 Gemini API / Google AI API 价格。ChatGPT Plus/Pro/Team/Business、
  Claude Pro/Max/Team、Gemini App、Workspace Gemini、Google Home Premium、Veo
  等订阅或非 API 产品价格不得进入 `pricing`。提示词用于降低模型误抽取，确定性 helper
  负责最终兜底。
- 新增回归测试：
  `test_default_model_product_pricing_query_targets_api_scope`、
  `test_chatgpt_subscription_pricing_is_removed_by_default_api_scope`、
  `test_claude_subscription_pricing_is_removed_by_default_api_scope`、
  `test_gemini_workspace_veo_and_home_pricing_are_removed`。这些用例覆盖：
  ChatGPT 官方页面中的 Plus/Team 订阅价被删除；Claude Max/Pro 订阅价和套餐级定位
  不进入画像；Gemini Workspace、Google Home Premium、Veo 价格被删除；Gemini API
  token price 继续保留。
- 聚焦验证：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py -q`
  结果 `79 passed`。完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果
  `151 passed, 7 deselected`。
- 真实 Extractor smoke：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests\test_live_extractor.py -q`
  结果 `1 passed`，耗时约 22.50 秒；配置来源仍为 `.env.example`，未记录任何变量值。
  本次同步收窄 live test 契约：Beacon Docs 只写出 Business 方案但没有价格时，模型可以
  不生成 pricing；如果生成，`price` 和 `billing_cycle` 必须保持 `None`。这更符合
  “找不到价格不要补位”的项目规则。
- 本次 notebook 复查：
  `F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2a-agent-tools.zh-CN.ipynb`
  的 `第 3 节：用代码提升 Agent 可靠性`、`3.1：内置代码执行器`、`3.2：更新 Agent 的 instruction 和 toolset`
  强调：模型可以根据 instruction 生成或调用工具，但稳定规则应交给代码执行。
  `F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1a-from-prompt-to-action.zh-CN.ipynb`
  的 `2.4 它是如何工作的？` 强调 instruction 告诉 Agent 何时使用工具。本项目对应关系是：
  prompt 定义 Extractor 的行为边界，`classify_pricing_source_scope()` 则把 API pricing
  scope 变成可测试、可复跑的普通 Python 规则，而不是只靠模型“记住不要混淆产品线”。
- 面试回答补充：我把 API pricing scope 拆成两层：Researcher 用产品相关查询词优先找到
  API/token 资料，Extractor 再用确定性规则过滤模型输出。这样即使官方域名里混进订阅页，
  污染也会在 ProductProfile 前被挡住。当前方案用关键词规则是因为 MVP 数据量小、需要
  可解释和易测试；如果未来产品线更多、召回率要求更高，可以把 scope 做成显式 Schema
  或专门的 profile validation 结果，并让 Reporter 展示“因范围不符被排除”的资料限制。
