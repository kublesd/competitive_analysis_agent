# Stage 06：Analyst

## 1. 本阶段目标

本阶段把多个 `ProductProfile` 转换成结构化竞品比较。

输入：

- 至少两个已经完成提取的 `ProductProfile`；
- 功能和价格项已经携带 Evidence ID。

输出：

- 定位比较；
- 功能比较；
- 价格比较；
- 机会点；
- 总结结论。

它位于完整流程中的位置：

```text
Extractor -> ProductProfile -> Analyst -> CompetitiveAnalysis -> Verifier
```

本阶段只根据画像进行比较，不搜索网页、不重新提取事实，也不验证最终报告。

## 2. 实现结果

- 完成的功能：
  - `AnalysisClaim` 使用 `claim_type` 区分事实和解释；
  - `CompetitiveAnalysis` 保存定位、功能、价格、机会点和总结；
  - `AnalystInput` 拒绝重复产品和跨产品重复 Evidence ID；
  - `Analyst` 使用结构化模型比较全部产品画像；
  - `validate_analyst_output()` 检查产品顺序、实际覆盖和结论范围；
  - `validate_claim_references()` 检查引用存在性、产品归属和逐产品支持；
  - `features` 章节中的价格、套餐、计费周期、用户数和存储限制语言会被确定性拒绝；
  - 单产品 feature fact 会被收窄成 `ProductProfile.features[].name` 对应的短句，避免营销扩写进入报告；
  - 当 Verifier 已点名 `conclusion` 不受支持时，下一轮结论会退回到“仅限已提供证据”的保守说明；
  - 格式、覆盖或引用错误允许一次修复；
  - Analyst 模型调用失败、修复调用失败，或修复后仍无效时，使用 `ProductProfile`
    生成保守 fallback 分析，保住已经完成的调研和提取结果；
  - `live_config.py` 为 Extractor 和 Analyst 真实入口统一加载敏感配置。
- 关键文件：
  - `competitive_analysis_agent/analyst.py`
  - `competitive_analysis_agent/live_analyst.py`
  - `competitive_analysis_agent/live_config.py`
  - `tests/test_analyst.py`
  - `tests/test_live_analyst.py`
  - `tests/fixtures/analyst_outputs.json`
- 核心数据流：

```text
ProductProfile 列表
  -> 一次结构化 Analyst 调用
  -> CompetitiveAnalysis 或 ProductProfile fallback
  -> 产品覆盖校验
  -> fact / interpretation 类型校验
  -> Evidence ID 存在性和产品归属校验
```

- 验证方式与结果：
  - `python -m pytest tests/test_analyst.py -q`：16 个测试通过；
  - `python -m pytest -q`：109 个离线测试通过，7 个真实用例默认排除；
  - `python -m compileall -q competitive_analysis_agent tests`：通过。
- 真实 LLM 测试：
  - 硅基流动模型成功返回结构化比较；
  - 产品覆盖、事实引用、机会点类型和结论范围全部通过。
- 暂未实现：
  - 不浏览网页；
  - 不读取原始 Evidence 摘要；
  - 不验证 claim 文本是否与引用语义完全一致；
  - 不进行 LangGraph 编排；
  - 不实现 Verifier 或报告渲染。

## 3. 设计决策

### 决策 1：为什么 Analyst 只消费 ProductProfile，不直接浏览网页？

**问题背景**

如果 Analyst 在比较过程中重新搜索，新的事实将绕过 Researcher 的证据编号和
Extractor 的结构化提取，难以判断结论来自哪个阶段。

**当前方案**

`competitive_analysis_agent/analyst.py` 中的 `AnalystInput` 只接收
`ProductProfile`。`ANALYST_SYSTEM_PROMPT` 明确禁止搜索和外部知识，
`Analyst` 本身也没有 SearchAdapter 依赖。

**替代方案**

让 Analyst 在发现资料不足时自行搜索并补充结论。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 只消费 ProductProfile | 数据边界清晰、可测试、来源可追踪 | 资料不足时只能明确保留限制 |
| Analyst 自主搜索 | 可能补充更多信息 | 搜索、提取和分析耦合，引用链难验证 |

**什么时候考虑切换**

不应让 Analyst 隐式搜索。如果未来需要补充资料，应由 Verifier 或图工作流明确
路由回 Planner/Researcher，并把新资料重新走 Evidence 和 Extractor 流程。

**面试回答参考**

我的 Analyst 不浏览网页，只消费 Extractor 生成的 ProductProfile。这样搜索、
事实提取和横向判断各自独立，结论能沿 Evidence ID 回溯来源。资料不足时由后续
工作流决定是否重新研究，而不是让 Analyst 绕过证据链。

### 决策 2：如何区分事实与解释？

**问题背景**

“Atlas Team 方案每月 12 美元”是事实，“Atlas 的价格透明度更高”是解释。
如果两者混在普通字符串中，后续 Verifier 无法采用不同规则检查。

**当前方案**

`AnalysisClaim` 使用 `claim` 保存陈述，并用 `claim_type` 标记
`fact` 或 `interpretation`。真实测试发现模型稳定使用 `claim` 而不是原始设计的
`text`，因此最终契约采用更贴近业务语义的 `claim` 字段。

**替代方案**

每个比较维度只返回一段自然语言。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 结构化 claim | 可分别校验事实和判断，便于报告渲染 | 输出结构更复杂 |
| 自然语言段落 | 阅读直接、字段少 | 事实和判断混杂，引用难精确绑定 |

**什么时候考虑切换**

最终面向用户的报告可以渲染成自然语言，但内部分析状态仍应保留结构化 claim。

**面试回答参考**

我把每条分析标记为 fact 或 interpretation。事实必须携带证据，解释可以基于多个
事实形成判断。这样 Verifier 可以严格检查事实引用，同时允许机会点和结论保留
合理的分析空间。

### 决策 3：为什么事实不仅要有 Evidence ID，还要校验证据归属？

**问题背景**

仅检查 Evidence ID 存在仍不够。模型可能用 Atlas 的 `E1` 支持一个只谈
Beacon Docs 的事实。

**当前方案**

`validate_claim_references()` 先计算 claim 中产品允许使用的 Evidence ID，
再检查引用范围。对于 `fact`，它还逐个检查涉及的产品是否都有自己的证据支持。

**替代方案**

只检查引用是否存在于任意 ProductProfile。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 检查存在性和产品归属 | 能拦截跨产品错引，比较事实更可靠 | 校验逻辑稍多 |
| 只检查全局存在 | 实现简单 | 无法发现引用了错误产品的真实 ID |

**什么时候考虑切换**

不建议降低校验强度。未来如果 Evidence 改为多产品共享来源，需要把归属模型升级
为多对多关系，而不是移除检查。

**面试回答参考**

有效 ID 不代表引用正确。我除了检查 ID 是否存在，还检查它是否属于 claim 中的
产品。多产品事实必须让每个产品都有自己的证据支持，这能确定性拦截跨产品错引。

### 决策 4：为什么机会点和结论必须标记为 interpretation？

**问题背景**

机会点和总体结论通常由多个事实归纳而来，不是网页直接陈述的原始事实。

**当前方案**

`CompetitiveAnalysis.validate_analysis_types()` 要求 opportunities 和
conclusion 都使用 `interpretation`。`validate_analyst_output()` 还要求结论包含
全部输入产品。

**替代方案**

允许模型把机会点标记为 fact。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 强制 interpretation | 避免把策略判断伪装成事实 | 某些直接来源中的市场结论仍需改写为解释 |
| 类型由模型自由决定 | 灵活 | 容易混淆来源事实与分析判断 |

**什么时候考虑切换**

如果未来增加独立的市场研究事实类型，可以扩展 claim taxonomy；当前 MVP 使用
fact/interpretation 两类最容易理解和验证。

**面试回答参考**

机会点和结论是分析产物，不是来源原文，所以我强制标记为 interpretation。
它们仍可携带相关 Evidence ID，表示判断依据，但不会被误认为来源直接陈述的事实。

### 决策 5：为什么仍然只修复一次？

**问题背景**

结构化分析可能漏产品、产生错误引用或返回格式错误，需要恢复能力，但模型调用
不能无限循环。

**当前方案**

`Analyst.analyze()` 首次捕获 `AnalystValidationError` 后，把具体错误加入修复
消息。第二次仍失败时不继续消耗模型调用，而是进入
`build_fallback_analysis()`：根据已提取的 `ProductProfile` 生成最短 feature facts、
pricing facts、空 opportunities 和范围型 conclusion。

**替代方案**

首次失败立即终止，或者无限重试。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 最多修复一次后 fallback | 能处理常见漏项和错引，成本有上限；模型不稳定时仍能产出保守报告 | fallback 只保留低风险事实，分析深度有限 |
| 立即失败 | 最简单、最低成本 | 对偶发模型错误不够宽容 |
| 多次或无限重试 | 成功概率可能提高 | 延迟和成本不可控，可能形成循环 |

**什么时候考虑切换**

进入异步生产任务后，可以结合预算、退避和失败统计增加有限重试，但必须设置上限。

**面试回答参考**

Analyst 仍然只允许一次修复，但它和前面阶段不同：如果模型服务不可用，或修复后
结构仍然不可信，就用 ProductProfile 生成保守 fallback。这样不会发明新结论，
但可以把已经收集到的功能、价格和来源继续交给 Verifier 和 Reporter。

## 4. 异常与边界情况

- 少于两个 ProductProfile：`AnalystInput` 在模型调用前拒绝。
- 产品名重复：无法形成明确比较，输入校验失败。
- 同一 Evidence ID 同时归属于不同产品：输入校验失败。
- 模型漏掉或新增产品：进入一次修复。
- 产品只出现在 `products` 字段、没有进入实际 claim：进入一次修复。
- `fact` 没有 Evidence ID：Pydantic 校验失败并进入修复。
- 引用不存在，或引用不属于 claim 中产品：进入一次修复。
- 多产品事实缺少某个产品自己的证据：进入一次修复。
- `features` 章节混入价格、套餐、计费周期、用户数或存储限制：进入一次修复。
- 单产品 feature fact 使用“perfectly written”“thinks like you”等证据未直接支持的扩写：
  输出前收窄为画像中的功能名。
- Verifier 反馈 `conclusion` 为 unsupported claim：下一轮输出前收窄为输入范围说明。
- opportunities 或 conclusion 被标记为 fact：Schema 校验失败。
- conclusion 没有包含全部产品：进入一次修复。
- 第二次仍无效：进入 `build_fallback_analysis()`，不会无限调用。
- 模型供应商异常：优先进入 `build_fallback_analysis()`；若 fallback 自身因输入数据不合法失败，
  才会继续暴露受控异常。
- 当前只能确定性验证引用关系，不能判断 claim 文字是否真正被证据语义支持；
  这是 Stage 7 Verifier 的职责。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：`python -m pytest -o addopts= -m live_llm tests/test_live_analyst.py -q`
- 真实调用的组件：`create_live_analyst()` 创建的 `Analyst`
- 验证的输出契约：
  - products 为 Atlas Notes 和 Beacon Docs；
  - features、pricing 和 opportunities 非空；
  - 所有事实都有 Evidence ID；
  - 引用只能属于 claim 中的产品；
  - opportunities 和 conclusion 为 interpretation；
  - conclusion 包含全部产品。
- 测试结果：最终通过，`1 passed`
- 耗时：27.29 秒
- 失败原因：无

验收过程中的有效发现：

- 第一次调用在 TLS 握手阶段超时，分类为网络/超时失败；
- 下一次成功响应使用 `claim` 作为正文键，而初始 Schema 使用 `text`；
- 根据真实输出和字段语义把契约调整为 `claim`，离线回归后真实测试通过；
- 以上记录均未包含 API Key 或请求头。

## 6. 与课程 Notebook 的对应

### Notebook source

`F:\大模型应用开发学习\3.Google_and_Kaggle\1-Day\codelabs\day-1b-agent-architectures.zh-CN.ipynb`

### Relevant sections

- `第 3 节：Sequential 工作流 - 装配线`
- `第 4 节：Parallel 工作流 - 独立研究员`
- `第 6 节：总结 - 选择正确的模式`

### Core concept

Sequential 适用于有明确依赖的流水线：后一步必须消费前一步结果。Parallel 适用于
互不依赖的任务，并在最后由聚合器汇总。工作流模式应由任务依赖决定，而不是为了
使用框架功能而选择。

本项目的 Extractor 和 Analyst 是顺序依赖：没有 ProductProfile，Analyst 就不能
比较。多个产品的搜索和提取理论上可以并行，但最终 Analyst 必须等待全部画像后
统一聚合。

### How it appears in this project

- `Extractor -> Analyst` 对应 Sequential 流水线；
- 多个产品画像相当于多个研究分支的汇总结果；
- `Analyst.analyze()` 扮演聚合器，一次读取全部 `ProductProfile`；
- 当前仍以普通 Python 调用表达依赖，Stage 8 才把节点连接成 LangGraph。

### ADK vs LangGraph

- Notebook 使用 ADK `SequentialAgent`、`ParallelAgent` 和 `output_key`；
- 当前项目使用普通 Python 类和 Pydantic 契约；
- 后续 LangGraph 会使用节点、边和共享 State 表达同样的依赖关系；
- Analyst 内部的 LLM 负责比较内容，但节点执行顺序不会交给 LLM 自由决定。

### Intentionally postponed

- 多产品 Researcher/Extractor 并行执行；
- LangGraph Sequential 边；
- Verifier 回到 Analyst 的受限 Loop；
- 运行 trace、并发调度和状态持久化。

## 7. 理解问题与参考思路

> Stage 8 变更记录：`AnalystInput` 新增默认空的 `revision_feedback`，用于接收
> Verifier issues；首次分析行为保持不变。真实整图验收还收紧了机会点规则：
> 简短证据不足时允许 `opportunities=[]`，不得推断用户需求、市场偏好或优越性。

### 问题 1：为什么 Analyst 不能重新浏览网页？

**参考思路：**

- 新搜索会绕过 Researcher 和 Extractor；
- 新事实没有稳定的 Evidence ID 和画像契约；
- 资料不足应由工作流明确路由回研究阶段。

### 问题 2：事实和解释有什么区别？

**参考思路：**

- fact 是 ProductProfile 直接支持的功能或价格陈述；
- interpretation 是多个事实形成的比较、机会点或结论；
- fact 必须有 Evidence ID，interpretation 必须明确标记。

### 问题 3：为什么只检查 Evidence ID 存在还不够？

**参考思路：**

- 一个真实 ID 也可能属于另一个产品；
- 多产品事实需要每个产品都有自己的证据；
- 因此要同时检查存在性、产品归属和逐产品覆盖。

### 问题 4：本阶段为什么一次调用全部产品，而不是逐产品调用？

**参考思路：**

- Extractor 是单产品任务，可以分开；
- Analyst 是横向比较，必须同时看到多个画像；
- 产品数量变大后可先分组比较，再做二次聚合。

## 8. 面试追问清单

- 如果产品数量扩大十倍，当前单次分析调用会遇到什么问题？
- 如果 Analyst 直接浏览网页，会造成什么耦合？
- 事实和解释分别应该由哪些规则验证？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 7：Verifier 将检查引用有效性和不受支持的 claim。本阶段不实现该行为。

## 10. 变更记录

- 2026-06-22 真实 Notion / Confluence 报告中，Verifier 多次指出 Analyst 把短功能标签
  扩写成证据不直接支持的能力描述。`ANALYST_SYSTEM_PROMPT` 已收紧：fact claim 应贴近
  `ProductProfile` 中的 `feature.name`、`feature.description`、`pricing.plan_name`、
  `price` 和 `main_limits` 原文，不得把短标签扩写成更大的能力。
- 同次变更还要求：收到多个 Verifier 反馈时，下一轮应显著减少 feature claims，只保留
  最直接、最短、证据措辞最接近的事实；不要用另一个推测替换 unsupported claim。
- 离线测试新增 `test_prompt_keeps_fact_claims_close_to_profile_text`。聚焦测试：
  `python -m pytest tests/test_analyst.py tests/test_workflow.py tests/test_verifier.py -q`，
  结果 `25 passed`。完整离线测试：`93 passed, 7 deselected`。
- 真实端到端复跑在本次提示词收紧后完成过一次：生成 19 条 Evidence、0 个搜索错误，
  定位和目标用户已生成，Verifier issue 数下降到 3。后续一次真实复跑在 Analyst 模型调用处
  遇到供应商连接断开，分类为外部模型网络失败，不是结构化输出或校验失败。
- 2026-06-22 最新 Notion / Confluence 报告继续暴露出两个 Analyst 边界问题：
  价格事实被放进 `features`，以及短功能标签被扩写成 “perfectly written”“thinks like you”
  等证据不直接支持的营销 claim。
- 本次修复在 `competitive_analysis_agent/analyst.py` 中新增确定性保护：
  `contains_pricing_language()` 拦截误入功能区的价格语言；
  `normalize_feature_fact_claims()` 把单产品功能事实收窄到画像功能名；
  `normalize_conclusion_after_feedback()` 在 Verifier 点名 conclusion 后生成保守结论。
- 离线回归新增：
  `test_pricing_claim_inside_features_is_repaired_once`、
  `test_feature_fact_is_narrowed_to_profile_feature_name`、
  `test_conclusion_feedback_uses_conservative_conclusion`。
  聚焦测试：`python -m pytest tests/test_analyst.py tests/test_workflow.py tests/test_verifier.py -q`，
  结果 `29 passed`。完整离线测试：`97 passed, 7 deselected`。
- 真实 Analyst 验收命令：
  `python -m pytest -o addopts= -m live_llm tests/test_live_analyst.py -q`，
  结果 `1 passed`，耗时约 21.95 秒；验证了真实模型输出中 facts 有引用，且
  `features` 不含价格类 claim。额外 Notion / Confluence 端到端复跑在 Analyst 调用处遇到
  服务端断连，失败分类为外部模型网络失败，未记录任何密钥或原始响应。
- 2026-06-22 端到端复跑发现 Verifier 会把 `Notion lists AI Meeting Notes` 这类
  规范化短句误判为“证据只 mentions，没有 explicitly lists”。因此单产品 feature fact
  的规范化模板从 `lists` 改为 `mentions`，更贴近搜索摘要中的证据措辞。
- 2026-06-22 真实端到端多次在 Analyst 模型调用处中断，错误类别为 `AnalystError`。
  本次修复新增 `build_fallback_analysis()`：当首次模型调用失败、修复调用失败，或修复后
  输出仍不符合 Analyst 校验时，改用已提取的 `ProductProfile` 生成保守分析。fallback 只生成
  最短 feature facts、pricing facts、空 opportunities 和范围型 conclusion，保留 Evidence ID，
  不判断优劣，不发明市场结论。
- 同次修复还把 fallback pricing claim 收窄为“方案名 + 价格 + 计费周期”。价格限制明细仍在
  产品概览展示，但不放入 pricing claim，避免 Verifier 因长限制文本的细节差异产生大量误报。
  回归测试：`python -m pytest tests/test_analyst.py -q`，结果 `16 passed`；完整离线测试：
  `109 passed, 7 deselected`。
- 2026-06-22 最新报告虽然完整跑到 Reporter，但 `verification_passed=False`。问题集中在
  Confluence 的 `features[15]` 到 `features[29]`：Verifier 已点名这些功能 claim
  缺少直接语义支撑，重试阶段的 fallback 却把 ProductProfile 中的功能项原样写回报告。
  本次修复没有放松 Verifier，而是在 `build_fallback_feature_claims()` 和
  `normalize_analysis_output()` 中消费 `revision_feedback`：只要反馈同时包含
  `features[...]` 与 `unsupported_claim`，且能匹配产品名和功能名，就删除对应 feature
  claim。这样第二轮 Analyst 模型不可用或忽略反馈时，也不会重复输出同一批未验证功能。
- 新增回归：
  `test_fallback_removes_unsupported_feature_feedback`、
  `test_model_output_removes_unsupported_feature_feedback`、
  `test_retry_fallback_removes_verifier_rejected_feature`。
  聚焦测试：`python -m pytest tests/test_analyst.py tests/test_workflow.py -q`，
  结果 `24 passed`。包含 Verifier/UI/Workflow 的聚焦回归：
  `python -m pytest tests/test_workflow.py tests/test_ui_service.py tests/test_verifier.py tests/test_analyst.py -q`，
  结果 `47 passed`。完整离线测试：`112 passed, 7 deselected`。
- 2026-06-22 后续报告仅剩价格误报：fallback 把 `Free` 方案写成
  `Free with monthly billing`，但 Evidence 只支持免费价和用户数限制，不支持月付周期。
  `format_fallback_pricing_claim()` 现在对 `Free` / `$0` 价格省略 billing cycle，只生成
  `Product lists the Free plan at Free.` 这类最小事实。新增回归：
  `test_free_fallback_pricing_claim_omits_billing_cycle`。
- 2026-06-23 最新报告显示 fallback pricing claim 仍会生成重复或不合理周期，
  例如 `$10 per seat/month with per month billing` 与 `Beta billing`。
  `format_fallback_pricing_claim()` 现在复用共享价格规则：价格文本已经包含月付/年付时不再
  追加 billing；`Beta` 等状态词不会进入 claim。新增回归：
  `test_fallback_pricing_claim_omits_redundant_or_invalid_billing`。
- 2026-06-23 最新报告虽然通过验证，但 `positioning=[]`、`opportunities=[]`，
  conclusion 也只有 “limited to the supplied product profiles” 这类范围说明。对这个个人项目
  来说，这种输出过于审计化，读者更需要“有证据约束但能给启发”的轻量分析。
- 本次修复把 `ANALYST_SYSTEM_PROMPT` 从“证据简短就返回空机会点”调整为：
  只要画像中能看到定位、目标用户、功能、价格透明度或公开信息缺口的差异，就生成 1 到 3 条
  谨慎机会点。仍然禁止编造市场规模、胜率和用户偏好。
- `build_fallback_analysis()` 不再只生成 feature/pricing facts 和空 opportunities。
  新增 `build_fallback_positioning_claims()`、`build_fallback_opportunity_claims()`、
  `build_fallback_conclusion()`：模型不可用、修复失败，或模型输出太保守时，也能生成定位分析、
  价格透明度机会点、功能差异机会点和更有信息量的结论。
- `normalize_analysis_output()` 新增 `fill_missing_lightweight_analysis_sections()`。
  即使真实模型返回空定位、空机会点或低信息结论，也会根据 `ProductProfile` 自动补齐。
  这是个人项目的实用性取舍：报告先做到可读可用，再由 Verifier 和引用规则挡住明显胡编。
- 新增/更新回归：
  `test_model_call_failure_uses_fallback_analysis`、
  `test_invalid_output_uses_fallback_after_one_failed_repair`、
  `test_sparse_model_output_gets_lightweight_sections`、
  `test_prompt_keeps_fact_claims_close_to_profile_text`。
  离线聚焦测试：`C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py -q`，
  结果 `21 passed`。相邻流程测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_verifier.py tests/test_workflow.py tests/test_reporter.py tests/test_ui_service.py -q`，
  结果 `46 passed`。完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果 `131 passed, 7 deselected`。
- 本次修改提高了离线评估的字段覆盖率，`average_field_coverage` 从旧基准 `0.8`
  提升到 `14 / 15`，因此同步更新 `tests/test_evaluation.py` 的指标预期。
- 真实 Analyst 验收命令：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_live_analyst.py -q -m live_llm`。
  结果 `1 passed`，耗时约 32.99 秒；验证真实模型路径返回结构化比较，且最终分析包含
  `positioning`、`features`、`pricing`、`opportunities`，事实 claim 保留 Evidence ID，
  `features` 不混入价格类 claim。配置仍来自 `.env.example`，未记录任何变量值。
- 2026-06-23 后续报告已经生成定位分析、机会点和结论，但最终验证仍卡在
  `positioning[0]`。根因是 fallback 把 `ProductProfile.positioning` 的长文案直接放进
  分析 claim，而该字段当前没有独立 Evidence ID，Verifier 会把它当成需要逐字支持的事实。
- 本次 Analyst 修复把 fallback 定位分析和结论改为只使用带 Evidence ID 的字段：
  `format_feature_mentions()` 生成 `mentions ...`，`format_pricing_mentions()` 生成
  `lists the ... plan at ...` 或 `names ... plan without a public price ...`。
  `build_fallback_opportunity_claims()` 也移除了 target users 机会点，因为
  `target_users` 当前同样没有独立 Evidence ID。
- 同次修复还调整了 fallback 文案细节：`choose_indefinite_article()` 避免
  `a Enterprise plan`，缺价结论复用价格事实句式，避免 `Business without public prices`
  这类过度压缩表达被 Verifier 误杀。
- 新增/更新回归仍集中在 `tests/test_analyst.py`：
  fallback 定位 claim 不再包含 `positioning around ...`，机会点不再包含 `audience fit`，
  缺价 Enterprise 文案使用 `an Enterprise plan`。聚焦测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py -q`，
  结果 `21 passed`。
- 2026-06-23 最新 ChatGPT / Claude / Gemini 报告显示，Extractor 已能抽到价格方案，
  但 Analyst 会在横向比较时改写价格事实：把 `price=None` 的 Business/Enterprise
  写成 `$0`，或把画像里的普通数字 `8` 改写为 `$8`。这不是搜索问题，而是 Analyst
  自由生成 pricing fact 时修改了结构化画像。
- 本次修复在 `competitive_analysis_agent/analyst.py` 中新增
  `normalize_pricing_fact_claims()`：对单产品 pricing fact，先用产品名、Evidence ID
  和 `plan_name` 映射回 `ProductProfile.pricing`，再复用
  `format_fallback_pricing_claim()` 生成标准句式。这样 `price=None` 只能输出
  “without a public price”，普通数字价格保持原样，不会自动加美元符号。
- 同次修复收紧 `ANALYST_SYSTEM_PROMPT`：明确要求 `pricing.price` 为 `null` 时不能写成
  `$0` 或 Free，普通数字价格也不能擅自补货币单位。提示词负责减少错误，确定性后处理负责
  兜住真实模型偶发改写。
- 新增回归：`test_pricing_fact_is_narrowed_to_profile_price`，覆盖 `$8` 和未知价格 `$0`
  两类真实报告问题。聚焦测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py -q`，
  结果 `22 passed`。相邻流程测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py tests/test_verifier.py tests/test_reporter.py tests/test_workflow.py -q`，
  结果 `61 passed`。完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果 `141 passed, 7 deselected`。
- 真实 Analyst 验收命令：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests/test_live_analyst.py -q`。
  结果 `1 passed`，耗时约 27.53 秒；验证真实模型路径仍能返回结构化 Analyst 输出，
  配置来源仍是 `.env.example`，未记录任何变量值。
- 完整离线测试中还发现 `tests/test_streamlit_app.py` 仍期待旧默认场景 Notion/Confluence，
  但 UI 默认值已经切换到 ChatGPT/Claude/Gemini。本次只同步测试断言到现有
  `DEFAULT_TARGET_PRODUCT`、`DEFAULT_COMPETITORS` 和默认官方域名，不改变 UI 业务逻辑。
- 2026-06-24 最新 ChatGPT / Claude / Gemini 报告显示，第一次 Analyst 输出已经较保守，
  但 Verifier 反馈进入第二轮后，模型被 `suggested_action` 带偏，把
  `Product mentions Feature` 改成了 `includes features like ...`，并生成了更宽泛的
  conclusion，导致最终报告仍未通过验证。
- 本次修复分两层处理：提示词层面，`ANALYST_SYSTEM_PROMPT`、`build_analyst_messages()` 和
  `build_repair_messages()` 明确要求不要照抄 Verifier suggested_action，不要写
  `includes/offers/provides features like`，features 被点名时只保留
  `Product mentions Feature.` 或删除；代码层面，`normalize_feature_claims_after_feedback()`
  在重试轮发现 features 反馈后直接用 `build_fallback_feature_claims()` 回退到画像里的
  最小功能事实，`normalize_conclusion_after_feedback()` 在任意重试反馈存在时回退到
  `build_fallback_conclusion()`。
- 同次修复还更新 `competitive_analysis_agent/workflow.py::build_revision_feedback()`：
  不再把 Verifier 的 `suggested_action` 原样拼回 Analyst 输入，而是按 claim 路径生成
  保守修订规则。这样 Verifier 可以继续给人类展示建议，但 Agent 修复轮不会把建议句当成事实。
- 新增回归：`test_revision_feedback_replaces_broad_feature_and_conclusion` 覆盖重试轮
  生成宽泛 feature/conclusion 的问题；`test_revision_feedback_does_not_copy_suggested_action`
  覆盖 Workflow 不再原样回灌 suggested_action。由于 feature unsupported 案例现在会被自动
  修复，`evaluation/cases.json` 的 warning 案例改为价格冲突连续失败，用来继续覆盖
  “有限重试后生成警告报告”。
- 验证结果：`C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py tests/test_workflow.py -q`
  结果 `30 passed`；相邻流程测试
  `tests/test_analyst.py tests/test_workflow.py tests/test_verifier.py tests/test_reporter.py tests/test_ui_service.py`
  结果 `75 passed`；完整离线测试
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q` 结果
  `143 passed, 7 deselected`。
- 真实 LLM 验证：真实 Analyst smoke
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests/test_live_analyst.py -q`
  结果 `1 passed`，耗时约 28.65 秒；真实 LangGraph smoke
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests/test_live_workflow.py -q`
  结果 `1 passed`，耗时约 72.23 秒。配置来源仍为 `.env.example`，未记录任何变量值。
- 离线评估结果已重新生成到 `docs/evaluation/evaluation-results.json` 和
  `docs/evaluation/evaluation-results.md`。任务成功率仍为 `66.7%`，因为 warning 案例
  仍按预期失败并生成警告报告；平均字段覆盖率提升为 `100.0%`。
- 2026-06-24 最新报告仍出现机会点验证失败：模型会写
  “Claude could offer more detailed pricing information”、
  “Gemini could provide clearer pricing structure” 这类听起来合理但没有证据支撑的建议。
  对个人项目来说，报告可以不那么严谨，但机会点仍应来自画像里看得到的差异，否则 Verifier
  会正确拦截。
- 本次在 `normalize_analysis_output()` 中新增 `normalize_opportunities()`：
  只要机会点点名具体产品却没有 `evidence_ids`，或 Verifier 已反馈
  `opportunities[...]` 不受支持/证据冲突，就用 `build_fallback_opportunity_claims()`
  重新生成轻量机会点。fallback 只基于价格透明度和功能差异等已有画像信息，并携带画像中的
  Evidence ID。
- 同步收紧 `ANALYST_SYSTEM_PROMPT`：机会点涉及具体产品时必须带上对应
  ProductProfile 的 `evidence_ids`，没有证据支撑时宁可少写。这样提示词和确定性后处理
  分工清晰：模型负责自然语言，代码负责最低可信边界。
- 新增回归：`test_unsourced_opportunities_are_replaced_with_fallback`。聚焦验证：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_extractor.py tests\test_analyst.py -q`
  结果 `43 passed`。
- 完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果
  `147 passed, 7 deselected`。真实 Extractor/Analyst smoke：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -o addopts= -m live_llm tests\test_live_extractor.py tests\test_live_analyst.py -q`，
  结果 `2 passed`，耗时约 62.19 秒；配置来源仍为 `.env.example`，未记录任何变量值。
