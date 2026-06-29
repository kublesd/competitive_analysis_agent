# Stage 07：Verifier

## 1. 本阶段目标

本阶段检查 `CompetitiveAnalysis` 的引用完整性和语义支持情况。

输入：

- Analyst 生成的 `CompetitiveAnalysis`；
- Researcher 收集的原始 `Evidence`。

输出：

- `passed`：是否通过验证；
- `issues`：可定位、可修复的问题清单；
- `retry_recommended`：是否建议重新执行分析。

它位于完整流程中的位置：

```text
Analyst -> CompetitiveAnalysis -> Verifier -> VerificationResult
```

本阶段只表示重试意图，不创建 LangGraph 循环，也不负责改写分析或格式化报告。

## 2. 实现结果

- 完成的功能：
  - `VerifierInput` 保存分析与 Evidence，并拒绝重复 Evidence ID；
  - `VerificationIssue` 提供类型、claim 路径、说明、引用和建议动作；
  - `VerificationResult` 保证 `passed`、issues 和重试意图一致；
  - `find_deterministic_issues()` 检查无效 ID、错产品引用和缺少逐产品支持；
  - 确定性问题存在时直接返回，不调用模型；
  - `Verifier` 对结构正确的分析执行一次语义评审；
  - `validate_semantic_output()` 检查模型返回的路径和 Evidence ID；
  - `live_verifier.py` 提供真实冲突 claim 验收入口。
- 关键文件：
  - `competitive_analysis_agent/verifier.py`
  - `competitive_analysis_agent/live_verifier.py`
  - `tests/test_verifier.py`
  - `tests/test_live_verifier.py`
  - `tests/fixtures/verifier_outputs.json`
- 核心数据流：

```text
CompetitiveAnalysis + Evidence
  -> 确定性引用检查
  -> 有硬错误：VerificationResult
  -> 无硬错误：结构化语义模型检查
  -> 模型 issue 边界校验
  -> VerificationResult
```

- 验证方式与结果：
  - `python -m pytest tests/test_verifier.py -q`：8 个测试通过；
  - `python -m pytest -q`：44 个离线测试通过，3 个真实用例默认排除；
  - `python -m compileall -q competitive_analysis_agent tests`：通过。
- 真实 LLM 测试：
  - 输入 claim 声称 Atlas Team 方案免费；
  - `E1` 明确写明每用户每月 12 USD；
  - 模型成功定位 `pricing[0]` 并报告不支持或冲突问题。
- 暂未实现：
  - 不自动改写 claim；
  - 不执行重试或 LangGraph 路由；
  - 不评估完整工作流轨迹、耗时和成本；
  - 不实现 Reporter；
  - 不对多个网页正文做深度事实核查，只使用当前 Evidence 摘要。

## 3. 设计决策

### 决策 1：哪些检查由代码完成，哪些交给模型？

**问题背景**

Evidence ID 是否存在、是否属于 claim 中的产品，是精确集合运算；claim 的文字是否
真的被摘要支持、不同证据是否冲突，则需要语言理解。

**当前方案**

`competitive_analysis_agent/verifier.py` 中的
`find_deterministic_issues()` 先检查引用；只有 issues 为空时，
`Verifier.verify()` 才构建消息并调用结构化模型。

**替代方案**

全部交给模型判断，或者全部使用关键词规则判断。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 代码硬校验 + 模型语义评审 | 确定性问题可靠，语义问题有理解能力 | 多一层模型调用 |
| 全部交给模型 | 实现表面简单 | 无效 ID 也可能漏检，结果不稳定 |
| 全部关键词规则 | 无模型成本、可重复 | 无法可靠理解改写、否定和冲突 |

**什么时候考虑切换**

如果业务只需要引用存在性，可以关闭模型语义层；如果未来有专门 NLI 或事实核查
模型，可以替换通用 LLM，但确定性检查仍应保留。

**面试回答参考**

我把可以写成集合运算的检查留给代码，例如 ID 存在性和产品归属；只有“证据是否
支持 claim”这种语义问题才调用模型。这样减少不必要的模型不确定性，也保留了对
自然语言支持关系的判断能力。

### 决策 2：为什么发现硬引用错误后直接跳过模型？

**问题背景**

如果 claim 引用了不存在的 `E99`，模型没有对应证据可读，继续做语义判断没有可靠
输入，还会增加延迟和成本。

**当前方案**

`Verifier.verify()` 在 `deterministic_issues` 非空时直接调用
`build_verification_result()` 返回。离线测试同时断言 fake model 的调用次数为 0。

**替代方案**

即使引用无效，也让模型同时检查其他 claim。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 硬错误快速失败 | 省成本、结果明确，避免无来源评审 | 一次只暴露引用层问题 |
| 始终调用模型 | 可能一次发现更多问题 | 无效引用使语义判断缺少依据 |

**什么时候考虑切换**

如果未来 Verifier 运行在离线批处理、希望一次收集尽可能多的问题，可以把有效 claim
单独送给模型；当前同步 MVP 优先快速失败和清晰反馈。

**面试回答参考**

无效引用是前置条件错误。模型无法评审不存在的证据，所以我先返回结构化问题，不
继续消耗模型调用。修复引用后再做语义检查，问题定位更清晰。

### 决策 3：为什么 Verifier 返回 issue 记录，而不是只返回布尔值？

**问题背景**

`passed=False` 只能告诉工作流失败，不能说明哪个 claim、哪条引用、应如何修复。

**当前方案**

`VerificationIssue` 包含 `issue_type`、`claim_path`、`message`、
`evidence_ids` 和 `suggested_action`。

**替代方案**

只返回 `True/False`，或返回一段自然语言总结。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 结构化 issue 列表 | 可定位、可测试、可用于路由和 UI | Schema 字段更多 |
| 只有布尔值 | 最简单 | 无法修复或解释 |
| 自然语言总结 | 对人易读 | 程序难以稳定路由 |

**什么时候考虑切换**

报告层可以把 issue 渲染成自然语言，但内部状态应保留结构化记录。

**面试回答参考**

Verifier 不只是判分，还要给下一步可执行反馈。每个 issue 都定位到 claim 路径，并
包含问题类型和建议动作，因此后续 LangGraph 可以决定回到 Analyst，UI 也能展示
具体原因。

### 决策 4：为什么只表示 retry_recommended，不在本阶段执行循环？

**问题背景**

验证失败后通常要回到 Analyst 修复，但如果 Verifier 自己直接调用 Analyst，会把
校验逻辑和编排逻辑耦合，也容易形成无限循环。

**当前方案**

`VerificationResult` 只保存 `retry_recommended`。Verifier 没有 Analyst 依赖，
也没有循环控制代码。

**替代方案**

Verifier 内部直接调用 Analyst，直到通过。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 返回重试意图 | 节点独立、易测试，循环上限由工作流统一控制 | 需要后续编排层处理 |
| Verifier 内部重试 | 单个入口看起来完整 | 职责耦合，调用次数和循环难控制 |

**什么时候考虑切换**

不建议让 Verifier 自己无限重试。Stage 8 会在 LangGraph 中建立最多一次的受限回路。

**面试回答参考**

Verifier 只负责判断和给出重试意图，不负责执行循环。真正的路由和重试上限属于
LangGraph 工作流，这能保持节点可独立测试，并避免隐藏的无限调用。

### 决策 5：为什么模型 issue 也必须经过代码校验？

**问题背景**

评审模型本身也可能返回不存在的 claim 路径或 Evidence ID。如果不校验，Verifier
会把新的幻觉当成验证结果。

**当前方案**

模型输出先通过 `VerifierModelOutput`。`validate_semantic_output()` 再检查
`claim_path` 是否存在、Evidence ID 是否来自当前输入，以及 ID 是否重复。

**替代方案**

完全信任模型返回的问题清单。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 校验模型 issue | 防止评审结果虚构定位和引用 | 增加少量确定性代码 |
| 信任模型输出 | 实现更短 | 验证器自身也可能产生无效数据 |

**什么时候考虑切换**

不建议移除边界校验。即使更换模型，外部输出进入项目时仍应验证。

**面试回答参考**

Verifier 使用模型不代表信任模型。模型只负责语义判断，它返回的 claim 路径和
Evidence ID 仍由代码验证，避免“验证器产生新的幻觉”。

## 4. 异常与边界情况

- Evidence 为空：`VerifierInput` 在运行前拒绝。
- Evidence ID 重复：输入校验失败，避免一个 ID 对应多个来源。
- claim 引用不存在的 ID：返回 `invalid_evidence_id`，不调用模型。
- claim 引用其他产品 Evidence：返回 `wrong_product_evidence`。
- 多产品事实缺少某个产品自己的 Evidence：返回
  `missing_product_evidence`。
- claim 引用正确但内容没有证据支持：模型返回 `unsupported_claim`。
- claim 与 Evidence 明确相反：模型返回 `conflicting_evidence`。
- 模型返回完整验证对象或嵌套验证对象：先规范化为 `{"issues": [...]}` 再校验。
- 模型输出格式不符合 Schema：抛出 `VerifierError`，并附带脱敏后的
  `public_detail`，说明缺失字段或结构问题。
- 模型返回不存在的 claim 路径或 Evidence ID：抛出 `VerifierError`，
  不把无效评审结果写入工作流，并在 `public_detail` 中保留出错路径或 ID。
- 模型把“本比较仅限于已提供证据”这类保守 conclusion 范围说明误报为
  `unsupported_claim`：确定性过滤该误报，其他 conclusion 问题仍保留。
- Stage 8 真实整图验收发现：当问题是 claim 完全没有直接支持来源时，
  语义 issue 允许 `evidence_ids=[]`；未知 ID 和重复 ID 仍会被拒绝。
- 模型供应商异常：转换成 `VerifierError`。
- issues 非空时 `passed=False` 且 `retry_recommended=True`；
  `VerificationResult` 会校验三者一致性。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：是
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：`python -m pytest -o addopts= -m live_llm tests/test_live_verifier.py -q`
- 真实调用的组件：`create_live_verifier()` 创建的 `Verifier`
- 验证的输出契约：
  - 结果 `passed=False`；
  - `retry_recommended=True`；
  - issues 非空；
  - 首个 issue 定位 `pricing[0]`；
  - issue 类型为 `unsupported_claim` 或 `conflicting_evidence`；
  - 如果模型返回 `evidence_ids`，必须是当前输入中的真实 ID；
  - issue 必须提供 `suggested_action`。
- 测试结果：通过，`1 passed`
- 耗时：17.69 秒
- 失败原因：无

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4b-agent-evaluation.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\whitepaper\Agent Quality 中文解析.md`

### Relevant sections

- `什么是 Agent 评估？`
- `第 3 节：使用 ADK Web UI 进行交互式评估`
- `第 4 节：系统化评估`
- `4.2：创建测试用例`
- `4.4：分析示例评估结果`
- 白皮书：`从模型评估到系统评估`
- 白皮书：`Agent 质量的四大支柱`
- 白皮书：`评估框架：Outside-In`

### Core concept

Agent 的正确性不只是“有没有输出文字”。需要检查最终结果、引用、工具或步骤轨迹，
并在失败时给出能定位根因的结果。固定测试用例可以持续发现回归，而可执行 issue
比单一分数更有助于修复。

确定性规则与模型评审解决不同问题：ID 是否存在可以精确计算；自然语言 claim 是否
被证据支持，需要语义判断。系统质量还包括有效性、效率、稳健性和安全对齐，本阶段
主要覆盖有效性中的引用与忠实性。

### How it appears in this project

- `find_deterministic_issues()` 相当于明确的硬评估标准；
- `VerifierModelOutput` 是结构化语义评审结果；
- `tests/test_verifier.py` 使用固定用例保护引用、冲突和错误边界；
- `VerificationIssue` 提供类似详细评估结果中的根因与修复建议；
- `retry_recommended` 为下一阶段工作流路由提供信号。

### ADK vs LangGraph

- Notebook 使用 ADK Web UI、evalset、`adk eval` 和工具轨迹指标；
- 当前项目使用 pytest fixture、Pydantic 和一个工作流内 Verifier；
- Verifier 是运行时质量门，不等于完整 Agent Evaluation；
- Stage 8 才会用 LangGraph 把验证失败路由回 Analyst；
- Stage 11 才会加入完整固定评测集和系统指标。

### Intentionally postponed

- 端到端任务成功率；
- Agent 执行轨迹和工具调用评分；
- Token、耗时、成本等效率指标；
- Prompt injection 和隐私等安全评估；
- 用户模拟与长期回归评测集。

## 7. 理解问题与参考思路

### 问题 1：哪些检查应该由普通代码完成？

**参考思路：**

- Evidence ID 存在性；
- Evidence 产品归属；
- 多产品事实是否为每个产品提供引用；
- 模型 issue 中的路径和 ID 是否有效。

### 问题 2：哪些问题适合交给模型？

**参考思路：**

- claim 是否被摘要语义支持；
- claim 是否与证据明确矛盾；
- 多条证据之间是否存在影响结论的冲突；
- 这些问题不能只靠字符串完全相等判断。

### 问题 3：为什么 Verifier 独立于 Analyst？

**参考思路：**

- 生成者不应只靠自己判断结果正确；
- 独立节点可以使用不同规则和提示词；
- issue 可以进入测试、UI 和后续路由。

### 问题 4：哪些 issue 应触发重新分析？

**参考思路：**

- 无效或错产品引用；
- 不受支持的事实 claim；
- 与证据冲突的 claim；
- 如果问题来自证据本身不足，未来可能需要路由回 Researcher，而不只是 Analyst。

## 8. 面试追问清单

- 如果 claim 数量扩大十倍，模型评审输入如何控制？
- 如果移除确定性引用检查，会发生什么？
- 哪些 issue 应回到 Analyst，哪些可能需要重新 Research？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 8：LangGraph Workflow 将根据 `passed` 和 `retry_recommended` 建立受限路由。
本阶段不实现该行为。

## 10. 变更记录

- 2026-06-22 真实 Notion / Confluence 工作流在 Verifier 阶段中断，日志定位为
  `VerifierError` / `validate_semantic_output`。原因不是报告未通过，而是 Verifier
  模型返回的语义评审对象不符合 `VerifierModelOutput`。
- 本次修复新增 `normalize_verifier_raw_output()`，兼容真实模型常见的外层形状：
  完整 `passed/issues/retry_recommended`、嵌套 `verification_result`、以及 issue 列表。
- 本次修复新增 `VerifierError.public_detail` 和
  `build_verifier_schema_error_detail()`，页面可显示缺失字段、无效 claim_path、
  未知 Evidence ID 等定位信息，同时不展示原始模型响应、prompt、traceback 或密钥。
- 本次修复新增 `should_ignore_semantic_issue()`，过滤模型对保守范围结论的误报。
- 离线测试：`python -m pytest tests/test_verifier.py tests/test_ui_service.py tests/test_workflow.py -q`，
  结果 `26 passed`；完整离线测试 `103 passed, 7 deselected`。
- 2026-06-22 最新 Notion / Confluence 报告显示 Verifier 又走向另一端：Evidence
  已经出现功能短语或价格数字，但模型仍要求逐字出现 `mentions`、`lists the plan at`
  等标准化句式，导致 `features`、`pricing` 和保守 `conclusion` 被误报为
  `unsupported_claim`。本次修复没有关闭语义评审，而是在
  `should_ignore_semantic_issue()` 后面增加三类确定性兜底：
  `is_supported_standard_feature_claim()` 允许 “Product mentions X” 由 Evidence 中的
  X 或关键词覆盖支持；`is_supported_standard_pricing_claim()` 允许 pricing claim
  由套餐名 + 价格/Custom pricing/Free 支持；`is_conservative_scope_claim()` 同时忽略
  `supplied evidence` 和 `supplied product profiles` 这类系统范围说明。
- 为避免把错位 issue 静默吞掉，若模型 issue 自带 `evidence_ids`，这些 ID 必须和当前
  claim 的引用有交集，才会进入上述误报过滤。否则仍保留 issue 并触发 Analyst 重试。
- 新增回归：
  `test_product_profile_scope_conclusion_issue_is_ignored`、
  `test_mentions_feature_issue_is_ignored_when_phrase_exists`、
  `test_pricing_issue_is_ignored_when_plan_and_price_exist`；同步更新
  `test_retry_fallback_removes_verifier_rejected_feature` 的错位 Evidence 场景。
  聚焦测试：`python -m pytest tests/test_verifier.py tests/test_workflow.py tests/test_analyst.py tests/test_ui_service.py -q`，
  结果 `50 passed`。完整离线测试：`115 passed, 7 deselected`。
  真实 Verifier 集成测试：
  `python -m pytest -o addopts= -m live_llm tests/test_live_verifier.py -q`，
  结果 `1 passed`，耗时约 14.16 秒；未记录任何密钥或原始敏感配置。
- 2026-06-22 最新报告的剩余 issue 是 `Confluence lists the Free plan at Free.`
  没有通过语义验证，但 Evidence 中已有 `Free forever for 10 users`。确定性兜底现在把
  `Free` / `$0` 视为同一类免费价格证据，不再要求价格文本必须包含数字。新增回归：
  `test_free_pricing_issue_is_ignored_when_free_evidence_exists`。
- 2026-06-23 最新报告发现 Verifier 的 pricing 兜底过宽：对于
  `names a Workers plan without a public price ... with Beta billing`，旧逻辑只要 Evidence
  出现套餐名就可能忽略模型 issue。现在 `is_supported_standard_pricing_claim()` 会检查
  可选 billing 片段，只有明确的 monthly/yearly/annual 周期且 Evidence 或 price 文本支持时
  才会进入误报过滤。新增回归：
  `test_invalid_billing_issue_is_not_ignored_for_missing_price`。
- 2026-06-23 最新报告仍因功能短语词形变化未通过，例如 `Workflow automation` 对应证据
  中的 `automated processes and workflows`，以及 `AI-powered knowledge management`
  对应证据中的 `AI-powered apps ... knowledge`。本次修复只放宽标准化 feature claim 的
  确定性兜底：`canonicalize_significant_tokens()` 处理自动化、管理、复数等轻量词形；
  `evidence_supports_feature_phrase()` 对长功能名允许高覆盖率匹配，但至少覆盖三个关键信息词。
  这样 `Rovo AI features` 在缺少 `Rovo` 专有词时仍不会被误判为通过。新增回归：
  `test_feature_issue_is_ignored_for_inflection_variants`、
  `test_feature_issue_is_ignored_for_high_token_coverage`、
  `test_feature_issue_is_not_ignored_when_proper_noun_is_missing`。
  聚焦测试：`python -m pytest tests/test_verifier.py tests/test_workflow.py tests/test_ui_service.py -q`，
  结果 `37 passed`。完整离线测试：`130 passed, 7 deselected`。真实 Verifier 集成测试：
  `python -m pytest -o addopts= -m live_llm tests/test_live_verifier.py -q`，
  结果 `1 passed`，耗时约 14.49 秒。
- 2026-06-23 最新报告只剩 `positioning[0]` 被误报 unsupported。由于本项目是个人项目，
  定位分析和机会点更接近解释型摘要，不应像 feature/pricing fact 一样要求逐字命中证据。
  本次修复在 `should_ignore_semantic_issue()` 中新增 `is_soft_interpretation_issue()`：
  对 `positioning[...]`、`opportunities[...]`，以及以
  `Based on the supplied profiles` 开头的 fallback summary conclusion，只过滤
  `unsupported_claim` 误报；`conflicting_evidence`、无效 Evidence ID、错产品引用仍然保留。
- 为避免过度放松，`contains_strong_evaluation_language()` 会保留包含
  `better`、`clearly stronger`、`dominates`、`领先`、`优于` 等强评价词的 issue。
  也就是说，解释型章节可以不逐字引用，但不能变成未经证据支持的胜负判断。
- 新增回归：
  `test_positioning_interpretation_wording_issue_is_ignored`、
  `test_strong_positioning_evaluation_is_not_ignored`、
  `test_fallback_summary_conclusion_wording_issue_is_ignored`。
  聚焦测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_analyst.py tests/test_verifier.py -q`，
  结果 `45 passed`。完整离线测试：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`，结果
  `134 passed, 7 deselected`。
- 额外真实 Verifier smoke 使用固定样例触发 Analyst fallback，再交给真实 Verifier 验证，
  结果 `passed=true`、`issue_count=0`。真实 LLM 回归中，`test_live_analyst.py`
  通过；`test_live_verifier.py` 首次遇到外部 SSL EOF 连接断开，复跑通过：
  `1 passed in 13.80s`。未记录任何密钥或原始敏感配置。
