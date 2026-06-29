# Agent 修复指导：API 价格范围控制

## 1. 修复目标

本次修复的核心目标是让 ChatGPT、Claude、Gemini 这类模型产品在默认场景下只分析
API 价格，而不是把网页上同品牌的消费端订阅、团队套餐、Workspace、Home、Veo 等
其他产品线价格混入报告。

用户已确认的产品行为：

- ChatGPT 默认分析 OpenAI / ChatGPT API 价格。
- Claude 默认分析 Anthropic Claude API 价格。
- Gemini 默认分析 Gemini API / Google AI API 价格。
- 默认不分析 ChatGPT Plus/Pro/Team/Business、Claude Pro/Max/Team、Gemini App、
  Workspace Gemini、Google Home Premium、Veo 等订阅或非 API 产品价格。

## 2. 当前主要问题

当前系统的主要缺陷不是单个 prompt，而是产品范围控制不足。官方域名只表示来源身份，
不能证明页面语义属于当前要分析的产品范围。

高风险数据流：

```text
SearchResult / Evidence
  -> ProductProfile
  -> CompetitiveAnalysis
  -> VerificationResult
  -> Markdown Report
```

如果 Evidence 或 ProductProfile 已经混入其他产品线，后面的 Analyst 和 Verifier 很难
可靠修复。Verifier 只能检查最终 claim，不能清洗已经污染的 profile。

## 3. 必须保持的边界

- Planner 只生成任务，不搜索。
- Researcher 只收集和规范化 Evidence，不解释产品事实。
- Extractor 负责从 Evidence 生成 ProductProfile，并应承担第一层产品范围过滤。
- Analyst 只比较 ProductProfile，不浏览网页，不重新解释网页来源。
- Verifier 检查引用和 unsupported/conflicting claims，不应该继续无限放宽。
- Reporter 只确定性渲染，不补充事实。

不要通过继续放松 Verifier 来掩盖上游污染。

## 4. 建议修复顺序

### P0-1：增加显式 API 价格 scope 规则

优先文件：

- `competitive_analysis_agent/researcher.py`
- `competitive_analysis_agent/extractor.py`
- `competitive_analysis_agent/schemas.py`
- `tests/test_researcher.py`
- `tests/test_extractor.py`

建议做法：

1. 为默认模型产品建立轻量 scope 规则。
   - ChatGPT/OpenAI：API、developer、platform、token、model pricing、input/output。
   - Claude/Anthropic：API、console、token、model pricing、input/output。
   - Gemini/Google AI：Gemini API、Google AI API、ai.google.dev、token、model pricing。
2. 对明显非 API 页面降权或排除。
   - ChatGPT Plus、Pro、Team、Business、Enterprise subscription。
   - Claude Pro、Max、Team、monthly plan、seat、collaborate often。
   - Gemini App、Workspace Gemini、Google Home Premium、Veo、Nest、consumer plan。
3. pricing Evidence 应同时看 title、url、snippet、raw_content，而不是只看官方域名。
4. 找不到 API 价格时应明确显示信息不足，不要用订阅价格补位。

### P0-2：阻止非 API pricing 进入 ProductProfile

优先文件：

- `competitive_analysis_agent/extractor.py`
- `tests/test_extractor.py`

建议做法：

1. 在 Extractor prompt 中明确写入默认 API pricing 范围。
2. 在 `normalize_profile_summary_fields()` 后处理链路中增加确定性过滤。
3. 当前 `PRODUCT_SCOPE_EXCLUSION_MARKERS` 只覆盖 Gemini Home Premium，过窄。
4. 不要只添加更多零散关键词；应抽出可测试 helper，例如：
   - `classify_pricing_source_scope(...)`
   - `pricing_plan_matches_requested_scope(...)`
   - `build_api_pricing_scope_rules(product_name)`
5. 对 scope 不确定的价格项，优先删除或标记为 ambiguous，不要进入普通 pricing claim。

### P0-3：让错误可验证，而不是只靠人工看报告

新增回归测试至少覆盖：

- Claude pricing Evidence 中出现 Max plan / daily users / monthly subscription 时，不生成
  product-level positioning，也不生成 API pricing。
- ChatGPT official Evidence 中出现 Plus/Pro/Team/Business 订阅价时，默认 API pricing
  场景应排除。
- Gemini Evidence 混入 Google Home Premium、Workspace Gemini、Veo 时，默认 API
  pricing 场景应排除。
- Gemini API token price 允许进入 Gemini API pricing。
- 找不到 API 价格时，report/data limitations 显示缺失，而不是用订阅价格替代。

### P1：补 profile validation

优先文件：

- `competitive_analysis_agent/schemas.py`
- `competitive_analysis_agent/extractor.py`
- `competitive_analysis_agent/workflow.py`
- `tests/test_workflow.py`

建议做法：

1. 在进入 Analyst 前增加 profile validation。
2. 检查同一产品的 pricing 是否混入多个 product scope。
3. 检查同名 plan + 同计费周期 + 不同价格冲突。
4. 检查 positioning 是否来自 pricing/support/subscription 语境。
5. validation 结果应进入 `research_errors` 或新的 limitations 字段，让 Reporter 显示。

### P1：收紧 opportunities 的验证策略

优先文件：

- `competitive_analysis_agent/analyst.py`
- `competitive_analysis_agent/verifier.py`
- `tests/test_analyst.py`
- `tests/test_verifier.py`

当前风险：

- `opportunities[...]` 被当成 soft interpretation。
- 如果机会点带了 evidence_ids，即使语义跳得太远，也可能被 Verifier 当误报忽略。

建议做法：

1. 只对确定性 fallback 模板放宽，例如 pricing clarity 和 feature contrast。
2. 模型自由生成的 opportunity 如果被 Verifier 标记 unsupported，应保留 issue。
3. 如果 opportunity 点名具体产品，必须能追溯到对应 profile 差异。
4. 不要生成 “Claude could offer clearer pricing” 这类没有直接数据差异支撑的建议。

### P1：修 Researcher URL 去重导致的 topic 丢失

优先文件：

- `competitive_analysis_agent/researcher.py`
- `tests/test_researcher.py`

当前风险：

同一 URL 如果先作为 features 被收录，后续 pricing 命中同 URL 会被全局去重跳过，
pricing raw_content 可能无法进入 Extractor。

建议做法：

- 将去重 key 从全局 URL 改成 `(product_name, topic, normalized_url)`；或
- 合并同一 URL 的多 topic 上下文；如果后续 pricing 命中同 URL，应升级已有 Evidence
  的 topic/raw_content 或创建 topic-specific Evidence。

### P2：文档和 UI 行为同步

优先文件：

- `README.md`
- `docs/stage-notes/stage-12-real-search-provider.md`
- `docs/stage-notes/stage-13-backend-logging.md`
- `competitive_analysis_agent/ui_service.py`

需要同步：

- README 当前默认场景仍有 Notion/Confluence 描述，但代码默认值已是 ChatGPT/Claude/Gemini。
- README 的 field coverage 仍有旧值，当前 evaluation 文档显示 100.0%。
- 日志文档说不记录官方域名和 Evidence 正文，但当前阶段 I/O 摘要会记录域名、URL、
  snippet preview 和 raw_content preview。需要决定是脱敏日志还是更新文档。

## 5. 不建议的修复方式

- 不要只在 prompt 里写 “不要混淆产品线”。
- 不要继续无限增加 Verifier 宽容度。
- 不要把 Claude Max、ChatGPT Plus、Gemini Workspace 等订阅价格映射成 API 价格。
- 不要为了字段覆盖率强行生成 pricing。
- 不要让 Reporter 在缺失价格时补写任何推测。
- 不要把真实 API key、完整 prompt、完整 raw_content 写进测试快照、日志或文档。

## 6. 推荐验证命令

先跑聚焦测试：

```powershell
C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests/test_researcher.py tests/test_extractor.py tests/test_analyst.py tests/test_verifier.py -q
```

再跑完整离线测试：

```powershell
C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q
```

如果修改了 LLM prompt、结构化输出、Workflow 或真实模型路径，再按项目约定运行对应
`live_llm` 测试，配置来源仍是：

```text
F:\大模型应用开发学习\competitive-analysis-agent\.env.example
```

不要打印或记录其中的变量值。

## 7. 建议后续 agent 使用的技能

- `competitive-analysis-agent-coach`：按项目阶段、测试和中文注释约定实施修复。
- `diagnose`：如果真实报告仍混入非 API 价格，用复现 -> 最小 fixture -> 假设 ->
  instrumentation -> fix -> regression test 的流程定位。
- `tdd`：本次非常适合先写 failing fixture，再做最小修复。

## 8. 可直接开始的最小里程碑

建议下一位 agent 从一个小切片开始：

1. 写 `tests/test_extractor.py` 的失败用例：ChatGPT/Claude/Gemini 默认 API pricing 不应接收
   订阅套餐价格。
2. 在 Extractor 中加入通用 API pricing scope helper。
3. 让测试通过。
4. 跑 `tests/test_extractor.py tests/test_analyst.py`。
5. 再决定是否把 scope 字段提升到 Schema。

这个顺序能先挡住最明显的报告错误，同时避免一次性重写整个数据模型。
