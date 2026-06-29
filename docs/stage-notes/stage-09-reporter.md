# Stage 09：Reporter

## 1. 本阶段目标

本阶段把结构化竞品分析结果确定性地渲染为可阅读、可下载的 Markdown 报告。

输入：

- `CompetitiveAnalysis`；
- `ProductProfile`；
- `Evidence`；
- `VerificationResult`；
- Researcher 记录的部分失败。

输出：

- 固定章节顺序的 Markdown 字符串；
- 可写入磁盘的 `.md` 文件。

完整数据流：

```text
Verifier
  -> analysis + profiles + evidence + verification + research errors
  -> Reporter
  -> Markdown report
```

本阶段不调用模型，不实现 Streamlit，也不保存跨会话历史。

## 2. 实现结果

- 完成的功能：
  - `ReporterInput` 校验产品顺序、Evidence ID 唯一性和全部报告引用；
  - `Reporter.render()` 按固定顺序生成 Markdown；
  - 产品画像渲染为横向概览表；
  - 定位、功能、价格和机会点使用统一 claim 表；
  - Evidence ID 渲染为直接指向来源 URL 的可点击链接；
  - 空字段使用“未提供”等明确文本，不补充未知事实；
  - 验证失败时保留警告、issue 类型、claim 路径和建议动作；
  - Researcher 错误、画像缺失和公开限制汇总到数据限制章节；
  - `Reporter.write()` 支持写入 `.md` 文件并创建父目录；
  - LangGraph 当前以 `Verifier -> Reporter -> END` 结束。
- 关键文件：
  - `competitive_analysis_agent/reporter.py`
  - `competitive_analysis_agent/workflow.py`
  - `tests/test_reporter.py`
  - `tests/test_workflow.py`
  - `docs/sample-report.md`
- 核心数据流：

```text
CompetitiveAnalysis
  + ProductProfile
  + Evidence
  + VerificationResult
  + ResearchError
  -> ReporterInput validation
  -> deterministic Markdown sections
  -> final_report / .md file
```

- 验证方式与结果：
  - `python -m pytest tests/test_reporter.py tests/test_workflow.py -q`：
    12 个测试通过；
  - `python -m pytest -q`：58 个离线测试通过，4 个真实 LLM 用例默认排除；
  - `python -m compileall -q competitive_analysis_agent tests`：通过；
  - `docs/sample-report.md`：生成 66 行、3818 字节的可检查样例。
- 真实 LLM 测试：本阶段不创建或修改模型行为，因此没有新增真实调用。
- 暂未实现：
  - Streamlit 展示与下载按钮；
  - HTML、PDF 或 DOCX 导出；
  - 品牌主题和多语言模板；
  - 报告历史数据库；
  - 人工批注和审批状态。

## 3. 设计决策

### 决策 1：为什么 Reporter 不调用大模型？

**问题背景**

Analyst 已经完成推理，Verifier 已经完成质量检查。Reporter 如果再次调用模型，可能
改写事实、丢失引用或增加未经验证的新结论。

**当前方案**

`competitive_analysis_agent/reporter.py` 中的 `Reporter.render()` 使用普通 Python
按固定模板渲染结构化数据。

**为什么这样选择**

输出稳定、无需额外 Token，并且同一输入可以得到完全相同的 Markdown。

**替代方案**

让 LLM 根据分析结果撰写自然语言报告。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 确定性模板 | 可重复、易测试、引用稳定 | 文风相对固定 | 结构化业务报告 |
| LLM 写报告 | 表达灵活 | 可能改写事实、成本和波动更高 | 创意叙事或人工复核后的润色 |

**什么时候考虑切换**

如果未来需要多种品牌语气，可以让模型只润色无事实风险的摘要，但引用表、警告和
数据表仍应由确定性代码生成。

**面试回答参考**

Reporter 是展示层，不是新的推理层。上游已经生成并验证结构化结论，因此我使用
确定性模板，保证相同输入产生相同报告，避免模型在最后一步新增事实或破坏引用。

### 决策 2：为什么引用要直接链接到 Evidence URL？

**问题背景**

只显示 `E1` 对读者没有足够价值，读者需要能快速打开来源并自行判断。

**当前方案**

`format_citations()` 把 claim 中的 Evidence ID 渲染为可点击链接，
`render_evidence_sources()` 在来源章节列出完整元数据。

**替代方案**

只在报告末尾显示普通 URL，或者只显示 Evidence ID。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 行内引用 + 来源表 | 追溯路径短，适合人工复核 | Markdown 稍长 | 研究和决策报告 |
| 只显示 ID | 输出简洁 | 必须另外查表 | 内部机器数据 |
| 只显示 URL | 可访问来源 | claim 与来源对应关系不清 | 简单资料列表 |

**什么时候考虑切换**

如果报告导出为 PDF 或 HTML，可以使用脚注或悬浮引用，但 Evidence ID 到来源的映射
仍应保留。

**面试回答参考**

我同时保留 Evidence ID 和可点击来源。ID 用于程序追踪，URL 方便人工复核；来源表
补充产品、主题、来源类型和采集时间，使报告结论能够回到原始证据。

### 决策 3：为什么验证失败仍然生成报告？

**问题背景**

第二次 Verifier 仍可能失败。如果完全不输出，用户看不到已经完成的调研和具体问题；
如果像正常结果一样输出，又可能误导用户。

**当前方案**

`render_verification_section()` 继续生成报告，但在顶部保留验证警告和结构化
issue 表。工作流在重试额度用完后仍进入 Reporter。

**替代方案**

验证失败时直接抛出异常，不生成任何报告。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 带警告的降级报告 | 保留部分成果和修复信息 | 调用方必须正确展示警告 | 研究辅助工具 |
| 失败即无报告 | 不会误把失败结果当成功 | 丢失可用信息，难以定位问题 | 强合规自动决策 |

**什么时候考虑切换**

如果报告会直接驱动自动交易、审批等高风险动作，应改为验证失败即阻止正式发布。

**面试回答参考**

当前产品是人工阅读的研究工具，所以验证失败时保留一份明确标红的降级报告，让用户
看到已有证据和具体问题。系统不会把它标记为通过，也不会隐藏 Verifier issues。

### 决策 4：为什么 Reporter 输入还要重新检查 Evidence 引用？

**问题背景**

上游 Analyst 和 Verifier 已经执行过校验，但 Reporter 也可能被独立调用，或者未来
从文件、数据库读取结构化结果。如果直接渲染，不存在的 ID 会形成失效或误导链接。

**当前方案**

`ReporterInput.validate_report_sources()` 检查产品顺序、Evidence ID 唯一性，以及
分析和画像中的全部引用是否存在。

**为什么这样选择**

Reporter 是数据离开系统、进入人类阅读界面的最后边界。小量重复校验能换来稳定的
来源链接，并使 Reporter 脱离 LangGraph 使用时仍然可靠。

**替代方案**

完全信任上游，只在渲染时使用字典查找；找不到时忽略引用。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Reporter 边界再校验 | 失败明确，不会生成断链报告 | 有少量重复检查 | 可独立调用的报告组件 |
| 完全信任上游 | 代码更短 | 外部数据可能生成错误报告 | 严格封闭且只有单一路径的内部函数 |

**什么时候考虑切换**

即使未来改成 HTML/PDF 渲染，也建议保留统一输入校验；可以把校验提取成共享发布契约，
而不是删除。

**面试回答参考**

虽然上游已经校验引用，但 Reporter 是发布边界，也可能独立消费持久化数据。我在
这里重新确认所有 Evidence ID 都能映射到来源，避免最终报告出现无法追溯的引用。

## 4. 异常与边界情况

- 产品画像顺序与分析产品顺序不同：`ReporterInput` 拒绝。
- Evidence ID 重复：`ReporterInput` 拒绝，避免一个引用对应多个来源。
- 分析或画像引用不存在的 Evidence ID：拒绝生成报告。
- 某分析章节为空：渲染明确的“未生成”或“资料不足”说明。
- 定位、目标用户、功能或价格为空：显示“未提供”，不猜测。
- 价格存在方案名但缺少金额或周期：分别显示“价格未提供”和“计费周期未提供”。
- Researcher 部分失败：成功结果仍渲染，错误进入数据限制章节。
- Verifier 未通过：生成带顶部警告的降级报告，issues 不被隐藏。
- Verifier issue 引用了不存在的 ID：该 ID 显示为“未找到来源”，不会生成假链接。
- 输出目录不存在：`Reporter.write()` 创建父目录后写入 UTF-8 Markdown。

## 5. 真实 LLM 测试

- 是否属于 LLM 相关阶段：否
- 配置来源：`F:\大模型应用开发学习\competitive-analysis-agent\.env.example`
- 测试命令：不适用
- 真实调用的组件：无
- 验证的输出契约：由确定性 fixture 测试覆盖
- 测试结果：不适用；本阶段没有模型调用
- 耗时：不适用
- 失败原因：无

## 6. 与课程 Notebook 的对应

### Notebook source

- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\codelabs\day-4b-agent-evaluation.zh-CN.ipynb`
- `F:\大模型应用开发学习\3.Google_and_Kaggle\4-Day\whitepaper\Agent Quality 中文解析.md`

### Relevant sections

- `3.2：创建你的第一个“完美”测试用例`
- `4.2：创建测试用例`
- `4.4：分析示例评估结果`
- `第一步：端到端评估`
- `Human-in-the-Loop`

### Core concept

Agent 的内部结构化结果和人类最终阅读的输出承担不同职责。内部对象要方便程序校验、
路由和回归测试；最终报告要让人快速理解结论、查看来源、发现限制并判断是否可信。

固定的 golden case 可以保护输出结构。失败时只说“不通过”不够，还需要展示 diff、
问题位置或具体原因。Human-in-the-Loop 的价值也依赖可访问的来源和上下文，否则人
无法真正复核 Agent。

### How it appears in this project

- `ReporterInput` 是发布前的结构化契约；
- `Reporter.render()` 把内部对象转换成人类可读 Markdown；
- `docs/sample-report.md` 相当于一个固定 golden output；
- `test_rendered_report_matches_inspectable_sample` 防止章节和格式静默回归；
- Evidence 链接、数据限制和验证 issue 为人工判断提供上下文；
- Reporter 不改变 Analyst 的结论，只改变展示形态。

### ADK vs LangGraph

- Notebook 使用 ADK Web UI 保存 session 为 evalset，并通过 `adk eval` 比较结果；
- 本项目使用 pytest fixture 和固定 Markdown 文件进行回归；
- LangGraph 只负责在 Verifier 后调度 Reporter；
- Markdown 渲染本身是框架无关的普通 Python，不需要 ADK 或 LangGraph API。

### Intentionally postponed

- Reviewer UI 和人工批注；
- 用户满意度等端到端质量指标；
- HTML/PDF 视觉快照测试；
- 线上失败自动转成评测用例；
- 多人标注一致性和审批工作流。

## 7. 理解问题与参考思路

### 问题 1：为什么 Reporter 不适合再调用模型？

**参考思路：**

- 推理已由 Analyst 完成；
- 验证已由 Verifier 完成；
- 再调用模型会引入新事实、格式波动和额外成本；
- Reporter 只负责结构化数据到展示文本的转换。

### 问题 2：Evidence ID 如何变成可点击来源？

**参考思路：**

- `ReporterInput` 先验证 ID 存在；
- `evidence_by_id` 建立 ID 到 Evidence 的映射；
- `format_citations()` 使用 Evidence URL 生成 Markdown 链接；
- 来源表继续展示标题、产品、主题、类型和采集时间。

### 问题 3：为什么缺失信息要明确显示，而不是省略？

**参考思路：**

- 省略容易被读者理解为“没有该功能”或“免费”；
- “未提供”说明资料不足，不代表事实不存在；
- 数据限制章节集中提示研究覆盖范围。

### 问题 4：为什么验证失败警告必须保留在报告中？

**参考思路：**

- Reporter 不能把未通过结果包装成正式结论；
- issue 路径和建议动作方便人工修复；
- 当前是研究辅助工具，所以保留降级输出比完全丢弃更有价值。

## 8. 面试追问清单

- 如果报告格式增加 HTML 和 PDF，哪些逻辑应复用？
- 如果移除 Reporter 输入校验，会出现什么引用风险？
- 验证失败时应该阻止输出还是降级输出？
- 哪些结论来自测试，哪些只是工程判断？

## 9. 下一阶段衔接

Stage 10 将在 Streamlit 中收集输入、运行图、展示并下载本阶段生成的 Markdown。

## 10. 变更记录

- 2026-06-22 最新报告显示免费方案被展示为 `Free / monthly`，容易把“每月使用限制”
  误读成“按月计费”。Reporter 现在对 `Free` / `$0` 价格省略计费周期，同时不会把免费方案
  缺少 billing cycle 记录为数据限制；未知价格的方案仍显示“价格未提供 / 计费周期未提供”。
  新增回归：`test_free_pricing_omits_billing_cycle_in_overview`。
- 2026-06-23 最新报告通过验证后仍有展示噪声：`$10 per seat/month / per month`、
  `Workers：价格未提供 / Beta`、以及 `Custom pricing` 被记录为缺少计费周期。
  Reporter 现在复用共享价格规则：价格里已有周期时不重复展示，`Beta` 显示为未知周期，
  `Custom pricing` 不产生缺失计费周期的数据限制。新增回归：
  `test_pricing_overview_filters_duplicate_and_invalid_billing`。
