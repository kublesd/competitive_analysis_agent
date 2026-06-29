# Stage 12：真实搜索 Provider

## 1. 本阶段目标

把 UI 从固定 Evidence 演示升级为真实网页搜索：

```text
用户输入产品、维度、官方域名
  -> Planner 生成 ResearchTask
  -> TavilySearchProvider 调用真实搜索 API
  -> SearchAdapter 规范化、去重并分类来源
  -> Researcher 生成 Evidence
  -> Extractor / Analyst / Verifier / Reporter
```

本阶段只接入一个同步搜索 Provider，不实现异步搜索、缓存或第二家供应商。pricing
任务会额外请求网页正文，并在 Researcher 中裁剪成短价格片段；其他主题仍只使用搜索摘要。

## 2. 实现结果

- 在 `competitive_analysis_agent/search.py` 新增 `TavilySearchProvider`，调用官方
  `POST https://api.tavily.com/search`，把 `title/url/content` 映射为项目的
  `ProviderSearchResult`。
- Tavily 请求固定使用 `search_depth="basic"`，关闭 answer 和 images。普通任务不请求
  raw content；pricing 任务通过 `SearchRequest.include_raw_content=True` 请求网页正文。
- `Researcher` 会从 pricing raw content 中裁剪套餐、价格、计费周期和限制相关短片段，
  写入 `Evidence.raw_content`，并附加到 `snippet` 供 Extractor 使用。
- 在 `competitive_analysis_agent/application_workflow.py` 新增
  `create_application_workflow_components()`，为 UI 装配真实模型和真实搜索。
- `competitive_analysis_agent/ui_service.py` 不再限制固定虚构产品，支持任意产品、
  五种分析维度和显式官方域名。
- `competitive_analysis_agent/streamlit_app.py` 默认使用 Notion 与 Confluence，并
  默认选择 features、pricing、positioning、target_users。
- `.env.example` 中的 `TAVILY_API_KEY` 会由 `load_live_settings()` 加载。
- `start.ps1` 会在双击启动前检查 Tavily 配置。
- `Settings` 的两个 Key 字段设置为 `repr=False`，避免错误输出泄露密钥。
- 离线验证：`83 passed, 7 deselected`。
- 真实 Tavily 验收：未通过，唯一阻塞是当前 `.env.example` 中
  `TAVILY_API_KEY` 为空。

## 3. 设计决策

### 决策 1：为什么 Tavily 原始响应不能直接进入 Researcher？

**问题背景**

Tavily 使用 `content` 字段表示摘要，其他供应商可能使用 `snippet`。如果 Researcher
直接依赖这些字段，替换供应商会影响工作流业务层。

**当前方案**

`TavilySearchProvider.search()` 只把 Tavily 的 `content` 映射为 `snippet`；
`SearchAdapter.search()` 继续负责 URL 规范化、去重、结果上限和官方来源分类；
`Researcher.research()` 不包含任何 Tavily 字段名。

**为什么这样选择**

保留现有 `SearchProvider -> SearchAdapter -> SearchResponse` 边界，让供应商变化只
影响最外层。

**替代方案**

让 Researcher 直接调用 Tavily SDK。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 独立 Provider | 易测试、易替换、业务层稳定 | 多一层映射 | 当前项目 |
| Researcher 直连 SDK | 代码短 | 强耦合，离线测试困难 | 一次性脚本 |

**什么时候考虑切换**

如果项目永远只使用 Tavily 且没有单元测试要求，直连可以更短，但不适合当前多阶段工程。

**面试回答参考**

我让 Tavily 只负责把外部结果映射成项目统一结构，Researcher 不知道供应商字段。这样
搜索实现可替换，超时、错误和 URL 规范化也能集中测试。

### 决策 2：为什么第一版继续使用同步搜索？

**问题背景**

每个产品和维度都会产生独立搜索任务，并行可以缩短总耗时。

**当前方案**

`Researcher` 继续按 Planner 任务顺序同步调用 `SearchAdapter`。Evidence ID 因此仍按
稳定任务与结果顺序生成。

**为什么这样选择**

当前任务规模小，同步流程更容易观察 API 配额、错误归属和 Evidence ID 顺序。

**替代方案**

使用 asyncio 或线程池并发请求。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 同步 | 简单、稳定、顺序确定 | 多任务总耗时较长 | MVP |
| 并发 | 延迟更低 | 配额、重试和稳定编号更复杂 | 大批量研究 |

**什么时候考虑切换**

当实测搜索耗时成为主要瓶颈，并且供应商配额允许并发时再切换。

**面试回答参考**

我先保留同步搜索，因为当前每次任务数量有限，而且顺序执行让错误处理和 Evidence ID
更可预测。等有延迟数据后再决定是否并发。

### 决策 3：为什么让用户提供官方域名？

**问题背景**

产品名称不能可靠推导官方网站，自动猜测可能把仿冒或同名网站标成官方来源。

**当前方案**

`AnalysisRequest.official_domains_by_product` 保存用户输入。UI 使用
`parse_official_domains()` 解析 `产品=域名`；Tavily 请求把域名传入
`include_domains`，SearchAdapter 再用同一域名执行确定性来源分类。

**为什么这样选择**

显式域名既可用于 Tavily 限定来源，也可供 SearchAdapter 做确定性官方来源分类。

**替代方案**

让模型或搜索结果自动判断官方网站。

**方案对比**

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 用户显式输入 | 可控、可审计 | 多一个输入字段 | 研究工具 |
| 自动推断 | 使用方便 | 可能误判和污染证据 | 有域名验证服务时 |

**什么时候考虑切换**

如果后续建立经过校验的产品目录，可从目录自动填充官方域名。

**面试回答参考**

官方来源身份属于确定性配置，不适合让模型猜。我允许用户显式填写域名，再由搜索层
限定和分类来源。

## 4. 异常与边界情况

- Tavily socket 超时转换为 Python `TimeoutError`，再由 SearchAdapter 转为
  `SearchError(code="timeout")`。
- HTTP、网络和响应契约错误转换为不包含 Key 和响应正文的
  `TavilySearchProviderError`。
- 单条缺字段或 URL 无效的搜索记录会被跳过，其他有效记录继续使用。
- 顶层响应没有 `results` 列表时，整个搜索任务返回 provider error。
- 官方域名配置包含请求以外的产品，或不使用 `产品=域名` 格式时，在 UI 输入边界拒绝。
- 某个搜索任务失败后，Researcher 仍记录错误并继续其他产品和维度。
- `Settings.__repr__` 不显示 LLM 与 Tavily Key。
- Tavily 请求格式依据官方 Search API：
  <https://docs.tavily.com/documentation/api-reference/endpoint/search>

## 5. 真实外部服务测试

- 配置来源：`.env.example`
- 所需变量：`TAVILY_API_KEY`
- 测试命令：
  `python -m pytest -o addopts='' tests/test_live_search.py -q`
- 验证契约：返回至少一条能通过 URL 规范化的 Notion 官方定价结果，来源分类为
  `official`，并且请求 raw content 后至少一条结果包含价格正文。
- 结果：通过，`1 passed`
- 耗时：2.71 秒
- 失败类别：无

## 6. 与课程 Notebook 的对应

- Notebook：
  `F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2a-agent-tools.zh-CN.ipynb`
- 相关章节：
  `为什么 Agents 需要工具？`、`第 2 节：什么是自定义工具？`、
  `2.1：构建自定义函数工具`、`2.2：如何定义一个工具？`
- 支持材料：
  `day-2b-agent-tools-best-practices.zh-CN.ipynb` 的
  `第 5 节：总结 - 高级工具的关键模式`
- 通用知识点：模型本身不能访问实时外部世界；工具需要清晰输入、类型约束、统一返回
  结构和可处理错误。
- 在本项目中的实现：`SearchRequest` 是输入契约，`SearchResponse` 是统一成功/错误
  结果，`TavilySearchProvider` 是外部连接，`SearchAdapter` 是稳定项目边界。
  pricing raw content 仍通过同一边界进入 Evidence，不让 Extractor 直接抓网页。
- ADK 与 LangGraph/本项目的差异：ADK 可以把 Python 函数注册进 `tools=[]`，由模型
  选择调用；本项目由 LangGraph 的 Researcher 节点按 Planner 任务显式调用搜索工具，
  调用顺序更确定。
- 本阶段有意简化的内容：不使用 MCP、不并发搜索、不暂停恢复、不缓存，也不做通用网页抓取。
  目前只为 pricing 任务抓取并裁剪价格相关正文。

## 7. 理解问题与参考思路

### 问题 1：为什么仍保留 SearchAdapter？

**参考思路：**

- Tavily 负责联网，Adapter 负责项目内统一语义。
- 更换供应商时 Researcher、Evidence 和测试不需要改。

### 问题 2：为什么官方域名由用户输入？

**参考思路：**

- 产品名与域名不是可靠的一对一关系。
- 显式配置可审计，也能避免仿冒域名被标为官方。

### 问题 3：为什么真实搜索还需要 Fake Provider？

**参考思路：**

- Fake 测试稳定验证分支、错误和格式，不消耗 credits。
- 真实测试只确认鉴权、端点和供应商响应兼容，二者不能互相替代。

## 8. 面试追问清单

- 为什么真实搜索仍需要 Fake Provider 测试？
- 为什么官方域名不能由模型直接猜？
- 搜索结果数量怎样影响 Token 成本和报告质量？

## 9. 下一阶段衔接

当前代码实现已完成，并且真实 Tavily 搜索验收已通过。后续可根据实测延迟决定是否
增加缓存、异步搜索或更稳定的网页正文抽取器。

## 10. 变更记录

- 2026-06-22 真实 Notion / Confluence 报告显示搜索成功但主题错位：
  `positioning` 命中 Notion 模板市场和 Confluence API `Position` 类，
  `target_users` 命中模板或开发者文档。`Researcher` 新增
  `build_focused_search_query()`，在调用搜索前把内部 topic 转换成官网常用表达：
  pricing 使用 `pricing plans price`，positioning 使用
  `product overview workspace teams business`，target_users 使用
  `use cases customers teams enterprise small business`。Planner 合约保持不变。
- UI 真实路径的每个任务搜索结果数从 2 提到 3，降低好页面排在第三位时被截断的概率。
  代价是 Tavily credit 和后续模型输入略有增加。
- `TavilySearchProvider` 对 timeout 和网络请求失败增加一次重试；HTTP 状态错误、坏 JSON
  和响应契约错误仍然快速失败。这样可以减少单个搜索任务因瞬时断网丢失整块 Evidence。
- 页面默认官方域名改为 `Notion=notion.com,notion.so` 和
  `Confluence=atlassian.com`，避免 Notion 新官网域名缺失导致定价页不稳定。
- 聚焦测试：`python -m pytest tests/test_search.py tests/test_researcher.py -q` 通过。
  完整离线测试：`93 passed, 7 deselected`。
- 2026-06-22 最新报告显示：Notion 和 Confluence pricing 官方页面已被搜到，但
  Plus、Business、Standard、Premium 等价格仍缺失。根因是 Tavily basic 搜索摘要没有稳定
  包含价格表正文，而 Extractor 被要求只能使用 Evidence。
- 本次修复新增 `SearchRequest.include_raw_content`、`ProviderSearchResult.raw_content`、
  `SearchResult.raw_content` 和 `Evidence.raw_content`。`Researcher` 仅对 `topic=pricing`
  打开 raw content，并用 `extract_pricing_excerpt()` 裁剪出套餐、价格、计费周期和限制相关
  片段，避免把整页正文塞进模型。
- `EXTRACTOR_SYSTEM_PROMPT` 已说明：pricing 可以同时使用 Evidence 的 `snippet` 和
  `raw_content` 中明确出现的价格页正文片段，仍不得使用 Evidence 外的信息。
- 离线测试：`python -m pytest tests/test_search.py tests/test_researcher.py tests/test_extractor.py -q`，
  结果 `30 passed`；完整离线测试 `107 passed, 7 deselected`。
- 真实 Tavily 测试：`python -m pytest -o addopts= tests/test_live_search.py -q`，
  结果 `1 passed`，确认 Notion 官方 pricing 搜索结果可返回 raw content 且包含价格符号。
- 真实窄链路验证 pricing Researcher + Extractor 成功提取：Notion Free `$0`、Plus `$10`、
  Business `$20`、Enterprise `Custom pricing`；Confluence Free `$0`、Standard
  `$5.42 per user / month`、Premium `$10.44 per user / month`。完整端到端复跑在 Analyst
  模型调用处遇到外部服务错误，分类为 Analyst 模型服务连接失败，不是 pricing 搜索或提取失败。
