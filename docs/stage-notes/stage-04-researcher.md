# Stage 4：Researcher 设计记录

## 1. 阶段目标

把 Planner 生成的 `ResearchTask` 逐个交给搜索适配器，并把搜索结果转换为带任务上下文的 `Evidence`。

本阶段只负责：

- 遍历研究任务
- 调用 `SearchAdapter`
- 为证据分配稳定 ID
- 按产品、主题和 URL 去重，同时保留跨主题上下文
- 记录单个任务的错误并继续运行

本阶段不负责特征提取、价格解析、竞品比较或报告生成。

## 2. 数据流

```text
ResearcherInput
  ├─ tasks: list[ResearchTask]
  ├─ official_domains_by_product
  └─ max_results_per_task
          |
          v
逐任务创建 SearchRequest
          |
          v
SearchAdapter.search()
          |
          v
错误 -> ResearchError，继续下一个任务
成功 -> 按产品/主题/URL 去重 -> Evidence(E1, E2, ...)
          |
          v
ResearchResult(evidence, errors)
```

## 3. 设计决策

### 决策一：由 Researcher 分配 Evidence ID

**当前方案：** Researcher 在接收搜索结果、完成去重后，按保留顺序生成 `E1`、`E2` 等 ID。

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| Researcher 分配顺序 ID | 简单、稳定，后续 Agent 可直接引用证据 | ID 只在本次运行内稳定 |
| SearchAdapter 分配 ID | 搜索层直接产出可引用对象 | 搜索层不知道哪些结果最终会被去重和采用 |
| 使用 URL 哈希作为 ID | 跨运行更稳定 | 可读性差，URL 变化会生成新 ID |

**切换条件：** 当项目需要跨运行保存、合并和追踪证据时，改为数据库 ID 或内容哈希。

**面试回答：** Evidence ID 应在“证据正式进入工作流”时创建。Researcher 正好掌握去重后的最终顺序，能避免重复结果或失败任务占用 ID。

### 决策二：单个任务失败时继续处理

**当前方案：** 每个失败任务生成一条 `ResearchError`，不抛出异常终止整批研究。

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| 记录错误并继续 | 保留部分成功结果，适合外部搜索这种不稳定依赖 | 调用方必须同时检查 evidence 和 errors |
| 首次失败立即终止 | 控制流简单 | 一个超时会丢掉全部可用结果 |
| 自动无限重试 | 可能提高成功率 | 容易拖慢或卡住工作流 |

**切换条件：** 后续可在 SearchAdapter 增加有限次数、带退避的重试，但最终仍应返回局部错误。

**面试回答：** 搜索是典型的不稳定外部服务。竞品分析允许部分完成，因此用结构化错误保留失败上下文，同时让成功任务继续产生证据。

### 决策三：为什么去重 key 使用产品、主题和规范化 URL？

**当前方案：** 同一次 Researcher 运行中，`Researcher.research()` 使用
`build_evidence_deduplication_key()` 生成
`(product_name, topic, normalized_url)`。同一产品、同一 topic、同一 URL
只保留第一次出现；但同一页面如果先作为 `features` 被收录，后续又作为
`pricing` 命中，会创建 topic-specific Evidence，保留 pricing 的
`raw_content` 和 task context。

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| 产品 + topic + URL 去重 | 保留价格、功能等不同研究上下文；避免 pricing raw_content 被 features 结果吞掉 | 同一 URL 可能以不同 topic 出现多次 |
| 全局 URL 去重，保留首次上下文 | 证据数量更少，后续 LLM 成本最低 | 同一页面关联多个主题的信息会丢失 |
| Evidence 保存多个任务上下文 | 信息最完整，一个 Evidence 可表达多 topic | 当前 schema、报告引用和测试复杂度更高 |

**切换条件：** 当同一页面经常支持多个 topic，且 Evidence 数量明显增加到影响模型输入时，
可以把 Evidence 与任务上下文改成多对多关系，或在 Researcher 中合并同 URL 的多 topic
上下文。

**面试回答：** 早期全局 URL 去重虽然省 token，但会让 features 命中的页面吞掉后续
pricing 命中的同 URL，导致价格正文无法进入 Extractor。我把去重 key 改成
产品、主题和 URL，保留每个研究任务的语境。同一 topic 内仍然去重，所以不会无限复制
完全相同的证据。未来如果证据量变大，可以升级成一个 Evidence 绑定多个 topic 的模型。

### 决策四：保持输入顺序，并注入时钟

**当前方案：** 串行处理任务和结果；每次运行只读取一次可注入时钟，所有证据共享该采集时间。

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| 串行顺序 + 注入时钟 | ID 和测试结果完全确定，代码适合初学者理解 | 搜索速度较慢 |
| 并发搜索，完成即写入 | 速度快 | 结果顺序和 ID 可能随网络时序变化 |
| 并发搜索，最后按输入排序 | 兼顾速度和稳定性 | 实现与错误处理更复杂 |

**切换条件：** 当真实任务数量增加、串行搜索成为性能瓶颈时，再引入并发并在汇总阶段恢复确定顺序。

**面试回答：** 在原型阶段，可测试性和可解释性比吞吐量重要。注入时钟还能让单元测试不依赖系统当前时间。

## 4. 实现结果

- 新增 `competitive_analysis_agent/researcher.py`：
  - `ResearcherInput` 约束任务、官方域名映射和单任务结果上限；
  - `ResearchError` 保存失败任务的产品、主题、查询、错误码和消息；
  - `ResearchResult` 同时返回 `evidence` 与 `errors`；
  - `Researcher.research()` 串行执行任务、调用 `SearchAdapter`、按产品/主题/URL 去重并分配 ID；
  - `build_evidence_deduplication_key()` 保留同一 URL 在不同 topic 下的证据上下文；
  - `utc_now()` 是默认生产时钟，测试可注入固定时钟。
- 新增 `tests/fixtures/researcher_search_results.json`，提供固定成功结果和跨任务重复 URL。
- 新增 `tests/test_researcher.py`，覆盖：
  - 固定输入重复运行得到相同输出；
  - Evidence ID 顺序稳定；
  - Evidence 保留产品与主题上下文；
  - 同一 URL 跨 topic 保留上下文；
  - 同一产品、同一 topic、同一 URL 仍然去重；
  - 单任务超时不删除成功证据；
  - 搜索成功但无结果时生成 `no_results` 错误。
- 更新 README，项目当前完成到 Stage 4。

实际数据流：

```text
ResearchTask
  -> SearchRequest
  -> SearchAdapter
  -> SearchResponse
  -> 按产品/主题/URL 去重后的 Evidence / ResearchError
  -> ResearchResult
```

## 5. 验证结果

- `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py -q`：
  Stage 4 的 **10 个测试通过**。
- `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py -q`：
  关键链路 **80 个测试通过**。
- `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`：
  全项目离线回归 **152 个测试通过，7 个真实用例默认排除**。
- 测试不访问网络，不需要 LLM 或搜索服务 API Key。
- 固定样例产生 5 条证据：
  - `E1`、`E2`：Atlas Notes / features；
  - `E3`：Atlas Notes / pricing，对应同一 overview URL 的 pricing 上下文；
  - `E4`：Atlas Notes / pricing，对应价格页；
  - `E5`：Beacon Docs / features。
- Beacon Docs / pricing 的超时被单独记录，前面 5 条成功证据仍然保留。

## 6. Notebook 知识映射

### 主要 Notebook

`F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2a-agent-tools.zh-CN.ipynb`

相关章节：

- `2.1：构建自定义函数工具`
- `3.3：Agent Tools 与 Sub-Agents：有什么区别？`
- `第 4 节：ADK 工具类型完整指南`

对应知识：

- Notebook 中普通 Python 函数可以作为工具，但需要清晰类型、文档和结构化错误。
- 本项目的 `SearchAdapter.search()` 就是 Researcher 使用的工具边界：
  输入 `SearchRequest`，输出统一的 `SearchResponse`。
- Notebook 中 Agent Tool 调用完成后，控制权回到原 Agent；本项目也是
  Researcher 调用搜索后继续决定去重、生成 Evidence 或记录错误。
- 它不是 sub-agent 控制权转移：SearchAdapter 不负责后续任务，也不知道完整工作流。

### 辅助 Notebook

`F:\大模型应用开发学习\3.Google_and_Kaggle\2-Day\codelabs\day-2b-agent-tools-best-practices.zh-CN.ipynb`

相关章节：

- `第 4 节：构建工作流`
- `4.1：在工作流中处理 Events`
- `4.4：工作流函数——把所有内容串起来！`

对应知识：

- Notebook 的工作流会收集并检查 events，再决定完成、暂停或恢复。
- 本项目暂时没有 ADK Events，但 Researcher 也必须检查每个
  `SearchResponse`，区分成功、错误和空结果。
- `run_shipping_workflow()` 把多个步骤串起来；本阶段的
  `Researcher.research()` 同样是一个确定性编排函数。

### 与 Notebook 的差异

- Notebook 使用 ADK Agent、Runner、Session 和 Events。
- 当前项目只用普通 Python 类和 Pydantic，尚未接入 LangGraph。
- 当前搜索任务串行执行，没有人工审批、暂停恢复和持久化 session。
- 这种简化让工具边界先被测试稳定，再在后续阶段加入图工作流。

## 7. Understanding Questions

### 问题一：ResearchTask 如何变成 Evidence？

Researcher 读取任务中的 `query` 调用 SearchAdapter，再把搜索结果与任务中的
`product_name`、`topic` 组合成 Evidence。因此 Evidence 不只是网页结果，还知道
“这条网页是为哪个产品、哪个研究主题收集的”。

### 问题二：为什么错误要放在 ResearchResult，而不是直接抛异常？

因为一次竞品分析包含多个独立搜索任务。某个价格查询超时，不代表其他功能查询
没有价值。结构化错误可以保留失败上下文，同时让调用方使用已成功的证据。

### 问题三：为什么 Evidence ID 在 Researcher 中创建？

SearchAdapter 只负责统一搜索结果，不知道结果是否会被其他任务重复引用。
Researcher 完成产品/主题/URL 去重后才知道最终证据集合，因此此时分配 ID 不会产生空号，
也不会让同一 topic 下的重复结果占用新 ID。

### 问题四：为什么同一 URL 在不同 topic 下可以出现两次？

同一网页可能既包含功能信息，也包含价格表。Researcher 不解释网页事实，但必须保留
“这条证据是为哪个 topic 收集的”这个上下文。pricing 任务还会请求和裁剪
`raw_content`，如果被 features 任务的同 URL 提前吞掉，Extractor 就拿不到价格正文。

### 问题五：为什么现在不并发搜索？

串行执行能自然保持任务与结果顺序，使 `E1`、`E2` 的含义在测试中稳定。
当前阶段先保证正确性和可解释性；任务规模变大后，可并发调用并在汇总时按原始
任务顺序排序。

## 8. 下一阶段

Stage 5：Extractor。它会读取 Evidence 并提取结构化产品信息，而不是继续扩大 Researcher 的职责。

## 9. 变更记录

- 2026-06-24 根据 `docs/agent-repair-guide-api-pricing-scope.md` 的 P1 建议，修复
  Researcher 全局 URL 去重导致的 topic 丢失问题。旧方案只按 URL 去重，如果
  `features` 任务先命中某个 URL，后续 `pricing` 任务命中同一 URL 时会被跳过，
  pricing 的 `raw_content` 无法进入 Extractor。
- 本次新增 `build_evidence_deduplication_key()`，将去重 key 改为
  `(product_name, topic, normalized_url)`。这样同一产品同一 topic 的重复 URL 仍然只保留一次，
  但同一 URL 可以分别作为 features 和 pricing Evidence 进入后续流程。
- 新增回归测试：
  `test_same_url_is_kept_for_different_topics` 和
  `test_same_url_is_deduplicated_within_same_product_and_topic`。同时更新固定样例断言，
  Atlas Notes 的 overview URL 现在会分别保留 features 和 pricing 两个上下文。
- 验证命令：
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py -q`
  结果 `10 passed`；
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest tests\test_researcher.py tests\test_extractor.py tests\test_analyst.py tests\test_verifier.py -q`
  结果 `80 passed`；
  `C:\Users\zoujunkai\miniconda3\python.exe -m pytest -q`
  结果 `152 passed, 7 deselected`。
- 真实 LLM 测试：本次属于 Stage 4 Researcher 的确定性调研编排修复，不修改 prompt、
  结构化模型输出或真实模型路径，因此不需要 live LLM 验收。
- Notebook 对应：Day 2a 的工具章节说明工具调用后控制权回到调用方；Day 2b 的 workflow
  章节说明工作流必须检查 events 和中间结果。本项目里的对应点是：`SearchAdapter`
  只返回统一搜索结果，Researcher 负责把工具结果连同产品、topic、错误和 Evidence ID
  一起写入工作流状态。去重 key 选择属于这个编排层的确定性规则，而不是模型推理。
- 面试回答补充：我没有完全取消去重，因为重复 Evidence 会增加模型输入成本；也没有马上把
  Evidence 改成多 topic 模型，因为当前 schema 和报告引用还不需要那种复杂度。按产品、主题和
  URL 去重是一个中间方案：保留对提取有用的上下文，同时保持实现简单、可测试。
