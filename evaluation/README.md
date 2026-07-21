# 评测

本目录保留两层互补评测：默认运行的三个固定案例负责快速、确定性的质量回归；DeepEval 负责可选的语义质量评审。固定案例实际运行完整 LangGraph，但使用 Fake Model 与固定搜索，因此不消耗 API 配额。

## 固定质量案例

- `fully_comparable`：同层级产品资料完整。
- `cross_product_line_contamination`：消费端套餐混入后必须被排除。
- `insufficient_in_scope_data`：核心维度缺资料时必须披露并阻止“比较可用”。

```powershell
.\.venv\Scripts\python.exe -m competitive_analysis_agent.evaluation
```

结果写入 `docs/evaluation/evaluation-results.json` 和 `evaluation-results.md`。指标包括范围一致率、引用有效率、核心维度覆盖率、价格上下文完整率、排除准确率、建议可行动率和运行耗时。

## DeepEval 指标

- `FaithfulnessMetric`：报告中的说法是否由本次检索到的 Evidence 支持。
- `G-Eval`：报告是否完成了指定的产品与维度比较，并且对未知信息保持保守。

Judge 与 Agent 共用 `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`；工作流还需要 `TAVILY_API_KEY`。密钥不会写入本目录。

## DeepEval 运行

```powershell
.\.venv\Scripts\deepeval.exe test run evaluation\test_deepeval_competitive_analysis.py
```

这会调用一次真实搜索、Agent 工作流和 DeepEval Judge，会消耗 API 配额。阈值暂定为 `0.7`；先积累失败案例，再调整阈值或增加指标。
