import pytest

from competitive_analysis_agent.ui_service import (
    AnalysisRequest,
    run_analysis,
)


@pytest.mark.live_llm
@pytest.mark.live_search
def test_ui_service_runs_real_model_workflow() -> None:
    """UI 等价路径应通过真实搜索和模型生成可下载报告。"""

    request = AnalysisRequest(
        target_product="Notion",
        competitors=["Confluence"],
        dimensions=["features", "pricing"],
        official_domains_by_product={
            "Notion": ["notion.so"],
            "Confluence": ["atlassian.com"],
        },
    )
    reported_stages: list[str] = []

    result = run_analysis(
        request,
        progress_callback=reported_stages.append,
    )

    assert result.verification_result.passed
    assert "# 竞品分析报告" in result.final_report
    assert "## 资料来源" in result.final_report
    assert reported_stages[-1] == "reporter"
    assert result.stage_history == reported_stages
    assert len(result.evidence) >= 2
