import pytest

from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_workflow import (
    create_live_workflow_components,
)
from competitive_analysis_agent.schemas import MarketDefinition
from competitive_analysis_agent.ui_service import (
    AnalysisRequest,
    run_analysis,
)


@pytest.mark.live_llm
def test_ui_service_result_summary_with_real_models() -> None:
    """固定搜索加真实模型应通过 UI 服务路径返回范围和验证摘要。"""

    market_definition = MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=["features"],
    )
    request = AnalysisRequest(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        market_definition=market_definition,
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
    )
    components = create_live_workflow_components(load_live_settings())

    result = run_analysis(request, components=components)

    assert result.market_definition == market_definition
    assert result.stage_history[-1] == "reporter"
    assert result.evidence_scope_counts.in_scope == 2
    assert result.evidence_scope_counts.out_of_scope == 0
    assert result.evidence_scope_counts.uncertain == 0
    verification_result = result.verification_result
    assert isinstance(verification_result.citations_valid, bool)
    assert isinstance(verification_result.scope_consistent, bool)
    assert isinstance(verification_result.comparison_usable, bool)
    assert verification_result.passed == (
        verification_result.citations_valid
        and verification_result.scope_consistent
        and verification_result.comparison_usable
    )
    assert "## 分析范围" in result.final_report


@pytest.mark.live_llm
@pytest.mark.live_search
def test_ui_service_runs_real_model_workflow() -> None:
    """UI 等价路径应通过真实搜索和模型生成可下载报告。"""

    request = AnalysisRequest(
        target_product="Notion",
        competitors=["Confluence"],
        market_definition=MarketDefinition(
            market_name="企业知识管理工具",
            product_category="SaaS 协作软件",
            target_buyer="中型企业 IT 与业务负责人",
            comparison_level="企业订阅产品",
            core_dimensions=["features", "pricing"],
            exclusions=["消费端套餐"],
        ),
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

    verification_result = result.verification_result
    failure_context = (
        "Real UI workflow verification failed.\n"
        f"issues={verification_result.model_dump_json(indent=2)}\n"
        f"report={result.final_report}"
    )
    assert verification_result.passed, failure_context
    assert result.verification_result.citations_valid
    assert result.verification_result.scope_consistent
    assert result.verification_result.comparison_usable
    assert "# 竞品分析报告" in result.final_report
    assert "## 分析范围" in result.final_report
    assert "## 范围内资料来源" in result.final_report
    assert reported_stages[-1] == "reporter"
    assert result.stage_history == reported_stages
    assert len(result.evidence) >= 2
    assert (
        result.evidence_scope_counts.in_scope
        + result.evidence_scope_counts.out_of_scope
        + result.evidence_scope_counts.uncertain
        == len(result.evidence)
    )
    assert result.market_definition == request.market_definition
