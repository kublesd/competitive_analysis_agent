"""Stage 29 API 竞品范围的三产品离线验收。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from competitive_analysis_agent.analyst import Analyst, FakeAnalystModel
from competitive_analysis_agent.evaluation import (
    calculate_core_dimension_coverage,
    calculate_pricing_context_completeness,
)
from competitive_analysis_agent.extractor import Extractor, FakeExtractorModel
from competitive_analysis_agent.planner import FakePlannerModel, Planner, PlannerInput
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.schemas import MarketDefinition
from competitive_analysis_agent.search import FakeSearchProvider, SearchAdapter
from competitive_analysis_agent.verifier import FakeVerifierModel, Verifier
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    create_initial_state,
    create_workflow_graph,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "api_competition_case.json"


def test_three_product_api_fixture_is_scoped_traceable_and_comparable() -> None:
    """验证 API 四维主线通过，订阅套餐只出现在已排除资料中。"""

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    market_definition = MarketDefinition.model_validate(
        fixture["market_definition"]
    )
    planner_input = PlannerInput(
        target_product=fixture["target_product"],
        competitors=fixture["competitors"],
        market_definition=market_definition,
    )
    components = WorkflowComponents(
        planner=Planner(FakePlannerModel([fixture["planner_output"]])),
        researcher=Researcher(
            SearchAdapter(FakeSearchProvider(fixture["search_results"])),
            clock=lambda: datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc),
        ),
        extractor=Extractor(
            FakeExtractorModel(fixture["extractor_outputs"])
        ),
        # 空响应列表会触发现有的确定性保守分析，不引入测试专用 Analyst。
        analyst=Analyst(FakeAnalystModel([])),
        verifier=Verifier(FakeVerifierModel({"issues": []})),
        reporter=Reporter(),
    )
    initial_state = create_initial_state(
        planner_input=planner_input,
        market_definition=market_definition,
        official_domains_by_product=fixture["official_domains_by_product"],
    )

    final_state = create_workflow_graph(components).invoke(initial_state)

    expected_pairs = {
        (product_name, dimension)
        for product_name in [fixture["target_product"], *fixture["competitors"]]
        for dimension in market_definition.core_dimensions
    }
    actual_pairs = {
        (task.product_name, task.topic)
        for task in final_state["research_tasks"]
    }
    assert actual_pairs == expected_pairs
    assert len(final_state["research_tasks"]) == 12
    assert len(final_state["evidence"]) == 12
    assert [item.title for item in final_state["excluded_evidence"]] == [
        "OpenAI API 消费端订阅套餐"
    ]

    for profile in final_state["product_profiles"]:
        findings = {
            finding.dimension: finding
            for finding in profile.dimension_findings
        }
        assert set(findings) == set(market_definition.core_dimensions)
        assert all(
            finding.facts and finding.evidence_ids
            for finding in findings.values()
        )
        assert len(profile.pricing) == 1
        assert "tokens" in (profile.pricing[0].unit or "")

    assert calculate_core_dimension_coverage(
        final_state["product_profiles"], market_definition
    ) == 1.0
    assert calculate_pricing_context_completeness(
        final_state["product_profiles"], market_definition
    ) == 1.0
    assert final_state["verification_result"].passed
    assert final_state["verification_result"].comparison_usable

    report_before_exclusions = final_state["final_report"].split(
        "## 已排除资料",
        maxsplit=1,
    )[0]
    assert "GPT-5 input tokens" in report_before_exclusions
    assert "ChatGPT Plus" not in report_before_exclusions
    assert "计费周期未提供" not in report_before_exclusions
