"""使用固定搜索与真实模型验收 Stage 24 的评估契约。"""

from dataclasses import replace
from datetime import datetime, timezone
from os import environ
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pytest

from competitive_analysis_agent.evaluation import (
    EvaluationCase,
    evaluate_workflow_state,
    run_evaluation_case,
)
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_workflow import create_live_workflow_components
from competitive_analysis_agent.planner import PlannerInput
from competitive_analysis_agent.pricing_utils import (
    api_pricing_plan_is_evidence_supported,
)
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.schemas import MarketDefinition, ResearchTask
from competitive_analysis_agent.search import (
    ProviderSearchResult,
    SearchAdapter,
    SearchRequest,
)
from competitive_analysis_agent.workflow import (
    create_initial_state,
    create_workflow_graph,
)


FIXED_TIME = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)


class FixedApiPricingPlanner:
    """固定 API 价格任务，避免评估结果受 Planner 措辞波动影响。"""

    def plan(self, planner_input: PlannerInput) -> list[ResearchTask]:
        """为每个产品只生成一条 api_pricing 任务。"""

        return [
            ResearchTask(
                product_name=product_name,
                topic="api_pricing",
                query=f"{product_name} official API pricing input tokens",
            )
            for product_name in planner_input.products
        ]


class FixedApiPricingSearchProvider:
    """提供可复现的完整 API 输入费率，不访问真实搜索服务。"""

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        """根据查询产品返回单条官方价格证据。"""

        normalized_query = request.query.casefold()
        if "openai api" in normalized_query:
            return [
                ProviderSearchResult(
                    title="OpenAI API Pricing",
                    url="https://openai.com/api/pricing",
                    snippet="OpenAI API pricing lists GPT-5 input token rates.",
                    raw_content=(
                        "GPT-5 input tokens cost $1.25 per 1M input tokens."
                    ),
                )
            ]
        if "gemini api" in normalized_query:
            return [
                ProviderSearchResult(
                    title="Gemini API Pricing",
                    url="https://ai.google.dev/gemini-api/docs/pricing",
                    snippet=(
                        "Gemini API pricing lists Gemini 2.0 Flash input rates."
                    ),
                    raw_content=(
                        "Gemini 2.0 Flash input costs "
                        "$0.15 per 1M input tokens."
                    ),
                )
            ]
        if "claude api" in normalized_query:
            return [
                ProviderSearchResult(
                    title="Claude API Pricing",
                    url="https://docs.anthropic.com/en/docs/about-claude/pricing",
                    snippet="Claude API pricing lists Claude Haiku 4.5 rates.",
                    raw_content=(
                        "Claude Haiku 4.5 | Input $1 / MTok | "
                        "Output $5 / MTok"
                    ),
                )
            ]
        return []


@pytest.mark.live_llm
def test_fixed_evaluation_case_with_real_models() -> None:
    """真实模型案例应完成结构化工作流，并由同一指标函数验收。"""

    case = EvaluationCase(
        case_id="live_fully_comparable",
        description="固定功能证据与真实模型的最小评估案例。",
        scenario="fully_comparable",
        core_dimensions=["features"],
        expected_verification_passed=True,
    )
    components = create_live_workflow_components(load_live_settings())

    result, final_state = run_evaluation_case(case, components=components)

    assert result.expected_behavior_passed
    assert result.scope_consistency == 1.0
    assert result.citation_validity == 1.0
    assert result.core_dimension_coverage == 1.0
    assert result.recommendation_actionability == 1.0
    assert final_state["stage_history"][-1] == "reporter"
    assert final_state["final_report"] is not None


@pytest.mark.live_llm
def test_fixed_api_pricing_case_with_real_models() -> None:
    """固定价格证据应得到完整费率，并在第一次验证中通过。"""

    env_file_name = environ.get("LIVE_TEST_ENV_FILE", ".env")
    env_file = Path(__file__).parents[1] / env_file_name
    with patch.dict(environ, {}, clear=True):
        settings = load_live_settings(env_file)
    components = create_live_workflow_components(settings)
    components = replace(
        components,
        planner=FixedApiPricingPlanner(),
        researcher=Researcher(
            SearchAdapter(FixedApiPricingSearchProvider()),
            clock=lambda: FIXED_TIME,
        ),
    )
    market_definition = MarketDefinition(
        market_name="生成式 AI API",
        product_category="大语言模型 API",
        target_buyer="开发团队",
        comparison_level="模型 API 服务",
        pricing_scope="api",
        core_dimensions=["api_pricing"],
        exclusions=["消费端订阅套餐"],
    )
    planner_input = PlannerInput(
        target_product="OpenAI API",
        competitors=["Claude API", "Gemini API"],
        market_definition=market_definition,
    )
    initial_state = create_initial_state(
        planner_input=planner_input,
        market_definition=market_definition,
        official_domains_by_product={
            "OpenAI API": ["openai.com"],
            "Claude API": ["anthropic.com"],
            "Gemini API": ["ai.google.dev"],
        },
        max_results_per_task=1,
    )
    started_at = perf_counter()

    final_state = create_workflow_graph(components).invoke(initial_state)
    evaluation_case = EvaluationCase(
        case_id="live_api_pricing_integrity",
        description="固定 API 输入价格证据与真实模型。",
        scenario="fully_comparable",
        core_dimensions=["api_pricing"],
        expected_verification_passed=True,
    )
    result = evaluate_workflow_state(
        evaluation_case,
        final_state,
        perf_counter() - started_at,
    )
    evidence_by_id = {
        item.evidence_id: item for item in final_state["evidence"]
    }

    assert result.verification_passed
    assert result.pricing_context_completeness == 1.0
    assert result.citation_validity == 1.0
    for profile in final_state["product_profiles"]:
        assert profile.pricing
        for pricing_plan in profile.pricing:
            assert api_pricing_plan_is_evidence_supported(
                pricing_plan,
                evidence_by_id,
            )
