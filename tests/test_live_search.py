"""Tavily 真实搜索集成测试；默认测试不会调用外部 API。"""

from os import environ
from pathlib import Path
from unittest.mock import patch

import pytest

from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.researcher import Researcher, ResearcherInput
from competitive_analysis_agent.search import (
    SearchAdapter,
    SearchRequest,
    TavilySearchProvider,
)
from competitive_analysis_agent.schemas import MarketDefinition, ResearchTask


LIVE_ENV_FILE = Path(__file__).parents[1] / ".env"


def load_exact_live_settings():
    """忽略 pytest 预加载环境，只从 Stage 29 指定文件读取配置。"""

    with patch.dict(environ, {}, clear=True):
        return load_live_settings(LIVE_ENV_FILE)


@pytest.mark.live_search
def test_tavily_returns_traceable_official_search_result() -> None:
    """真实 Tavily 搜索应返回可规范化且来自指定官方域名的结果。"""

    settings = load_exact_live_settings()
    if settings.tavily_api_key is None:
        pytest.fail("Missing configuration: TAVILY_API_KEY")

    adapter = SearchAdapter(TavilySearchProvider(settings.tavily_api_key))
    response = adapter.search(
        SearchRequest(
            query=(
                "Notion official pricing plans price Free Plus "
                "Business Enterprise"
            ),
            official_domains=["notion.com", "notion.so"],
            max_results=2,
            include_raw_content=True,
        )
    )

    assert response.status == "success"
    assert response.results
    assert all(result.source_type == "official" for result in response.results)
    raw_contents = [
        result.raw_content
        for result in response.results
        if result.raw_content is not None
    ]
    assert raw_contents
    assert any("$" in raw_content for raw_content in raw_contents)


@pytest.mark.live_search
def test_researcher_searches_planner_query_without_exclusion_clause() -> None:
    """Planner 范围任务聚焦后应恢复功能与价格官方结果。"""

    settings = load_exact_live_settings()
    if settings.tavily_api_key is None:
        pytest.fail("Missing configuration: TAVILY_API_KEY")

    researcher = Researcher(
        SearchAdapter(TavilySearchProvider(settings.tavily_api_key))
    )
    market_definition = MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=["features", "pricing"],
        exclusions=["消费端套餐"],
    )
    result = researcher.research(
        ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Notion",
                    topic="features",
                    query=(
                        "Notion SaaS 协作软件 企业订阅产品 features "
                        "official exclude 消费端套餐"
                    ),
                ),
                ResearchTask(
                    product_name="Notion",
                    topic="pricing",
                    query=(
                        "Notion SaaS 协作软件 企业订阅产品 pricing "
                        "official exclude 消费端套餐"
                    ),
                )
            ],
            market_definition=market_definition,
            official_domains_by_product={
                "Notion": ["notion.com", "notion.so"]
            },
            max_results_per_task=1,
        )
    )

    assert result.evidence
    assert not result.errors
    assert all(item.source_type == "official" for item in result.evidence)
    assert {item.topic for item in result.evidence} == {
        "features",
        "pricing",
    }
    assert any(
        item.raw_content is not None and "$" in item.raw_content
        for item in result.evidence
    )


@pytest.mark.live_search
@pytest.mark.parametrize(
    ("product_name", "official_domains"),
    [
        ("OpenAI API", ["openai.com", "platform.openai.com"]),
        ("Claude API", ["anthropic.com", "docs.anthropic.com"]),
        ("Gemini API", ["ai.google.dev", "cloud.google.com"]),
    ],
)
def test_api_pricing_probe_returns_topic_matched_official_evidence(
    product_name: str,
    official_domains: list[str],
) -> None:
    """三个 API 价格探针都应返回范围内官方正文和 Token 计价信号。"""

    settings = load_exact_live_settings()
    if settings.tavily_api_key is None:
        pytest.fail("Missing configuration: TAVILY_API_KEY")

    market_definition = MarketDefinition(
        market_name="生成式 AI API",
        product_category="大语言模型 API",
        target_buyer="开发团队、AI 产品负责人、企业技术团队",
        comparison_level="模型 API 服务",
        pricing_scope="api",
        core_dimensions=["api_pricing"],
        exclusions=["消费端订阅套餐", "按席位企业套餐"],
    )
    task = ResearchTask(
        product_name=product_name,
        topic="api_pricing",
        query=(
            f"{product_name} 大语言模型 API 模型 API 服务 api_pricing "
            "official exclude 消费端订阅套餐 exclude 按席位企业套餐"
        ),
    )
    researcher = Researcher(
        SearchAdapter(TavilySearchProvider(settings.tavily_api_key))
    )

    result = researcher.research(
        ResearcherInput(
            tasks=[task],
            market_definition=market_definition,
            official_domains_by_product={product_name: official_domains},
            max_results_per_task=2,
        )
    )

    assert result.evidence
    assert not result.errors
    assert all(item.source_type == "official" for item in result.evidence)
    assert all(item.topic == "api_pricing" for item in result.evidence)
    assert any(item.raw_content for item in result.evidence)
    pricing_text = " ".join(
        f"{item.snippet} {item.raw_content or ''}"
        for item in result.evidence
    ).casefold()
    assert "token" in pricing_text
