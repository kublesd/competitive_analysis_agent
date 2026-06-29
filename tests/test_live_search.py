"""Tavily 真实搜索集成测试；默认测试不会调用外部 API。"""

import pytest

from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.search import (
    SearchAdapter,
    SearchRequest,
    TavilySearchProvider,
)


@pytest.mark.live_search
def test_tavily_returns_traceable_official_search_result() -> None:
    """真实 Tavily 搜索应返回可规范化且来自指定官方域名的结果。"""

    settings = load_live_settings()
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
