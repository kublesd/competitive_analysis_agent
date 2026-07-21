import json
import socket
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from competitive_analysis_agent.search import (
    FakeSearchProvider,
    ProviderSearchResult,
    SearchAdapter,
    SearchRequest,
    TavilySearchProvider,
    classify_source,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "search_results.json"


def _load_search_results() -> dict:
    """读取固定搜索结果，确保测试不访问真实网络。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class SearchAdapterTest(unittest.TestCase):
    def test_fake_provider_returns_normalized_results(self) -> None:
        # 适配器负责统一 URL、来源类型和结果结构。
        provider = FakeSearchProvider(_load_search_results())
        adapter = SearchAdapter(provider)
        request = SearchRequest(
            query="Atlas Notes pricing",
            official_domains=["example.com"],
            max_results=3,
        )

        response = adapter.search(request)

        self.assertEqual(response.status, "success")
        self.assertIsNone(response.error)
        self.assertEqual(len(response.results), 3)
        self.assertEqual(
            str(response.results[0].url),
            "https://example.com/atlas/pricing",
        )
        self.assertEqual(response.results[0].source_type, "official")
        self.assertEqual(response.results[1].source_type, "official")
        self.assertEqual(response.results[2].source_type, "third_party")

    def test_raw_content_is_preserved_when_provider_supplies_it(self) -> None:
        # pricing 页面正文需要继续传给 Researcher，不能只保留搜索摘要。
        provider = FakeSearchProvider(
            {
                "Notion pricing": [
                    ProviderSearchResult(
                        title="Notion Pricing",
                        url="https://www.notion.com/pricing",
                        snippet="Notion pricing plans.",
                        raw_content=(
                            "Free $0 per member / month\n"
                            "Plus $10 per member / month\n"
                            "Business $20 per member / month"
                        ),
                    )
                ]
            }
        )
        adapter = SearchAdapter(provider)

        response = adapter.search(
            SearchRequest(
                query="Notion pricing",
                official_domains=["notion.com"],
            )
        )

        self.assertEqual(response.status, "success")
        self.assertIsNotNone(response.results[0].raw_content)
        self.assertIn("$10", response.results[0].raw_content)

    def test_duplicate_urls_are_removed_before_result_cap(self) -> None:
        # 第一条等价 URL 被保留，后续重复项不会占用结果名额。
        provider = FakeSearchProvider(_load_search_results())
        adapter = SearchAdapter(provider)
        request = SearchRequest(
            query="Atlas Notes pricing",
            official_domains=["example.com"],
            max_results=2,
        )

        response = adapter.search(request)

        result_titles = [result.title for result in response.results]
        self.assertEqual(
            result_titles,
            ["Atlas Notes Pricing", "Atlas Notes Documentation"],
        )

    def test_timeout_becomes_controlled_error(self) -> None:
        # 一个搜索任务超时后，调用方仍能读取响应并继续其他任务。
        provider = FakeSearchProvider(
            results_by_query={},
            failures_by_query={
                "slow query": TimeoutError("fixture timeout"),
            },
        )
        adapter = SearchAdapter(provider)

        response = adapter.search(SearchRequest(query="slow query"))

        self.assertEqual(response.status, "error")
        self.assertEqual(response.results, [])
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.code, "timeout")

    def test_provider_failure_becomes_controlled_error(self) -> None:
        # 未知供应商异常也应转为统一错误，而不是向上抛出。
        provider = FakeSearchProvider(
            results_by_query={},
            failures_by_query={
                "broken query": RuntimeError("fixture failure"),
            },
        )
        adapter = SearchAdapter(provider)

        response = adapter.search(SearchRequest(query="broken query"))

        self.assertEqual(response.status, "error")
        self.assertEqual(response.error.code, "provider_error")
        self.assertIn("fixture failure", response.error.message)

    def test_lookalike_domain_is_not_official(self) -> None:
        # 后缀匹配必须包含点号，防止仿冒域名被误判为官方来源。
        source_type = classify_source(
            "https://example.com.evil.test/pricing",
            ["example.com"],
        )

        self.assertEqual(source_type, "third_party")

    def test_vendor_forum_subdomain_is_third_party(self) -> None:
        source_type = classify_source(
            "https://discuss.ai.google.dev/t/pricing-question/123",
            ["ai.google.dev"],
        )

        self.assertEqual(source_type, "third_party")


class FakeHttpResponse:
    """模拟 urllib 响应上下文，避免测试访问真实 Tavily。"""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class TavilySearchProviderTest(unittest.TestCase):
    def test_maps_tavily_content_and_limits_to_official_domains(self) -> None:
        # Tavily 的 content 必须转换为项目统一的 snippet。
        provider_response = {
            "results": [
                {
                    "title": "Notion Pricing",
                    "url": "https://www.notion.so/pricing",
                    "content": "Notion publishes Free and Plus plans.",
                    "score": 0.9,
                }
            ]
        }
        provider = TavilySearchProvider("test-key")

        with patch(
            "competitive_analysis_agent.search.urlopen",
            return_value=FakeHttpResponse(provider_response),
        ) as mocked_urlopen:
            results = provider.search(
                SearchRequest(
                    query="Notion official pricing",
                    official_domains=["notion.so"],
                    max_results=2,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].snippet,
            "Notion publishes Free and Plus plans.",
        )
        http_request = mocked_urlopen.call_args.args[0]
        request_payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(request_payload["search_depth"], "basic")
        self.assertEqual(request_payload["max_results"], 2)
        self.assertFalse(request_payload["include_raw_content"])
        self.assertEqual(
            request_payload["include_domains"],
            ["notion.so"],
        )
        self.assertNotIn("test-key", http_request.data.decode("utf-8"))

    def test_tavily_raw_content_is_requested_and_mapped(self) -> None:
        # pricing 任务会请求原始正文，Provider 需要把 raw_content 带回项目结构。
        provider_response = {
            "results": [
                {
                    "title": "Notion Pricing",
                    "url": "https://www.notion.com/pricing",
                    "content": "Notion pricing plans.",
                    "raw_content": (
                        "Free $0 per member / month\n"
                        "Plus $10 per member / month"
                    ),
                }
            ]
        }
        provider = TavilySearchProvider("test-key")

        with patch(
            "competitive_analysis_agent.search.urlopen",
            return_value=FakeHttpResponse(provider_response),
        ) as mocked_urlopen:
            results = provider.search(
                SearchRequest(
                    query="Notion official pricing",
                    include_raw_content=True,
                )
            )

        self.assertEqual(
            results[0].raw_content,
            provider_response["results"][0]["raw_content"],
        )
        http_request = mocked_urlopen.call_args.args[0]
        request_payload = json.loads(http_request.data.decode("utf-8"))
        self.assertTrue(request_payload["include_raw_content"])

    def test_advanced_search_extracts_top_pricing_result(self) -> None:
        # 价格检索先召回 URL，再用 Tavily Extract 返回查询聚焦的 Markdown。
        search_response = {
            "results": [
                {
                    "title": "OpenAI API Pricing",
                    "url": "https://openai.com/api/pricing",
                    "content": "OpenAI API model pricing.",
                    "score": 0.97,
                }
            ]
        }
        extract_response = {
            "results": [
                {
                    "url": "https://openai.com/api/pricing",
                    "raw_content": (
                        "| Model | Input | Output |\n"
                        "| GPT-5 | $1.25 / 1M tokens | $10 / 1M tokens |"
                    ),
                }
            ]
        }
        provider = TavilySearchProvider("test-key")

        with patch(
            "competitive_analysis_agent.search.urlopen",
            side_effect=[
                FakeHttpResponse(search_response),
                FakeHttpResponse(extract_response),
            ],
        ) as mocked_urlopen:
            results = provider.search(
                SearchRequest(
                    query="OpenAI API official pricing",
                    official_domains=["openai.com"],
                    search_depth="advanced",
                    chunks_per_source=3,
                    include_raw_content=True,
                    extract_query="OpenAI API input output token pricing",
                    extract_top_results=1,
                )
            )

        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertTrue(results[0].extracted_content)
        self.assertIn("| Model |", results[0].raw_content)
        search_request = mocked_urlopen.call_args_list[0].args[0]
        search_payload = json.loads(search_request.data.decode("utf-8"))
        self.assertEqual(search_payload["search_depth"], "advanced")
        self.assertEqual(search_payload["chunks_per_source"], 3)
        self.assertFalse(search_payload["include_raw_content"])
        extract_request = mocked_urlopen.call_args_list[1].args[0]
        self.assertEqual(extract_request.full_url, "https://api.tavily.com/extract")
        extract_payload = json.loads(extract_request.data.decode("utf-8"))
        self.assertEqual(extract_payload["extract_depth"], "advanced")
        self.assertEqual(
            extract_payload["query"],
            "OpenAI API input output token pricing",
        )

    def test_extract_failure_keeps_search_result_with_warning(self) -> None:
        # Extract 是增强步骤；失败时保留 Search 摘要，供 Researcher 继续产出证据。
        search_response = {
            "results": [
                {
                    "title": "OpenAI API Pricing",
                    "url": "https://openai.com/api/pricing",
                    "content": "OpenAI API input and output token pricing.",
                }
            ]
        }
        provider = TavilySearchProvider("test-key", max_retries=0)

        with patch(
            "competitive_analysis_agent.search.urlopen",
            side_effect=[
                FakeHttpResponse(search_response),
                URLError("extract unavailable"),
            ],
        ):
            results = provider.search(
                SearchRequest(
                    query="OpenAI API pricing",
                    extract_query="OpenAI API token pricing",
                    extract_top_results=1,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].raw_content)
        self.assertFalse(results[0].extracted_content)
        self.assertIn("Tavily extract", results[0].extraction_error)

    def test_malformed_result_is_skipped_without_losing_valid_result(self) -> None:
        # 单条坏记录不应让同一次搜索中的其他来源全部失败。
        provider = TavilySearchProvider("test-key")
        provider_response = {
            "results": [
                {"title": "", "url": "not-a-url", "content": ""},
                {
                    "title": "Valid result",
                    "url": "https://example.com/result",
                    "content": "Useful source summary.",
                },
            ]
        }

        with patch(
            "competitive_analysis_agent.search.urlopen",
            return_value=FakeHttpResponse(provider_response),
        ):
            results = provider.search(SearchRequest(query="valid query"))

        self.assertEqual([result.title for result in results], ["Valid result"])

    def test_tavily_retries_transient_network_failure_once(self) -> None:
        # Tavily 临时断网时重试一次，避免整项调研直接丢失。
        provider_response = {
            "results": [
                {
                    "title": "Recovered result",
                    "url": "https://example.com/recovered",
                    "content": "Search succeeded after one retry.",
                }
            ]
        }
        provider = TavilySearchProvider("test-key")

        with patch(
            "competitive_analysis_agent.search.urlopen",
            side_effect=[
                URLError("temporary network failure"),
                FakeHttpResponse(provider_response),
            ],
        ) as mocked_urlopen:
            results = provider.search(SearchRequest(query="retry query"))

        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual([result.title for result in results], ["Recovered result"])

    def test_tavily_timeout_becomes_adapter_timeout(self) -> None:
        # 底层 socket timeout 应沿用项目已有的可恢复错误语义。
        adapter = SearchAdapter(TavilySearchProvider("test-key"))

        with patch(
            "competitive_analysis_agent.search.urlopen",
            side_effect=socket.timeout("network timeout"),
        ):
            response = adapter.search(SearchRequest(query="slow query"))

        self.assertEqual(response.status, "error")
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.code, "timeout")


if __name__ == "__main__":
    unittest.main()
