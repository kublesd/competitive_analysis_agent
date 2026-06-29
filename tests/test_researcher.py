import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from competitive_analysis_agent.researcher import (
    Researcher,
    ResearcherInput,
    build_focused_search_query,
)
from competitive_analysis_agent.schemas import ResearchTask
from competitive_analysis_agent.search import (
    FakeSearchProvider,
    ProviderSearchResult,
    SearchAdapter,
    SearchRequest,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "researcher_search_results.json"
)
FIXED_TIME = datetime(2026, 6, 11, 8, 0, tzinfo=timezone.utc)


def _load_search_results() -> dict:
    """读取 Researcher 的固定搜索数据。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_input() -> ResearcherInput:
    """创建覆盖两个产品、两个主题的固定任务列表。"""

    return ResearcherInput(
        tasks=[
            ResearchTask(
                product_name="Atlas Notes",
                topic="features",
                query="Atlas Notes official features",
            ),
            ResearchTask(
                product_name="Atlas Notes",
                topic="pricing",
                query="Atlas Notes official pricing",
            ),
            ResearchTask(
                product_name="Beacon Docs",
                topic="features",
                query="Beacon Docs official features",
            ),
            ResearchTask(
                product_name="Beacon Docs",
                topic="pricing",
                query="Beacon Docs official pricing",
            ),
        ],
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
        max_results_per_task=3,
    )


def _build_researcher() -> Researcher:
    """创建带固定数据、固定失败和固定时钟的 Researcher。"""

    provider = FakeSearchProvider(
        results_by_query=_load_search_results(),
        failures_by_query={
            "Beacon Docs official pricing plans price Free Plus Standard Business Enterprise": TimeoutError(
                "fixture timeout"
            ),
        },
    )
    return Researcher(
        search_adapter=SearchAdapter(provider),
        clock=lambda: FIXED_TIME,
    )


class RecordingSearchProvider:
    """记录收到的搜索请求，用于检查 Researcher 是否打开正文抓取。"""

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        self.requests.append(request)
        return [
            ProviderSearchResult(
                title="Notion Pricing",
                url="https://www.notion.com/pricing",
                snippet="Notion pricing plans.",
                raw_content=(
                    "Header navigation that should be ignored.\n"
                    "Free\n"
                    "$0 per member / month\n"
                    "Plus\n"
                    "$10 per member / month\n"
                    "Business\n"
                    "$20 per member / month\n"
                    "Enterprise\n"
                    "Custom pricing\n"
                ),
            )
        ]


class ResearcherTest(unittest.TestCase):
    def test_topic_builds_focused_search_query(self) -> None:
        # 内部 topic 名称不一定适合搜索，应转换成官网页面常用词。
        positioning_query = build_focused_search_query(
            ResearchTask(
                product_name="Confluence",
                topic="positioning",
                query="Confluence positioning official",
            )
        )
        target_users_query = build_focused_search_query(
            ResearchTask(
                product_name="Notion",
                topic="target_users",
                query="Notion target_users official",
            )
        )

        self.assertEqual(
            positioning_query,
            "Confluence official product overview workspace teams business",
        )
        self.assertNotIn("positioning", positioning_query)
        self.assertEqual(
            target_users_query,
            (
                "Notion official use cases customers teams enterprise "
                "small business"
            ),
        )
        self.assertNotIn("target_users", target_users_query)

    def test_default_model_product_pricing_query_targets_api_scope(self) -> None:
        # ChatGPT / Claude / Gemini 默认分析 API 价格，不应主动搜索消费端订阅套餐。
        chatgpt_query = build_focused_search_query(
            ResearchTask(
                product_name="ChatGPT",
                topic="pricing",
                query="ChatGPT pricing",
            )
        )
        claude_query = build_focused_search_query(
            ResearchTask(
                product_name="Claude",
                topic="pricing",
                query="Claude pricing",
            )
        )
        gemini_query = build_focused_search_query(
            ResearchTask(
                product_name="Gemini",
                topic="pricing",
                query="Gemini pricing",
            )
        )

        self.assertIn("API pricing", chatgpt_query)
        self.assertIn("developer platform", chatgpt_query)
        self.assertIn("token", chatgpt_query)
        self.assertIn("Anthropic Claude API pricing", claude_query)
        self.assertIn("console", claude_query)
        self.assertIn("Gemini API Google AI API pricing", gemini_query)
        self.assertIn("ai.google.dev", gemini_query)
        for query in [chatgpt_query, claude_query, gemini_query]:
            self.assertNotIn("Plus", query)
            self.assertNotIn("Business", query)
            self.assertNotIn("Enterprise", query)

    def test_tasks_become_deterministic_evidence(self) -> None:
        result = _build_researcher().research(_build_input())
        repeated_result = _build_researcher().research(_build_input())

        self.assertEqual(
            [item.evidence_id for item in result.evidence],
            ["E1", "E2", "E3", "E4", "E5"],
        )
        self.assertEqual(
            result.model_dump(mode="json"),
            repeated_result.model_dump(mode="json"),
        )
        self.assertEqual(
            [str(item.url) for item in result.evidence],
            [
                "https://example.com/atlas/overview",
                "https://example.com/atlas/features",
                "https://example.com/atlas/overview",
                "https://example.com/atlas/pricing",
                "https://example.com/beacon/features",
            ],
        )
        self.assertTrue(
            all(item.collected_at == FIXED_TIME for item in result.evidence)
        )

    def test_evidence_keeps_task_context(self) -> None:
        result = _build_researcher().research(_build_input())

        contexts = [
            (item.product_name, item.topic) for item in result.evidence
        ]
        self.assertEqual(
            contexts,
            [
                ("Atlas Notes", "features"),
                ("Atlas Notes", "features"),
                ("Atlas Notes", "pricing"),
                ("Atlas Notes", "pricing"),
                ("Beacon Docs", "features"),
            ],
        )

    def test_same_url_is_kept_for_different_topics(self) -> None:
        # 同一网页可能同时支持功能和价格；不同 topic 应保留各自上下文。
        result = _build_researcher().research(_build_input())

        overview_contexts = [
            (item.product_name, item.topic)
            for item in result.evidence
            if str(item.url) == "https://example.com/atlas/overview"
        ]
        self.assertEqual(
            overview_contexts,
            [
                ("Atlas Notes", "features"),
                ("Atlas Notes", "pricing"),
            ],
        )

    def test_same_url_is_deduplicated_within_same_product_and_topic(
        self,
    ) -> None:
        # 完全相同的产品 + topic + URL 仍然只保留一次，避免重复 Evidence。
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Atlas Notes",
                    topic="features",
                    query="Atlas Notes official features",
                ),
                ResearchTask(
                    product_name="Atlas Notes",
                    topic="features",
                    query="Atlas Notes official features duplicate",
                ),
            ],
            official_domains_by_product={"Atlas Notes": ["example.com"]},
        )
        provider = FakeSearchProvider(
            results_by_query={
                "Atlas Notes official product features capabilities": [
                    {
                        "title": "Atlas Notes Features",
                        "url": "https://example.com/atlas/features",
                        "snippet": "Feature documentation.",
                    }
                ]
            }
        )
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        )

        result = researcher.research(researcher_input)

        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].evidence_id, "E1")
        self.assertEqual(result.evidence[0].topic, "features")

    def test_failed_task_does_not_remove_successful_evidence(self) -> None:
        result = _build_researcher().research(_build_input())

        self.assertEqual(len(result.evidence), 5)
        self.assertEqual(len(result.errors), 1)
        error = result.errors[0]
        self.assertEqual(error.product_name, "Beacon Docs")
        self.assertEqual(error.topic, "pricing")
        self.assertEqual(error.code, "timeout")

    def test_empty_search_results_are_recorded(self) -> None:
        researcher = Researcher(
            search_adapter=SearchAdapter(
                FakeSearchProvider(results_by_query={})
            ),
            clock=lambda: FIXED_TIME,
        )
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Atlas Notes",
                    topic="security",
                    query="Atlas Notes official security",
                )
            ]
        )

        result = researcher.research(researcher_input)

        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "no_results")

    def test_pricing_task_requests_raw_content_and_keeps_price_excerpt(self) -> None:
        # pricing 不能只依赖搜索摘要，应把网页正文里的价格片段带入 Evidence。
        provider = RecordingSearchProvider()
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        )
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Notion",
                    topic="pricing",
                    query="Notion official pricing",
                )
            ],
            official_domains_by_product={"Notion": ["notion.com"]},
        )

        result = researcher.research(researcher_input)

        self.assertTrue(provider.requests[0].include_raw_content)
        self.assertEqual(len(result.evidence), 1)
        evidence = result.evidence[0]
        self.assertIsNotNone(evidence.raw_content)
        self.assertIn("$10 per member / month", evidence.raw_content)
        self.assertIn("$20 per member / month", evidence.raw_content)
        self.assertIn("Pricing page excerpt", evidence.snippet)

    def test_feature_task_does_not_request_raw_content(self) -> None:
        # 只有 pricing 需要正文片段，其他主题继续用轻量搜索摘要。
        provider = RecordingSearchProvider()
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        )
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Notion",
                    topic="features",
                    query="Notion official features",
                )
            ]
        )

        researcher.research(researcher_input)

        self.assertFalse(provider.requests[0].include_raw_content)


if __name__ == "__main__":
    unittest.main()
