import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from competitive_analysis_agent.researcher import (
    Researcher,
    ResearcherInput,
    build_focused_search_query,
    should_request_raw_content,
)
from competitive_analysis_agent.schemas import MarketDefinition, ResearchTask
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


def _build_market_definition() -> MarketDefinition:
    """创建 Researcher 测试共用的市场范围。"""

    return MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=["features", "pricing"],
        exclusions=["消费端套餐"],
    )


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
        market_definition=_build_market_definition(),
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
            "Beacon Docs official pricing": TimeoutError(
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


class ExtractionWarningSearchProvider:
    """模拟 Extract 失败但 Search 结果仍然可用。"""

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        return [
            ProviderSearchResult(
                title="Notion Pricing",
                url="https://www.notion.com/pricing",
                snippet="Notion pricing plans include Free and Plus.",
                extraction_error="Tavily extract network request failed.",
            )
        ]


class ResearcherTest(unittest.TestCase):
    def test_search_query_removes_exclusion_control_clause(self) -> None:
        # 排除项约束仍留在任务中，但不能作为关键词发送给搜索供应商。
        task = ResearchTask(
            product_name="Atlas Notes",
            topic="pricing",
            query="Atlas Notes pricing official exclude 消费端套餐",
        )

        focused_query = build_focused_search_query(
            task,
            _build_market_definition().model_copy(
                update={"exclusions": ["消费端套餐"]}
            ),
        )

        self.assertEqual(
            focused_query,
            "Atlas Notes pricing official",
        )

    def test_over_scoped_query_becomes_provider_focused_query(self) -> None:
        # 市场控制词保留在任务中，但供应商查询只聚焦产品和价格主题。
        task = ResearchTask(
            product_name="Notion",
            topic="pricing",
            query=(
                "Notion SaaS 协作软件 企业订阅产品 pricing official "
                "exclude 消费端套餐"
            ),
        )

        focused_query = build_focused_search_query(
            task,
            _build_market_definition().model_copy(
                update={"exclusions": ["消费端套餐"]}
            ),
        )

        self.assertEqual(
            focused_query,
            "Notion official pricing plans price",
        )

        feature_task = task.model_copy(
            update={
                "topic": "features",
                "query": (
                    "Notion SaaS 协作软件 企业订阅产品 features official "
                    "exclude 消费端套餐"
                ),
            }
        )
        self.assertEqual(
            build_focused_search_query(
                feature_task,
                _build_market_definition().model_copy(
                    update={"exclusions": ["消费端套餐"]}
                ),
            ),
            "Notion official product features collaboration",
        )

    def test_api_dimensions_use_fixed_provider_queries(self) -> None:
        # 四个 API topic 使用短而稳定的专用召回词，不引入查询改写模型。
        market_definition = MarketDefinition(
            market_name="生成式 AI API",
            product_category="大语言模型 API",
            target_buyer="开发团队",
            comparison_level="模型 API 服务",
            pricing_scope="api",
            core_dimensions=[
                "model_capabilities",
                "api_pricing",
                "developer_platform",
                "usage_limits",
            ],
        )
        expected_topics = {
            "model_capabilities": "models multimodal capabilities",
            "api_pricing": "API pricing input output tokens",
            "developer_platform": "API documentation SDK tools",
            "usage_limits": "context window rate limits",
        }

        for topic, provider_topic in expected_topics.items():
            with self.subTest(topic=topic):
                task = ResearchTask(
                    product_name="OpenAI API",
                    topic=topic,
                    query=(
                        "OpenAI API 大语言模型 API 模型 API 服务 "
                        f"{topic} official"
                    ),
                )
                self.assertEqual(
                    build_focused_search_query(task, market_definition),
                    f"OpenAI API official {provider_topic}",
                )

        self.assertTrue(
            should_request_raw_content(
                ResearchTask(
                    product_name="OpenAI API",
                    topic="api_pricing",
                    query="OpenAI API official API pricing",
                )
            )
        )

    def test_official_api_page_without_topic_match_is_uncertain(self) -> None:
        # 官方域名和产品名都命中，但价格页不能支撑开发平台维度。
        market_definition = MarketDefinition(
            market_name="生成式 AI API",
            product_category="大语言模型 API",
            target_buyer="开发团队",
            comparison_level="模型 API 服务",
            pricing_scope="api",
            core_dimensions=["developer_platform"],
        )
        provider = FakeSearchProvider(
            {
                "OpenAI API official developer_platform": [
                    {
                        "title": "OpenAI API Pricing",
                        "url": "https://openai.com/api/pricing",
                        "snippet": (
                            "OpenAI API pricing lists input and output tokens."
                        ),
                    }
                ]
            }
        )
        result = Researcher(
            SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        ).research(
            ResearcherInput(
                tasks=[
                    ResearchTask(
                        product_name="OpenAI API",
                        topic="developer_platform",
                        query="OpenAI API official developer_platform",
                    )
                ],
                market_definition=market_definition,
                official_domains_by_product={"OpenAI API": ["openai.com"]},
            )
        )

        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.uncertain_evidence), 1)
        self.assertIn("没有回答当前 API 主题", result.uncertain_evidence[0].scope_reason)

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
            market_definition=_build_market_definition(),
            official_domains_by_product={"Atlas Notes": ["example.com"]},
        )
        provider = FakeSearchProvider(
            results_by_query={
                "Atlas Notes official features": [
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
            ],
            market_definition=_build_market_definition(),
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
            market_definition=_build_market_definition(),
            official_domains_by_product={"Notion": ["notion.com"]},
        )

        result = researcher.research(researcher_input)

        self.assertTrue(provider.requests[0].include_raw_content)
        self.assertEqual(provider.requests[0].search_depth, "advanced")
        self.assertEqual(provider.requests[0].chunks_per_source, 3)
        self.assertEqual(provider.requests[0].extract_top_results, 2)
        self.assertEqual(
            provider.requests[0].extract_query,
            "Notion official pricing",
        )
        self.assertEqual(len(result.evidence), 1)
        evidence = result.evidence[0]
        self.assertIsNotNone(evidence.raw_content)
        self.assertIn("$10 per member / month", evidence.raw_content)
        self.assertIn("$20 per member / month", evidence.raw_content)
        self.assertIn("Pricing page excerpt", evidence.snippet)

    def test_extract_failure_is_a_limitation_not_a_failed_task(self) -> None:
        # 增强抓取失败时保留 Search 证据，并显式记录可展示的数据限制。
        result = Researcher(
            search_adapter=SearchAdapter(ExtractionWarningSearchProvider()),
            clock=lambda: FIXED_TIME,
        ).research(
            ResearcherInput(
                tasks=[
                    ResearchTask(
                        product_name="Notion",
                        topic="pricing",
                        query="Notion official pricing",
                    )
                ],
                market_definition=_build_market_definition(),
                official_domains_by_product={"Notion": ["notion.com"]},
            )
        )

        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "provider_error")
        self.assertIn("Extract", result.errors[0].message)

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
            ],
            market_definition=_build_market_definition(),
        )

        researcher.research(researcher_input)

        self.assertFalse(provider.requests[0].include_raw_content)

    def test_excluded_product_line_is_recorded_but_not_forwarded(self) -> None:
        # 同品牌但命中排除项的资料必须可追溯，且不能进入 Extractor 输入列表。
        market_definition = _build_market_definition().model_copy(
            update={"exclusions": ["Consumer Plan"]}
        )
        provider = FakeSearchProvider(
            {
                "Atlas Notes pricing official": [
                    {
                        "title": "Atlas Notes Consumer Plan Pricing",
                        "url": "https://example.com/atlas/consumer-pricing",
                        "snippet": "Consumer Plan costs 9 USD monthly.",
                    }
                ]
            }
        )
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        )
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Atlas Notes",
                    topic="pricing",
                    query="Atlas Notes pricing official",
                )
            ],
            market_definition=market_definition,
            official_domains_by_product={"Atlas Notes": ["example.com"]},
        )

        result = researcher.research(researcher_input)

        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.excluded_evidence), 1)
        excluded = result.excluded_evidence[0]
        self.assertEqual(excluded.evidence_id, "E1")
        self.assertEqual(excluded.scope_status, "out_of_scope")
        self.assertIn("Consumer Plan", excluded.scope_reason)

    def test_unconfirmed_third_party_result_is_kept_for_review(self) -> None:
        # 无法确认产品边界的第三方资料应保留待核验，不能直接支撑事实。
        provider = FakeSearchProvider(
            {
                "Atlas Notes security official": [
                    {
                        "title": "Market comparison",
                        "url": "https://reviews.example.net/comparison",
                        "snippet": "A possible option for business teams.",
                    }
                ]
            }
        )
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
        )
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="Atlas Notes",
                    topic="security",
                    query="Atlas Notes security official",
                )
            ],
            market_definition=_build_market_definition(),
        )

        result = researcher.research(researcher_input)

        self.assertEqual(result.evidence, [])
        self.assertEqual(result.excluded_evidence, [])
        self.assertEqual(len(result.uncertain_evidence), 1)
        self.assertEqual(
            result.uncertain_evidence[0].scope_status,
            "uncertain",
        )

    def test_api_pricing_community_post_is_kept_for_review(self) -> None:
        # 官方域名下的用户帖子仍不是权威价表，不能进入 Extractor 生成价格事实。
        provider = FakeSearchProvider(
            {
                "OpenAI API official API pricing input output tokens": [
                    {
                        "title": "ChatGPT4o API Pricing for Input and Output",
                        "url": "https://community.openai.com/t/pricing/746258",
                        "snippet": "OpenAI API GPT4o input pricing is $5/1M.",
                        "raw_content": "A user asks how API token pricing works.",
                    }
                ]
            }
        )
        researcher = Researcher(
            search_adapter=SearchAdapter(provider),
            clock=lambda: FIXED_TIME,
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
        researcher_input = ResearcherInput(
            tasks=[
                ResearchTask(
                    product_name="OpenAI API",
                    topic="api_pricing",
                    query=(
                        "OpenAI API 大语言模型 API 模型 API 服务 "
                        "api_pricing official"
                    ),
                )
            ],
            market_definition=market_definition,
            official_domains_by_product={"OpenAI API": ["openai.com"]},
        )

        result = researcher.research(researcher_input)

        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.uncertain_evidence), 1)
        self.assertIn("社区", result.uncertain_evidence[0].scope_reason)


if __name__ == "__main__":
    unittest.main()
