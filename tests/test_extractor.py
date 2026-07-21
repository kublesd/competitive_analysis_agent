import json
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_MAX_EVIDENCE_PER_PRODUCT,
    EXTRACTOR_SNIPPET_MAX_CHARS,
    Extractor,
    ExtractorError,
    ExtractorInput,
    ExtractorOutput,
    ExtractorValidationError,
    FakeExtractorModel,
    LangChainExtractorModel,
    build_extractor_messages,
    build_repair_messages,
    merge_model_items,
    normalize_extractor_raw_output,
    select_evidence_for_extraction,
    validate_extractor_output,
)
from competitive_analysis_agent.schemas import (
    Evidence,
    MarketDefinition,
    WorkflowState,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


def _load_json(file_name: str) -> dict:
    """读取固定 JSON，确保 Extractor 单元测试不调用真实模型。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _load_sample_evidence() -> list[Evidence]:
    """从 Stage 1 样例读取两个产品的固定 Evidence。"""

    sample_case = _load_json("sample_case.json")
    workflow_state = WorkflowState.model_validate(sample_case)
    return workflow_state.evidence


def _build_market_definition(
    dimensions: list[str] | None = None,
) -> MarketDefinition:
    """创建 Extractor 测试共用的市场定义。"""

    return MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=dimensions or ["features", "pricing"],
        exclusions=["消费端套餐"],
    )


def _build_extractor_input(
    evidence: list[Evidence],
    dimensions: list[str] | None = None,
) -> ExtractorInput:
    """把证据和固定市场范围组合成 ExtractorInput。"""

    return ExtractorInput(
        evidence=evidence,
        market_definition=_build_market_definition(dimensions),
    )


def _build_api_extractor_input(evidence: list[Evidence]) -> ExtractorInput:
    """创建只比较 API 价格的 ExtractorInput。"""

    market_definition = _build_market_definition(
        ["api_pricing"]
    ).model_copy(
        update={
            "market_name": "大模型 API",
            "product_category": "模型 API",
            "comparison_level": "开发者 API",
            "pricing_scope": "api",
            "exclusions": ["消费端订阅"],
        }
    )
    return ExtractorInput(
        evidence=evidence,
        market_definition=market_definition,
    )


def _build_pricing_output(
    product_name: str,
    pricing: list[dict],
) -> dict:
    """创建价格契约测试使用的最小模型输出。"""

    return {
        "profile": {
            "product_name": product_name,
            "positioning": None,
            "target_users": [],
            "features": [],
            "dimension_findings": [],
            "pricing": pricing,
            "strengths": [],
            "limitations": [],
        }
    }


def _build_api_evidence(
    *,
    topic: str,
    snippet: str,
    raw_content: str | None = None,
    evidence_id: str = "E1",
    source_type: str = "official",
) -> Evidence:
    """创建统一字段抽取测试使用的固定 API Evidence。"""

    return Evidence(
        evidence_id=evidence_id,
        product_name="OpenAI API",
        topic=topic,
        title=f"OpenAI API {topic}",
        url=f"https://example.com/{evidence_id.lower()}",
        snippet=snippet,
        raw_content=raw_content,
        source_type=source_type,
        collected_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )


def _build_source_reference(evidence: Evidence) -> dict[str, object]:
    """把 Evidence 转成统一模型输出需要的来源引用。"""

    return {
        "evidence_id": evidence.evidence_id,
        "title": evidence.title,
        "url": str(evidence.url),
        "source_type": evidence.source_type,
        "collected_at": evidence.collected_at.isoformat(),
    }


def _build_unified_output(
    *,
    models: list[dict[str, object]] | None = None,
    positioning: str | None = None,
    target_users: list[str] | None = None,
    features: list[dict[str, object]] | None = None,
    dimension_findings: list[dict[str, object]] | None = None,
    field_evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """创建新旧字段并存的最小统一 Extractor 输出。"""

    return {
        "profile": {
            "product_name": "OpenAI API",
            "positioning": positioning,
            "target_users": target_users or [],
            "models": models or [],
            "features": features or [],
            "dimension_findings": dimension_findings or [],
            "field_evidence": field_evidence or [],
            "pricing": [],
            "strengths": [],
            "limitations": [],
        }
    }


class FakeChatModel:
    """模拟 LangChain ChatModel 的 with_structured_output 接口。"""

    def __init__(self, structured_model: FakeExtractorModel) -> None:
        self.structured_model = structured_model
        self.received_schema: type[ExtractorOutput] | None = None
        self.received_method: str | None = None
        self.received_include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type[ExtractorOutput],
        *,
        method: str,
        include_raw: bool,
    ) -> FakeExtractorModel:
        self.received_schema = schema
        self.received_method = method
        self.received_include_raw = include_raw
        return self.structured_model


class FakeRawMessage:
    """模拟 LangChain 在解析失败时返回的原始 AIMessage。"""

    def __init__(self, content: str) -> None:
        self.content = content


class RaisingExtractorModel:
    """模拟模型服务调用失败，用于检查页面可显示的安全定位信息。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """抛出包含敏感文本的异常，Extractor 不应把原文暴露到 public_detail。"""

        raise RuntimeError("secret-token-must-not-be-shown")


class ExtractorTest(unittest.TestCase):
    def test_fixed_evidence_produces_two_grounded_profiles(self) -> None:
        # 两个产品分别提取，所有功能和价格都应引用已有 Evidence。
        fixture = _load_json("extractor_outputs.json")
        model = FakeExtractorModel(
            [fixture["valid_atlas"], fixture["valid_beacon"]]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(
            _build_extractor_input(_load_sample_evidence())
        )

        self.assertEqual(
            [profile.product_name for profile in profiles],
            ["Atlas Notes", "Beacon Docs"],
        )
        self.assertEqual(model.invocation_count, 2)
        self.assertEqual(profiles[0].features[0].evidence_ids, ["E1"])
        self.assertEqual(profiles[0].pricing[0].evidence_ids, ["E2"])
        self.assertIsNone(profiles[1].pricing[0].price)
        self.assertIsNone(profiles[1].pricing[0].billing_cycle)
        self.assertIsNone(profiles[1].positioning)
        self.assertEqual(profiles[1].target_users, [])
        self.assertEqual(
            [item.dimension for item in profiles[0].dimension_findings],
            ["features", "pricing"],
        )

    def test_missing_custom_dimension_is_explicitly_empty(self) -> None:
        # 证据没有治理信息时保留空 finding，不能让模型猜测。
        fixture = _load_json("extractor_outputs.json")
        extractor = Extractor(FakeExtractorModel([fixture["valid_atlas"]]))

        profiles = extractor.extract(
            _build_extractor_input(
                _load_sample_evidence()[:2],
                dimensions=["features", "governance"],
            )
        )

        governance = profiles[0].dimension_findings[1]
        self.assertEqual(governance.dimension, "governance")
        self.assertEqual(governance.facts, [])
        self.assertEqual(governance.evidence_ids, [])

    def test_out_of_scope_evidence_is_rejected_before_model_call(self) -> None:
        # Researcher 分流若被绕过，Extractor 输入边界仍不能接受范围外资料。
        evidence = _load_sample_evidence()[0].model_copy(
            update={
                "scope_status": "out_of_scope",
                "scope_reason": "命中排除项：消费端套餐",
            }
        )

        with self.assertRaises(ValidationError):
            _build_extractor_input([evidence])

    def test_unknown_evidence_id_is_repaired_once(self) -> None:
        # 首次虚构 E99 时，Extractor 应反馈引用错误并接受一次修复。
        fixture = _load_json("extractor_outputs.json")
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [fixture["invalid_reference"], fixture["valid_atlas"]]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(atlas_evidence))

        self.assertEqual(len(profiles), 1)
        self.assertEqual(model.invocation_count, 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn("unknown evidence IDs: E99", repair_message)

    def test_malformed_output_stops_after_one_failed_repair(self) -> None:
        # 连续两次结构错误后停止，防止无限模型调用。
        atlas_evidence = _load_sample_evidence()[:2]
        invalid_output = {"profile": {"product_name": "Wrong product"}}
        model = FakeExtractorModel([invalid_output, invalid_output])
        extractor = Extractor(model)

        with self.assertRaises(ExtractorError):
            extractor.extract(_build_extractor_input(atlas_evidence))

        self.assertEqual(model.invocation_count, 2)

    def test_api_bare_amount_is_left_for_workflow_filtering(self) -> None:
        # Extractor 只校验结构与引用；价格语义由 Analyst 前的统一过滤器处理。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="pricing",
                title="OpenAI API pricing",
                url="https://example.com/openai/pricing",
                snippet="GPT-5.6-sol input pricing.",
                raw_content=(
                    "GPT-5.6-sol input costs $5.00 per million input tokens."
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        ]
        invalid_output = _build_pricing_output(
            "OpenAI API",
            [
                {
                    "plan_name": "GPT-5.6-sol input",
                    "price": "5.00",
                    "unit": "USD",
                    "billing_cycle": None,
                    "service_level": None,
                    "threshold": None,
                    "main_limits": [
                        "input",
                        "cached_input",
                        "output",
                    ],
                    "evidence_ids": ["E1"],
                }
            ],
        )
        model = FakeExtractorModel([invalid_output])

        profiles = Extractor(model).extract(
            _build_api_extractor_input(evidence)
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(profiles[0].pricing[0].price, "5.00")

    def test_gemini_api_price_semantics_do_not_trigger_repair(self) -> None:
        # 单条价格不合格不应让整个产品画像进入修复或全局失败。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Gemini API",
                topic="pricing",
                title="Gemini API pricing",
                url="https://example.com/gemini/pricing",
                snippet=(
                    "Gemini 2.0 Flash input costs $0.15 per 1M input tokens."
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        ]
        invalid_plan = {
            "plan_name": "Gemini 2.0 Flash input",
            "price": "$0.15",
            "unit": "USD",
            "billing_cycle": None,
            "service_level": None,
            "threshold": None,
            "main_limits": [],
            "evidence_ids": ["E1"],
        }
        invalid_model = FakeExtractorModel(
            [_build_pricing_output("Gemini API", [invalid_plan])]
        )

        profiles = Extractor(invalid_model).extract(
            _build_api_extractor_input(evidence)
        )

        self.assertEqual(invalid_model.invocation_count, 1)
        self.assertEqual(profiles[0].pricing[0].price, "$0.15")

        valid_plan = {
            **invalid_plan,
            "price": "$0.15 per 1M input tokens",
            "unit": "per 1M input tokens",
        }
        profiles = Extractor(
            FakeExtractorModel(
                [_build_pricing_output("Gemini API", [valid_plan])]
            )
        ).extract(_build_api_extractor_input(evidence))

        self.assertEqual(
            profiles[0].pricing[0].price,
            "$0.15 per 1M input tokens",
        )

    def test_claude_split_input_and_output_rates_are_valid(self) -> None:
        # 输入和输出可以各自成为一条完整费率，不强制两个方向捆在同一项。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Claude API",
                topic="pricing",
                title="Claude API pricing",
                url="https://example.com/claude/pricing",
                snippet=(
                    "Claude Sonnet 4 input is $3 per million input tokens; "
                    "Claude Sonnet 4 output is $15 per million output tokens."
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        ]
        rates = [
            {
                "plan_name": "Claude Sonnet 4 input",
                "price": "$3 per million input tokens",
                "unit": "per million input tokens",
                "billing_cycle": None,
                "service_level": None,
                "threshold": None,
                "main_limits": [],
                "evidence_ids": ["E1"],
            },
            {
                "plan_name": "Claude Sonnet 4 output",
                "price": "$15 per million output tokens",
                "unit": "per million output tokens",
                "billing_cycle": None,
                "service_level": None,
                "threshold": None,
                "main_limits": [],
                "evidence_ids": ["E1"],
            },
        ]
        model = FakeExtractorModel(
            [_build_pricing_output("Claude API", rates)]
        )

        profiles = Extractor(model).extract(
            _build_api_extractor_input(evidence)
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(len(profiles[0].pricing), 2)

    def test_claude_mtok_rates_are_valid(self) -> None:
        # 官方价表使用 MTok 时，Extractor 的规范化完整费率仍应一次通过。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Claude API",
                topic="api_pricing",
                title="Claude API pricing",
                url="https://example.com/claude/pricing",
                snippet=(
                    "Claude Haiku 4.5 | Input $1 / MTok | "
                    "Output $5 / MTok"
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        ]
        rates = [
            {
                "plan_name": "Claude Haiku 4.5 input",
                "price": "$1 per million input tokens",
                "unit": "per million input tokens",
                "billing_cycle": None,
                "service_level": None,
                "threshold": None,
                "main_limits": [],
                "evidence_ids": ["E1"],
            },
            {
                "plan_name": "Claude Haiku 4.5 output",
                "price": "$5 per million output tokens",
                "unit": "per million output tokens",
                "billing_cycle": None,
                "service_level": None,
                "threshold": None,
                "main_limits": [],
                "evidence_ids": ["E1"],
            },
        ]
        model = FakeExtractorModel(
            [_build_pricing_output("Claude API", rates)]
        )

        profiles = Extractor(model).extract(
            _build_api_extractor_input(evidence)
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(len(profiles[0].pricing), 2)

    def test_unknown_api_price_is_omitted_with_derived_fact(self) -> None:
        # price=null 不能进入 API 画像，也不能在 api_pricing Finding 留下旧事实。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="pricing",
                title="OpenAI API models",
                url="https://example.com/openai/models",
                snippet="OpenAI API lists the GPT-5.5 model.",
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        ]
        output = _build_pricing_output(
            "OpenAI API",
            [
                {
                    "plan_name": "GPT-5.5",
                    "price": None,
                    "unit": None,
                    "billing_cycle": None,
                    "service_level": None,
                    "threshold": None,
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
                {
                    "plan_name": "US-only inference",
                    "price": "USD",
                    "unit": "USD",
                    "billing_cycle": None,
                    "service_level": None,
                    "threshold": None,
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
                {
                    "plan_name": "US-only inference",
                    "price": "1.1x token pricing",
                    "unit": "USD",
                    "billing_cycle": None,
                    "service_level": None,
                    "threshold": None,
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
            ],
        )
        output["profile"]["dimension_findings"] = [
            {
                "dimension": "api_pricing",
                "facts": ["GPT-5.5 has no public price."],
                "evidence_ids": ["E1"],
            }
        ]
        model = FakeExtractorModel([output])

        profiles = Extractor(model).extract(
            _build_api_extractor_input(evidence)
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(profiles[0].pricing, [])
        self.assertEqual(profiles[0].dimension_findings[0].facts, [])
        self.assertEqual(profiles[0].dimension_findings[0].evidence_ids, [])

    def test_pricing_main_limits_objects_are_normalized(self) -> None:
        # 真实模型有时会把限制写成对象；Extractor 应压平成字符串后再校验。
        beacon_evidence = _load_sample_evidence()[2:]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Beacon Docs",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Business",
                                "price": None,
                                "billing_cycle": None,
                                "main_limits": [
                                    {
                                        "name": "user limit",
                                        "description": (
                                            "Supports more than 10 users."
                                        ),
                                    },
                                    {
                                        "description": (
                                            "Provides more storage than "
                                            "the Free plan."
                                        ),
                                    },
                                ],
                                "evidence_ids": ["E4"],
                            }
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(beacon_evidence))

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(
            profiles[0].pricing[0].main_limits,
            [
                "user limit: Supports more than 10 users.",
                "Provides more storage than the Free plan.",
            ],
        )

    def test_pricing_null_main_limits_are_normalized_without_repair(
        self,
    ) -> None:
        # 未知限制的 null 与空列表语义相同，不应触发第二次模型调用。
        fixture = _load_json("extractor_outputs.json")
        fixture["valid_atlas"]["profile"]["pricing"][0][
            "main_limits"
        ] = None
        model = FakeExtractorModel([fixture["valid_atlas"]])

        profiles = Extractor(model).extract(
            _build_extractor_input(_load_sample_evidence()[:2])
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(profiles[0].pricing[0].main_limits, [])

    def test_extractor_messages_compact_long_evidence_text(self) -> None:
        # 多维度真实搜索会产生长摘要，进入模型前应裁剪并移除无关字段。
        evidence = Evidence(
            evidence_id="E1",
            product_name="Atlas Notes",
            topic="features",
            title="Atlas Notes Features",
            url="https://example.com/atlas/features",
            snippet="A" * (EXTRACTOR_SNIPPET_MAX_CHARS + 200),
            raw_content="B" * 4000,
            source_type="official",
            collected_at=datetime(
                2026,
                6,
                22,
                8,
                0,
                tzinfo=timezone.utc,
            ),
        )

        messages = build_extractor_messages(
            product_name="Atlas Notes",
            evidence=[evidence],
            market_definition=_build_market_definition(),
        )

        user_message = messages[-1]["content"]
        self.assertIn("...[truncated]", user_message)
        self.assertIn('"raw_content"', user_message)
        self.assertIn('"scope_status": "in_scope"', user_message)
        self.assertIn('"scope_reason"', user_message)
        self.assertIn("collected_at", user_message)
        self.assertIn('"comparison_level": "企业订阅产品"', user_message)
        self.assertLess(len(user_message), 5000)

    def test_select_evidence_budgets_context_and_keeps_pricing_depth(
        self,
    ) -> None:
        # 先覆盖所有 topic，再把有限的补充上下文优先留给高密度价格资料。
        collected_at = datetime(2026, 6, 22, 8, 0, tzinfo=timezone.utc)
        evidence: list[Evidence] = []
        topics = [
            "positioning",
            "positioning",
            "positioning",
            "target_users",
            "target_users",
            "features",
            "features",
            "features",
            "pricing",
            "pricing",
            "pricing",
        ]
        for index, topic in enumerate(topics, start=1):
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    product_name="ChatGPT",
                    topic=topic,
                    title=f"Evidence {index}",
                    url=f"https://example.com/{index}",
                    snippet=f"Snippet {index}",
                    source_type="official",
                    collected_at=collected_at,
                )
            )

        selected = select_evidence_for_extraction(evidence)

        self.assertEqual(
            len(selected),
            EXTRACTOR_MAX_EVIDENCE_PER_PRODUCT,
        )
        self.assertEqual(
            [item.topic for item in selected],
            [
                "pricing",
                "features",
                "positioning",
                "target_users",
                "pricing",
                "pricing",
            ],
        )

    def test_model_call_failure_exposes_safe_public_detail(self) -> None:
        # Extractor 调用失败时，页面需要能定位产品和输入规模，但不能暴露原始异常。
        evidence = _load_sample_evidence()[:2]
        extractor = Extractor(RaisingExtractorModel())

        with self.assertRaises(ExtractorError) as captured_error:
            extractor.extract(_build_extractor_input(evidence))

        public_detail = captured_error.exception.public_detail
        self.assertIn("产品：Atlas Notes", public_detail)
        self.assertIn("原始证据条数：2", public_detail)
        self.assertIn("送入模型证据条数：2", public_detail)
        self.assertIn("模型输入约", public_detail)
        self.assertIn("底层异常类型：RuntimeError", public_detail)
        self.assertIn("模型输入已自动限制", public_detail)
        self.assertNotIn("secret-token-must-not-be-shown", public_detail)

    def test_prompt_allows_explicit_positioning_and_target_users(self) -> None:
        # 提示词应允许提取官网明说的定位和用户群，但不能让模型反推。
        self.assertIn("官网标题、产品标语或产品概览", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("target_users 可以来自 use case", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("不得从功能名称反推用户", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("定义、正例和反例", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("quote、rationale", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("raw_content", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("价格页正文片段", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("同一模型只能返回一个 ModelProfile", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("context_window_tokens", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("max_output_tokens", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("requests_per_minute", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("搜索召回的补充上下文", EXTRACTOR_SYSTEM_PROMPT)

    def test_missing_positioning_is_filled_from_evidence_sentence(self) -> None:
        # 模型漏填定位时，可以从 positioning Evidence 的明示首句保守补齐。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Beacon Docs",
                topic="positioning",
                title="Beacon Docs Overview",
                url="https://example.com/beacon/overview",
                snippet=(
                    "Beacon Docs is a team workspace where knowledge "
                    "and collaboration meet. It supports shared pages."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    22,
                    8,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Beacon Docs",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertEqual(
            profiles[0].positioning,
            "Beacon Docs is a team workspace where knowledge and collaboration meet.",
        )

    def test_plan_level_positioning_from_pricing_page_is_removed(self) -> None:
        # 套餐适用人群不是产品定位，尤其不能从 pricing 页提升成产品级定位。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Claude",
                topic="pricing",
                title="Claude pricing",
                url="https://www.anthropic.com/pricing",
                snippet=(
                    "Claude Max is for daily users who collaborate often "
                    "with Claude for most tasks."
                ),
                raw_content=(
                    "Max plan for daily users who collaborate often with "
                    "Claude for most tasks. Billed monthly."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Claude",
                        "positioning": (
                            "Claude is positioned for daily users who "
                            "collaborate often with Claude for most tasks."
                        ),
                        "target_users": [],
                        "features": [],
                        "pricing": [],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertIsNone(profiles[0].positioning)

    def test_extractor_does_not_assume_chatgpt_means_api_scope(
        self,
    ) -> None:
        # 市场边界来自 MarketDefinition，Extractor 不再暗含 ChatGPT API 规则。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="ChatGPT",
                topic="pricing",
                title="OpenAI API pricing",
                url="https://platform.openai.com/docs/pricing",
                snippet="OpenAI API pricing lists token-based model prices.",
                raw_content=(
                    "GPT-4.1 input tokens $2.00 / 1M tokens. "
                    "GPT-4.1 output tokens $8.00 / 1M tokens."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            ),
            Evidence(
                evidence_id="E2",
                product_name="ChatGPT",
                topic="pricing",
                title="ChatGPT plans",
                url="https://openai.com/chatgpt/pricing",
                snippet=(
                    "ChatGPT Plus, Pro, Team and Business subscription "
                    "plans are billed per month."
                ),
                raw_content=(
                    "ChatGPT Plus plan $20 per month. ChatGPT Pro plan "
                    "$200 per month. Team plan billed per seat."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "ChatGPT",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "GPT-4.1 input tokens",
                                "price": "$2.00 / 1M tokens",
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                            {
                                "plan_name": "ChatGPT Plus",
                                "price": "$20",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "ChatGPT Team",
                                "price": "$25",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertEqual(
            [plan.plan_name for plan in profiles[0].pricing],
            [
                "GPT-4.1 input tokens",
                "ChatGPT Plus",
                "ChatGPT Team",
            ],
        )

    def test_extractor_does_not_assume_claude_means_api_scope(
        self,
    ) -> None:
        # 没有范围状态排除时，Extractor 不得套用厂商专属 API 规则。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Claude",
                topic="pricing",
                title="Claude pricing",
                url="https://www.anthropic.com/pricing",
                snippet=(
                    "Claude Max is for daily users who collaborate often. "
                    "Monthly subscription plans are available."
                ),
                raw_content=(
                    "Claude Pro plan $20 per month. Claude Max plan for "
                    "daily users who collaborate often with Claude for most "
                    "tasks. Monthly subscription."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Claude",
                        "positioning": (
                            "Claude is for daily users who collaborate "
                            "often."
                        ),
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Claude Max",
                                "price": "$100",
                                "billing_cycle": "monthly",
                                "main_limits": ["daily users"],
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertIsNone(profiles[0].positioning)
        self.assertEqual(len(profiles[0].pricing), 1)
        self.assertEqual(profiles[0].pricing[0].plan_name, "Claude Max")

    def test_conflicting_side_product_prices_are_removed_generically(self) -> None:
        # 无论产品线名称是什么，同一上下文中的冲突价格都应确定性删除。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Gemini",
                topic="pricing",
                title="Gemini Developer API pricing",
                url="https://ai.google.dev/gemini-api/docs/pricing",
                snippet="Gemini API pricing lists token-based model prices.",
                raw_content="Gemini API input tokens $0.10 / 1M tokens.",
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            ),
            Evidence(
                evidence_id="E2",
                product_name="Gemini",
                topic="pricing",
                title="Google Home Premium pricing",
                url="https://gemini.google.com/veo",
                snippet=(
                    "Google Home Premium Standard plan starts at $4.99 "
                    "per month."
                ),
                raw_content=(
                    "Google Home Premium Standard plan $4.99 per month. "
                    "Google Home Premium Standard plan $19.99 per month."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Gemini",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Gemini API input tokens",
                                "price": "$0.10 / 1M tokens",
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                            {
                                "plan_name": (
                                    "Google Home Premium (Standard plan)"
                                ),
                                "price": "$4.99",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": (
                                    "Google Home Premium (Standard plan)"
                                ),
                                "price": "$19.99",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertEqual(len(profiles[0].pricing), 1)
        self.assertEqual(
            profiles[0].pricing[0].plan_name,
            "Gemini API input tokens",
        )

    def test_extractor_does_not_assume_gemini_means_api_scope(self) -> None:
        # Gemini 也只服从显式市场定义，不在 Extractor 中写死产品线。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Gemini",
                topic="pricing",
                title="Gemini API pricing",
                url="https://ai.google.dev/gemini-api/docs/pricing",
                snippet="Gemini API pricing for token input and output.",
                raw_content="Gemini 2.5 Pro input tokens $1.25 / 1M tokens.",
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            ),
            Evidence(
                evidence_id="E2",
                product_name="Gemini",
                topic="pricing",
                title="Gemini for Workspace, Google Home Premium and Veo",
                url="https://workspace.google.com/solutions/ai/",
                snippet=(
                    "Workspace Gemini Business and Enterprise add-ons are "
                    "priced per user per month. Google Home Premium and "
                    "Veo are consumer subscription plans."
                ),
                raw_content=(
                    "Gemini Business plan $20 per user per month. Gemini "
                    "Enterprise plan $30 per user per month. Google Home "
                    "Premium Standard plan $4.99 per month. Veo plan "
                    "subscription $19.99 per month."
                ),
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Gemini",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Gemini 2.5 Pro input tokens",
                                "price": "$1.25 / 1M tokens",
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                            {
                                "plan_name": "Workspace Gemini Business",
                                "price": "$20",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Veo subscription",
                                "price": "$19.99",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertEqual(
            [plan.plan_name for plan in profiles[0].pricing],
            [
                "Gemini 2.5 Pro input tokens",
                "Workspace Gemini Business",
                "Veo subscription",
            ],
        )

    def test_conflicting_duplicate_pricing_plan_is_removed(self) -> None:
        # 同一套餐同一计费周期出现两个价格时，个人项目里先删除而不是猜哪个对。
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Atlas Notes",
                topic="pricing",
                title="Atlas Notes pricing",
                url="https://example.com/atlas/pricing",
                snippet="Standard plan $4.99/month. Standard plan $19.99/month.",
                source_type="official",
                collected_at=datetime(
                    2026,
                    6,
                    24,
                    2,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
        ]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Standard",
                                "price": "$4.99",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                            {
                                "plan_name": "Standard",
                                "price": "$19.99",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                            {
                                "plan_name": "Pro",
                                "price": "$29",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E1"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(evidence))

        self.assertEqual(
            [pricing.plan_name for pricing in profiles[0].pricing],
            ["Pro"],
        )

    def test_different_pricing_units_are_not_merged(self) -> None:
        # 同名方案按不同单位计费时必须分别保留，不能伪装成价格冲突。
        evidence = [_load_sample_evidence()[1]]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Team",
                                "price": "$10",
                                "unit": "per user",
                                "billing_cycle": "monthly",
                                "service_level": "standard",
                                "threshold": None,
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Team",
                                "price": "$100",
                                "unit": "per workspace",
                                "billing_cycle": "monthly",
                                "service_level": "standard",
                                "threshold": "up to 50 users",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )

        profiles = Extractor(model).extract(
            _build_extractor_input(evidence, dimensions=["pricing"])
        )

        self.assertEqual(len(profiles[0].pricing), 2)
        self.assertEqual(
            [plan.unit for plan in profiles[0].pricing],
            ["per user", "per workspace"],
        )

    def test_pricing_defaults_are_normalized_from_plan_text(self) -> None:
        # Free 方案和价格文本中的 /month 可以由代码确定性补齐。
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Free",
                                "price": None,
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Plus",
                                "price": "$10 per seat/month",
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(atlas_evidence))

        self.assertEqual(profiles[0].pricing[0].price, "$0")
        self.assertIsNone(profiles[0].pricing[0].billing_cycle)
        self.assertEqual(profiles[0].pricing[1].billing_cycle, "monthly")

    def test_free_price_clears_model_supplied_billing_cycle(self) -> None:
        # 免费价旁边的每月使用限制不能被误保留成 monthly billing。
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Free",
                                "price": "Free",
                                "billing_cycle": "monthly",
                                "main_limits": ["10 users"],
                                "evidence_ids": ["E2"],
                            }
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(atlas_evidence))

        self.assertEqual(profiles[0].pricing[0].price, "Free")
        self.assertIsNone(profiles[0].pricing[0].billing_cycle)

    def test_pricing_normalization_removes_status_and_duplicates(
        self,
    ) -> None:
        # 真实价格页会混入 $0 周期、Beta 状态和 monthly credits 文案。
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": "Free",
                                "price": "$0 per seat/month",
                                "billing_cycle": "per month",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Workers",
                                "price": None,
                                "billing_cycle": "Beta",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Custom Agents",
                                "price": (
                                    "Free to try, then $10 per 1,000 "
                                    "monthly Notion credits"
                                ),
                                "billing_cycle": None,
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                            {
                                "plan_name": "Enterprise",
                                "price": "Custom pricing",
                                "billing_cycle": "annual",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            },
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(_build_extractor_input(atlas_evidence))
        pricing = profiles[0].pricing

        self.assertEqual(pricing[0].price, "$0")
        self.assertIsNone(pricing[0].billing_cycle)
        self.assertIsNone(pricing[1].billing_cycle)
        self.assertIsNone(pricing[2].billing_cycle)
        self.assertEqual(pricing[3].price, "Custom pricing")
        self.assertIsNone(pricing[3].billing_cycle)

    def test_pricing_without_plan_name_is_discarded(self) -> None:
        # 无法识别套餐名时宁可丢弃该项，也不能猜测名称。
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [
                {
                    "profile": {
                        "product_name": "Atlas Notes",
                        "positioning": None,
                        "target_users": [],
                        "features": [],
                        "pricing": [
                            {
                                "plan_name": None,
                                "price": "$10",
                                "billing_cycle": "monthly",
                                "main_limits": [],
                                "evidence_ids": ["E2"],
                            }
                        ],
                        "strengths": [],
                        "limitations": [],
                    }
                }
            ]
        )

        profiles = Extractor(model).extract(
            _build_extractor_input(atlas_evidence)
        )

        self.assertEqual(profiles[0].pricing, [])
        self.assertEqual(model.invocation_count, 1)

    def test_langchain_wrapper_binds_extractor_output_schema(self) -> None:
        # 真实模型边界必须绑定 ExtractorOutput，并保留 raw 解析结果。
        fixture = _load_json("extractor_outputs.json")
        structured_model = FakeExtractorModel([fixture["valid_atlas"]])
        chat_model = FakeChatModel(structured_model)
        extractor_model = LangChainExtractorModel(chat_model)
        extractor = Extractor(extractor_model)

        profiles = extractor.extract(
            _build_extractor_input(_load_sample_evidence()[:2])
        )

        self.assertIs(chat_model.received_schema, ExtractorOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertEqual(len(profiles), 1)

    def test_langchain_parse_failure_enters_repair_flow(self) -> None:
        # LangChain 解析失败时，原始文本仍应进入一次修复流程。
        fixture = _load_json("extractor_outputs.json")
        invalid_json = json.dumps(
            {"profile": {"product_name": "Wrong product"}},
            ensure_ascii=False,
        )
        responses = [
            {
                "raw": FakeRawMessage(invalid_json),
                "parsed": None,
                "parsing_error": ValueError("fixture parse failure"),
            },
            {
                "raw": FakeRawMessage(""),
                "parsed": ExtractorOutput.model_validate(
                    fixture["valid_atlas"]
                ),
                "parsing_error": None,
            },
        ]
        structured_model = FakeExtractorModel(responses)
        extractor_model = LangChainExtractorModel(
            FakeChatModel(structured_model)
        )
        extractor = Extractor(extractor_model)

        profiles = extractor.extract(
            _build_extractor_input(_load_sample_evidence()[:2])
        )

        self.assertEqual(len(profiles), 1)
        self.assertEqual(structured_model.invocation_count, 2)

    def test_duplicate_evidence_ids_are_rejected_before_model_call(self) -> None:
        # 重复 ID 无法建立唯一引用关系，应在模型调用前拒绝。
        evidence = _load_sample_evidence()

        with self.assertRaises(ValidationError):
            _build_extractor_input([evidence[0], evidence[0]])

    def test_repair_message_repeats_required_nested_fields(self) -> None:
        # 真实模型漏字段时，修复指令必须明确重复嵌套对象契约。
        repair_messages = build_repair_messages(
            initial_messages=[],
            raw_output={"profile": {"features": [{"name": "Feature"}]}},
            validation_error="description is required",
        )

        repair_content = repair_messages[-1]["content"]
        self.assertIn(
            "name、description、evidence_ids",
            repair_content,
        )
        self.assertIn(
            "plan_name、price、unit、billing_cycle",
            repair_content,
        )
        self.assertIn("同一模型只保留一个 ModelProfile", repair_content)
        self.assertIn("input、output、cached input、audio 和 Batch", repair_content)
        self.assertIn("effective_from、effective_to", repair_content)
        self.assertIn("完整对象，不能只写 evidence_id 字符串", repair_content)
        self.assertIn("quote、rationale", repair_content)
        self.assertIn("price=null 的条目必须整条删除", repair_content)
        self.assertIn("非价格文本也必须整条删除", repair_content)
        self.assertIn("倍率不是独立绝对费率", repair_content)

    def test_authentication_instruction_cannot_be_positioning(self) -> None:
        # 认证步骤即使被模型逐字复制，也必须降级为缺失定位。
        evidence = _build_api_evidence(
            topic="developer_platform",
            snippet="Authenticate with access tokens before sending requests.",
        )
        output = _build_unified_output(
            positioning="Authenticate with access tokens before sending requests.",
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0]

        self.assertIsNone(profile.positioning)

    def test_input_and_output_prices_merge_into_one_model(self) -> None:
        # 同一模型的输入和输出价格不能成为两个模型或两个套餐。
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet="GPT-5 costs $2 input and $10 output per 1M tokens.",
        )
        source = _build_source_reference(evidence)
        common = {
            "model_name": "GPT-5",
            "source_evidence": [source],
            "extraction_confidence": 0.95,
        }
        output = _build_unified_output(
            models=[
                {
                    **common,
                    "pricing": {
                        "input_price": {
                            "amount": "2",
                            "currency": "USD",
                            "per_quantity": 1_000_000,
                            "unit": "token",
                            "evidence_ids": ["E1"],
                        }
                    },
                },
                {
                    **common,
                    "pricing": {
                        "output_price": {
                            "amount": "10",
                            "currency": "USD",
                            "per_quantity": 1_000_000,
                            "unit": "token",
                            "evidence_ids": ["E1"],
                        }
                    },
                },
            ]
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0]

        self.assertEqual(len(profile.models), 1)
        self.assertEqual(
            profile.models[0].pricing.input_price.amount,
            Decimal("2"),
        )
        self.assertEqual(
            profile.models[0].pricing.output_price.amount,
            Decimal("10"),
        )
        self.assertTrue(profile.dimension_findings[0].facts)

    def test_source_evidence_ids_and_condition_tier_prices_are_normalized(self) -> None:
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet=(
                "GPT-5 standard input is $2 and output is $8 per 1M tokens. "
                "Priority input is $4 and output is $16 per 1M tokens."
            ),
        )
        def price(amount: str, condition: str) -> dict[str, object]:
            return {
                "amount": amount,
                "currency": "USD",
                "per_quantity": 1_000_000,
                "unit": "token",
                "condition": condition,
                "evidence_ids": ["E1"],
            }
        output = _build_unified_output(
            models=[
                {
                    "model_name": "GPT-5",
                    "pricing": {
                        "input_price": [price("2", "Standard"), price("4", "Priority")],
                        "output_price": [price("8", "Standard"), price("16", "Priority")],
                    },
                    "source_evidence": ["E1"],
                    "extraction_confidence": 0.95,
                }
            ]
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0]

        self.assertEqual(profile.models[0].source_evidence[0].evidence_id, "E1")
        self.assertEqual(len(profile.models[0].pricing.input_price), 2)

    def test_same_price_type_merges_different_effective_periods(self) -> None:
        # 同一模型两段有效期价格必须合并为同一价格类型的历史列表。
        first_price = {"amount": "2", "effective_to": "2026-06-30"}
        second_price = {"amount": "2.5", "effective_from": "2026-07-01"}

        merged = merge_model_items(
            [
                {
                    "model_name": "GPT-5",
                    "pricing": {"input_price": first_price},
                },
                {
                    "model_name": "gpt-5",
                    "pricing": {"input_price": second_price},
                },
            ]
        )

        self.assertEqual(
            merged[0]["pricing"]["input_price"],
            [first_price, second_price],
        )

    def test_cache_audio_and_batch_prices_keep_distinct_fields(self) -> None:
        # 缓存、音频和 Batch 价格必须保留各自分类与明确有效期。
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet=(
                "Cached input is $0.20 per 1M tokens. "
                "Audio is $32 per 1M tokens. "
                "Batch input is $1 per 1M tokens, effective 2026-01-01 "
                "through 2026-06-30."
            ),
        )
        source = _build_source_reference(evidence)

        def price_rate(amount: str) -> dict[str, object]:
            return {
                "amount": amount,
                "currency": "USD",
                "per_quantity": 1_000_000,
                "unit": "token",
                "evidence_ids": ["E1"],
            }

        output = _build_unified_output(
            models=[
                {
                    "model_name": "GPT-5",
                    "batch_api": "supported",
                    "pricing": {
                        "cached_input_price": price_rate("0.20"),
                        "audio_price": price_rate("32"),
                        "batch_input_price": {
                            **price_rate("1"),
                            "effective_from": "2026-01-01",
                            "effective_to": "2026-06-30",
                        },
                    },
                    "source_evidence": [source],
                    "extraction_confidence": 0.95,
                }
            ],
            field_evidence=[
                {
                    "field_path": "models.GPT-5.pricing.cached_input_price",
                    "evidence_id": "E1",
                    "quote": "Cached input is $0.20 per 1M tokens.",
                    "rationale": "The quote labels cached input pricing.",
                    "confidence": 0.95,
                },
                {
                    "field_path": "models.GPT-5.pricing.audio_price",
                    "evidence_id": "E1",
                    "quote": "Audio is $32 per 1M tokens.",
                    "rationale": "The quote labels audio pricing.",
                    "confidence": 0.95,
                },
                {
                    "field_path": "models.GPT-5.pricing.batch_input_price",
                    "evidence_id": "E1",
                    "quote": (
                        "Batch input is $1 per 1M tokens, effective "
                        "2026-01-01 through 2026-06-30."
                    ),
                    "rationale": "The quote labels Batch pricing and dates.",
                    "confidence": 0.95,
                },
            ],
        )

        pricing = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0].models[0].pricing

        self.assertEqual(pricing.cached_input_price.amount, Decimal("0.20"))
        self.assertEqual(pricing.audio_price.amount, Decimal("32"))
        self.assertEqual(pricing.batch_input_price.amount, Decimal("1"))
        self.assertEqual(
            pricing.batch_input_price.effective_to.isoformat(),
            "2026-06-30",
        )

    def test_context_and_max_output_use_separate_fields(self) -> None:
        # 上下文窗口和最大输出必须保持两个独立 Token 上限。
        quote = "GPT-5 has a 1M context window and 128k max output tokens."
        evidence = _build_api_evidence(
            topic="model_capabilities",
            snippet=quote,
        )
        output = _build_unified_output(
            models=[
                {
                    "model_name": "GPT-5",
                    "context_window_tokens": 1_000_000,
                    "max_output_tokens": 128_000,
                    "source_evidence": [_build_source_reference(evidence)],
                    "extraction_confidence": 0.95,
                }
            ],
            field_evidence=[
                {
                    "field_path": "models.GPT-5.context_window_tokens",
                    "evidence_id": "E1",
                    "quote": quote,
                    "rationale": "The quote explicitly labels the context window.",
                    "confidence": 0.95,
                },
                {
                    "field_path": "models.GPT-5.max_output_tokens",
                    "evidence_id": "E1",
                    "quote": quote,
                    "rationale": "The quote explicitly labels maximum output.",
                    "confidence": 0.95,
                },
            ],
        )

        model = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0].models[0]

        self.assertEqual(model.context_window_tokens, 1_000_000)
        self.assertEqual(model.max_output_tokens, 128_000)

    def test_rate_limit_cannot_be_positioning(self) -> None:
        # RPM、TPM、RPD 属于限流字段，不能提升为产品定位。
        quote = "Rate limits: 500 RPM, 2,000,000 TPM, and 10,000 RPD."
        evidence = _build_api_evidence(
            topic="usage_limits",
            snippet=quote,
        )
        output = _build_unified_output(
            positioning=quote,
            models=[
                {
                    "model_name": "GPT-5",
                    "rate_limits": [
                        {
                            "metric": "requests_per_minute",
                            "limit": 500,
                            "evidence_ids": ["E1"],
                        },
                        {
                            "metric": "tokens_per_minute",
                            "limit": 2_000_000,
                            "evidence_ids": ["E1"],
                        },
                        {
                            "metric": "requests_per_day",
                            "limit": 10_000,
                            "evidence_ids": ["E1"],
                        },
                    ],
                    "source_evidence": [_build_source_reference(evidence)],
                    "extraction_confidence": 0.95,
                }
            ],
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0]

        self.assertIsNone(profile.positioning)
        self.assertEqual(
            [limit.metric.value for limit in profile.models[0].rate_limits],
            [
                "requests_per_minute",
                "tokens_per_minute",
                "requests_per_day",
            ],
        )

    def test_feature_facts_override_false_missing_finding(self) -> None:
        # features 已有证据时，模型返回的空维度不能让报告显示资料不足。
        evidence = _build_api_evidence(
            topic="model_capabilities",
            snippet="OpenAI API supports function calling and structured outputs.",
        )
        output = _build_unified_output(
            features=[
                {
                    "name": "Function calling",
                    "description": "The API supports function calling.",
                    "evidence_ids": ["E1"],
                }
            ],
            dimension_findings=[
                {
                    "dimension": "model_capabilities",
                    "facts": [],
                    "evidence_ids": [],
                }
            ],
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_extractor_input(
                [evidence],
                dimensions=["model_capabilities"],
            )
        )[0]

        self.assertEqual(
            profile.dimension_findings[0].facts,
            ["Function calling: The API supports function calling."],
        )

    def test_instructions_and_rate_limits_are_not_features(self) -> None:
        # 认证步骤和限流说明不是可比较的产品功能。
        evidence = _build_api_evidence(
            topic="features",
            snippet=(
                "Authenticate with access tokens. "
                "Rate limits are 500 RPM."
            ),
        )
        output = _build_unified_output(
            features=[
                {
                    "name": "Authenticate with access tokens",
                    "description": "Authenticate with access tokens.",
                    "evidence_ids": ["E1"],
                },
                {
                    "name": "Rate limits",
                    "description": "Rate limits are 500 RPM.",
                    "evidence_ids": ["E1"],
                },
            ]
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_extractor_input([evidence], dimensions=["features"])
        )[0]

        self.assertEqual(profile.features, [])
        self.assertEqual(profile.dimension_findings[0].facts, [])

    def test_unsupported_target_user_is_removed(self) -> None:
        # 只有直接描述目标群体的文本才能支持 target_users。
        evidence = _build_api_evidence(
            topic="target_users",
            snippet=(
                "Built for enterprise engineering teams. "
                "The API also supports dashboards."
            ),
        )
        output = _build_unified_output(
            target_users=["enterprise engineering teams", "marketing teams"]
        )

        profile = Extractor(FakeExtractorModel([output])).extract(
            _build_api_extractor_input([evidence])
        )[0]

        self.assertEqual(profile.target_users, ["enterprise engineering teams"])

    def test_field_evidence_quote_must_exist_and_meet_confidence(self) -> None:
        # 字段说明不能引用不存在的原文，高层字段置信度必须更高。
        evidence = _build_api_evidence(
            topic="positioning",
            snippet="OpenAI API is an AI platform for developers.",
        )
        output = _build_unified_output(
            positioning="OpenAI API is an AI platform for developers.",
            field_evidence=[
                {
                    "field_path": "positioning",
                    "evidence_id": "E1",
                    "quote": "A quote that is absent from the evidence.",
                    "rationale": "Claims this defines the product.",
                    "confidence": 0.7,
                }
            ],
        )

        with self.assertRaises(ExtractorValidationError):
            validate_extractor_output(
                raw_output=output,
                product_name="OpenAI API",
                evidence=[evidence],
                market_definition=_build_api_extractor_input(
                    [evidence]
                ).market_definition,
            )

    def test_common_field_evidence_and_dimension_aliases_are_normalized(self) -> None:
        normalized = normalize_extractor_raw_output(
            {
                "profile": {
                    "field_evidence": [
                        {
                            "field_path": "features.shared workspace",
                            "evidence_id": "E1",
                            "evidence_quote": "Shared workspace.",
                            "field_rationale": "Direct feature statement.",
                            "confidence": 0.9,
                        }
                    ],
                    "dimension_findings": {
                        "features": {
                            "facts": ["Shared workspace."],
                            "evidence_ids": ["E1"],
                        }
                    },
                }
            }
        )

        profile = normalized["profile"]
        self.assertEqual(profile["field_evidence"][0]["quote"], "Shared workspace.")
        self.assertNotIn("evidence_quote", profile["field_evidence"][0])
        self.assertEqual(profile["dimension_findings"][0]["dimension"], "features")

    def test_features_without_descriptions_are_dropped(self) -> None:
        normalized = normalize_extractor_raw_output(
            {
                "profile": {
                    "features": [
                        {"name": "Incomplete", "description": None},
                        {"name": "Complete", "description": "A capability."},
                    ]
                }
            }
        )

        self.assertEqual(
            normalized["profile"]["features"],
            [{"name": "Complete", "description": "A capability."}],
        )

    def test_missing_feature_description_is_recovered_from_its_evidence(self) -> None:
        evidence = _build_api_evidence(
            topic="features",
            snippet="Atlas Notes provides reusable templates for project plans.",
        )
        normalized = normalize_extractor_raw_output(
            {
                "profile": {
                    "features": [
                        {
                            "name": "reusable templates",
                            "description": None,
                            "evidence_ids": ["E1"],
                        }
                    ]
                }
            },
            [evidence],
        )

        self.assertEqual(
            normalized["profile"]["features"][0]["description"],
            "Atlas Notes provides reusable templates for project plans.",
        )

    def test_field_evidence_quote_not_in_evidence_is_dropped(self) -> None:
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet="GPT-5 input is $2 per 1M tokens.",
        )
        normalized = normalize_extractor_raw_output(
            {
                "profile": {
                    "field_evidence": [
                        {
                            "field_path": "models",
                            "evidence_id": "E1",
                            "quote": "GPT-5 input is $2...",
                            "rationale": "Pricing.",
                            "confidence": 0.9,
                        }
                    ]
                }
            },
            [evidence],
        )

        self.assertEqual(normalized["profile"]["field_evidence"], [])

    def test_bare_model_price_cell_in_markdown_table_is_dropped(self) -> None:
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet="| Model | Input | Output |\n| GPT-5 | $2.00 | $8.00 |",
        )
        normalized = normalize_extractor_raw_output(
            {
                "profile": {
                    "field_evidence": [
                        {
                            "field_path": "models.GPT-5.pricing.input_price",
                            "evidence_id": "E1",
                            "quote": "$2.00",
                            "rationale": "Pricing table cell.",
                            "confidence": 0.9,
                        }
                    ]
                }
            },
            [evidence],
        )

        self.assertEqual(normalized["profile"]["field_evidence"], [])

    def test_missing_high_level_field_cannot_have_field_evidence(self) -> None:
        evidence = _build_api_evidence(
            topic="model_capabilities",
            snippet="OpenAI API supports function calling.",
        )
        output = _build_unified_output(
            field_evidence=[
                {
                    "field_path": "positioning",
                    "evidence_id": "E1",
                    "quote": "OpenAI API supports function calling.",
                    "rationale": "A feature is not positioning.",
                    "confidence": 0.95,
                }
            ],
        )

        with self.assertRaises(ExtractorValidationError):
            validate_extractor_output(
                raw_output=output,
                product_name="OpenAI API",
                evidence=[evidence],
                market_definition=_build_api_extractor_input(
                    [evidence]
                ).market_definition,
            )

    def test_cached_quote_cannot_support_plain_input_price(self) -> None:
        # quote 存在也不够，cached input 不能被挂到普通 input_price。
        evidence = _build_api_evidence(
            topic="api_pricing",
            snippet="Cached input is $0.20 per 1M tokens.",
        )
        source = _build_source_reference(evidence)
        output = _build_unified_output(
            models=[
                {
                    "model_name": "GPT-5",
                    "pricing": {
                        "input_price": {
                            "amount": "0.20",
                            "currency": "USD",
                            "per_quantity": 1_000_000,
                            "unit": "token",
                            "evidence_ids": ["E1"],
                        }
                    },
                    "source_evidence": [source],
                    "extraction_confidence": 0.95,
                }
            ],
            field_evidence=[
                {
                    "field_path": "models.GPT-5.pricing.input_price",
                    "evidence_id": "E1",
                    "quote": "Cached input is $0.20 per 1M tokens.",
                    "rationale": "Incorrectly treats cached input as input.",
                    "confidence": 0.95,
                }
            ],
        )

        with self.assertRaises(ExtractorValidationError):
            validate_extractor_output(
                raw_output=output,
                product_name="OpenAI API",
                evidence=[evidence],
                market_definition=_build_api_extractor_input(
                    [evidence]
                ).market_definition,
            )

    def test_official_evidence_is_selected_before_third_party(self) -> None:
        # 同 topic 有多个来源时，官方资料必须先进入有限上下文。
        third_party = _build_api_evidence(
            topic="features",
            snippet="Community summary.",
            evidence_id="E1",
            source_type="third_party",
        )
        official = _build_api_evidence(
            topic="features",
            snippet="Official product capabilities.",
            evidence_id="E2",
        )

        selected = select_evidence_for_extraction([third_party, official])

        self.assertEqual(selected[0].evidence_id, "E2")


if __name__ == "__main__":
    unittest.main()
