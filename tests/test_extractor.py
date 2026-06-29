import json
import unittest
from datetime import datetime, timezone
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
    FakeExtractorModel,
    LangChainExtractorModel,
    build_extractor_messages,
    build_repair_messages,
    select_evidence_for_extraction,
)
from competitive_analysis_agent.schemas import Evidence, WorkflowState


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
            ExtractorInput(evidence=_load_sample_evidence())
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

    def test_unknown_evidence_id_is_repaired_once(self) -> None:
        # 首次虚构 E99 时，Extractor 应反馈引用错误并接受一次修复。
        fixture = _load_json("extractor_outputs.json")
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [fixture["invalid_reference"], fixture["valid_atlas"]]
        )
        extractor = Extractor(model)

        profiles = extractor.extract(ExtractorInput(evidence=atlas_evidence))

        self.assertEqual(len(profiles), 1)
        self.assertEqual(model.invocation_count, 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn("unknown evidence IDs: E99", repair_message)

    def test_malformed_output_stops_after_one_failed_repair(self) -> None:
        # 连续两次结构错误后停止，防止无限模型调用。
        fixture = _load_json("extractor_outputs.json")
        atlas_evidence = _load_sample_evidence()[:2]
        model = FakeExtractorModel(
            [fixture["invalid_shape"], fixture["invalid_shape"]]
        )
        extractor = Extractor(model)

        with self.assertRaises(ExtractorError):
            extractor.extract(ExtractorInput(evidence=atlas_evidence))

        self.assertEqual(model.invocation_count, 2)

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

        profiles = extractor.extract(ExtractorInput(evidence=beacon_evidence))

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(
            profiles[0].pricing[0].main_limits,
            [
                "user limit: Supports more than 10 users.",
                "Provides more storage than the Free plan.",
            ],
        )

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
        )

        user_message = messages[-1]["content"]
        self.assertIn("...[truncated]", user_message)
        self.assertIn('"raw_content"', user_message)
        self.assertNotIn("collected_at", user_message)
        self.assertLess(len(user_message), 5000)

    def test_select_evidence_round_robins_topics_before_model_call(
        self,
    ) -> None:
        # 单产品证据很多时，应跨 topic 取代表项，而不是只保留最前面的网页。
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
                "features",
            ],
        )

    def test_model_call_failure_exposes_safe_public_detail(self) -> None:
        # Extractor 调用失败时，页面需要能定位产品和输入规模，但不能暴露原始异常。
        evidence = _load_sample_evidence()[:2]
        extractor = Extractor(RaisingExtractorModel())

        with self.assertRaises(ExtractorError) as captured_error:
            extractor.extract(ExtractorInput(evidence=evidence))

        public_detail = captured_error.exception.public_detail
        self.assertIn("产品：Atlas Notes", public_detail)
        self.assertIn("原始证据条数：2", public_detail)
        self.assertIn("送入模型证据条数：2", public_detail)
        self.assertIn("模型输入约", public_detail)
        self.assertIn("底层异常类型：RuntimeError", public_detail)
        self.assertNotIn("secret-token-must-not-be-shown", public_detail)

    def test_prompt_allows_explicit_positioning_and_target_users(self) -> None:
        # 提示词应允许提取官网明说的定位和用户群，但不能让模型反推。
        self.assertIn("官网标题、产品标语或产品概览", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("target_users 可以来自 use case", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("不得从功能名称反推用户", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("raw_content", EXTRACTOR_SYSTEM_PROMPT)
        self.assertIn("价格页正文片段", EXTRACTOR_SYSTEM_PROMPT)

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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertIsNone(profiles[0].positioning)

    def test_chatgpt_subscription_pricing_is_removed_by_default_api_scope(
        self,
    ) -> None:
        # 默认 ChatGPT pricing 是 API/token 范围，Plus/Pro/Team 等订阅价不能进画像。
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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertEqual(len(profiles[0].pricing), 1)
        self.assertEqual(
            profiles[0].pricing[0].plan_name,
            "GPT-4.1 input tokens",
        )

    def test_claude_subscription_pricing_is_removed_by_default_api_scope(
        self,
    ) -> None:
        # Claude Max/Pro 是消费端订阅语境，默认 API 价格画像应保持 pricing 为空。
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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertIsNone(profiles[0].positioning)
        self.assertEqual(profiles[0].pricing, [])

    def test_gemini_home_premium_pricing_is_removed(self) -> None:
        # Gemini 官网域名也可能出现其他 Google 产品价格，应从 Gemini 画像中删除。
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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertEqual(len(profiles[0].pricing), 1)
        self.assertEqual(
            profiles[0].pricing[0].plan_name,
            "Gemini API input tokens",
        )

    def test_gemini_workspace_veo_and_home_pricing_are_removed(self) -> None:
        # Gemini 默认 API pricing 不应接收 Workspace、Veo、Home 等非 API 产品价格。
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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertEqual(len(profiles[0].pricing), 1)
        self.assertEqual(
            profiles[0].pricing[0].plan_name,
            "Gemini 2.5 Pro input tokens",
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

        profiles = extractor.extract(ExtractorInput(evidence=evidence))

        self.assertEqual(
            [pricing.plan_name for pricing in profiles[0].pricing],
            ["Pro"],
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

        profiles = extractor.extract(ExtractorInput(evidence=atlas_evidence))

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

        profiles = extractor.extract(ExtractorInput(evidence=atlas_evidence))

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

        profiles = extractor.extract(ExtractorInput(evidence=atlas_evidence))
        pricing = profiles[0].pricing

        self.assertEqual(pricing[0].price, "$0")
        self.assertIsNone(pricing[0].billing_cycle)
        self.assertIsNone(pricing[1].billing_cycle)
        self.assertIsNone(pricing[2].billing_cycle)
        self.assertEqual(pricing[3].price, "Custom pricing")
        self.assertIsNone(pricing[3].billing_cycle)

    def test_langchain_wrapper_binds_extractor_output_schema(self) -> None:
        # 真实模型边界必须绑定 ExtractorOutput，并保留 raw 解析结果。
        fixture = _load_json("extractor_outputs.json")
        structured_model = FakeExtractorModel([fixture["valid_atlas"]])
        chat_model = FakeChatModel(structured_model)
        extractor_model = LangChainExtractorModel(chat_model)
        extractor = Extractor(extractor_model)

        profiles = extractor.extract(
            ExtractorInput(evidence=_load_sample_evidence()[:2])
        )

        self.assertIs(chat_model.received_schema, ExtractorOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertEqual(len(profiles), 1)

    def test_langchain_parse_failure_enters_repair_flow(self) -> None:
        # LangChain 解析失败时，原始文本仍应进入一次修复流程。
        fixture = _load_json("extractor_outputs.json")
        invalid_json = json.dumps(
            fixture["invalid_shape"],
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
            ExtractorInput(evidence=_load_sample_evidence()[:2])
        )

        self.assertEqual(len(profiles), 1)
        self.assertEqual(structured_model.invocation_count, 2)

    def test_duplicate_evidence_ids_are_rejected_before_model_call(self) -> None:
        # 重复 ID 无法建立唯一引用关系，应在模型调用前拒绝。
        evidence = _load_sample_evidence()

        with self.assertRaises(ValidationError):
            ExtractorInput(evidence=[evidence[0], evidence[0]])

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
            "plan_name、price、billing_cycle",
            repair_content,
        )


if __name__ == "__main__":
    unittest.main()
