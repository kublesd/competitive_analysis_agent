import json
import unittest
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.analyst import (
    ANALYST_SYSTEM_PROMPT,
    Analyst,
    AnalystInput,
    AnalystOutput,
    FakeAnalystModel,
    LangChainAnalystModel,
    build_analyst_messages,
    collect_analysis_claims,
    build_fallback_pricing_claims,
    contains_pricing_language,
    format_fallback_pricing_claim,
)
from competitive_analysis_agent.schemas import (
    Currency,
    DimensionFinding,
    MarketDefinition,
    ModelPricing,
    ModelProfile,
    PriceRate,
    PricingUnit,
    ProductProfile,
    SourceReference,
    SourceType,
    SupportStatus,
    WorkflowState,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


def _load_json(file_name: str) -> dict:
    """读取固定 JSON，确保 Analyst 单元测试不调用真实模型。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _load_sample_profiles() -> list[ProductProfile]:
    """从 Stage 1 样例读取两个带 Evidence ID 的产品画像。"""

    sample_case = _load_json("sample_case.json")
    workflow_state = WorkflowState.model_validate(sample_case)
    return workflow_state.product_profiles


def _build_market_definition() -> MarketDefinition:
    """创建 Analyst 范围与目标购买者测试使用的固定市场。"""

    return MarketDefinition(
        market_name="Team knowledge workspace",
        product_category="SaaS collaboration software",
        target_buyer="Mid-sized company IT and business leaders",
        comparison_level="Team subscription product",
        core_dimensions=["features", "pricing"],
        exclusions=["consumer plans", "API usage pricing"],
    )


class FakeChatModel:
    """模拟 LangChain ChatModel 的 with_structured_output 接口。"""

    def __init__(self, structured_model: FakeAnalystModel) -> None:
        self.structured_model = structured_model
        self.received_schema: type[AnalystOutput] | None = None
        self.received_method: str | None = None
        self.received_include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type[AnalystOutput],
        *,
        method: str,
        include_raw: bool,
    ) -> FakeAnalystModel:
        self.received_schema = schema
        self.received_method = method
        self.received_include_raw = include_raw
        return self.structured_model


class AnalystTest(unittest.TestCase):
    def test_model_input_and_output_prices_keep_complete_semantics(self) -> None:
        # 同一模型的价格方向必须留在 claim 中，不能退化成两个同名套餐数字。
        source = SourceReference(
            evidence_id="E1",
            title="Claude pricing",
            url="https://docs.anthropic.com/pricing",
            source_type=SourceType.OFFICIAL,
            collected_at="2026-07-20T00:00:00Z",
        )
        profile = ProductProfile(
            product_name="Claude API",
            models=[
                ModelProfile(
                    model_name="Claude Sonnet 5",
                    pricing=ModelPricing(
                        input_price=PriceRate(
                            amount="2",
                            currency=Currency.USD,
                            per_quantity=1_000_000,
                            unit=PricingUnit.TOKEN,
                            evidence_ids=["E1"],
                        ),
                        output_price=PriceRate(
                            amount="10",
                            currency=Currency.USD,
                            per_quantity=1_000_000,
                            unit=PricingUnit.TOKEN,
                            evidence_ids=["E1"],
                        ),
                    ),
                    batch_api=SupportStatus.MISSING,
                    source_evidence=[source],
                    extraction_confidence=0.98,
                )
            ],
        )

        claims = [
            item.claim for item in build_fallback_pricing_claims([profile])
        ]

        self.assertEqual(
            claims,
            [
                "Claude Sonnet 5 input price is $2 per million tokens.",
                "Claude Sonnet 5 output price is $10 per million tokens.",
            ],
        )
        self.assertFalse(any("plan is" in claim for claim in claims))

    def test_valid_output_compares_all_products_with_citations(self) -> None:
        # 固定输出应覆盖两个产品，并让所有事实保留 Evidence ID。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel([fixture["valid"]])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(
            analysis.products,
            ["Atlas Notes", "Beacon Docs"],
        )
        factual_claims = [
            claim
            for claim in collect_analysis_claims(analysis)
            if claim.claim_type == "fact"
        ]
        interpretation_claims = [
            claim
            for claim in collect_analysis_claims(analysis)
            if claim.claim_type == "interpretation"
        ]
        self.assertTrue(factual_claims)
        self.assertTrue(interpretation_claims)
        self.assertTrue(
            all(claim.evidence_ids for claim in factual_claims)
        )
        self.assertTrue(
            all(
                claim.claim_type == "interpretation"
                for claim in analysis.opportunities
            )
        )
        self.assertEqual(
            set(analysis.conclusion.product_names),
            {"Atlas Notes", "Beacon Docs"},
        )

    def test_unknown_evidence_id_is_repaired_once(self) -> None:
        # 首次虚构 E99 时，Analyst 应反馈错误并接受一次修复。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel(
            [fixture["unknown_reference"], fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(len(analysis.products), 2)
        self.assertEqual(model.invocation_count, 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn(
            "outside its products: E99",
            repair_message,
        )

    def test_cross_product_reference_is_repaired_once(self) -> None:
        # 真实存在的 E1 也不能用于只描述 Beacon Docs 的事实。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel(
            [fixture["cross_product_reference"], fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(len(analysis.products), 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn(
            "outside its products: E1",
            repair_message,
        )

    def test_fact_without_evidence_enters_repair_flow(self) -> None:
        # Schema 应拒绝没有引用的事实，再让模型修复一次。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel(
            [fixture["fact_without_evidence"], fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(len(analysis.products), 2)
        self.assertEqual(model.invocation_count, 2)

    def test_unreferenced_facts_do_not_discard_valid_analysis(self) -> None:
        # 个别无引用事实应被移除，其余横向判断不应整体退回 fallback。
        fixture = _load_json("analyst_outputs.json")
        raw_output = deepcopy(fixture["valid"])
        raw_output["analysis"]["dimension_comparisons"] = [
            {
                "claim": (
                    "Atlas Notes documents templates while Beacon Docs "
                    "documents collaborative pages."
                ),
                "claim_type": "interpretation",
                "product_names": ["Atlas Notes", "Beacon Docs"],
                "evidence_ids": ["E1", "E3"],
            }
        ]
        raw_output["analysis"]["pricing"].extend(
            [
                {
                    "claim": "Atlas Notes has no public price fact.",
                    "claim_type": "fact",
                    "product_names": ["Atlas Notes"],
                    "evidence_ids": [],
                },
                {
                    "claim": "Beacon Docs has no public price fact.",
                    "claim_type": "fact",
                    "product_names": ["Beacon Docs"],
                    "evidence_ids": [],
                },
            ]
        )
        model = FakeAnalystModel(
            [json.dumps(raw_output, ensure_ascii=False)]
        )

        analysis = Analyst(model).analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(len(analysis.dimension_comparisons), 1)
        self.assertIn(
            "documents templates",
            analysis.dimension_comparisons[0].claim,
        )
        self.assertNotIn(
            "no public price fact",
            " ".join(claim.claim for claim in analysis.pricing),
        )

    def test_missing_product_is_repaired_once(self) -> None:
        # 产品只出现在 products 标题中还不够，必须进入实际比较 claim。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel(
            [fixture["missing_product"], fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(len(analysis.products), 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn(
            "Products missing from comparison claims: Beacon Docs",
            repair_message,
        )

    def test_conclusion_without_evidence_is_repaired_once(self) -> None:
        # 有可用 Evidence 时，最终结论也必须保留可追溯引用。
        fixture = _load_json("analyst_outputs.json")
        conclusion_without_evidence = deepcopy(fixture["valid"])
        conclusion_without_evidence["analysis"]["conclusion"][
            "evidence_ids"
        ] = []
        model = FakeAnalystModel(
            [conclusion_without_evidence, fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertTrue(analysis.conclusion.evidence_ids)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn(
            "Conclusion must cite supplied Evidence",
            repair_message,
        )

    def test_pricing_claim_inside_features_is_removed_locally(self) -> None:
        # 误放的价格事实可安全删除时，不应让整份有效分析进入修复。
        fixture = _load_json("analyst_outputs.json")
        pricing_inside_features = deepcopy(fixture["valid"])
        pricing_inside_features["analysis"]["features"].append(
            {
                "claim": (
                    "Beacon Docs offers a Standard plan priced at "
                    "USD 6.40 per user per month."
                ),
                "claim_type": "fact",
                "product_names": ["Beacon Docs"],
                "evidence_ids": ["E4"],
            }
        )
        model = FakeAnalystModel(
            [pricing_inside_features, fixture["valid"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(len(analysis.products), 2)
        self.assertEqual(model.invocation_count, 1)
        self.assertFalse(
            any(
                "Standard plan priced" in claim.claim
                for claim in analysis.features
            )
        )

    def test_pricing_language_detector_avoids_enterprise_search(self) -> None:
        # “Enterprise Search” 是功能名，不应因为 Enterprise 一词被误杀。
        self.assertFalse(
            contains_pricing_language("Notion provides Enterprise Search.")
        )
        self.assertTrue(
            contains_pricing_language(
                "Confluence offers an Enterprise plan with annual billing."
            )
        )
        self.assertTrue(
            contains_pricing_language(
                "Confluence offers a Standard plan priced at USD 6.40 "
                "per user per month."
            )
        )

    def test_feature_fact_is_narrowed_to_profile_feature_name(self) -> None:
        # 功能事实应回到画像中的功能名，避免把短标签扩写成营销承诺。
        fixture = _load_json("analyst_outputs.json")
        overstated_feature = deepcopy(fixture["valid"])
        overstated_feature["analysis"]["features"][0]["claim"] = (
            "Atlas Notes creates perfectly written reusable templates "
            "for every workflow."
        )
        model = FakeAnalystModel([overstated_feature])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(model.invocation_count, 1)
        self.assertEqual(
            analysis.features[0].claim,
            "Atlas Notes mentions Reusable templates.",
        )
        self.assertNotIn("perfectly written", analysis.features[0].claim)

    def test_revision_feedback_replaces_broad_feature_and_conclusion(
        self,
    ) -> None:
        # 重试轮模型可能被 suggested action 带偏，写出更宽泛的功能和结论。
        fixture = _load_json("analyst_outputs.json")
        broad_retry_output = deepcopy(fixture["valid"])
        broad_retry_output["analysis"]["features"] = [
            {
                "claim": (
                    "Atlas Notes includes features like Reusable templates."
                ),
                "claim_type": "interpretation",
                "product_names": ["Atlas Notes"],
                "evidence_ids": ["E1"],
            }
        ]
        broad_retry_output["analysis"]["conclusion"]["claim"] = (
            "Atlas Notes and Beacon Docs offer different feature sets, with "
            "Atlas Notes providing more template-focused options."
        )
        model = FakeAnalystModel([broad_retry_output])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                revision_feedback=[
                    (
                        "features[0] [unsupported_claim]: The broad feature "
                        "sentence is not directly supported."
                    )
                ],
            )
        )

        feature_claims = [claim.claim for claim in analysis.features]
        self.assertIn(
            "Atlas Notes mentions Reusable templates.",
            feature_claims,
        )
        self.assertFalse(
            any("includes features like" in claim for claim in feature_claims)
        )
        self.assertEqual(
            analysis.conclusion.claim,
            (
                "Based on the supplied profiles, Atlas Notes mentions "
                "Reusable templates and lists the Team plan at 12 USD per "
                "user; Beacon Docs mentions Collaborative pages and names "
                "a Business plan without a "
                "public price in the supplied profile."
            ),
        )

    def test_pricing_fact_is_narrowed_to_profile_price(self) -> None:
        # 模型不能把普通数字价格改成美元价，也不能把未知价格写成 $0。
        profiles = [
            ProductProfile(
                product_name="ChatGPT",
                pricing=[
                    {
                        "plan_name": "Go",
                        "price": "8",
                        "billing_cycle": "monthly",
                        "main_limits": [],
                        "evidence_ids": ["E1"],
                    },
                    {
                        "plan_name": "Business",
                        "price": None,
                        "billing_cycle": None,
                        "main_limits": [],
                        "evidence_ids": ["E2"],
                    },
                ],
            ),
            ProductProfile(
                product_name="Claude",
                pricing=[
                    {
                        "plan_name": "Pro",
                        "price": "$20/month",
                        "billing_cycle": "monthly",
                        "main_limits": [],
                        "evidence_ids": ["E3"],
                    }
                ],
            ),
        ]
        overstated_pricing = {
            "analysis": {
                "products": ["ChatGPT", "Claude"],
                "positioning": [],
                "features": [],
                "pricing": [
                    {
                        "claim": "ChatGPT offers the Go plan at $8 per month.",
                        "claim_type": "fact",
                        "product_names": ["ChatGPT"],
                        "evidence_ids": ["E1"],
                    },
                    {
                        "claim": "ChatGPT lists the Business plan at $0.",
                        "claim_type": "fact",
                        "product_names": ["ChatGPT"],
                        "evidence_ids": ["E2"],
                    },
                    {
                        "claim": "Claude lists the Pro plan at $20/month.",
                        "claim_type": "fact",
                        "product_names": ["Claude"],
                        "evidence_ids": ["E3"],
                    },
                ],
                "opportunities": [],
                "conclusion": {
                    "claim": (
                        "The supplied profiles compare ChatGPT and Claude "
                        "pricing."
                    ),
                    "claim_type": "interpretation",
                    "product_names": ["ChatGPT", "Claude"],
                    "evidence_ids": ["E1", "E2", "E3"],
                },
            }
        }
        model = FakeAnalystModel([overstated_pricing])
        analyst = Analyst(model)

        analysis = analyst.analyze(AnalystInput(profiles=profiles))

        pricing_claims = [claim.claim for claim in analysis.pricing]
        self.assertIn(
            "ChatGPT lists the Go plan at 8 with monthly billing.",
            pricing_claims,
        )
        self.assertIn(
            (
                "ChatGPT names a Business plan without a public price in the "
                "supplied profile."
            ),
            pricing_claims,
        )
        self.assertFalse(any("$8" in claim for claim in pricing_claims))
        self.assertFalse(
            any("Business plan at $0" in claim for claim in pricing_claims)
        )

    def test_unsourced_opportunities_are_replaced_with_fallback(self) -> None:
        # 机会点如果点名产品但没有 evidence，容易变成泛泛建议，应退回画像差异。
        fixture = _load_json("analyst_outputs.json")
        unsupported_opportunity = deepcopy(fixture["valid"])
        unsupported_opportunity["analysis"]["opportunities"] = [
            {
                "claim": (
                    "Beacon Docs could offer more detailed pricing "
                    "information for its plans."
                ),
                "claim_type": "interpretation",
                "product_names": ["Beacon Docs"],
                "evidence_ids": [],
            }
        ]
        model = FakeAnalystModel([unsupported_opportunity])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        opportunity_claims = [claim.claim for claim in analysis.opportunities]
        self.assertFalse(
            any(
                "could offer more detailed pricing" in claim
                for claim in opportunity_claims
            )
        )
        self.assertTrue(
            all(claim.evidence_ids for claim in analysis.opportunities)
        )
        self.assertTrue(
            any("pricing clarity" in claim for claim in opportunity_claims)
        )

    def test_conclusion_feedback_uses_conservative_conclusion(self) -> None:
        # Verifier 点名 conclusion 不受支持后，应退回可见画像摘要。
        fixture = _load_json("analyst_outputs.json")
        overstated_conclusion = deepcopy(fixture["valid"])
        overstated_conclusion["analysis"]["conclusion"]["claim"] = (
            "Atlas Notes is clearly stronger than Beacon Docs."
        )
        model = FakeAnalystModel([overstated_conclusion])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                revision_feedback=[
                    "conclusion [unsupported_claim]: The conclusion is too broad."
                ],
            )
        )

        self.assertEqual(
            analysis.conclusion.claim,
            (
                "Based on the supplied profiles, Atlas Notes mentions "
                "Reusable templates and lists the Team plan at 12 USD per "
                "user; Beacon Docs mentions Collaborative pages and names "
                "a Business plan without a "
                "public price in the supplied profile."
            ),
        )
        self.assertEqual(
            analysis.conclusion.evidence_ids,
            ["E1", "E2", "E3", "E4"],
        )

    def test_invalid_output_uses_fallback_after_one_failed_repair(self) -> None:
        # 连续两次结构错误后停止模型调用，并保守使用已提取画像。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel(
            [fixture["invalid_shape"], fixture["invalid_shape"]]
        )
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(model.invocation_count, 2)
        self.assertEqual(
            analysis.products,
            ["Atlas Notes", "Beacon Docs"],
        )
        self.assertTrue(analysis.positioning)
        self.assertTrue(analysis.opportunities)
        self.assertIn(
            "Based on the supplied profiles",
            analysis.conclusion.claim,
        )

    def test_model_call_failure_uses_fallback_analysis(self) -> None:
        # 真实服务临时不可用时，应保留已提取画像并生成轻量分析。
        model = FakeAnalystModel([])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertEqual(
            analysis.products,
            ["Atlas Notes", "Beacon Docs"],
        )
        self.assertTrue(analysis.features)
        self.assertTrue(analysis.pricing)
        self.assertTrue(analysis.positioning)
        self.assertTrue(analysis.opportunities)
        self.assertIn(
            "mentions Reusable templates",
            analysis.positioning[0].claim,
        )
        self.assertNotIn(
            "A collaborative workspace for small teams",
            analysis.positioning[0].claim,
        )
        self.assertEqual(
            analysis.features[0].claim,
            "Atlas Notes mentions Reusable templates.",
        )
        self.assertFalse(
            any(
                "main limits include" in claim.claim
                for claim in analysis.pricing
            )
        )
        self.assertIn(
            "Based on the supplied profiles",
            analysis.conclusion.claim,
        )
        self.assertNotIn(
            "positioning around",
            analysis.conclusion.claim,
        )
        self.assertFalse(
            any("audience fit" in claim.claim for claim in analysis.opportunities)
        )
        self.assertTrue(analysis.conclusion.evidence_ids)

    def test_sparse_model_output_gets_lightweight_sections(self) -> None:
        # 模型只给事实和兜底结论时，Analyst 会补齐个人项目更需要的分析段落。
        sparse_output = {
            "analysis": {
                "products": ["Atlas Notes", "Beacon Docs"],
                "positioning": [],
                "features": [
                    {
                        "claim": "Atlas Notes mentions Reusable templates.",
                        "claim_type": "fact",
                        "product_names": ["Atlas Notes"],
                        "evidence_ids": ["E1"],
                    },
                    {
                        "claim": "Beacon Docs mentions Collaborative pages.",
                        "claim_type": "fact",
                        "product_names": ["Beacon Docs"],
                        "evidence_ids": ["E3"],
                    },
                ],
                "pricing": [],
                "opportunities": [],
                "conclusion": {
                    "claim": (
                        "The comparison is limited to the supplied product "
                        "profiles for Atlas Notes and Beacon Docs."
                    ),
                    "claim_type": "interpretation",
                    "product_names": ["Atlas Notes", "Beacon Docs"],
                    "evidence_ids": ["E1", "E3"],
                },
            }
        }
        model = FakeAnalystModel([sparse_output])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertTrue(analysis.positioning)
        self.assertTrue(analysis.opportunities)
        self.assertIn(
            "mentions Reusable templates",
            analysis.positioning[0].claim,
        )
        self.assertIn(
            "Based on the supplied profiles",
            analysis.conclusion.claim,
        )
        self.assertNotIn("limited to", analysis.conclusion.claim)

    def test_free_fallback_pricing_claim_omits_billing_cycle(self) -> None:
        # Free/$0 是价格本身，不应再拼成 “monthly billing”。
        claim = format_fallback_pricing_claim(
            product_name="Confluence",
            plan_name="Free",
            price="Free",
            billing_cycle="monthly",
            main_limits=["10 users"],
        )

        self.assertEqual(
            claim,
            "Confluence lists the Free plan at Free.",
        )

    def test_fallback_pricing_claim_omits_redundant_or_invalid_billing(
        self,
    ) -> None:
        # price 已包含 /month 时不重复拼 billing；Beta 不是 billing。
        paid_claim = format_fallback_pricing_claim(
            product_name="Notion",
            plan_name="Plus",
            price="$10 per seat/month",
            billing_cycle="per month",
            main_limits=[],
        )
        beta_claim = format_fallback_pricing_claim(
            product_name="Notion",
            plan_name="Workers",
            price=None,
            billing_cycle="Beta",
            main_limits=[],
        )
        enterprise_claim = format_fallback_pricing_claim(
            product_name="Confluence",
            plan_name="Enterprise",
            price=None,
            billing_cycle=None,
            main_limits=[],
        )

        self.assertEqual(
            paid_claim,
            "Notion lists the Plus plan at $10 per seat/month.",
        )
        self.assertEqual(
            beta_claim,
            (
                "Notion names a Workers plan without a public price in the "
                "supplied profile."
            ),
        )
        self.assertEqual(
            enterprise_claim,
            (
                "Confluence names an Enterprise plan without a public price "
                "in the supplied profile."
            ),
        )

    def test_fallback_removes_unsupported_feature_feedback(self) -> None:
        # Verifier 已点名 unsupported 的功能，fallback 重试时不应原样写回报告。
        model = FakeAnalystModel([])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                revision_feedback=[
                    (
                        "features[1] [unsupported_claim]: The claim that "
                        "Beacon Docs mentions Collaborative pages is not "
                        "supported by the provided evidence. Suggested "
                        "action: Remove this claim."
                    )
                ],
            )
        )

        feature_claims = [claim.claim for claim in analysis.features]
        self.assertNotIn(
            "Beacon Docs mentions Collaborative pages.",
            feature_claims,
        )
        self.assertIn(
            "Atlas Notes mentions Reusable templates.",
            feature_claims,
        )

    def test_model_output_removes_unsupported_feature_feedback(self) -> None:
        # 即使模型第二轮忽略反馈，确定性规范化也要删除被点名的功能 claim。
        fixture = _load_json("analyst_outputs.json")
        model = FakeAnalystModel([fixture["valid"]])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                revision_feedback=[
                    (
                        "features[1] [unsupported_claim]: The claim that "
                        "Beacon Docs mentions Collaborative pages is not "
                        "supported by the provided evidence. Suggested "
                        "action: Remove this claim."
                    )
                ],
            )
        )

        feature_claims = [claim.claim for claim in analysis.features]
        self.assertNotIn(
            "Beacon Docs mentions Collaborative pages.",
            feature_claims,
        )
        self.assertTrue(
            all("collaborative pages" not in claim for claim in feature_claims)
        )

    def test_langchain_wrapper_binds_analyst_output_schema(self) -> None:
        # 真实模型边界必须绑定 AnalystOutput，并使用项目统一 JSON mode。
        fixture = _load_json("analyst_outputs.json")
        structured_model = FakeAnalystModel([fixture["valid"]])
        chat_model = FakeChatModel(structured_model)
        analyst_model = LangChainAnalystModel(chat_model)
        analyst = Analyst(analyst_model)

        analysis = analyst.analyze(
            AnalystInput(profiles=_load_sample_profiles())
        )

        self.assertIs(chat_model.received_schema, AnalystOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertEqual(len(analysis.products), 2)

    def test_duplicate_product_profiles_are_rejected(self) -> None:
        # 重复产品无法形成明确比较，应在模型调用前拒绝。
        profiles = _load_sample_profiles()

        with self.assertRaises(ValidationError):
            AnalystInput(profiles=[profiles[0], profiles[0]])

    def test_revision_feedback_is_added_to_user_message(self) -> None:
        # 工作流重试时，Verifier 问题必须进入 Analyst 修订上下文。
        analyst_input = AnalystInput(
            profiles=_load_sample_profiles(),
            revision_feedback=[
                "pricing[0]: Correct the unsupported price claim."
            ],
        )

        messages = build_analyst_messages(analyst_input)

        self.assertIn("上一次分析未通过 Verifier", messages[1]["content"])
        self.assertIn("pricing[0]", messages[1]["content"])

    def test_market_definition_is_added_to_user_message(self) -> None:
        # Analyst 必须看到目标购买者、比较层级、核心维度和排除项。
        analyst_input = AnalystInput(
            profiles=_load_sample_profiles(),
            market_definition=_build_market_definition(),
        )

        messages = build_analyst_messages(analyst_input)
        user_message = messages[1]["content"]

        self.assertIn("Mid-sized company IT", user_message)
        self.assertIn("Team subscription product", user_message)
        self.assertIn('"features"', user_message)
        self.assertIn("API usage pricing", user_message)

    def test_recommendation_references_are_repaired_once(self) -> None:
        # 建议也必须遵守产品到 Evidence ID 的归属关系。
        fixture = _load_json("analyst_outputs.json")
        invalid_output = deepcopy(fixture["valid"])
        invalid_output["analysis"]["recommendations"][0][
            "evidence_ids"
        ] = ["E99"]
        model = FakeAnalystModel([invalid_output, fixture["valid"]])
        analyst = Analyst(model)

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                market_definition=_build_market_definition(),
            )
        )

        self.assertEqual(model.invocation_count, 2)
        self.assertTrue(analysis.recommendations)
        self.assertNotIn("E99", analysis.recommendations[0].evidence_ids)

    def test_fallback_recommendation_has_action_evidence_and_boundary(
        self,
    ) -> None:
        # 两次无效模型输出后，保守 fallback 仍应给出可追溯的购买者建议。
        analyst = Analyst(FakeAnalystModel([{"bad": 1}, {"bad": 1}]))

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                market_definition=_build_market_definition(),
            )
        )

        self.assertTrue(analysis.recommendations)
        recommendation = analysis.recommendations[0]
        self.assertEqual(
            recommendation.target_scenario,
            "Mid-sized company IT and business leaders",
        )
        self.assertTrue(recommendation.recommended_action)
        self.assertTrue(recommendation.evidence_ids)
        self.assertIn("资料不足", recommendation.limitations[0])

    def test_unselected_pricing_dimension_is_removed_by_fallback(self) -> None:
        # 用户只选功能时，模型写出的 pricing 比较不能进入最终结果。
        fixture = _load_json("analyst_outputs.json")
        analyst = Analyst(
            FakeAnalystModel([fixture["valid"], fixture["valid"]])
        )
        market = _build_market_definition().model_copy(
            update={"core_dimensions": ["features"]}
        )

        analysis = analyst.analyze(
            AnalystInput(
                profiles=_load_sample_profiles(),
                market_definition=market,
            )
        )

        self.assertEqual(analysis.pricing, [])
        self.assertTrue(analysis.features)

    def test_custom_dimension_findings_become_grounded_comparisons(
        self,
    ) -> None:
        # 自定义维度不能丢失；无资料的产品保持空白并在建议中披露边界。
        profiles = _load_sample_profiles()
        profiles[0] = profiles[0].model_copy(
            update={
                "dimension_findings": [
                    DimensionFinding(
                        dimension="governance",
                        facts=["Workspace roles are documented."],
                        evidence_ids=["E1"],
                    )
                ]
            }
        )
        profiles[1] = profiles[1].model_copy(
            update={
                "dimension_findings": [
                    DimensionFinding(dimension="governance")
                ]
            }
        )
        market = _build_market_definition().model_copy(
            update={"core_dimensions": ["governance"]}
        )
        analyst = Analyst(FakeAnalystModel([{"bad": 1}, {"bad": 1}]))

        analysis = analyst.analyze(
            AnalystInput(profiles=profiles, market_definition=market)
        )

        self.assertEqual(len(analysis.dimension_comparisons), 1)
        self.assertIn(
            "Workspace roles are documented",
            analysis.dimension_comparisons[0].claim,
        )
        self.assertEqual(analysis.dimension_comparisons[0].evidence_ids, ["E1"])
        self.assertTrue(analysis.recommendations)
        self.assertIn("资料不足", analysis.recommendations[0].limitations[0])

    def test_prompt_keeps_fact_claims_close_to_profile_text(self) -> None:
        # 真实模型应避免把短功能标签扩写成证据没有支持的大 claim。
        self.assertIn("贴近 ProductProfile", ANALYST_SYSTEM_PROMPT)
        self.assertIn("不得把短标签扩写成更大的能力", ANALYST_SYSTEM_PROMPT)
        self.assertIn("价格、套餐、计费周期", ANALYST_SYSTEM_PROMPT)
        self.assertIn("features 章节", ANALYST_SYSTEM_PROMPT)
        self.assertIn("不要因为证据简短就全部留空", ANALYST_SYSTEM_PROMPT)
        self.assertIn("更需要可读分析", ANALYST_SYSTEM_PROMPT)
        self.assertIn("显著减少", ANALYST_SYSTEM_PROMPT)
        self.assertIn("recommendations 与事实", ANALYST_SYSTEM_PROMPT)
        self.assertIn("不得把不同产品层级", ANALYST_SYSTEM_PROMPT)

    def test_unified_fields_generate_safe_scenario_analysis(self) -> None:
        source_one = SourceReference(
            evidence_id="E1",
            title="One model docs",
            url="https://example.com/one",
            source_type=SourceType.OFFICIAL,
            collected_at="2026-07-20T00:00:00Z",
        )
        source_two = source_one.model_copy(
            update={"evidence_id": "E2", "title": "Two model docs"}
        )
        profiles = [
            ProductProfile(
                product_name="One API",
                models=[
                    ModelProfile(
                        model_name="One",
                        model_capabilities=["coding"],
                        supported_modalities=["text", "image"],
                        context_window_tokens=200_000,
                        tool_calling=SupportStatus.SUPPORTED,
                        batch_api=SupportStatus.SUPPORTED,
                        pricing=ModelPricing(
                            input_price=PriceRate(
                                amount="2", currency="USD", per_quantity=1_000_000,
                                unit="token", evidence_ids=["E1"],
                            ),
                            output_price=PriceRate(
                                amount="8", currency="USD", per_quantity=1_000_000,
                                unit="token", evidence_ids=["E1"],
                            ),
                        ),
                        source_evidence=[source_one],
                        extraction_confidence=1,
                    )
                ],
            ),
            ProductProfile(
                product_name="Two API",
                models=[
                    ModelProfile(
                        model_name="Two",
                        source_evidence=[source_two],
                        extraction_confidence=1,
                    )
                ],
            ),
        ]

        analysis = Analyst(FakeAnalystModel([])).analyze(
            AnalystInput(profiles=profiles)
        )

        self.assertEqual(
            {item.scenario for item in analysis.scenario_recommendations},
            {
                "编程 Agent", "RAG 与知识库问答", "长文档分析", "实时语音应用",
                "多模态理解", "低成本批处理", "企业级 API 集成",
            },
        )
        coding = next(
            item for item in analysis.scenario_recommendations
            if item.scenario == "编程 Agent"
        )
        voice = next(
            item for item in analysis.scenario_recommendations
            if item.scenario == "实时语音应用"
        )
        self.assertEqual(coding.recommended_product, "One API")
        self.assertEqual(voice.recommended_product, "暂无可验证推荐")
        self.assertFalse(any(
            "不支持" in gap
            for assessment in analysis.product_assessments
            for gap in assessment.data_gaps
        ))
        self.assertEqual(len(analysis.market_opportunities), 3)
        self.assertTrue(all(
            item.user_pain and item.product_direction
            for item in analysis.market_opportunities
        ))


if __name__ == "__main__":
    unittest.main()
