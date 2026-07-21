import json
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.schemas import (
    AvailabilityStatus,
    Currency,
    Evidence,
    MarketDefinition,
    Modality,
    ModelPricing,
    ModelProfile,
    PriceRate,
    PricingUnit,
    ProductProfile,
    SourceReference,
    SupportStatus,
    VerificationStatus,
    WorkflowState,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_case.json"


def _load_sample_case() -> dict:
    """读取固定样例，确保测试不依赖模型或网络。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class SchemaValidationTest(unittest.TestCase):
    def _build_source(self, evidence_id: str = "E1") -> SourceReference:
        """创建统一模型字段使用的固定来源引用。"""

        return SourceReference(
            evidence_id=evidence_id,
            title="Official model documentation",
            url="https://example.com/models",
            source_type="official",
            collected_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )

    def test_valid_sample_case_parses_two_products(self) -> None:
        # 整个固定 JSON 应能转换成有类型的工作流状态。
        sample_data = _load_sample_case()

        state = WorkflowState.model_validate(sample_data)

        self.assertEqual(state.target_product, "Atlas Notes")
        self.assertEqual(
            state.market_definition.comparison_level,
            "企业订阅产品",
        )
        self.assertEqual(len(state.product_profiles), 2)
        self.assertEqual(len(state.evidence), 4)

    def test_missing_evidence_url_fails_clearly(self) -> None:
        # URL 是证据追溯的必需字段，缺失时应在边界立即失败。
        sample_data = _load_sample_case()
        invalid_evidence = sample_data["evidence"][0].copy()
        invalid_evidence.pop("url")

        with self.assertRaises(ValidationError) as raised:
            Evidence.model_validate(invalid_evidence)

        error_locations = [error["loc"] for error in raised.exception.errors()]
        self.assertIn(("url",), error_locations)

    def test_market_definition_rejects_missing_required_scope(self) -> None:
        # 市场名称、产品类别和比较层级缺失时，入口必须在运行前失败。
        with self.assertRaises(ValidationError) as raised:
            MarketDefinition.model_validate(
                {
                    "market_name": "团队知识管理工具",
                    "core_dimensions": ["features"],
                }
            )

        error_locations = [error["loc"] for error in raised.exception.errors()]
        self.assertIn(("product_category",), error_locations)
        self.assertIn(("comparison_level",), error_locations)

    def test_pricing_scope_defaults_to_subscription_and_accepts_api(self) -> None:
        # 旧 SaaS 请求保持订阅语义，API 主线必须显式声明 api。
        sample_market = _load_sample_case()["market_definition"]

        subscription_market = MarketDefinition.model_validate(sample_market)
        api_market = subscription_market.model_copy(
            update={"pricing_scope": "api"}
        )

        self.assertEqual(subscription_market.pricing_scope, "subscription")
        self.assertEqual(api_market.pricing_scope, "api")

        with self.assertRaises(ValidationError):
            MarketDefinition.model_validate(
                {**sample_market, "pricing_scope": "mixed"}
            )

    def test_unknown_pricing_is_explicit(self) -> None:
        # 没有公开价格时保留 None，不能补写一个猜测值。
        sample_data = _load_sample_case()

        state = WorkflowState.model_validate(sample_data)
        beacon_profile = state.product_profiles[1]
        business_plan = beacon_profile.pricing[0]

        self.assertIsNone(beacon_profile.positioning)
        self.assertIsNone(business_plan.price)
        self.assertIsNone(business_plan.billing_cycle)

    def test_profile_references_point_to_fixture_evidence(self) -> None:
        # evidence_ids 是产品事实与原始来源之间的连接键。
        sample_data = _load_sample_case()
        state = WorkflowState.model_validate(sample_data)
        available_evidence_ids = {
            evidence.evidence_id for evidence in state.evidence
        }

        referenced_evidence_ids: set[str] = set()
        for profile in state.product_profiles:
            for feature in profile.features:
                referenced_evidence_ids.update(feature.evidence_ids)
            for pricing_plan in profile.pricing:
                referenced_evidence_ids.update(pricing_plan.evidence_ids)

        self.assertTrue(
            referenced_evidence_ids.issubset(available_evidence_ids)
        )

    def test_model_prices_are_grouped_under_one_model(self) -> None:
        # 输入、缓存输入、输出和 Batch 价格必须属于同一个模型对象。
        source = self._build_source()
        model = ModelProfile(
            model_name="GPT-5.4",
            model_capabilities=["reasoning", "tool use"],
            supported_modalities=[Modality.TEXT, Modality.IMAGE],
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            tool_calling=SupportStatus.SUPPORTED,
            structured_output=SupportStatus.SUPPORTED,
            realtime=SupportStatus.UNCERTAIN,
            prompt_caching=SupportStatus.SUPPORTED,
            batch_api=SupportStatus.SUPPORTED,
            pricing=ModelPricing(
                input_price=PriceRate(
                    amount="2.50",
                    currency=Currency.USD,
                    per_quantity=1_000_000,
                    unit=PricingUnit.TOKEN,
                    evidence_ids=["E1"],
                ),
                cached_input_price=PriceRate(
                    amount="0.25",
                    currency=Currency.USD,
                    per_quantity=1_000_000,
                    unit=PricingUnit.TOKEN,
                    evidence_ids=["E1"],
                ),
                output_price=PriceRate(
                    amount="10.00",
                    currency=Currency.USD,
                    per_quantity=1_000_000,
                    unit=PricingUnit.TOKEN,
                    evidence_ids=["E1"],
                ),
                batch_discount_percent="50",
                batch_evidence_ids=["E1"],
            ),
            source_evidence=[source],
            extraction_confidence=0.95,
            verification_status=VerificationStatus.PASSED,
        )

        profile = WorkflowState.model_validate(
            {
                **_load_sample_case(),
                "product_profiles": [
                    {
                        "product_name": "OpenAI API",
                        "positioning": "通用模型 API 平台",
                        "target_users": ["开发团队"],
                        "models": [model.model_dump(mode="json")],
                        "source_evidence": [source.model_dump(mode="json")],
                        "extraction_confidence": 0.95,
                        "verification_status": "passed",
                    }
                ],
            }
        ).product_profiles[0]

        self.assertEqual(len(profile.models), 1)
        self.assertEqual(
            profile.models[0].pricing.input_price.amount,
            Decimal("2.50"),
        )
        self.assertEqual(
            profile.models[0].pricing.output_price.amount,
            Decimal("10.00"),
        )
        self.assertEqual(
            profile.models[0].pricing.batch_discount_percent,
            Decimal("50"),
        )

    def test_unified_profile_uses_explicit_missing_defaults(self) -> None:
        # 旧画像未提供统一字段时仍可解析，但缺失状态必须明确。
        state = WorkflowState.model_validate(_load_sample_case())
        profile = state.product_profiles[0]

        self.assertEqual(
            profile.positioning_status,
            AvailabilityStatus.AVAILABLE,
        )
        self.assertEqual(
            profile.target_users_status,
            AvailabilityStatus.AVAILABLE,
        )
        self.assertEqual(profile.models, [])
        self.assertIsNone(profile.enterprise_capabilities)
        self.assertEqual(
            profile.enterprise_capabilities_status,
            AvailabilityStatus.MISSING,
        )
        self.assertIsNone(profile.extraction_confidence)
        self.assertEqual(
            profile.verification_status,
            VerificationStatus.UNVERIFIED,
        )

    def test_legacy_model_copy_reinfers_implicit_availability(self) -> None:
        # 旧抽取器会先建画像再补定位，未显式状态必须随新值更新。
        profile = ProductProfile(product_name="Atlas Notes")
        updated_profile = profile.model_copy(
            update={"positioning": "团队知识管理产品"}
        )

        state = WorkflowState.model_validate(
            {
                **_load_sample_case(),
                "product_profiles": [updated_profile],
            }
        )

        self.assertEqual(
            state.product_profiles[0].positioning_status,
            AvailabilityStatus.AVAILABLE,
        )

    def test_duplicate_model_names_are_rejected_case_insensitively(self) -> None:
        # 同一产品不能用重复模型对象分别保存 input 和 output 价格。
        source = self._build_source()
        model_data = {
            "model_name": "Claude Sonnet 5",
            "source_evidence": [source.model_dump(mode="json")],
            "extraction_confidence": 0.9,
        }

        with self.assertRaises(ValidationError):
            WorkflowState.model_validate(
                {
                    **_load_sample_case(),
                    "product_profiles": [
                        {
                            "product_name": "Claude API",
                            "models": [
                                model_data,
                                {**model_data, "model_name": "claude sonnet 5"},
                            ],
                        }
                    ],
                }
            )

    def test_price_direction_cannot_be_encoded_in_model_name(self) -> None:
        # input/output 是 ModelPricing 字段，不得伪装成两个模型名称。
        with self.assertRaises(ValidationError):
            ModelProfile(
                model_name="Claude Sonnet 5 output",
                source_evidence=[self._build_source()],
                extraction_confidence=0.9,
            )

    def test_model_price_must_reference_model_evidence(self) -> None:
        # 模型价格不能引用当前模型来源列表之外的 Evidence ID。
        with self.assertRaises(ValidationError):
            ModelProfile(
                model_name="Gemini 2.5 Pro",
                pricing=ModelPricing(
                    input_price=PriceRate(
                        amount="1.25",
                        currency="USD",
                        per_quantity=1_000_000,
                        unit="token",
                        evidence_ids=["E2"],
                    )
                ),
                source_evidence=[self._build_source("E1")],
                extraction_confidence=0.9,
            )

    def test_invalid_enum_and_price_values_are_rejected(self) -> None:
        # 枚举、负价格和超范围折扣必须在 Schema 边界失败。
        with self.assertRaises(ValidationError):
            PriceRate(
                amount="-1",
                currency="bitcoin",
                per_quantity=0,
                unit="tokens",
                evidence_ids=["E1"],
            )

        with self.assertRaises(ValidationError):
            ModelPricing(
                batch_discount_percent="120",
                batch_evidence_ids=["E1"],
            )

    def test_audio_price_preserves_effective_period(self) -> None:
        # 音频价格有独立分类，且不同有效期不能丢失。
        audio_price = PriceRate(
            amount="32",
            currency="USD",
            per_quantity=1_000_000,
            unit="token",
            effective_from=date(2026, 1, 1),
            effective_to=date(2026, 6, 30),
            evidence_ids=["E1"],
        )

        pricing = ModelPricing(audio_price=audio_price)

        self.assertEqual(pricing.audio_price.amount, Decimal("32"))
        self.assertEqual(
            pricing.audio_price.effective_from,
            date(2026, 1, 1),
        )

    def test_price_rejects_reversed_effective_period(self) -> None:
        # 价格结束日期不能早于开始日期。
        with self.assertRaises(ValidationError):
            PriceRate(
                amount="2",
                currency="USD",
                per_quantity=1_000_000,
                unit="token",
                effective_from=date(2026, 7, 1),
                effective_to=date(2026, 6, 30),
                evidence_ids=["E1"],
            )

    def test_one_price_type_keeps_multiple_effective_periods(self) -> None:
        # 同模型同价格类型的历史价和新价格保存在同一 pricing 对象中。
        old_price = PriceRate(
            amount="2",
            currency="USD",
            per_quantity=1_000_000,
            unit="token",
            effective_to=date(2026, 6, 30),
            evidence_ids=["E1"],
        )
        new_price = PriceRate(
            amount="2.5",
            currency="USD",
            per_quantity=1_000_000,
            unit="token",
            effective_from=date(2026, 7, 1),
            evidence_ids=["E2"],
        )

        pricing = ModelPricing(input_price=[old_price, new_price])

        self.assertEqual(len(pricing.input_price), 2)

    def test_multiple_undated_prices_are_rejected(self) -> None:
        # 无日期的冲突价格不能伪装成有效的历史价格列表。
        price = PriceRate(
            amount="2",
            currency="USD",
            per_quantity=1_000_000,
            unit="token",
            evidence_ids=["E1"],
        )

        with self.assertRaises(ValidationError):
            ModelPricing(input_price=[price, price])

    def test_multiple_undated_prices_with_distinct_conditions_are_allowed(self) -> None:
        pricing = ModelPricing(
            input_price=[
                PriceRate(
                    amount="2",
                    currency=Currency.USD,
                    per_quantity=1_000_000,
                    unit="token",
                    condition="Standard",
                    evidence_ids=["E1"],
                ),
                PriceRate(
                    amount="4",
                    currency=Currency.USD,
                    per_quantity=1_000_000,
                    unit="token",
                    condition="Priority",
                    evidence_ids=["E1"],
                ),
            ]
        )

        self.assertEqual(len(pricing.input_price), 2)


if __name__ == "__main__":
    unittest.main()
