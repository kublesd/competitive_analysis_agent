import unittest
from datetime import datetime, timezone

from competitive_analysis_agent.pricing_utils import (
    api_pricing_evidence_error,
    is_free_price_text,
    normalize_billing_cycle_value,
    normalize_price_value,
    should_include_billing_cycle,
    should_report_missing_billing_cycle,
)
from competitive_analysis_agent.schemas import Evidence, PricingPlan


class PricingUtilsTest(unittest.TestCase):
    def test_api_pricing_evidence_contract(self) -> None:
        # 共享边界同时覆盖完整费率、裸金额、未知价格和自定义报价。
        evidence_by_id = {
            "E1": Evidence(
                evidence_id="E1",
                product_name="Model API",
                topic="api_pricing",
                title="Model API pricing",
                url="https://example.com/pricing",
                snippet=(
                    "Model A input costs $2.50 per million input tokens. "
                    "Enterprise API uses Custom pricing."
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        }
        valid_rate = PricingPlan(
            plan_name="Model A input",
            price="$2.50 per million input tokens",
            unit="per million input tokens",
            evidence_ids=["E1"],
        )
        bare_rate = valid_rate.model_copy(
            update={"price": "$2.50", "unit": "USD"}
        )
        split_rate = valid_rate.model_copy(
            update={
                "price": "$2.50",
                "unit": "per million input tokens",
            }
        )
        unknown_rate = valid_rate.model_copy(update={"price": None})
        custom_rate = PricingPlan(
            plan_name="Enterprise API",
            price="Custom pricing",
            evidence_ids=["E1"],
        )

        self.assertIsNone(
            api_pricing_evidence_error(valid_rate, evidence_by_id)
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(bare_rate, evidence_by_id)
        )
        self.assertIsNone(
            api_pricing_evidence_error(split_rate, evidence_by_id)
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(unknown_rate, evidence_by_id)
        )
        self.assertIsNone(
            api_pricing_evidence_error(custom_rate, evidence_by_id)
        )

    def test_api_pricing_accepts_mtok_without_crossing_rate_columns(
        self,
    ) -> None:
        # Anthropic 的 MTok 是 million tokens；输入金额不能借用输出列通过校验。
        evidence_by_id = {
            "E1": Evidence(
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
        }
        input_rate = PricingPlan(
            plan_name="Claude Haiku 4.5 input",
            price="$1 per million input tokens",
            unit="per million input tokens",
            evidence_ids=["E1"],
        )
        output_rate = PricingPlan(
            plan_name="Claude Haiku 4.5 output",
            price="$5 per million output tokens",
            unit="per million output tokens",
            evidence_ids=["E1"],
        )

        self.assertIsNone(
            api_pricing_evidence_error(input_rate, evidence_by_id)
        )
        self.assertIsNone(
            api_pricing_evidence_error(output_rate, evidence_by_id)
        )
        for price_text in ["$1 / MTok", "$1 per M tokens", "$1 per 1M tokens"]:
            with self.subTest(price_text=price_text):
                self.assertIsNone(
                    api_pricing_evidence_error(
                        input_rate.model_copy(
                            update={"price": price_text}
                        ),
                        evidence_by_id,
                    )
                )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                input_rate.model_copy(
                    update={"price": "$5 per million input tokens"}
                ),
                evidence_by_id,
            )
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                input_rate.model_copy(
                    update={"plan_name": "Claude Sonnet 4.5 input"}
                ),
                evidence_by_id,
            )
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                input_rate.model_copy(
                    update={"price": "$1 per million output tokens"}
                ),
                evidence_by_id,
            )
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                input_rate.model_copy(
                    update={"price": "$1", "unit": "USD"}
                ),
                evidence_by_id,
            )
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                input_rate.model_copy(
                    update={"price": "$1 | USD", "unit": "USD"}
                ),
                evidence_by_id,
            )
        )

    def test_api_pricing_preserves_markdown_table_columns(self) -> None:
        # 真实价格页把单位、方向和金额拆在表头与数据行中，校验时必须按列重组。
        evidence_by_id = {
            "E1": Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="api_pricing",
                title="Pricing | OpenAI API",
                url="https://developers.openai.com/api/docs/pricing",
                snippet="Prices per 1M tokens.",
                raw_content=(
                    "Prices per 1M tokens. Standard | | Model | Input | "
                    "Cached input | Output | | gpt-5.6-sol | $5.00 | "
                    "$0.50 | $30.00 | | gpt-5.6-terra | $2.50 | "
                    "$0.25 | $15.00 |"
                ),
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        }
        valid_input = PricingPlan(
            plan_name="gpt-5.6-sol",
            price="$5.00 per million input tokens",
            unit="per million input tokens",
            evidence_ids=["E1"],
        )
        valid_output = valid_input.model_copy(
            update={"price": "$30.00 per million output tokens"}
        )

        self.assertIsNone(
            api_pricing_evidence_error(valid_input, evidence_by_id)
        )
        self.assertIsNone(
            api_pricing_evidence_error(valid_output, evidence_by_id)
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                valid_input.model_copy(
                    update={"price": "$0.50 per million input tokens"}
                ),
                evidence_by_id,
            )
        )
        self.assertIsNotNone(
            api_pricing_evidence_error(
                valid_input.model_copy(
                    update={"price": "$15.00 per million output tokens"}
                ),
                evidence_by_id,
            )
        )

    def test_api_pricing_accepts_evidenced_scale_tier_unit_per_day(
        self,
    ) -> None:
        # Scale Tier 是 API 容量价格；per unit/day 不是 Token 单位，但仍是完整费率。
        evidence_by_id = {
            "E1": Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="api_pricing",
                title="Scale Tier for API customers",
                url="https://example.com/openai/scale-tier",
                snippet="GPT-5.5 | 50,000 TPM | $750.00 per unit/day",
                source_type="official",
                collected_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
            )
        }
        scale_tier_rate = PricingPlan(
            plan_name="GPT-5.5",
            price="50,000 TPM $750.00 per unit/day",
            unit="per unit/day",
            evidence_ids=["E1"],
        )

        self.assertIsNone(
            api_pricing_evidence_error(scale_tier_rate, evidence_by_id)
        )

    def test_free_price_text_detection_is_exact(self) -> None:
        # 0 价格可以支持免费口径，但用户数和版本号不能被误认成价格。
        self.assertTrue(is_free_price_text("Free forever for 10 users"))
        self.assertTrue(is_free_price_text("$0/mo"))
        self.assertTrue(is_free_price_text("USD 0 per user"))
        self.assertFalse(
            is_free_price_text(
                "Free to try, then $10 per 1,000 monthly Notion credits"
            )
        )
        self.assertFalse(is_free_price_text("10 users"))
        self.assertFalse(is_free_price_text("version 0.1"))

    def test_price_value_normalization_keeps_only_zero_price(self) -> None:
        # 免费价的计费周期不要留在 price 字段里，避免报告重复展示。
        self.assertEqual(normalize_price_value("$0 per seat/month"), "$0")
        self.assertEqual(
            normalize_price_value("Free to try, then $10 per 1,000 credits"),
            "Free to try, then $10 per 1,000 credits",
        )

    def test_billing_cycle_value_filters_status_words(self) -> None:
        # Beta 是发布状态，不是月付或年付周期。
        self.assertEqual(normalize_billing_cycle_value("per user / month"), "per user / month")
        self.assertIsNone(normalize_billing_cycle_value("Beta"))

    def test_free_price_omits_billing_cycle_and_missing_gap(self) -> None:
        self.assertFalse(should_include_billing_cycle("Free", "monthly"))
        self.assertFalse(should_report_missing_billing_cycle("Free", None))
        self.assertTrue(should_include_billing_cycle("$10", "monthly"))
        self.assertTrue(should_report_missing_billing_cycle("$10", None))
        self.assertFalse(
            should_include_billing_cycle("$10 per seat/month", "per month")
        )
        self.assertFalse(
            should_report_missing_billing_cycle("Custom pricing", None)
        )
        self.assertFalse(
            should_report_missing_billing_cycle(
                "10",
                None,
                "per member / month",
            )
        )
        self.assertFalse(should_include_billing_cycle(None, "Beta"))
        self.assertTrue(should_report_missing_billing_cycle(None, "Beta"))


if __name__ == "__main__":
    unittest.main()
