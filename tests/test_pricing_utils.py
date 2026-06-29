import unittest

from competitive_analysis_agent.pricing_utils import (
    is_free_price_text,
    normalize_billing_cycle_value,
    normalize_price_value,
    should_include_billing_cycle,
    should_report_missing_billing_cycle,
)


class PricingUtilsTest(unittest.TestCase):
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
        self.assertFalse(should_include_billing_cycle(None, "Beta"))
        self.assertTrue(should_report_missing_billing_cycle(None, "Beta"))


if __name__ == "__main__":
    unittest.main()
