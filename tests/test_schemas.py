import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.schemas import Evidence, WorkflowState


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_case.json"


def _load_sample_case() -> dict:
    """读取固定样例，确保测试不依赖模型或网络。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class SchemaValidationTest(unittest.TestCase):
    def test_valid_sample_case_parses_two_products(self) -> None:
        # 整个固定 JSON 应能转换成有类型的工作流状态。
        sample_data = _load_sample_case()

        state = WorkflowState.model_validate(sample_data)

        self.assertEqual(state.target_product, "Atlas Notes")
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


if __name__ == "__main__":
    unittest.main()
