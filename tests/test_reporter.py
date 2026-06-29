import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from competitive_analysis_agent.analyst import AnalystOutput
from competitive_analysis_agent.reporter import Reporter, ReporterInput
from competitive_analysis_agent.schemas import WorkflowState
from competitive_analysis_agent.verifier import (
    VerificationIssue,
    VerificationResult,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
SAMPLE_REPORT_PATH = Path(__file__).parents[1] / "docs" / "sample-report.md"


def _load_json(file_name: str) -> dict:
    """读取 Stage 9 使用的固定结构化输入。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _build_reporter_input(
    verification_result: VerificationResult | None = None,
) -> ReporterInput:
    """组合 Stage 1、6、7 的 fixture，构造稳定 Reporter 输入。"""

    workflow_state = WorkflowState.model_validate(
        _load_json("sample_case.json")
    )
    analysis = AnalystOutput.model_validate(
        _load_json("analyst_outputs.json")["valid"]
    ).analysis
    current_verification = verification_result or VerificationResult(
        passed=True,
        issues=[],
        retry_recommended=False,
    )
    return ReporterInput(
        analysis=analysis,
        product_profiles=workflow_state.product_profiles,
        evidence=workflow_state.evidence,
        verification_result=current_verification,
    )


class ReporterTest(unittest.TestCase):
    def test_same_input_produces_same_markdown(self) -> None:
        # Reporter 不调用模型，相同输入必须产生逐字相同的报告。
        reporter = Reporter()
        reporter_input = _build_reporter_input()

        first_report = reporter.render(reporter_input)
        second_report = reporter.render(reporter_input)

        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report.endswith("\n"))

    def test_rendered_report_matches_inspectable_sample(self) -> None:
        # 样例报告也是回归 fixture，章节或格式变化必须被明确确认。
        expected_report = SAMPLE_REPORT_PATH.read_text(encoding="utf-8")
        actual_report = Reporter().render(_build_reporter_input())

        self.assertEqual(actual_report, expected_report)

    def test_report_contains_comparison_tables_and_source_links(self) -> None:
        # 每个分析引用都应成为可点击来源，而不是孤立的 Evidence ID。
        reporter_input = _build_reporter_input()
        report = Reporter().render(reporter_input)

        self.assertIn("## 产品概览", report)
        self.assertIn("## 功能对比", report)
        self.assertIn("## 价格对比", report)
        self.assertIn("## 资料来源", report)
        for evidence in reporter_input.evidence:
            expected_link = (
                f"[{evidence.evidence_id}]({str(evidence.url)})"
            )
            self.assertIn(expected_link, report)
            self.assertIn(str(evidence.url), report)

    def test_missing_fields_are_rendered_without_guessing(self) -> None:
        # Beacon 的未知定位和公开价格应明确显示缺失，不允许补写。
        report = Reporter().render(_build_reporter_input())

        self.assertIn("Beacon Docs | 未提供 | 未提供", report)
        self.assertIn("价格未提供 / 计费周期未提供", report)
        self.assertIn("Beacon Docs：未提供定位信息。", report)
        self.assertIn(
            "Beacon Docs 的 Business 方案未提供公开价格。",
            report,
        )

    def test_free_pricing_omits_billing_cycle_in_overview(self) -> None:
        # Free 方案不展示 “Free / monthly”，也不记录缺失计费周期。
        reporter_input = _build_reporter_input()
        atlas_profile = reporter_input.product_profiles[0]
        free_plan = atlas_profile.pricing[0].model_copy(
            update={
                "plan_name": "Free",
                "price": "Free",
                "billing_cycle": None,
            }
        )
        free_profile = atlas_profile.model_copy(
            update={"pricing": [free_plan]}
        )
        adjusted_input = ReporterInput(
            analysis=reporter_input.analysis,
            product_profiles=[
                free_profile,
                reporter_input.product_profiles[1],
            ],
            evidence=reporter_input.evidence,
            verification_result=reporter_input.verification_result,
        )

        report = Reporter().render(adjusted_input)

        self.assertIn(
            "Free：Free [E2](https://example.com/atlas/pricing)",
            report,
        )
        self.assertNotIn("Free：Free /", report)
        self.assertNotIn(
            "Atlas Notes 的 Free 方案未提供计费周期。",
            report,
        )

    def test_pricing_overview_filters_duplicate_and_invalid_billing(
        self,
    ) -> None:
        # 报告展示层也要防御重复周期、Beta 状态和 Custom pricing 缺口。
        reporter_input = _build_reporter_input()
        atlas_profile = reporter_input.product_profiles[0]
        base_plan = atlas_profile.pricing[0]
        plans = [
            base_plan.model_copy(
                update={
                    "plan_name": "Plus",
                    "price": "$10 per seat/month",
                    "billing_cycle": "per month",
                }
            ),
            base_plan.model_copy(
                update={
                    "plan_name": "Workers",
                    "price": None,
                    "billing_cycle": "Beta",
                }
            ),
            base_plan.model_copy(
                update={
                    "plan_name": "Enterprise",
                    "price": "Custom pricing",
                    "billing_cycle": None,
                }
            ),
        ]
        adjusted_profile = atlas_profile.model_copy(
            update={"pricing": plans}
        )
        adjusted_input = ReporterInput(
            analysis=reporter_input.analysis,
            product_profiles=[
                adjusted_profile,
                reporter_input.product_profiles[1],
            ],
            evidence=reporter_input.evidence,
            verification_result=reporter_input.verification_result,
        )

        report = Reporter().render(adjusted_input)

        self.assertIn(
            "Plus：$10 per seat/month [E2](https://example.com/atlas/pricing)",
            report,
        )
        self.assertNotIn("$10 per seat/month / per month", report)
        self.assertIn("Workers：价格未提供 / 计费周期未提供", report)
        self.assertNotIn("Workers：价格未提供 / Beta", report)
        self.assertIn(
            "Enterprise：Custom pricing [E2](https://example.com/atlas/pricing)",
            report,
        )
        self.assertNotIn("Atlas Notes 的 Enterprise 方案未提供计费周期。", report)

    def test_failed_verification_remains_visible(self) -> None:
        # 验证失败仍输出降级报告，但顶部必须保留问题和警告。
        verification_result = VerificationResult(
            passed=False,
            issues=[
                VerificationIssue(
                    issue_type="unsupported_claim",
                    claim_path="features[0]",
                    message="The claim is not supported.",
                    evidence_ids=["E1"],
                    suggested_action="Remove the unsupported claim.",
                )
            ],
            retry_recommended=True,
        )
        report = Reporter().render(
            _build_reporter_input(verification_result)
        )

        self.assertIn("本报告未通过最终验证", report)
        self.assertIn("features[0]", report)
        self.assertIn("unsupported_claim", report)
        self.assertIn("[E1](https://example.com/atlas/features)", report)

    def test_unknown_analysis_citation_is_rejected(self) -> None:
        # 无法映射到 Evidence 的引用不能进入最终报告。
        workflow_state = WorkflowState.model_validate(
            _load_json("sample_case.json")
        )
        invalid_analysis = AnalystOutput.model_validate(
            _load_json("analyst_outputs.json")["unknown_reference"]
        ).analysis

        with self.assertRaises(ValidationError):
            ReporterInput(
                analysis=invalid_analysis,
                product_profiles=workflow_state.product_profiles,
                evidence=workflow_state.evidence,
                verification_result=VerificationResult(
                    passed=True,
                    issues=[],
                    retry_recommended=False,
                ),
            )

    def test_write_creates_exact_markdown_file(self) -> None:
        # 文件输出必须与内存渲染完全一致，供下一阶段直接下载。
        reporter = Reporter()
        reporter_input = _build_reporter_input()
        expected_report = reporter.render(reporter_input)

        with TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory) / "reports" / "sample.md"
            )
            written_path = reporter.write(reporter_input, output_path)

            self.assertEqual(written_path, output_path)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                expected_report,
            )


if __name__ == "__main__":
    unittest.main()
