import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from competitive_analysis_agent.analyst import (
    AnalystInput,
    AnalystOutput,
    build_fallback_analysis,
)
from competitive_analysis_agent.reporter import Reporter, ReporterInput
from competitive_analysis_agent.schemas import WorkflowState
from competitive_analysis_agent.verifier import (
    VerificationIssue,
    VerificationResult,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
SAMPLE_REPORT_PATH = Path(__file__).parents[1] / "docs" / "sample-report.md"


def _load_json(file_name: str) -> dict:
    return json.loads((FIXTURE_DIRECTORY / file_name).read_text(encoding="utf-8"))


def _build_reporter_input(
    verification_result: VerificationResult | None = None,
) -> ReporterInput:
    workflow_state = WorkflowState.model_validate(_load_json("sample_case.json"))
    analysis = AnalystOutput.model_validate(
        _load_json("analyst_outputs.json")["valid"]
    ).analysis
    return ReporterInput(
        analysis=analysis,
        market_definition=workflow_state.market_definition,
        product_profiles=workflow_state.product_profiles,
        evidence=workflow_state.evidence,
        verification_result=verification_result or VerificationResult(
            passed=True, issues=[], retry_recommended=False
        ),
    )


class ReporterTest(unittest.TestCase):
    def test_report_matches_golden_file(self) -> None:
        self.assertEqual(
            Reporter().render(_build_reporter_input()),
            SAMPLE_REPORT_PATH.read_text(encoding="utf-8"),
        )

    def test_decision_report_has_required_structure_and_chinese_summary(self) -> None:
        report = Reporter().render(_build_reporter_input())

        for heading in (
            "## 执行摘要", "## 验证状态", "## 关键结论", "## 统一能力矩阵",
            "## 模型与价格对比", "## 场景成本估算", "## 使用限制和企业能力",
            "## 产品优势与短板", "## 场景化选择建议", "## 市场机会点",
            "## 数据缺口和待核验内容", "## 来源附录",
        ):
            self.assertIn(heading, report)
        summary = report.split("## 执行摘要", 1)[1].split("## 验证状态", 1)[0].strip()
        self.assertGreaterEqual(len(summary), 300)
        self.assertLessEqual(len(summary), 500)
        self.assertIn("✅ 已验证，可用于初步决策", report)
        self.assertIn("https://example.com/atlas/features", report)
        self.assertIn("来源类型", report)
        self.assertIn("采集时间", report)

    def test_failed_verification_marks_draft_and_blocks_scenario_advice(self) -> None:
        reporter_input = _build_reporter_input()
        api_market = reporter_input.market_definition.model_copy(
            update={"pricing_scope": "api"}
        )
        analysis = build_fallback_analysis(
            AnalystInput(
                profiles=reporter_input.product_profiles,
                market_definition=api_market,
            )
        )
        failed = VerificationResult(
            passed=False,
            issues=[
                VerificationIssue(
                    issue_type="unsupported_claim",
                    claim_path="conclusion",
                    message="需要核验。",
                    evidence_ids=["E1"],
                    suggested_action="收窄结论。",
                )
            ],
            retry_recommended=True,
        )
        report = Reporter().render(
            reporter_input.model_copy(
                update={
                    "analysis": analysis,
                    "market_definition": api_market,
                    "verification_result": failed,
                }
            )
        )

        self.assertIn("草稿状态", report)
        self.assertIn("待验证判断", report)
        self.assertIn("验证未通过，未输出确定性场景购买建议。", report)
        self.assertNotIn("编程 Agent | One API", report)

    def test_api_report_contains_cost_table_and_confidence_per_scenario(self) -> None:
        reporter_input = _build_reporter_input()
        api_market = reporter_input.market_definition.model_copy(
            update={"pricing_scope": "api"}
        )
        analysis = build_fallback_analysis(
            AnalystInput(
                profiles=reporter_input.product_profiles,
                market_definition=api_market,
            )
        )
        report = Reporter().render(
            reporter_input.model_copy(
                update={"analysis": analysis, "market_definition": api_market}
            )
        )

        self.assertIn("## 场景成本估算", report)
        self.assertIn("## 场景化选择建议", report)
        self.assertIn("编程 Agent、RAG 与知识库问答", report)
        self.assertIn("未输出正式购买建议", report)
        self.assertIn("| 机会点 | 用户痛点 |", report)

    def test_write_creates_exact_markdown_file(self) -> None:
        reporter = Reporter()
        reporter_input = _build_reporter_input()
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "reports" / "report.md"
            self.assertEqual(reporter.write(reporter_input, output_path), output_path)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                reporter.render(reporter_input),
            )


if __name__ == "__main__":
    unittest.main()
