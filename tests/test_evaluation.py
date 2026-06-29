import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from competitive_analysis_agent.analyst import AnalystOutput
from competitive_analysis_agent.evaluation import (
    calculate_citation_validity,
    calculate_source_coverage,
    EvaluationCase,
    load_evaluation_cases,
    render_evaluation_markdown,
    run_evaluation_case,
    run_offline_evaluation_suite,
    write_evaluation_results,
)
from competitive_analysis_agent.schemas import WorkflowState


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


class EvaluationTest(unittest.TestCase):
    def test_three_fixed_cases_are_loaded(self) -> None:
        # 评测集必须固定且包含成功、重试恢复和最终警告三种行为。
        cases = load_evaluation_cases()

        self.assertEqual(len(cases), 3)
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "complete_success",
                "retry_recovery",
                "verification_warning",
            ],
        )

    def test_offline_suite_reports_measured_metrics(self) -> None:
        # 三个案例实际运行 LangGraph 后应产生可解释的汇总指标。
        suite = run_offline_evaluation_suite()

        self.assertEqual(suite.summary.case_count, 3)
        self.assertEqual(suite.summary.case_pass_rate, 1.0)
        self.assertAlmostEqual(
            suite.summary.task_success_rate,
            2 / 3,
        )
        self.assertEqual(suite.summary.average_field_coverage, 1.0)
        self.assertEqual(suite.summary.citation_validity, 1.0)
        self.assertEqual(suite.summary.source_coverage, 1.0)
        self.assertGreater(suite.summary.total_duration_seconds, 0)
        self.assertIsNone(suite.summary.estimated_cost_usd)

    def test_warning_case_passes_behavior_but_not_user_task(self) -> None:
        # 有限循环和警告报告属于正确行为，但不能算业务任务成功。
        suite = run_offline_evaluation_suite()
        warning_case = suite.cases[2]

        self.assertTrue(warning_case.expected_behavior_passed)
        self.assertFalse(warning_case.task_succeeded)
        self.assertFalse(warning_case.verification_passed)
        self.assertTrue(warning_case.final_report_generated)
        self.assertEqual(warning_case.retry_count, 1)

    def test_invalid_citation_reduces_metric(self) -> None:
        # 引用有效率必须能发现存在于 claim 但不存在于 Evidence 的 ID。
        workflow_state = WorkflowState.model_validate(
            json.loads(
                (FIXTURE_DIRECTORY / "sample_case.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        analyst_outputs = json.loads(
            (FIXTURE_DIRECTORY / "analyst_outputs.json").read_text(
                encoding="utf-8"
            )
        )
        invalid_analysis = AnalystOutput.model_validate(
            analyst_outputs["unknown_reference"]
        ).analysis

        validity = calculate_citation_validity(
            invalid_analysis,
            workflow_state.evidence,
        )

        self.assertEqual(validity, 0.0)

    def test_partial_research_reduces_source_coverage(self) -> None:
        # 来源覆盖关注任务是否有证据，而不是单纯统计 Evidence 条数。
        workflow_state = WorkflowState.model_validate(
            json.loads(
                (FIXTURE_DIRECTORY / "sample_case.json").read_text(
                    encoding="utf-8"
                )
            )
        )

        coverage = calculate_source_coverage(
            workflow_state.research_tasks,
            workflow_state.evidence[:2],
        )

        self.assertEqual(coverage, 0.5)

    def test_runtime_failure_records_only_error_category(self) -> None:
        # 评测失败应保留可聚合类别，但不能把请求内容写进结果文件。
        evaluation_case = EvaluationCase(
            case_id="runtime_failure",
            description="工作流异常路径",
            analyst_output_keys=["valid"],
            verifier_output_keys=["supported"],
            expected_verification_passed=True,
        )
        with patch(
            "competitive_analysis_agent.evaluation.build_fixture_components",
            side_effect=RuntimeError("sensitive request content"),
        ):
            result = run_evaluation_case(evaluation_case)

        self.assertFalse(result.expected_behavior_passed)
        self.assertEqual(result.error_category, "RuntimeError")
        self.assertNotIn(
            "sensitive request content",
            result.model_dump_json(),
        )

    def test_result_files_contain_same_summary(self) -> None:
        # JSON 供程序读取，Markdown 供 README 和人工审阅。
        suite = run_offline_evaluation_suite()

        with TemporaryDirectory() as temporary_directory:
            json_path, markdown_path = write_evaluation_results(
                suite,
                Path(temporary_directory),
            )
            saved_json = json.loads(
                json_path.read_text(encoding="utf-8")
            )
            saved_markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(saved_json["summary"]["case_count"], 3)
        self.assertIn("Task success rate: 66.7%", saved_markdown)
        self.assertEqual(
            saved_markdown,
            render_evaluation_markdown(suite),
        )


if __name__ == "__main__":
    unittest.main()
