"""验证三个固定案例会运行真实 LangGraph 并产生可解释指标。"""

from competitive_analysis_agent.evaluation import (
    load_evaluation_cases,
    run_evaluation_case,
    run_offline_evaluation_suite,
)


def test_fixed_quality_cases_report_measured_metrics() -> None:
    """三类案例应全部符合预期行为，且耗时来自实际运行。"""

    cases = load_evaluation_cases()
    suite = run_offline_evaluation_suite()

    assert [item.case_id for item in cases] == [
        "fully_comparable",
        "cross_product_line_contamination",
        "insufficient_in_scope_data",
    ]
    assert suite.summary.case_count == 3
    assert suite.summary.case_pass_rate == 1.0
    assert suite.summary.task_success_rate == 2 / 3
    assert suite.summary.citation_validity == 1.0
    assert suite.summary.pricing_context_completeness == 1.0
    assert suite.summary.exclusion_accuracy == 1.0
    assert suite.summary.total_duration_seconds > 0


def test_cross_product_line_case_excludes_contamination_before_extraction() -> None:
    """消费端套餐必须进入排除分桶，不能成为正式来源或画像事实。"""

    case = load_evaluation_cases()[1]
    result, final_state = run_evaluation_case(case)

    assert result.expected_behavior_passed
    assert result.excluded_evidence_count == 1
    assert result.exclusion_accuracy == 1.0
    assert "Beacon Docs Consumer Plan" in final_state["final_report"]
    assert all(
        "E5" not in plan.evidence_ids
        for profile in final_state["product_profiles"]
        for plan in profile.pricing
    )


def test_insufficient_case_exposes_missing_core_dimension() -> None:
    """治理资料缺失时工作流应完成报告，但不能标记比较可用。"""

    case = load_evaluation_cases()[2]
    result, final_state = run_evaluation_case(case)

    assert result.expected_behavior_passed
    assert not result.task_succeeded
    assert not result.verification_passed
    assert result.core_dimension_coverage == 2 / 3
    assert result.research_error_count == 2
    assert "资料不足" in final_state["final_report"]
