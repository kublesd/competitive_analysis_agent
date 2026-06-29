import pytest

from competitive_analysis_agent.evaluation import (
    run_live_evaluation_case,
)


@pytest.mark.live_llm
def test_fixed_evaluation_case_with_real_model() -> None:
    """真实模型固定案例应通过核心可靠性和覆盖指标。"""

    result = run_live_evaluation_case()

    assert result.expected_behavior_passed
    assert result.task_succeeded
    assert result.verification_passed
    assert result.final_report_generated
    assert result.citation_validity == 1.0
    assert result.source_coverage == 1.0
    assert result.field_coverage >= 0.4
    assert result.duration_seconds > 0
    assert result.error_category is None
