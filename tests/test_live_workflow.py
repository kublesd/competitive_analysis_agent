import pytest

from competitive_analysis_agent.live_workflow import run_smoke_test


@pytest.mark.live_llm
def test_live_langgraph_workflow_reaches_verified_result() -> None:
    """真实模型节点应经完整图到达已验证且重试受限的终态。"""

    final_state = run_smoke_test()

    assert len(final_state["research_tasks"]) == 2
    assert len(final_state["evidence"]) == 2
    assert final_state["research_errors"] == []
    assert len(final_state["product_profiles"]) == 2
    assert final_state["analysis_result"] is not None
    assert final_state["analysis_result"].products == [
        "Atlas Notes",
        "Beacon Docs",
    ]
    assert final_state["verification_result"] is not None
    assert final_state["verification_result"].passed
    assert final_state["final_report"] is not None
    assert "# 竞品分析报告" in final_state["final_report"]
    assert final_state["retry_count"] <= 1
    assert final_state["stage_history"][:3] == [
        "planner",
        "researcher",
        "extractor",
    ]
    assert final_state["stage_history"][-1] == "reporter"
    assert final_state["stage_history"].count("analyst") <= 2
    assert final_state["stage_history"].count("verifier") <= 2
    assert final_state["stage_history"].count("reporter") == 1
