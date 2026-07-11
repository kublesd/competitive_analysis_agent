"""运行固定 LangGraph 评测案例并生成可重复的质量指标。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import Field

from competitive_analysis_agent.analyst import (
    Analyst,
    CompetitiveAnalysis,
    FakeAnalystModel,
    collect_analysis_claims,
)
from competitive_analysis_agent.extractor import (
    Extractor,
    FakeExtractorModel,
)
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_workflow import (
    create_live_workflow_components,
)
from competitive_analysis_agent.planner import (
    FakePlannerModel,
    Planner,
    PlannerInput,
)
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    ResearchTask,
    RequiredText,
)
from competitive_analysis_agent.search import (
    FakeSearchProvider,
    SearchAdapter,
)
from competitive_analysis_agent.verifier import Verifier
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    WorkflowGraphState,
    create_initial_state,
    create_workflow_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIRECTORY = PROJECT_ROOT / "tests" / "fixtures"
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "cases.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "reports"
FIXED_EVALUATION_TIME = datetime(
    2026,
    6,
    14,
    8,
    0,
    tzinfo=timezone.utc,
)


class EvaluationCase(ContractModel):
    """描述一个固定输入、模型响应序列和预期终态。"""

    case_id: RequiredText
    description: RequiredText
    target_product: RequiredText = "Atlas Notes"
    competitors: list[RequiredText] = Field(
        default_factory=lambda: ["Beacon Docs"]
    )
    dimensions: list[RequiredText] = Field(
        default_factory=lambda: ["features", "pricing"]
    )
    analyst_output_keys: list[RequiredText] = Field(default_factory=list)
    verifier_output_keys: list[RequiredText] = Field(default_factory=list)
    expected_verification_passed: bool
    expected_retry_count: int | None = Field(default=None, ge=0)


class EvaluationCaseResult(ContractModel):
    """保存单个案例的行为结果、质量指标和实际耗时。"""

    case_id: RequiredText
    description: RequiredText
    expected_behavior_passed: bool
    task_succeeded: bool
    verification_passed: bool
    final_report_generated: bool
    field_coverage: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    source_coverage: float = Field(ge=0, le=1)
    duration_seconds: float = Field(ge=0)
    retry_count: int = Field(ge=0)
    research_error_count: int = Field(ge=0)
    stage_history: list[str] = Field(default_factory=list)
    error_category: str | None = None


class EvaluationSummary(ContractModel):
    """汇总评测集规模、通过率、覆盖率、耗时和可用成本数据。"""

    case_count: int = Field(ge=1)
    case_pass_rate: float = Field(ge=0, le=1)
    task_success_rate: float = Field(ge=0, le=1)
    average_field_coverage: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    source_coverage: float = Field(ge=0, le=1)
    total_duration_seconds: float = Field(ge=0)
    average_duration_seconds: float = Field(ge=0)
    estimated_cost_usd: float | None = None


class EvaluationSuiteResult(ContractModel):
    """保存评测生成时间、汇总指标和全部案例结果。"""

    generated_at: datetime
    mode: RequiredText
    summary: EvaluationSummary
    cases: list[EvaluationCaseResult] = Field(min_length=1)


class SequenceVerifierModel:
    """按固定顺序返回多个 Verifier 输出，供评测循环案例使用。"""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self._index = 0

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回下一条响应，超出预设次数时明确失败。"""

        if self._index >= len(self._responses):
            raise RuntimeError("No verifier evaluation response left.")
        response = self._responses[self._index]
        self._index += 1
        return response


def load_evaluation_cases(
    cases_path: Path = DEFAULT_CASES_PATH,
) -> list[EvaluationCase]:
    """读取并校验固定评测案例。"""

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return [
        EvaluationCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]


def load_fixture(file_name: str) -> dict[str, Any]:
    """读取评测所需的既有固定模型或搜索输出。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def build_fixture_components(
    evaluation_case: EvaluationCase,
) -> WorkflowComponents:
    """根据案例指定的响应序列创建离线 LangGraph 组件。"""

    if not evaluation_case.analyst_output_keys:
        raise ValueError("Offline evaluation case requires analyst outputs.")
    if not evaluation_case.verifier_output_keys:
        raise ValueError("Offline evaluation case requires verifier outputs.")

    planner_outputs = load_fixture("planner_outputs.json")
    extractor_outputs = load_fixture("extractor_outputs.json")
    analyst_outputs = load_fixture("analyst_outputs.json")
    verifier_outputs = load_fixture("verifier_outputs.json")
    search_results = load_fixture("workflow_search_results.json")

    analyst_responses = [
        analyst_outputs[key]
        for key in evaluation_case.analyst_output_keys
    ]
    verifier_responses = [
        verifier_outputs[key]
        for key in evaluation_case.verifier_output_keys
    ]

    return WorkflowComponents(
        planner=Planner(FakePlannerModel([planner_outputs["valid"]])),
        researcher=Researcher(
            SearchAdapter(FakeSearchProvider(search_results)),
            clock=lambda: FIXED_EVALUATION_TIME,
        ),
        extractor=Extractor(
            FakeExtractorModel(
                [
                    extractor_outputs["valid_atlas"],
                    extractor_outputs["valid_beacon"],
                ]
            )
        ),
        analyst=Analyst(FakeAnalystModel(analyst_responses)),
        verifier=Verifier(SequenceVerifierModel(verifier_responses)),
        reporter=Reporter(),
    )


def create_case_initial_state(
    evaluation_case: EvaluationCase,
    max_results_per_task: int = 3,
) -> WorkflowGraphState:
    """根据评测案例创建完整初始 State。"""

    planner_input = PlannerInput(
        target_product=evaluation_case.target_product,
        competitors=evaluation_case.competitors,
        dimensions=evaluation_case.dimensions,
    )
    return create_initial_state(
        planner_input=planner_input,
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
        max_results_per_task=max_results_per_task,
    )


def run_evaluation_case(
    evaluation_case: EvaluationCase,
    components: WorkflowComponents | None = None,
    max_results_per_task: int = 3,
) -> EvaluationCaseResult:
    """执行一个实际 LangGraph 案例并计算终态指标。"""

    started_at = perf_counter()
    try:
        current_components = components
        if current_components is None:
            current_components = build_fixture_components(evaluation_case)

        graph = create_workflow_graph(current_components)
        initial_state = create_case_initial_state(
            evaluation_case,
            max_results_per_task=max_results_per_task,
        )
        final_state = graph.invoke(initial_state)
    except Exception as error:
        duration_seconds = perf_counter() - started_at
        return build_failed_case_result(
            evaluation_case=evaluation_case,
            duration_seconds=duration_seconds,
            error=error,
        )

    duration_seconds = perf_counter() - started_at
    return evaluate_workflow_state(
        evaluation_case=evaluation_case,
        final_state=final_state,
        duration_seconds=duration_seconds,
    )


def evaluate_workflow_state(
    evaluation_case: EvaluationCase,
    final_state: WorkflowGraphState,
    duration_seconds: float,
) -> EvaluationCaseResult:
    """从 LangGraph 最终 State 计算单案例确定性指标。"""

    analysis = final_state["analysis_result"]
    verification = final_state["verification_result"]
    final_report = final_state["final_report"]
    if analysis is None or verification is None:
        return build_failed_case_result(
            evaluation_case=evaluation_case,
            duration_seconds=duration_seconds,
            error=ValueError("Workflow final state is incomplete."),
        )

    verification_matches = (
        verification.passed
        == evaluation_case.expected_verification_passed
    )
    retry_matches = True
    if evaluation_case.expected_retry_count is not None:
        retry_matches = (
            final_state["retry_count"]
            == evaluation_case.expected_retry_count
        )

    report_generated = final_report is not None
    warning_matches = True
    if not evaluation_case.expected_verification_passed:
        warning_matches = (
            report_generated
            and "本报告未通过最终验证" in final_report
        )

    stage_history = final_state["stage_history"]
    terminal_stage_matches = bool(stage_history)
    terminal_stage_matches = (
        terminal_stage_matches and stage_history[-1] == "reporter"
    )
    expected_behavior_passed = all(
        [
            verification_matches,
            retry_matches,
            report_generated,
            warning_matches,
            terminal_stage_matches,
        ]
    )
    task_succeeded = verification.passed and report_generated

    return EvaluationCaseResult(
        case_id=evaluation_case.case_id,
        description=evaluation_case.description,
        expected_behavior_passed=expected_behavior_passed,
        task_succeeded=task_succeeded,
        verification_passed=verification.passed,
        final_report_generated=report_generated,
        field_coverage=calculate_field_coverage(analysis),
        citation_validity=calculate_citation_validity(
            analysis,
            final_state["evidence"],
        ),
        source_coverage=calculate_source_coverage(
            final_state["research_tasks"],
            final_state["evidence"],
        ),
        duration_seconds=duration_seconds,
        retry_count=final_state["retry_count"],
        research_error_count=len(final_state["research_errors"]),
        stage_history=stage_history,
    )


def build_failed_case_result(
    evaluation_case: EvaluationCase,
    duration_seconds: float,
    error: Exception,
) -> EvaluationCaseResult:
    """把运行异常转换成可汇总的失败案例，而不是终止整套评测。"""

    return EvaluationCaseResult(
        case_id=evaluation_case.case_id,
        description=evaluation_case.description,
        expected_behavior_passed=False,
        task_succeeded=False,
        verification_passed=False,
        final_report_generated=False,
        field_coverage=0,
        citation_validity=0,
        source_coverage=0,
        duration_seconds=duration_seconds,
        retry_count=0,
        research_error_count=0,
        stage_history=[],
        error_category=type(error).__name__,
    )


def calculate_field_coverage(analysis: CompetitiveAnalysis) -> float:
    """计算五个报告分析字段中非空字段的比例。"""

    populated_fields = [
        bool(analysis.positioning),
        bool(analysis.features),
        bool(analysis.pricing),
        bool(analysis.opportunities),
        bool(analysis.conclusion.claim),
    ]
    return sum(populated_fields) / len(populated_fields)


def calculate_citation_validity(
    analysis: CompetitiveAnalysis,
    evidence: list[Evidence],
) -> float:
    """计算分析引用中能映射到当前 Evidence 的 ID 比例。"""

    known_evidence_ids = {item.evidence_id for item in evidence}
    referenced_evidence_ids: list[str] = []
    for claim in collect_analysis_claims(analysis):
        referenced_evidence_ids.extend(claim.evidence_ids)

    if not referenced_evidence_ids:
        return 1.0

    valid_reference_count = 0
    for evidence_id in referenced_evidence_ids:
        if evidence_id in known_evidence_ids:
            valid_reference_count += 1
    return valid_reference_count / len(referenced_evidence_ids)


def calculate_source_coverage(
    research_tasks: list[ResearchTask],
    evidence: list[Evidence],
) -> float:
    """计算计划中的产品与主题组合有 Evidence 覆盖的比例。"""

    planned_pairs = {
        (task.product_name, task.topic)
        for task in research_tasks
    }
    if not planned_pairs:
        return 0.0

    covered_pairs = {
        (item.product_name, item.topic)
        for item in evidence
    }
    covered_planned_pairs = planned_pairs.intersection(covered_pairs)
    return len(covered_planned_pairs) / len(planned_pairs)


def summarize_evaluation_cases(
    case_results: list[EvaluationCaseResult],
) -> EvaluationSummary:
    """把案例级指标汇总为评测集指标。"""

    case_count = len(case_results)
    case_pass_count = sum(
        result.expected_behavior_passed for result in case_results
    )
    task_success_count = sum(
        result.task_succeeded for result in case_results
    )
    total_duration = sum(
        result.duration_seconds for result in case_results
    )

    return EvaluationSummary(
        case_count=case_count,
        case_pass_rate=case_pass_count / case_count,
        task_success_rate=task_success_count / case_count,
        average_field_coverage=(
            sum(result.field_coverage for result in case_results)
            / case_count
        ),
        citation_validity=(
            sum(result.citation_validity for result in case_results)
            / case_count
        ),
        source_coverage=(
            sum(result.source_coverage for result in case_results)
            / case_count
        ),
        total_duration_seconds=total_duration,
        average_duration_seconds=total_duration / case_count,
        # 当前模型工厂未暴露统一 Token usage，因此不估算虚假成本。
        estimated_cost_usd=None,
    )


def run_offline_evaluation_suite(
    cases_path: Path = DEFAULT_CASES_PATH,
) -> EvaluationSuiteResult:
    """运行三个无网络固定案例并返回实测评测结果。"""

    evaluation_cases = load_evaluation_cases(cases_path)
    case_results: list[EvaluationCaseResult] = []

    for evaluation_case in evaluation_cases:
        case_result = run_evaluation_case(evaluation_case)
        case_results.append(case_result)

    return EvaluationSuiteResult(
        generated_at=datetime.now(timezone.utc),
        mode="offline_fixture",
        summary=summarize_evaluation_cases(case_results),
        cases=case_results,
    )


def run_live_evaluation_case() -> EvaluationCaseResult:
    """使用真实模型节点运行一个短固定案例并计算同一组指标。"""

    evaluation_case = EvaluationCase(
        case_id="live_feature_analysis",
        description="真实模型运行两个产品的 features 分析。",
        dimensions=["features"],
        expected_verification_passed=True,
        expected_retry_count=None,
    )
    settings = load_live_settings()
    components = create_live_workflow_components(settings)
    return run_evaluation_case(
        evaluation_case=evaluation_case,
        components=components,
        max_results_per_task=1,
    )


def render_evaluation_markdown(
    suite_result: EvaluationSuiteResult,
) -> str:
    """把评测结果渲染成适合 README 引用的 Markdown 摘要。"""

    summary = suite_result.summary
    lines = [
        "# Evaluation Results",
        "",
        f"- Mode: `{suite_result.mode}`",
        f"- Generated at: `{suite_result.generated_at.isoformat()}`",
        f"- Cases: {summary.case_count}",
        f"- Case pass rate: {format_percentage(summary.case_pass_rate)}",
        f"- Task success rate: {format_percentage(summary.task_success_rate)}",
        (
            "- Average field coverage: "
            f"{format_percentage(summary.average_field_coverage)}"
        ),
        (
            "- Citation validity: "
            f"{format_percentage(summary.citation_validity)}"
        ),
        (
            "- Source coverage: "
            f"{format_percentage(summary.source_coverage)}"
        ),
        (
            "- Average duration: "
            f"{summary.average_duration_seconds:.4f} seconds"
        ),
        "- Estimated cost: not available",
        "",
        "| Case | Expected behavior | Task success | Field coverage | "
        "Citation validity | Source coverage | Retry | Duration (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in suite_result.cases:
        lines.append(
            "| "
            f"{result.case_id} | "
            f"{format_boolean(result.expected_behavior_passed)} | "
            f"{format_boolean(result.task_succeeded)} | "
            f"{format_percentage(result.field_coverage)} | "
            f"{format_percentage(result.citation_validity)} | "
            f"{format_percentage(result.source_coverage)} | "
            f"{result.retry_count} | "
            f"{result.duration_seconds:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Metric Boundaries",
            "",
            "- Citation validity only checks whether referenced IDs exist; "
            "it does not prove the claim is semantically true.",
            "- Source coverage checks planned product/topic pairs with "
            "Evidence; it does not measure source authority or freshness.",
            "- Field coverage measures populated report sections; more "
            "fields do not automatically mean better analysis.",
            "- Cost remains unavailable until model usage metadata is "
            "captured consistently.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_results(
    suite_result: EvaluationSuiteResult,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """将完整 JSON 和人类可读 Markdown 评测结果写入磁盘。"""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "evaluation-results.json"
    markdown_path = output_directory / "evaluation-results.md"

    json_text = json.dumps(
        suite_result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    json_path.write_text(json_text + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_evaluation_markdown(suite_result),
        encoding="utf-8",
    )
    return json_path, markdown_path


def format_percentage(value: float) -> str:
    """把 0 到 1 的比例转换成一位小数百分比。"""

    return f"{value * 100:.1f}%"


def format_boolean(value: bool) -> str:
    """把布尔结果转换成易读文本。"""

    return "pass" if value else "fail"


def main() -> None:
    """运行离线评测集并输出生成文件和汇总指标。"""

    parser = argparse.ArgumentParser(
        description="Run the fixed competitive-analysis evaluation suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    suite_result = run_offline_evaluation_suite()
    json_path, markdown_path = write_evaluation_results(
        suite_result,
        output_directory=arguments.output_dir,
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(
        "Task success rate: "
        f"{format_percentage(suite_result.summary.task_success_rate)}"
    )


if __name__ == "__main__":
    main()
