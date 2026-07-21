"""用固定案例运行真实工作流，并从终态计算确定性质量指标。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from competitive_analysis_agent.analyst import (
    Analyst,
    CompetitiveAnalysis,
    FakeAnalystModel,
    collect_analysis_claims,
)
from competitive_analysis_agent.extractor import Extractor, FakeExtractorModel
from competitive_analysis_agent.planner import FakePlannerModel, Planner, PlannerInput
from competitive_analysis_agent.pricing_utils import should_report_missing_billing_cycle
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    MarketDefinition,
    ProductProfile,
)
from competitive_analysis_agent.search import FakeSearchProvider, SearchAdapter
from competitive_analysis_agent.verifier import (
    FakeVerifierModel,
    Verifier,
    pricing_plan_has_unit,
    pricing_plan_requires_context,
)
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    WorkflowGraphState,
    create_initial_state,
    create_workflow_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "cases.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "evaluation"
FIXTURE_DIRECTORY = PROJECT_ROOT / "tests" / "fixtures"


class EvaluationCase(ContractModel):
    """描述一个固定场景及其可确定判断的预期行为。"""

    case_id: str
    description: str
    scenario: Literal[
        "fully_comparable",
        "cross_product_line_contamination",
        "insufficient_in_scope_data",
    ]
    core_dimensions: list[str] = Field(min_length=1)
    expected_verification_passed: bool
    expected_excluded_titles: list[str] = Field(default_factory=list)


class EvaluationCaseResult(ContractModel):
    """保存一次实际工作流运行的质量指标和行为结果。"""

    case_id: str
    description: str
    expected_behavior_passed: bool
    task_succeeded: bool
    verification_passed: bool
    scope_consistency: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    core_dimension_coverage: float = Field(ge=0, le=1)
    pricing_context_completeness: float = Field(ge=0, le=1)
    exclusion_accuracy: float = Field(ge=0, le=1)
    recommendation_actionability: float = Field(ge=0, le=1)
    duration_seconds: float = Field(ge=0)
    retry_count: int = Field(ge=0)
    in_scope_evidence_count: int = Field(ge=0)
    excluded_evidence_count: int = Field(ge=0)
    uncertain_evidence_count: int = Field(ge=0)
    research_error_count: int = Field(ge=0)
    stage_history: list[str]
    error_category: str | None = None


class EvaluationSummary(ContractModel):
    """汇总多个案例的平均质量指标和实测耗时。"""

    case_count: int = Field(gt=0)
    case_pass_rate: float = Field(ge=0, le=1)
    task_success_rate: float = Field(ge=0, le=1)
    scope_consistency_rate: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    core_dimension_coverage: float = Field(ge=0, le=1)
    pricing_context_completeness: float = Field(ge=0, le=1)
    exclusion_accuracy: float = Field(ge=0, le=1)
    recommendation_actionability: float = Field(ge=0, le=1)
    total_duration_seconds: float = Field(ge=0)
    average_duration_seconds: float = Field(ge=0)


class EvaluationSuiteResult(ContractModel):
    """保存一轮固定评测的生成时间、汇总和案例明细。"""

    generated_at: datetime
    mode: str
    summary: EvaluationSummary
    cases: list[EvaluationCaseResult]


def load_evaluation_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    """读取并校验固定评估案例；文件缺失或格式错误时直接失败。"""

    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(item) for item in raw_cases]


def calculate_citation_validity(
    analysis: CompetitiveAnalysis,
    evidence: list[Evidence],
) -> float:
    """计算 claim 与建议中能映射到范围内 Evidence 的引用比例。"""

    known_ids = {item.evidence_id for item in evidence}
    referenced_ids: list[str] = []
    for claim in collect_analysis_claims(analysis):
        referenced_ids.extend(claim.evidence_ids)
    for recommendation in analysis.recommendations:
        referenced_ids.extend(recommendation.evidence_ids)

    if not referenced_ids:
        return 1.0
    valid_count = sum(evidence_id in known_ids for evidence_id in referenced_ids)
    return valid_count / len(referenced_ids)


def calculate_core_dimension_coverage(
    profiles: list[ProductProfile],
    market_definition: MarketDefinition,
) -> float:
    """计算“产品 × 核心维度”组合中有证据化事实的比例。"""

    expected_count = len(profiles) * len(market_definition.core_dimensions)
    if expected_count == 0:
        return 0.0

    covered_count = 0
    for profile in profiles:
        findings = {item.dimension: item for item in profile.dimension_findings}
        for dimension in market_definition.core_dimensions:
            finding = findings.get(dimension)
            if finding is not None and finding.facts and finding.evidence_ids:
                covered_count += 1
    return covered_count / expected_count


def calculate_pricing_context_completeness(
    profiles: list[ProductProfile],
    market_definition: MarketDefinition,
) -> float:
    """计算价格项是否具有当前 API 或订阅范围所需的上下文。"""

    relevant_plans = []
    for profile in profiles:
        for pricing_plan in profile.pricing:
            if pricing_plan_requires_context(pricing_plan):
                relevant_plans.append(pricing_plan)

    if not relevant_plans:
        return 1.0

    complete_count = 0
    for pricing_plan in relevant_plans:
        has_billing_cycle = not should_report_missing_billing_cycle(
            price_text=pricing_plan.price,
            billing_cycle=pricing_plan.billing_cycle,
            unit_text=pricing_plan.unit,
        )
        context_complete = pricing_plan_has_unit(pricing_plan)
        if market_definition.pricing_scope == "subscription":
            context_complete = context_complete and has_billing_cycle
        if context_complete:
            complete_count += 1
    return complete_count / len(relevant_plans)


def calculate_exclusion_accuracy(
    excluded_evidence: list[Evidence],
    expected_titles: list[str],
) -> float:
    """用标题集合的 Jaccard 比例同时惩罚漏排与误排。"""

    actual = {item.title for item in excluded_evidence}
    expected = set(expected_titles)
    if not actual and not expected:
        return 1.0
    return len(actual & expected) / len(actual | expected)


def calculate_recommendation_actionability(
    analysis: CompetitiveAnalysis,
    evidence: list[Evidence],
) -> float:
    """计算建议中产品和 Evidence 均能映射到当前运行数据的比例。"""

    if not analysis.recommendations:
        return 0.0

    known_products = set(analysis.products)
    known_evidence_ids = {item.evidence_id for item in evidence}
    actionable_count = 0
    for recommendation in analysis.recommendations:
        products_valid = set(recommendation.product_names) <= known_products
        evidence_valid = set(recommendation.evidence_ids) <= known_evidence_ids
        if products_valid and evidence_valid:
            actionable_count += 1
    return actionable_count / len(analysis.recommendations)


def evaluate_workflow_state(
    evaluation_case: EvaluationCase,
    final_state: WorkflowGraphState,
    duration_seconds: float,
) -> EvaluationCaseResult:
    """从实际 LangGraph 终态计算一个案例的全部确定性指标。"""

    analysis = final_state["analysis_result"]
    verification = final_state["verification_result"]
    report = final_state["final_report"]
    if analysis is None or verification is None or report is None:
        raise ValueError("Workflow final state is incomplete.")

    exclusion_accuracy = calculate_exclusion_accuracy(
        final_state["excluded_evidence"],
        evaluation_case.expected_excluded_titles,
    )
    expected_behavior_passed = all(
        [
            verification.passed
            == evaluation_case.expected_verification_passed,
            exclusion_accuracy == 1.0,
            bool(final_state["stage_history"]),
            final_state["stage_history"][-1] == "reporter",
        ]
    )

    return EvaluationCaseResult(
        case_id=evaluation_case.case_id,
        description=evaluation_case.description,
        expected_behavior_passed=expected_behavior_passed,
        task_succeeded=verification.passed,
        verification_passed=verification.passed,
        scope_consistency=float(verification.scope_consistent),
        citation_validity=calculate_citation_validity(
            analysis, final_state["evidence"]
        ),
        core_dimension_coverage=calculate_core_dimension_coverage(
            final_state["product_profiles"], final_state["market_definition"]
        ),
        pricing_context_completeness=calculate_pricing_context_completeness(
            final_state["product_profiles"],
            final_state["market_definition"],
        ),
        exclusion_accuracy=exclusion_accuracy,
        recommendation_actionability=calculate_recommendation_actionability(
            analysis, final_state["evidence"]
        ),
        duration_seconds=duration_seconds,
        retry_count=final_state["retry_count"],
        in_scope_evidence_count=len(final_state["evidence"]),
        excluded_evidence_count=len(final_state["excluded_evidence"]),
        uncertain_evidence_count=len(final_state["uncertain_evidence"]),
        research_error_count=len(final_state["research_errors"]),
        stage_history=final_state["stage_history"],
    )


def summarize_evaluation_cases(
    case_results: list[EvaluationCaseResult],
) -> EvaluationSummary:
    """对案例指标做简单平均；空列表属于调用错误。"""

    if not case_results:
        raise ValueError("At least one evaluation result is required.")
    count = len(case_results)
    total_duration = sum(item.duration_seconds for item in case_results)
    def average(field: str) -> float:
        return sum(getattr(item, field) for item in case_results) / count
    return EvaluationSummary(
        case_count=count,
        case_pass_rate=sum(item.expected_behavior_passed for item in case_results)
        / count,
        task_success_rate=sum(item.task_succeeded for item in case_results) / count,
        scope_consistency_rate=average("scope_consistency"),
        citation_validity=average("citation_validity"),
        core_dimension_coverage=average("core_dimension_coverage"),
        pricing_context_completeness=average("pricing_context_completeness"),
        exclusion_accuracy=average("exclusion_accuracy"),
        recommendation_actionability=average("recommendation_actionability"),
        total_duration_seconds=total_duration,
        average_duration_seconds=total_duration / count,
    )


def build_fixture_components(evaluation_case: EvaluationCase) -> WorkflowComponents:
    """按案例组装已有 Fake Model 与固定搜索，不访问网络或付费 API。"""

    extractor_outputs = _load_fixture("extractor_outputs.json")
    analyst_outputs = _load_fixture("analyst_outputs.json")
    verifier_outputs = _load_fixture("verifier_outputs.json")
    search_results = _load_fixture("workflow_search_results.json")

    if evaluation_case.scenario == "cross_product_line_contamination":
        query = _build_provider_query("Beacon Docs", "pricing")
        search_results[query].append(
            {
                "title": "Beacon Docs Consumer Plan",
                "url": "https://example.com/beacon/consumer",
                "snippet": "Beacon Docs 消费端套餐面向个人用户。",
            }
        )

    planner_response = {
        "tasks": [
            {
                "product_name": product_name,
                "topic": dimension,
                "query": _build_query(product_name, dimension),
            }
            for product_name in ["Atlas Notes", "Beacon Docs"]
            for dimension in evaluation_case.core_dimensions
        ]
    }
    return WorkflowComponents(
        planner=Planner(FakePlannerModel([planner_response])),
        researcher=Researcher(
            SearchAdapter(FakeSearchProvider(search_results)),
            clock=lambda: datetime(2026, 6, 13, 8, 0, tzinfo=timezone.utc),
        ),
        extractor=Extractor(
            FakeExtractorModel(
                [
                    extractor_outputs["valid_atlas"],
                    extractor_outputs["valid_beacon"],
                ]
            )
        ),
        analyst=Analyst(FakeAnalystModel([analyst_outputs["valid"]])),
        verifier=Verifier(FakeVerifierModel(verifier_outputs["supported"])),
        reporter=Reporter(),
    )


def create_case_initial_state(evaluation_case: EvaluationCase) -> WorkflowGraphState:
    """把案例转换为正式 Planner 输入和完整工作流初始 State。"""

    market_definition = MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=evaluation_case.core_dimensions,
        exclusions=["消费端套餐", "API 用量价格"],
    )
    planner_input = PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        market_definition=market_definition,
    )
    return create_initial_state(
        planner_input,
        market_definition=market_definition,
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
    )


def run_evaluation_case(
    evaluation_case: EvaluationCase,
    components: WorkflowComponents | None = None,
) -> tuple[EvaluationCaseResult, WorkflowGraphState]:
    """实际运行一个案例，并同时返回指标和可重生成报告的终态。"""

    started_at = perf_counter()
    graph = create_workflow_graph(components or build_fixture_components(evaluation_case))
    final_state = graph.invoke(create_case_initial_state(evaluation_case))
    duration_seconds = perf_counter() - started_at
    result = evaluate_workflow_state(evaluation_case, final_state, duration_seconds)
    return result, final_state


def run_offline_evaluation_suite() -> EvaluationSuiteResult:
    """运行三个固定离线案例并返回实际测量结果。"""

    results = [run_evaluation_case(case)[0] for case in load_evaluation_cases()]
    return EvaluationSuiteResult(
        generated_at=datetime.now(timezone.utc),
        mode="offline_fixture",
        summary=summarize_evaluation_cases(results),
        cases=results,
    )


def render_evaluation_markdown(suite: EvaluationSuiteResult) -> str:
    """将评测结果渲染为适合人工复查的 Markdown 摘要。"""

    summary = suite.summary
    lines = [
        "# Quality Evaluation Results",
        "",
        f"- Mode: `{suite.mode}`",
        f"- Cases: {summary.case_count}",
        f"- Case pass rate: {_percentage(summary.case_pass_rate)}",
        f"- Task success rate: {_percentage(summary.task_success_rate)}",
        f"- Scope consistency rate: {_percentage(summary.scope_consistency_rate)}",
        f"- Citation validity: {_percentage(summary.citation_validity)}",
        f"- Core dimension coverage: {_percentage(summary.core_dimension_coverage)}",
        f"- Pricing context completeness: {_percentage(summary.pricing_context_completeness)}",
        f"- Exclusion accuracy: {_percentage(summary.exclusion_accuracy)}",
        f"- Recommendation actionability: {_percentage(summary.recommendation_actionability)}",
        f"- Average duration: {summary.average_duration_seconds:.4f} seconds",
        "",
        "| Case | Expected behavior | Verified | Scope | Citations | Dimensions | Pricing | Exclusion | Actionability | Duration (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in suite.cases:
        lines.append(
            "| "
            f"{result.case_id} | {_pass(result.expected_behavior_passed)} | "
            f"{_pass(result.verification_passed)} | {_percentage(result.scope_consistency)} | "
            f"{_percentage(result.citation_validity)} | {_percentage(result.core_dimension_coverage)} | "
            f"{_percentage(result.pricing_context_completeness)} | {_percentage(result.exclusion_accuracy)} | "
            f"{_percentage(result.recommendation_actionability)} | {result.duration_seconds:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metric boundaries",
            "",
            "- Citation validity checks ID mapping, not semantic truth.",
            "- Coverage measures supplied evidence, not whether missing facts exist in the market.",
            "- Actionability checks structured, traceable recommendations, not business outcome.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_results(
    suite: EvaluationSuiteResult,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """写入机器可读 JSON 和人工可读 Markdown 结果。"""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "evaluation-results.json"
    markdown_path = output_directory / "evaluation-results.md"
    json_path.write_text(
        json.dumps(suite.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_evaluation_markdown(suite), encoding="utf-8")
    return json_path, markdown_path


def _load_fixture(file_name: str) -> dict:
    """读取评测复用的项目测试 fixture。"""

    return json.loads((FIXTURE_DIRECTORY / file_name).read_text(encoding="utf-8"))


def _build_query(product_name: str, dimension: str) -> str:
    """生成与 Planner 范围校验一致的固定查询。"""

    return (
        f"{product_name} SaaS 协作软件 企业订阅产品 {dimension} official "
        "exclude 消费端套餐 exclude API 用量价格"
    )


def _build_provider_query(product_name: str, dimension: str) -> str:
    """生成供应商实际收到的聚焦查询，不携带市场控制词。"""

    if dimension == "pricing":
        search_topic = "pricing plans price"
    elif dimension == "features":
        search_topic = "product features collaboration"
    else:
        search_topic = dimension
    return f"{product_name} official {search_topic}"


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pass(value: bool) -> str:
    return "pass" if value else "fail"


def main() -> None:
    """运行离线评测并输出结果文件位置，不输出模型或密钥内容。"""

    parser = argparse.ArgumentParser(description="Run fixed quality evaluation cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    suite = run_offline_evaluation_suite()
    json_path, markdown_path = write_evaluation_results(suite, arguments.output_dir)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Case pass rate: {_percentage(suite.summary.case_pass_rate)}")


if __name__ == "__main__":
    main()
