"""使用 LangGraph 编排竞品分析节点、共享状态和有限验证回路。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from competitive_analysis_agent.agent_hooks import HookManager
from competitive_analysis_agent.model_io import model_io_context
from competitive_analysis_agent.analyst import (
    Analyst,
    AnalystInput,
    CompetitiveAnalysis,
)
from competitive_analysis_agent.extractor import (
    Extractor,
    ExtractorInput,
    build_pricing_duplicate_key,
    classify_pricing_source_scope,
    normalize_scope_text,
    remove_plan_level_positioning,
)
from competitive_analysis_agent.planner import Planner, PlannerInput
from competitive_analysis_agent.reporter import Reporter, ReporterInput
from competitive_analysis_agent.researcher import (
    ResearchError,
    Researcher,
    ResearcherInput,
)
from competitive_analysis_agent.schemas import (
    Evidence,
    ProductProfile,
    PricingPlan,
    ResearchTask,
)
from competitive_analysis_agent.verifier import (
    VerificationResult,
    Verifier,
    VerifierInput,
)
from competitive_analysis_agent.observability import StagePayloadSummarizer


MAX_ANALYSIS_RETRIES = 1
RouteName = Literal["retry_analyst", "reporter"]
LOGGER = logging.getLogger(__name__)


class WorkflowGraphState(TypedDict):
    """保存一次 LangGraph 竞品分析运行的全部共享状态。"""

    target_product: str
    competitors: list[str]
    dimensions: list[str]
    official_domains_by_product: dict[str, list[str]]
    max_results_per_task: int
    research_tasks: list[ResearchTask]
    evidence: list[Evidence]
    research_errors: list[ResearchError]
    product_profiles: list[ProductProfile]
    analysis_result: CompetitiveAnalysis | None
    verification_result: VerificationResult | None
    final_report: str | None
    retry_count: int
    retry_pending: bool
    stage_history: list[str]


WorkflowNodeRunner = Callable[[WorkflowGraphState], dict[str, object]]


@dataclass(frozen=True, slots=True)
class WorkflowComponents:
    """集中保存图节点依赖的六个可独立测试组件。"""

    planner: Planner
    researcher: Researcher
    extractor: Extractor
    analyst: Analyst
    verifier: Verifier
    reporter: Reporter


def create_initial_state(
    planner_input: PlannerInput,
    official_domains_by_product: dict[str, list[str]] | None = None,
    max_results_per_task: int = 3,
) -> WorkflowGraphState:
    """根据用户输入创建字段完整、尚未执行任何节点的初始 State。"""

    return WorkflowGraphState(
        target_product=planner_input.target_product,
        competitors=list(planner_input.competitors),
        dimensions=list(planner_input.dimensions),
        official_domains_by_product=official_domains_by_product or {},
        max_results_per_task=max_results_per_task,
        research_tasks=[],
        evidence=[],
        research_errors=[],
        product_profiles=[],
        analysis_result=None,
        verification_result=None,
        final_report=None,
        retry_count=0,
        retry_pending=False,
        stage_history=[],
    )


def run_planner_node(
    state: WorkflowGraphState,
    planner: Planner,
) -> dict[str, object]:
    """调用 Planner，并只返回任务和阶段历史的部分更新。"""

    planner_input = PlannerInput(
        target_product=state["target_product"],
        competitors=state["competitors"],
        dimensions=state["dimensions"],
    )
    research_tasks = planner.plan(planner_input)
    return {
        "research_tasks": research_tasks,
        "stage_history": [*state["stage_history"], "planner"],
    }


def run_researcher_node(
    state: WorkflowGraphState,
    researcher: Researcher,
) -> dict[str, object]:
    """调用 Researcher，把任务转换成 Evidence 和局部错误。"""

    researcher_input = ResearcherInput(
        tasks=state["research_tasks"],
        official_domains_by_product=state[
            "official_domains_by_product"
        ],
        max_results_per_task=state["max_results_per_task"],
    )
    research_result = researcher.research(researcher_input)
    return {
        "evidence": research_result.evidence,
        "research_errors": research_result.errors,
        "stage_history": [*state["stage_history"], "researcher"],
    }


def run_extractor_node(
    state: WorkflowGraphState,
    extractor: Extractor,
) -> dict[str, object]:
    """调用 Extractor，把 Evidence 转换成产品画像。"""

    extractor_input = ExtractorInput(evidence=state["evidence"])
    extracted_profiles = extractor.extract(extractor_input)
    product_profiles, validation_errors = validate_product_profiles_for_analysis(
        product_profiles=extracted_profiles,
        evidence=state["evidence"],
    )
    return {
        "product_profiles": product_profiles,
        "research_errors": [*state["research_errors"], *validation_errors],
        "stage_history": [*state["stage_history"], "extractor"],
    }


def validate_product_profiles_for_analysis(
    product_profiles: list[ProductProfile],
    evidence: list[Evidence],
) -> tuple[list[ProductProfile], list[ResearchError]]:
    """在 Analyst 前清理画像污染，并把删除原因记录为数据限制。"""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    evidence_by_product = group_evidence_by_product(evidence)
    validated_profiles: list[ProductProfile] = []
    validation_errors: list[ResearchError] = []

    for profile in product_profiles:
        product_evidence = evidence_by_product.get(profile.product_name, [])

        # 输入：Extractor 生成的画像；转换：删除套餐级定位；输出：更保守的画像。
        positioned_profile = remove_plan_level_positioning(
            profile=profile,
            evidence=product_evidence,
        )
        if positioned_profile.positioning != profile.positioning:
            validation_errors.append(
                build_profile_validation_error(
                    product_name=profile.product_name,
                    topic="positioning",
                    message=(
                        f"{profile.product_name} positioning was removed "
                        "because it looked like plan-level, pricing, or "
                        "subscription wording."
                    ),
                )
            )

        scoped_profile, scope_errors = remove_out_of_scope_pricing(
            profile=positioned_profile,
            evidence_by_id=evidence_by_id,
        )
        conflict_checked_profile, conflict_errors = (
            remove_conflicting_profile_pricing(scoped_profile)
        )
        validated_profiles.append(conflict_checked_profile)
        validation_errors.extend(scope_errors)
        validation_errors.extend(conflict_errors)

    return validated_profiles, validation_errors


def group_evidence_by_product(
    evidence: list[Evidence],
) -> dict[str, list[Evidence]]:
    """按产品名分组 Evidence，供画像校验读取同产品来源文本。"""

    evidence_groups: dict[str, list[Evidence]] = {}
    for item in evidence:
        if item.product_name not in evidence_groups:
            evidence_groups[item.product_name] = []
        evidence_groups[item.product_name].append(item)

    return evidence_groups


def remove_out_of_scope_pricing(
    profile: ProductProfile,
    evidence_by_id: dict[str, Evidence],
) -> tuple[ProductProfile, list[ResearchError]]:
    """删除默认 API pricing 范围外的价格项，并生成可展示的数据限制。"""

    kept_pricing: list[PricingPlan] = []
    validation_errors: list[ResearchError] = []

    for pricing_plan in profile.pricing:
        scope_classification = classify_pricing_source_scope(
            product_name=profile.product_name,
            pricing_plan=pricing_plan,
            evidence_by_id=evidence_by_id,
        )
        if scope_classification in {"non_api_pricing", "ambiguous"}:
            validation_errors.append(
                build_profile_validation_error(
                    product_name=profile.product_name,
                    topic="pricing",
                    message=(
                        f"{profile.product_name} pricing plan "
                        f"{pricing_plan.plan_name!r} was removed because "
                        f"it was classified as {scope_classification} for "
                        "the requested default API pricing scope."
                    ),
                )
            )
            continue

        kept_pricing.append(pricing_plan)

    if len(kept_pricing) == len(profile.pricing):
        return profile, validation_errors

    return profile.model_copy(update={"pricing": kept_pricing}), validation_errors


def remove_conflicting_profile_pricing(
    profile: ProductProfile,
) -> tuple[ProductProfile, list[ResearchError]]:
    """删除同名套餐同一计费周期下价格互相冲突的价格项。"""

    pricing_groups: dict[tuple[str, str], list[PricingPlan]] = {}
    for pricing_plan in profile.pricing:
        duplicate_key = build_pricing_duplicate_key(pricing_plan)
        if duplicate_key not in pricing_groups:
            pricing_groups[duplicate_key] = []
        pricing_groups[duplicate_key].append(pricing_plan)

    conflicting_keys: set[tuple[str, str]] = set()
    for duplicate_key, pricing_plans in pricing_groups.items():
        normalized_prices = {
            normalize_scope_text(pricing_plan.price or "")
            for pricing_plan in pricing_plans
        }
        if len(normalized_prices) > 1:
            conflicting_keys.add(duplicate_key)

    if not conflicting_keys:
        return profile, []

    kept_pricing: list[PricingPlan] = []
    validation_errors: list[ResearchError] = []
    for pricing_plan in profile.pricing:
        duplicate_key = build_pricing_duplicate_key(pricing_plan)
        if duplicate_key not in conflicting_keys:
            kept_pricing.append(pricing_plan)
            continue

        validation_errors.append(
            build_profile_validation_error(
                product_name=profile.product_name,
                topic="pricing",
                message=(
                    f"{profile.product_name} pricing plan "
                    f"{pricing_plan.plan_name!r} was removed because "
                    "the profile contained conflicting prices for the same "
                    "plan and billing cycle."
                ),
            )
        )

    return profile.model_copy(update={"pricing": kept_pricing}), validation_errors


def build_profile_validation_error(
    product_name: str,
    topic: str,
    message: str,
) -> ResearchError:
    """把画像入场校验问题转换成 Reporter 已能展示的结构化错误。"""

    return ResearchError(
        product_name=product_name,
        topic=topic,
        query=f"{product_name} profile validation",
        code="profile_validation",
        message=message,
    )


def run_analyst_node(
    state: WorkflowGraphState,
    analyst: Analyst,
) -> dict[str, object]:
    """调用 Analyst；发生图级重试时注入上一轮验证反馈。"""

    revision_feedback = build_revision_feedback(
        state["verification_result"]
    )
    analyst_input = AnalystInput(
        profiles=state["product_profiles"],
        revision_feedback=revision_feedback,
    )
    analysis_result = analyst.analyze(analyst_input)
    return {
        "analysis_result": analysis_result,
        "retry_pending": False,
        "stage_history": [*state["stage_history"], "analyst"],
    }


def run_verifier_node(
    state: WorkflowGraphState,
    verifier: Verifier,
) -> dict[str, object]:
    """调用 Verifier，并计算是否允许一次返回 Analyst 的条件路由。"""

    analysis_result = state["analysis_result"]
    if analysis_result is None:
        raise ValueError("Verifier node requires analysis_result.")

    verifier_input = VerifierInput(
        analysis=analysis_result,
        evidence=state["evidence"],
    )
    verification_result = verifier.verify(verifier_input)

    retry_pending = (
        verification_result.retry_recommended
        and state["retry_count"] < MAX_ANALYSIS_RETRIES
    )
    retry_count = state["retry_count"]
    if retry_pending:
        retry_count += 1

    return {
        "verification_result": verification_result,
        "retry_count": retry_count,
        "retry_pending": retry_pending,
        "stage_history": [*state["stage_history"], "verifier"],
    }


def route_after_verifier(state: WorkflowGraphState) -> RouteName:
    """根据 Verifier 结果和重试上限选择返回 Analyst 或生成报告。"""

    if state["retry_pending"]:
        return "retry_analyst"
    return "reporter"


def run_reporter_node(
    state: WorkflowGraphState,
    reporter: Reporter,
) -> dict[str, object]:
    """调用 Reporter，把最终结构化状态渲染成 Markdown。"""

    analysis_result = state["analysis_result"]
    verification_result = state["verification_result"]
    if analysis_result is None:
        raise ValueError("Reporter node requires analysis_result.")
    if verification_result is None:
        raise ValueError("Reporter node requires verification_result.")

    reporter_input = ReporterInput(
        analysis=analysis_result,
        product_profiles=state["product_profiles"],
        evidence=state["evidence"],
        verification_result=verification_result,
        research_errors=state["research_errors"],
    )
    final_report = reporter.render(reporter_input)
    return {
        "final_report": final_report,
        "stage_history": [*state["stage_history"], "reporter"],
    }


def build_revision_feedback(
    verification_result: VerificationResult | None,
) -> list[str]:
    """把结构化 issues 转成 Analyst 可读且仍可定位的修订提示。"""

    if verification_result is None or verification_result.passed:
        return []

    feedback: list[str] = []
    for issue in verification_result.issues:
        feedback_text = (
            f"{issue.claim_path} [{issue.issue_type}]: "
            f"{issue.message} "
            "Verifier suggested_action is diagnostic only; do not copy its "
            "wording into the next analysis."
        )
        guidance = build_revision_guidance(
            claim_path=issue.claim_path,
            issue_type=issue.issue_type,
        )
        feedback_text += f" Revision rule: {guidance}"
        feedback.append(feedback_text)

    return feedback


def build_revision_guidance(claim_path: str, issue_type: str) -> str:
    """根据 issue 位置生成不会带偏 Analyst 的保守修订规则。"""

    if claim_path.startswith("features["):
        return (
            "For features, use exact ProductProfile feature names as "
            "`Product mentions Feature.` facts, or remove the claim. Do not "
            "write `includes/offers/provides features like`."
        )

    if claim_path == "conclusion":
        return (
            "Replace the conclusion with a conservative `Based on the "
            "supplied profiles` summary assembled only from supported "
            "feature and pricing facts."
        )

    if claim_path.startswith("pricing["):
        return (
            "For pricing, map the claim back to ProductProfile.pricing; "
            "unknown prices must remain `without a public price`."
        )

    if issue_type == "unsupported_claim":
        return (
            "If ProductProfile does not directly support a narrower version, "
            "remove this claim instead of replacing it with another "
            "inference."
        )

    if issue_type == "conflicting_evidence":
        return (
            "Remove the conflicting wording or rewrite it as a narrower "
            "observation directly visible in ProductProfile."
        )

    return "Keep only statements directly supported by ProductProfile."


def create_workflow_graph(
    components: WorkflowComponents,
    hook_manager: HookManager | None = None,
) -> CompiledStateGraph:
    """创建并编译带一次受限验证回路的 LangGraph。"""

    graph_builder = StateGraph(WorkflowGraphState)
    summarizer = StagePayloadSummarizer() if hook_manager is not None else None

    # node 只做依赖注入和部分 State 更新，业务逻辑仍保留在独立组件中。
    graph_builder.add_node(
        "planner",
        build_observed_node(
            "planner",
            lambda state: run_planner_node(state, components.planner),
            hook_manager,
            summarizer,
        ),
    )
    graph_builder.add_node(
        "researcher",
        build_observed_node(
            "researcher",
            lambda state: run_researcher_node(state, components.researcher),
            hook_manager,
            summarizer,
        ),
    )
    graph_builder.add_node(
        "extractor",
        build_observed_node(
            "extractor",
            lambda state: run_extractor_node(state, components.extractor),
            hook_manager,
            summarizer,
        ),
    )
    graph_builder.add_node(
        "analyst",
        build_observed_node(
            "analyst",
            lambda state: run_analyst_node(state, components.analyst),
            hook_manager,
            summarizer,
        ),
    )
    graph_builder.add_node(
        "verifier",
        build_observed_node(
            "verifier",
            lambda state: run_verifier_node(state, components.verifier),
            hook_manager,
            summarizer,
        ),
    )
    graph_builder.add_node(
        "reporter",
        build_observed_node(
            "reporter",
            lambda state: run_reporter_node(state, components.reporter),
            hook_manager,
            summarizer,
        ),
    )

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    graph_builder.add_edge("researcher", "extractor")
    graph_builder.add_edge("extractor", "analyst")
    graph_builder.add_edge("analyst", "verifier")
    graph_builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "retry_analyst": "analyst",
            "reporter": "reporter",
        },
    )
    graph_builder.add_edge("reporter", END)

    return graph_builder.compile(name="competitive-analysis-workflow")


def build_observed_node(
    stage_name: str,
    node_runner: WorkflowNodeRunner,
    hook_manager: HookManager | None,
    summarizer: StagePayloadSummarizer | None,
) -> WorkflowNodeRunner:
    """为 LangGraph 节点增加 Hook 调用，不改变节点业务函数。"""

    if hook_manager is None or summarizer is None:
        return node_runner

    def observed_node(state: WorkflowGraphState) -> dict[str, object]:
        retry_count = int(state["retry_count"])
        stage_context = hook_manager.create_stage_context(
            stage_name=stage_name,
            retry_count=retry_count,
        )
        input_summary = build_safe_stage_summary(
            summarizer=summarizer,
            stage_name=stage_name,
            state=state,
            direction="input",
        )
        hook_manager.on_stage_started(stage_context, input_summary)

        try:
            with model_io_context(
                analysis_id=hook_manager.run_context.analysis_id,
                entrypoint=hook_manager.run_context.entrypoint,
                stage=stage_name,
                attempt_index=stage_context.attempt_index,
                retry_count=stage_context.retry_count,
            ):
                update = node_runner(state)
        except Exception as error:
            error_summary = build_safe_error_summary(
                summarizer=summarizer,
                error=error,
                failed_stage=stage_name,
            )
            hook_manager.on_stage_failed(stage_context, error_summary)
            raise

        output_state = dict(state)
        output_state.update(update)
        output_summary = build_safe_stage_summary(
            summarizer=summarizer,
            stage_name=stage_name,
            state=output_state,
            direction="output",
        )
        hook_manager.on_stage_completed(stage_context, output_summary)
        return update

    return observed_node


def build_safe_stage_summary(
    summarizer: StagePayloadSummarizer,
    stage_name: str,
    state: Mapping[str, object],
    direction: Literal["input", "output"],
) -> dict[str, object]:
    """构建阶段摘要；摘要失败时不影响主工作流。"""

    try:
        if direction == "input":
            return summarizer.build_stage_input_summary(stage_name, state)
        return summarizer.build_stage_output_summary(stage_name, state)
    except Exception as error:
        LOGGER.warning(
            "hook_summary_failed stage=%s direction=%s error_type=%s",
            stage_name,
            direction,
            type(error).__name__,
        )
        return {"summary_error_type": type(error).__name__}


def build_safe_error_summary(
    summarizer: StagePayloadSummarizer,
    error: Exception,
    failed_stage: str,
) -> dict[str, object]:
    """构建脱敏错误摘要；摘要失败时仍保留异常类型。"""

    try:
        return summarizer.build_error_summary(
            error=error,
            failed_stage=failed_stage,
        )
    except Exception as summary_error:
        LOGGER.warning(
            "hook_error_summary_failed failed_stage=%s error_type=%s",
            failed_stage,
            type(summary_error).__name__,
        )
        return {
            "error_type": type(error).__name__,
            "failed_stage": failed_stage,
            "summary_error_type": type(summary_error).__name__,
        }
