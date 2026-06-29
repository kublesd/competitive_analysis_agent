"""连接 Streamlit 输入与 LangGraph 工作流的可测试应用服务。"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
import re
from time import perf_counter
from traceback import extract_tb
from uuid import uuid4

from pydantic import Field, model_validator

from competitive_analysis_agent.application_workflow import (
    ApplicationSearchConfigurationError,
    create_application_workflow_components,
)
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.planner import PlannerInput
from competitive_analysis_agent.researcher import ResearchError
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    RequiredText,
)
from competitive_analysis_agent.verifier import VerificationResult
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    WorkflowGraphState,
    build_revision_feedback,
    create_initial_state,
    create_workflow_graph,
)


DEFAULT_TARGET_PRODUCT = "ChatGPT"
DEFAULT_COMPETITORS = ["Claude, Gemini"]
AVAILABLE_DIMENSIONS = [
    "features",
    "pricing",
    "positioning",
    "target_users",
    "limitations",
]
DEFAULT_DIMENSIONS = [
    "features",
    "pricing",
    "positioning",
    "target_users",
]
DEFAULT_OFFICIAL_DOMAINS_TEXT = "\n".join(
    [
        "ChatGPT=openai.com,chatgpt.com",
        "Claude=anthropic.com",
        "Gemini=one.google.com,workspace.google.com,ai.google.dev,gemini.google.com",
    ]
)

STAGE_LABELS = {
    "planner": "规划调研任务",
    "researcher": "收集并整理证据",
    "extractor": "提取产品画像",
    "analyst": "生成竞品比较",
    "verifier": "验证结论与引用",
    "reporter": "生成 Markdown 报告",
}
NEXT_STAGE_AFTER_SUCCESS = {
    "planner": "researcher",
    "researcher": "extractor",
    "extractor": "analyst",
    "analyst": "verifier",
    "verifier": "reporter",
}

ProgressCallback = Callable[[str], None]
LOGGER = logging.getLogger(__name__)
STAGE_IO_ITEM_LIMIT = 8
STAGE_IO_TEXT_PREVIEW_LIMIT = 500
WIDE_ANALYSIS_DIMENSION_THRESHOLD = 5
VERY_WIDE_ANALYSIS_DIMENSION_THRESHOLD = 7


class AnalysisRunError(RuntimeError):
    """表示图没有产生 UI 所需的完整终态。"""


class AnalysisRequest(ContractModel):
    """保存 UI 提交的目标产品、竞品和分析维度。"""

    target_product: RequiredText
    competitors: list[RequiredText] = Field(min_length=1)
    dimensions: list[RequiredText] = Field(min_length=1)
    official_domains_by_product: dict[str, list[RequiredText]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_unique_values(self) -> "AnalysisRequest":
        """拒绝重复产品和维度，避免生成含义相同的任务。"""

        products = [self.target_product, *self.competitors]
        if len(products) != len(set(products)):
            raise ValueError("Target product and competitors must be unique.")

        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("Dimensions must be unique.")

        known_products = set(products)
        unknown_domain_products = (
            set(self.official_domains_by_product) - known_products
        )
        if unknown_domain_products:
            unknown_text = ", ".join(sorted(unknown_domain_products))
            raise ValueError(
                "Official domains contain unknown products: "
                f"{unknown_text}"
            )

        return self


class AnalysisRunResult(ContractModel):
    """保存页面需要展示和下载的工作流终态。"""

    final_report: RequiredText
    stage_history: list[RequiredText] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    verification_result: VerificationResult
    research_errors: list[ResearchError] = Field(default_factory=list)


def parse_competitors(raw_value: str) -> list[str]:
    """按换行、中英文逗号拆分竞品输入，并删除空白项。"""

    raw_items = re.split(r"[\n,，]+", raw_value)
    competitors: list[str] = []
    for raw_item in raw_items:
        competitor = raw_item.strip()
        if not competitor:
            continue
        competitors.append(competitor)
    return competitors


def parse_custom_dimensions(raw_value: str) -> list[str]:
    """按常见分隔符拆分自定义分析维度，并删除空白项。"""

    raw_items = re.split(r"[\n,，;；]+", raw_value)
    dimensions: list[str] = []
    for raw_item in raw_items:
        dimension = raw_item.strip()
        if not dimension:
            continue
        dimensions.append(dimension)
    return dimensions


def build_analysis_dimensions(
    selected_dimensions: list[str],
    custom_dimensions_text: str,
) -> list[str]:
    """合并常用维度和自定义维度，按首次出现顺序去重。"""

    custom_dimensions = parse_custom_dimensions(custom_dimensions_text)
    merged_dimensions: list[str] = []
    seen_dimension_keys: set[str] = set()

    # 输入层先做温和去重，避免用户把已勾选维度又写进自定义框后触发校验错误。
    for raw_dimension in [*selected_dimensions, *custom_dimensions]:
        dimension = raw_dimension.strip()
        if not dimension:
            continue

        dimension_key = dimension.casefold()
        if dimension_key in seen_dimension_keys:
            continue

        seen_dimension_keys.add(dimension_key)
        merged_dimensions.append(dimension)

    return merged_dimensions


def create_analysis_request(
    target_product: str,
    competitors_text: str,
    dimensions: list[str],
    official_domains_text: str = "",
) -> AnalysisRequest:
    """把 Streamlit 原始控件值转换成经过校验的请求。"""

    competitors = parse_competitors(competitors_text)
    products = [target_product.strip(), *competitors]
    official_domains = parse_official_domains(
        official_domains_text,
        products,
    )
    return AnalysisRequest(
        target_product=target_product,
        competitors=competitors,
        dimensions=dimensions,
        official_domains_by_product=official_domains,
    )


def parse_official_domains(
    raw_value: str,
    products: list[str],
) -> dict[str, list[str]]:
    """解析“产品=域名”配置，并拒绝未知产品或格式错误。"""

    known_products = set(products)
    official_domains: dict[str, list[str]] = {}
    for line_number, raw_line in enumerate(raw_value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(
                f"Official domain line {line_number} must use product=domain."
            )

        product_name, raw_domains = line.split("=", maxsplit=1)
        product_name = product_name.strip()
        if product_name not in known_products:
            raise ValueError(
                f"Official domain product is not in the request: {product_name}"
            )

        domains = parse_competitors(raw_domains)
        if not domains:
            raise ValueError(
                f"Official domain line {line_number} has no domain."
            )
        official_domains[product_name] = domains

    return official_domains


def run_analysis(
    analysis_request: AnalysisRequest,
    progress_callback: ProgressCallback | None = None,
    components: WorkflowComponents | None = None,
) -> AnalysisRunResult:
    """运行 UI 提交的真实搜索分析，并通过回调报告节点进度。"""

    analysis_id = uuid4().hex[:12]
    started_at = perf_counter()
    LOGGER.info(
        "analysis_started analysis_id=%s competitor_count=%d "
        "dimension_count=%d official_domain_product_count=%d",
        analysis_id,
        len(analysis_request.competitors),
        len(analysis_request.dimensions),
        len(analysis_request.official_domains_by_product),
    )

    try:
        result = _run_analysis_workflow(
            analysis_request=analysis_request,
            progress_callback=progress_callback,
            components=components,
            analysis_id=analysis_id,
            started_at=started_at,
        )
    except Exception as error:
        elapsed_seconds = perf_counter() - started_at
        failure_function, failure_line = _describe_failure_location(error)
        attach_runtime_failure_context(
            error=error,
            analysis_id=analysis_id,
            failure_function=failure_function,
            failure_line=failure_line,
            elapsed_seconds=elapsed_seconds,
        )
        failed_stage = getattr(error, "workflow_failed_stage", "unknown")
        LOGGER.error(
            "analysis_failed analysis_id=%s error_type=%s "
            "failed_stage=%s failure_function=%s failure_line=%s "
            "elapsed_seconds=%.3f",
            analysis_id,
            type(error).__name__,
            failed_stage,
            failure_function,
            failure_line,
            elapsed_seconds,
        )
        raise

    elapsed_seconds = perf_counter() - started_at
    LOGGER.info(
        "analysis_completed analysis_id=%s elapsed_seconds=%.3f "
        "stage_count=%d evidence_count=%d research_error_count=%d "
        "verification_passed=%s",
        analysis_id,
        elapsed_seconds,
        len(result.stage_history),
        len(result.evidence),
        len(result.research_errors),
        result.verification_result.passed,
    )
    return result


def _describe_failure_location(error: Exception) -> tuple[str, str]:
    """返回异常最后发生的函数和行号，不把可能含敏感信息的异常原文写入日志。"""

    traceback_frames = extract_tb(error.__traceback__)
    if not traceback_frames:
        return "unknown", "unknown"

    final_frame = traceback_frames[-1]
    return final_frame.name, str(final_frame.lineno)


def attach_runtime_failure_context(
    error: Exception,
    analysis_id: str,
    failure_function: str,
    failure_line: str,
    elapsed_seconds: float,
) -> None:
    """给异常补充一次运行的脱敏定位信息，供 UI 和日志共用。"""

    setattr(error, "analysis_id", analysis_id)
    setattr(error, "failure_function", failure_function)
    setattr(error, "failure_line", failure_line)
    setattr(error, "elapsed_seconds", f"{elapsed_seconds:.3f}")


def attach_workflow_failure_context(
    error: Exception,
    final_state: WorkflowGraphState | None,
) -> None:
    """根据最后一个成功 State 推断失败阶段，不读取模型原始响应。"""

    if final_state is None:
        setattr(error, "workflow_stage_history", [])
        setattr(error, "workflow_failed_stage", "planner")
        return

    stage_history = list(final_state["stage_history"])
    setattr(error, "workflow_stage_history", stage_history)
    setattr(
        error,
        "workflow_failed_stage",
        infer_failed_stage_from_state(final_state),
    )


def infer_failed_stage_from_state(state: WorkflowGraphState) -> str:
    """根据最后完成的 State 推断下一次失败发生在哪个工作流阶段。"""

    stage_history = list(state["stage_history"])
    if state["retry_pending"]:
        return "analyst"

    if not stage_history:
        return "planner"

    last_completed_stage = stage_history[-1]
    return NEXT_STAGE_AFTER_SUCCESS.get(last_completed_stage, "unknown")


def _run_analysis_workflow(
    analysis_request: AnalysisRequest,
    progress_callback: ProgressCallback | None,
    components: WorkflowComponents | None,
    analysis_id: str,
    started_at: float,
) -> AnalysisRunResult:
    """执行原有工作流，并在每个 State 快照后记录阶段元数据。"""

    current_components = components
    if current_components is None:
        settings = load_live_settings()
        current_components = create_application_workflow_components(settings)

    graph = create_workflow_graph(current_components)
    planner_input = PlannerInput(
        target_product=analysis_request.target_product,
        competitors=analysis_request.competitors,
        dimensions=analysis_request.dimensions,
    )
    initial_state = create_initial_state(
        planner_input=planner_input,
        official_domains_by_product=(
            analysis_request.official_domains_by_product
        ),
        max_results_per_task=choose_max_results_per_task(
            analysis_request.dimensions
        ),
    )

    # values 模式在每个节点后返回完整 State，页面可显示进度并取得最终结果。
    final_state: WorkflowGraphState | None = None
    previous_state: WorkflowGraphState | None = initial_state
    reported_stage_count = 0
    try:
        for state_snapshot in graph.stream(
            initial_state,
            stream_mode="values",
        ):
            final_state = state_snapshot
            stage_history = state_snapshot["stage_history"]
            new_stages = stage_history[reported_stage_count:]
            for stage_name in new_stages:
                elapsed_seconds = perf_counter() - started_at
                log_stage_io(
                    analysis_id=analysis_id,
                    stage_name=stage_name,
                    input_state=previous_state,
                    output_state=state_snapshot,
                )
                LOGGER.info(
                    "analysis_stage_completed analysis_id=%s stage=%s "
                    "elapsed_seconds=%.3f task_count=%d evidence_count=%d "
                    "profile_count=%d research_error_count=%d retry_count=%d",
                    analysis_id,
                    stage_name,
                    elapsed_seconds,
                    len(state_snapshot["research_tasks"]),
                    len(state_snapshot["evidence"]),
                    len(state_snapshot["product_profiles"]),
                    len(state_snapshot["research_errors"]),
                    state_snapshot["retry_count"],
                )
                if progress_callback is not None:
                    progress_callback(stage_name)
            reported_stage_count = len(stage_history)
            previous_state = state_snapshot
    except Exception as error:
        attach_workflow_failure_context(error, final_state)
        raise

    if final_state is None:
        raise AnalysisRunError("Workflow completed without a final state.")

    final_report = final_state["final_report"]
    verification_result = final_state["verification_result"]
    if final_report is None:
        raise AnalysisRunError("Workflow did not produce a final report.")
    if verification_result is None:
        raise AnalysisRunError(
            "Workflow did not produce a verification result."
        )

    return AnalysisRunResult(
        final_report=final_report,
        stage_history=final_state["stage_history"],
        evidence=final_state["evidence"],
        verification_result=verification_result,
        research_errors=final_state["research_errors"],
    )


def choose_max_results_per_task(dimensions: list[str]) -> int:
    """按维度数量控制每个调研任务的搜索结果数，避免 Evidence 过量。"""

    dimension_count = len(dimensions)
    if dimension_count >= VERY_WIDE_ANALYSIS_DIMENSION_THRESHOLD:
        return 1
    if dimension_count >= WIDE_ANALYSIS_DIMENSION_THRESHOLD:
        return 2
    return 3


def log_stage_io(
    analysis_id: str,
    stage_name: str,
    input_state: WorkflowGraphState | None,
    output_state: WorkflowGraphState,
) -> None:
    """记录某个阶段的输入和输出摘要，避免日志里出现完整网页正文。"""

    if input_state is not None:
        log_stage_payload(
            analysis_id=analysis_id,
            stage_name=stage_name,
            direction="input",
            payload=build_stage_input_payload(stage_name, input_state),
        )
    log_stage_payload(
        analysis_id=analysis_id,
        stage_name=stage_name,
        direction="output",
        payload=build_stage_output_payload(stage_name, output_state),
    )


def log_stage_payload(
    analysis_id: str,
    stage_name: str,
    direction: str,
    payload: dict[str, object],
) -> None:
    """把阶段 I/O payload 写成单行 JSON，便于按 analysis_id 搜索。"""

    try:
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception as error:
        LOGGER.warning(
            "analysis_stage_io_skipped analysis_id=%s stage=%s "
            "direction=%s error_type=%s",
            analysis_id,
            stage_name,
            direction,
            type(error).__name__,
        )
        return

    LOGGER.info(
        "analysis_stage_io analysis_id=%s stage=%s direction=%s "
        "payload_json=%s",
        analysis_id,
        stage_name,
        direction,
        payload_json,
    )


def build_stage_input_payload(
    stage_name: str,
    state: WorkflowGraphState,
) -> dict[str, object]:
    """按阶段提取即将交给节点的输入契约。"""

    if stage_name == "planner":
        return {
            "target_product": state["target_product"],
            "competitors": list(state["competitors"]),
            "dimensions": list(state["dimensions"]),
        }

    if stage_name == "researcher":
        return {
            "research_tasks": summarize_research_tasks(
                state["research_tasks"]
            ),
            "official_domains_by_product": (
                state["official_domains_by_product"]
            ),
            "max_results_per_task": state["max_results_per_task"],
        }

    if stage_name == "extractor":
        return {
            "evidence": summarize_evidence_items(state["evidence"]),
        }

    if stage_name == "analyst":
        return {
            "product_profiles": summarize_product_profiles(
                state["product_profiles"]
            ),
            "revision_feedback": summarize_text_items(
                build_revision_feedback(state["verification_result"])
            ),
            "retry_count": state["retry_count"],
        }

    if stage_name == "verifier":
        return {
            "analysis_result": summarize_analysis(
                state["analysis_result"]
            ),
            "evidence": summarize_evidence_items(state["evidence"]),
        }

    if stage_name == "reporter":
        return {
            "analysis_result": summarize_analysis(
                state["analysis_result"]
            ),
            "product_profiles": summarize_product_profiles(
                state["product_profiles"]
            ),
            "verification_result": summarize_verification_result(
                state["verification_result"]
            ),
            "research_errors": summarize_research_errors(
                state["research_errors"]
            ),
            "evidence": summarize_evidence_items(state["evidence"]),
        }

    return {"stage_history": list(state["stage_history"])}


def build_stage_output_payload(
    stage_name: str,
    state: WorkflowGraphState,
) -> dict[str, object]:
    """按阶段提取节点完成后写入 State 的输出契约。"""

    if stage_name == "planner":
        return {
            "research_tasks": summarize_research_tasks(
                state["research_tasks"]
            ),
        }

    if stage_name == "researcher":
        return {
            "evidence": summarize_evidence_items(state["evidence"]),
            "research_errors": summarize_research_errors(
                state["research_errors"]
            ),
        }

    if stage_name == "extractor":
        return {
            "product_profiles": summarize_product_profiles(
                state["product_profiles"]
            ),
        }

    if stage_name == "analyst":
        return {
            "analysis_result": summarize_analysis(
                state["analysis_result"]
            ),
            "retry_pending": state["retry_pending"],
        }

    if stage_name == "verifier":
        return {
            "verification_result": summarize_verification_result(
                state["verification_result"]
            ),
            "retry_count": state["retry_count"],
            "retry_pending": state["retry_pending"],
        }

    if stage_name == "reporter":
        final_report = state["final_report"] or ""
        return {
            "final_report_chars": len(final_report),
            "final_report_preview": truncate_text(final_report),
        }

    return {"stage_history": list(state["stage_history"])}


def summarize_research_tasks(tasks: list[object]) -> dict[str, object]:
    """摘要 ResearchTask 列表，保留产品、主题和查询文本。"""

    items: list[dict[str, object]] = []
    for task in tasks[:STAGE_IO_ITEM_LIMIT]:
        items.append(
            {
                "product_name": getattr(task, "product_name", ""),
                "topic": getattr(task, "topic", ""),
                "query": getattr(task, "query", ""),
            }
        )

    return build_limited_collection_summary(items, len(tasks))


def summarize_evidence_items(evidence_items: list[object]) -> dict[str, object]:
    """摘要 Evidence 列表，正文只保留短预览和长度。"""

    items: list[dict[str, object]] = []
    for evidence in evidence_items[:STAGE_IO_ITEM_LIMIT]:
        raw_content = getattr(evidence, "raw_content", None)
        items.append(
            {
                "evidence_id": getattr(evidence, "evidence_id", ""),
                "product_name": getattr(evidence, "product_name", ""),
                "topic": getattr(evidence, "topic", ""),
                "source_type": getattr(evidence, "source_type", ""),
                "title": truncate_text(getattr(evidence, "title", "")),
                "url": str(getattr(evidence, "url", "")),
                "snippet_preview": truncate_text(
                    getattr(evidence, "snippet", "")
                ),
                "raw_content_chars": len(raw_content or ""),
                "raw_content_preview": truncate_text(raw_content or ""),
            }
        )

    return build_limited_collection_summary(items, len(evidence_items))


def summarize_product_profiles(profiles: list[object]) -> dict[str, object]:
    """摘要产品画像，便于观察 Extractor 输出是否缺字段。"""

    items: list[dict[str, object]] = []
    for profile in profiles[:STAGE_IO_ITEM_LIMIT]:
        items.append(
            {
                "product_name": getattr(profile, "product_name", ""),
                "positioning": truncate_text(
                    getattr(profile, "positioning", "") or ""
                ),
                "target_users": summarize_text_items(
                    getattr(profile, "target_users", [])
                ),
                "features": summarize_feature_items(
                    getattr(profile, "features", [])
                ),
                "pricing": summarize_pricing_plans(
                    getattr(profile, "pricing", [])
                ),
                "limitations": summarize_text_items(
                    getattr(profile, "limitations", [])
                ),
            }
        )

    return build_limited_collection_summary(items, len(profiles))


def summarize_feature_items(features: list[object]) -> dict[str, object]:
    """摘要功能项，保留名称、短描述和引用。"""

    items: list[dict[str, object]] = []
    for feature in features[:STAGE_IO_ITEM_LIMIT]:
        items.append(
            {
                "name": getattr(feature, "name", ""),
                "description": truncate_text(
                    getattr(feature, "description", "")
                ),
                "evidence_ids": list(getattr(feature, "evidence_ids", [])),
            }
        )

    return build_limited_collection_summary(items, len(features))


def summarize_pricing_plans(pricing_plans: list[object]) -> dict[str, object]:
    """摘要价格项，保留方案名、价格、周期、限制和引用。"""

    items: list[dict[str, object]] = []
    for pricing_plan in pricing_plans[:STAGE_IO_ITEM_LIMIT]:
        items.append(
            {
                "plan_name": getattr(pricing_plan, "plan_name", ""),
                "price": getattr(pricing_plan, "price", None),
                "billing_cycle": getattr(pricing_plan, "billing_cycle", None),
                "main_limits": summarize_text_items(
                    getattr(pricing_plan, "main_limits", [])
                ),
                "evidence_ids": list(
                    getattr(pricing_plan, "evidence_ids", [])
                ),
            }
        )

    return build_limited_collection_summary(items, len(pricing_plans))


def summarize_analysis(analysis: object | None) -> dict[str, object] | None:
    """摘要 Analyst 输出，保留各章节 claim 和 Evidence ID。"""

    if analysis is None:
        return None

    return {
        "products": list(getattr(analysis, "products", [])),
        "positioning": summarize_claims(getattr(analysis, "positioning", [])),
        "features": summarize_claims(getattr(analysis, "features", [])),
        "pricing": summarize_claims(getattr(analysis, "pricing", [])),
        "opportunities": summarize_claims(
            getattr(analysis, "opportunities", [])
        ),
        "conclusion": summarize_claim(getattr(analysis, "conclusion", None)),
    }


def summarize_claims(claims: list[object]) -> dict[str, object]:
    """摘要 claim 列表。"""

    items = [
        summarize_claim(claim)
        for claim in claims[:STAGE_IO_ITEM_LIMIT]
    ]
    return build_limited_collection_summary(items, len(claims))


def summarize_claim(claim: object | None) -> dict[str, object] | None:
    """摘要单条 claim。"""

    if claim is None:
        return None

    return {
        "claim": truncate_text(getattr(claim, "claim", "")),
        "claim_type": getattr(claim, "claim_type", ""),
        "product_names": list(getattr(claim, "product_names", [])),
        "evidence_ids": list(getattr(claim, "evidence_ids", [])),
    }


def summarize_verification_result(
    verification_result: object | None,
) -> dict[str, object] | None:
    """摘要 Verifier 输出，保留问题路径和建议动作。"""

    if verification_result is None:
        return None

    issues = getattr(verification_result, "issues", [])
    issue_items: list[dict[str, object]] = []
    for issue in issues[:STAGE_IO_ITEM_LIMIT]:
        issue_items.append(
            {
                "claim_path": getattr(issue, "claim_path", ""),
                "issue_type": getattr(issue, "issue_type", ""),
                "message": truncate_text(getattr(issue, "message", "")),
                "evidence_ids": list(getattr(issue, "evidence_ids", [])),
                "suggested_action": truncate_text(
                    getattr(issue, "suggested_action", "")
                ),
            }
        )

    return {
        "passed": getattr(verification_result, "passed", False),
        "retry_recommended": getattr(
            verification_result,
            "retry_recommended",
            False,
        ),
        "issues": build_limited_collection_summary(
            issue_items,
            len(issues),
        ),
    }


def summarize_research_errors(errors: list[object]) -> dict[str, object]:
    """摘要 Researcher 的局部失败。"""

    items: list[dict[str, object]] = []
    for error in errors[:STAGE_IO_ITEM_LIMIT]:
        items.append(
            {
                "product_name": getattr(error, "product_name", ""),
                "topic": getattr(error, "topic", ""),
                "query": truncate_text(getattr(error, "query", "")),
                "message": truncate_text(getattr(error, "message", "")),
            }
        )

    return build_limited_collection_summary(items, len(errors))


def summarize_text_items(items: list[str]) -> dict[str, object]:
    """摘要普通字符串列表。"""

    preview_items = [
        truncate_text(item)
        for item in items[:STAGE_IO_ITEM_LIMIT]
    ]
    return build_limited_collection_summary(preview_items, len(items))


def build_limited_collection_summary(
    items: list[object],
    total_count: int,
) -> dict[str, object]:
    """返回带总数和省略数量的列表摘要。"""

    omitted_count = max(total_count - len(items), 0)
    return {
        "count": total_count,
        "items": items,
        "omitted_count": omitted_count,
    }


def truncate_text(value: object) -> str:
    """把任意文本压缩成单行短预览，避免日志体积失控。"""

    text = "" if value is None else str(value)
    compact_text = re.sub(r"\s+", " ", text).strip()
    if len(compact_text) <= STAGE_IO_TEXT_PREVIEW_LIMIT:
        return compact_text

    return compact_text[:STAGE_IO_TEXT_PREVIEW_LIMIT] + "...[truncated]"


def build_stage_summary(stage_history: list[str]) -> str:
    """把内部节点名称转换成适合页面展示的中文执行轨迹。"""

    labels: list[str] = []
    for stage_name in stage_history:
        labels.append(STAGE_LABELS.get(stage_name, stage_name))
    return " → ".join(labels)


def describe_user_error(error: Exception) -> str:
    """把异常转换成不包含 traceback 或敏感配置的用户提示。"""

    error_name = type(error).__name__
    configuration_error_names = {
        "LiveModelConfigurationError",
        "LivePlannerConfigurationError",
        "LiveExtractorConfigurationError",
        "LiveAnalystConfigurationError",
        "LiveVerifierConfigurationError",
        "ApplicationSearchConfigurationError",
    }
    if error_name in configuration_error_names:
        if isinstance(error, ApplicationSearchConfigurationError):
            return (
                "搜索配置不完整，请在 .env.example 中设置 "
                "TAVILY_API_KEY 后重新启动。"
            )
        return "模型配置不完整，请检查项目环境配置后重试。"

    if isinstance(error, ValueError):
        return (
            "输入格式不正确，请确认产品、竞品、分析维度和官方域名格式。"
        )

    lines = [
        "分析未完成。模型服务或工作流暂时不可用，请稍后重试。",
        f"错误类别：{error_name}",
    ]
    analysis_id = getattr(error, "analysis_id", None)
    if analysis_id:
        lines.append(f"分析编号：{analysis_id}")

    failed_stage = getattr(error, "workflow_failed_stage", None)
    if failed_stage:
        failed_stage_label = STAGE_LABELS.get(failed_stage, failed_stage)
        lines.append(f"失败阶段：{failed_stage_label}（{failed_stage}）")

    stage_history = getattr(error, "workflow_stage_history", None)
    if stage_history:
        lines.append(f"已完成阶段：{build_stage_summary(stage_history)}")

    failure_function = getattr(error, "failure_function", None)
    failure_line = getattr(error, "failure_line", None)
    if failure_function and failure_line:
        lines.append(f"失败位置：{failure_function}:{failure_line}")

    public_detail = getattr(error, "public_detail", None)
    if public_detail:
        lines.append(f"定位信息：{truncate_user_error_detail(public_detail)}")

    return "\n".join(lines)


def truncate_user_error_detail(public_detail: str) -> str:
    """限制页面错误详情长度，避免模型异常文本占满页面。"""

    max_length = 500
    if len(public_detail) <= max_length:
        return public_detail
    return public_detail[:max_length] + "..."
