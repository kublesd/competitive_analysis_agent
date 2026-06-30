"""连接 Streamlit 输入与 LangGraph 工作流的可测试应用服务。"""

from __future__ import annotations

from collections.abc import Callable
import logging
import re
from time import perf_counter
from traceback import extract_tb
from uuid import uuid4

from pydantic import Field, model_validator

from competitive_analysis_agent.agent_hooks import (
    AgentHook,
    AgentRunContext,
    HookManager,
)
from competitive_analysis_agent.application_workflow import (
    ApplicationSearchConfigurationError,
    create_application_workflow_components,
)
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.observability import (
    JsonlLoggingHook,
    StagePayloadSummarizer,
)
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
    *,
    entrypoint: str = "service",
    hooks: list[AgentHook] | None = None,
) -> AnalysisRunResult:
    """运行 UI 提交的真实搜索分析，并通过回调报告节点进度。"""

    analysis_id = uuid4().hex[:12]
    started_at = perf_counter()
    hook_manager = build_hook_manager(
        analysis_id=analysis_id,
        entrypoint=entrypoint,
        started_at=started_at,
        analysis_request=analysis_request,
        hooks=hooks,
    )
    event_summarizer = StagePayloadSummarizer()
    LOGGER.info(
        "analysis_started analysis_id=%s competitor_count=%d "
        "dimension_count=%d official_domain_product_count=%d entrypoint=%s",
        analysis_id,
        len(analysis_request.competitors),
        len(analysis_request.dimensions),
        len(analysis_request.official_domains_by_product),
        entrypoint,
    )
    hook_manager.on_run_started()

    try:
        result = _run_analysis_workflow(
            analysis_request=analysis_request,
            progress_callback=progress_callback,
            components=components,
            hook_manager=hook_manager,
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
        error_summary = event_summarizer.build_error_summary(
            error=error,
            failed_stage=failed_stage,
        )
        hook_manager.on_run_failed(error_summary)
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
    hook_manager.on_run_completed(
        build_run_result_summary(result)
    )
    return result


def build_hook_manager(
    analysis_id: str,
    entrypoint: str,
    started_at: float,
    analysis_request: AnalysisRequest,
    hooks: list[AgentHook] | None,
) -> HookManager:
    """创建一次运行的 HookManager，并默认启用本地 JSONL 日志 Hook。"""

    run_context = AgentRunContext(
        analysis_id=analysis_id,
        entrypoint=entrypoint,
        started_at=started_at,
        configuration_summary=build_run_configuration_summary(
            analysis_request
        ),
    )
    active_hooks: list[AgentHook] = [JsonlLoggingHook()]
    active_hooks.extend(hooks or [])
    return HookManager(run_context=run_context, hooks=active_hooks)


def build_run_configuration_summary(
    analysis_request: AnalysisRequest,
) -> dict[str, object]:
    """生成运行开始事件使用的脱敏配置摘要。"""

    return {
        "target_product": analysis_request.target_product,
        "competitor_count": len(analysis_request.competitors),
        "dimensions": list(analysis_request.dimensions),
        "dimension_count": len(analysis_request.dimensions),
        "official_domain_product_count": len(
            analysis_request.official_domains_by_product
        ),
    }


def build_run_result_summary(
    result: AnalysisRunResult,
) -> dict[str, object]:
    """生成运行完成事件使用的结果摘要。"""

    return {
        "stage_count": len(result.stage_history),
        "stage_history": list(result.stage_history),
        "evidence_count": len(result.evidence),
        "research_error_count": len(result.research_errors),
        "verification_passed": result.verification_result.passed,
    }


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
    hook_manager: HookManager | None,
) -> AnalysisRunResult:
    """执行原有工作流，并通过 HookManager 记录阶段生命周期。"""

    current_components = components
    if current_components is None:
        settings = load_live_settings()
        current_components = create_application_workflow_components(settings)

    graph = create_workflow_graph(
        current_components,
        hook_manager=hook_manager,
    )
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
                if progress_callback is not None:
                    progress_callback(stage_name)
            reported_stage_count = len(stage_history)
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
