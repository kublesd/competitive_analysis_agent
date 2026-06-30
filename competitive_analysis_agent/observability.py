"""Agent Hooks 的本地结构化日志实现和脱敏摘要策略。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import logging
import re

from competitive_analysis_agent.agent_hooks import (
    AgentRunContext,
    AgentStageContext,
)
from competitive_analysis_agent.logging_config import AGENT_EVENT_LOGGER_NAME


LOGGER = logging.getLogger(AGENT_EVENT_LOGGER_NAME)
SUMMARY_ITEM_LIMIT = 8
SUMMARY_TEXT_LIMIT = 180


class JsonlLoggingHook:
    """把 Agent 生命周期事件写入本地 JSONL 日志。"""

    schema_version = 1

    def on_run_started(self, context: AgentRunContext) -> None:
        """记录一次运行开始事件。"""

        self._log_event(
            event_type="run_started",
            run_context=context,
            stage_name=None,
            status="started",
            duration_ms=0,
            summary=context.configuration_summary,
        )

    def on_stage_started(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        input_summary: dict[str, object],
    ) -> None:
        """记录阶段开始事件和脱敏输入摘要。"""

        self._log_stage_event(
            event_type="stage_started",
            run_context=run_context,
            stage_context=stage_context,
            status="started",
            duration_ms=0,
            payload_key="input",
            payload=input_summary,
        )

    def on_stage_completed(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        output_summary: dict[str, object],
    ) -> None:
        """记录阶段成功事件和脱敏输出摘要。"""

        self._log_stage_event(
            event_type="stage_completed",
            run_context=run_context,
            stage_context=stage_context,
            status="completed",
            duration_ms=stage_context.elapsed_ms(),
            payload_key="output",
            payload=output_summary,
        )

    def on_stage_failed(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        error_summary: dict[str, object],
    ) -> None:
        """记录阶段失败事件，不写异常原文。"""

        self._log_stage_event(
            event_type="stage_failed",
            run_context=run_context,
            stage_context=stage_context,
            status="failed",
            duration_ms=stage_context.elapsed_ms(),
            payload_key="error",
            payload=error_summary,
        )

    def on_run_completed(
        self,
        context: AgentRunContext,
        result_summary: dict[str, object],
    ) -> None:
        """记录一次运行成功结束事件。"""

        self._log_event(
            event_type="run_completed",
            run_context=context,
            stage_name=None,
            status="completed",
            duration_ms=context.elapsed_ms(),
            summary=result_summary,
        )

    def on_run_failed(
        self,
        context: AgentRunContext,
        error_summary: dict[str, object],
    ) -> None:
        """记录一次运行失败事件，不写异常原文。"""

        self._log_event(
            event_type="run_failed",
            run_context=context,
            stage_name=None,
            status="failed",
            duration_ms=context.elapsed_ms(),
            summary=error_summary,
        )

    def _log_stage_event(
        self,
        event_type: str,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        status: str,
        duration_ms: int,
        payload_key: str,
        payload: dict[str, object],
    ) -> None:
        """把阶段上下文和阶段 payload 合并到 summary 中。"""

        summary = {
            "attempt_index": stage_context.attempt_index,
            "retry_count": stage_context.retry_count,
            payload_key: payload,
        }
        self._log_event(
            event_type=event_type,
            run_context=run_context,
            stage_name=stage_context.stage_name,
            status=status,
            duration_ms=duration_ms,
            summary=summary,
        )

    def _log_event(
        self,
        event_type: str,
        run_context: AgentRunContext,
        stage_name: str | None,
        status: str,
        duration_ms: int,
        summary: dict[str, object],
    ) -> None:
        """写入一条字段稳定的 JSONL 事件。"""

        event = {
            "schema_version": self.schema_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "analysis_id": run_context.analysis_id,
            "entrypoint": run_context.entrypoint,
            "stage": stage_name,
            "status": status,
            "duration_ms": duration_ms,
            "summary": summary,
        }
        event_json = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        LOGGER.info(event_json)


class StagePayloadSummarizer:
    """把工作流 State 转换成适合日志保存的脱敏摘要。"""

    def build_stage_input_summary(
        self,
        stage_name: str,
        state: Mapping[str, object],
    ) -> dict[str, object]:
        """按阶段提取即将交给节点的输入摘要。"""

        if stage_name == "planner":
            return {
                "target_product": state.get("target_product", ""),
                "competitors": list(state.get("competitors", [])),
                "dimensions": list(state.get("dimensions", [])),
            }

        if stage_name == "researcher":
            return {
                "research_tasks": summarize_research_tasks(
                    state.get("research_tasks", [])
                ),
                "official_domain_product_count": len(
                    state.get("official_domains_by_product", {})
                ),
                "official_domain_counts_by_product": (
                    summarize_official_domain_counts(
                        state.get("official_domains_by_product", {})
                    )
                ),
                "max_results_per_task": state.get("max_results_per_task", 0),
            }

        if stage_name == "extractor":
            return {
                "evidence": summarize_evidence_items(
                    state.get("evidence", [])
                ),
            }

        if stage_name == "analyst":
            return {
                "product_profiles": summarize_product_profiles(
                    state.get("product_profiles", [])
                ),
                "previous_verification_result": summarize_verification_result(
                    state.get("verification_result")
                ),
                "retry_count": state.get("retry_count", 0),
            }

        if stage_name == "verifier":
            return {
                "analysis_result": summarize_analysis(
                    state.get("analysis_result")
                ),
                "evidence": summarize_evidence_items(
                    state.get("evidence", [])
                ),
            }

        if stage_name == "reporter":
            return {
                "analysis_result": summarize_analysis(
                    state.get("analysis_result")
                ),
                "product_profiles": summarize_product_profiles(
                    state.get("product_profiles", [])
                ),
                "verification_result": summarize_verification_result(
                    state.get("verification_result")
                ),
                "research_errors": summarize_research_errors(
                    state.get("research_errors", [])
                ),
                "evidence": summarize_evidence_items(
                    state.get("evidence", [])
                ),
            }

        return {"stage_history": list(state.get("stage_history", []))}

    def build_stage_output_summary(
        self,
        stage_name: str,
        state: Mapping[str, object],
    ) -> dict[str, object]:
        """按阶段提取节点完成后写入 State 的输出摘要。"""

        if stage_name == "planner":
            return {
                "research_tasks": summarize_research_tasks(
                    state.get("research_tasks", [])
                ),
            }

        if stage_name == "researcher":
            return {
                "evidence": summarize_evidence_items(
                    state.get("evidence", [])
                ),
                "research_errors": summarize_research_errors(
                    state.get("research_errors", [])
                ),
            }

        if stage_name == "extractor":
            return {
                "product_profiles": summarize_product_profiles(
                    state.get("product_profiles", [])
                ),
            }

        if stage_name == "analyst":
            return {
                "analysis_result": summarize_analysis(
                    state.get("analysis_result")
                ),
                "retry_pending": state.get("retry_pending", False),
            }

        if stage_name == "verifier":
            return {
                "verification_result": summarize_verification_result(
                    state.get("verification_result")
                ),
                "retry_count": state.get("retry_count", 0),
                "retry_pending": state.get("retry_pending", False),
            }

        if stage_name == "reporter":
            final_report = state.get("final_report") or ""
            return {
                "final_report_chars": len(str(final_report)),
                "final_report_section_count": count_markdown_sections(
                    final_report
                ),
            }

        return {"stage_history": list(state.get("stage_history", []))}

    def build_error_summary(
        self,
        error: Exception,
        failed_stage: str | None = None,
    ) -> dict[str, object]:
        """把异常转换为脱敏错误摘要，不记录异常消息文本。"""

        return {
            "error_type": type(error).__name__,
            "failed_stage": failed_stage or getattr(
                error,
                "workflow_failed_stage",
                "unknown",
            ),
            "failure_function": getattr(
                error,
                "failure_function",
                "unknown",
            ),
            "failure_line": getattr(error, "failure_line", "unknown"),
            "completed_stages": list(
                getattr(error, "workflow_stage_history", [])
            ),
            "public_detail_chars": len(
                str(getattr(error, "public_detail", "") or "")
            ),
        }


def summarize_research_tasks(tasks: object) -> dict[str, object]:
    """摘要 ResearchTask 列表，保留任务定位字段而不记录模型 Prompt。"""

    task_items = list(tasks or [])
    items: list[dict[str, object]] = []
    for task in task_items[:SUMMARY_ITEM_LIMIT]:
        items.append(
            {
                "product_name": getattr(task, "product_name", ""),
                "topic": getattr(task, "topic", ""),
                "query": truncate_summary_text(getattr(task, "query", "")),
            }
        )

    return build_limited_collection_summary(items, len(task_items))


def summarize_official_domain_counts(domains_by_product: object) -> dict[str, int]:
    """只记录每个产品的官方域名数量，不记录用户填写的域名原文。"""

    if not isinstance(domains_by_product, Mapping):
        return {}

    counts: dict[str, int] = {}
    for product_name, domains in domains_by_product.items():
        counts[str(product_name)] = len(list(domains or []))
    return counts


def summarize_evidence_items(evidence_items: object) -> dict[str, object]:
    """摘要 Evidence 列表，不记录 snippet、raw_content 或完整 URL。"""

    evidence_list = list(evidence_items or [])
    items: list[dict[str, object]] = []
    for evidence in evidence_list[:SUMMARY_ITEM_LIMIT]:
        snippet = getattr(evidence, "snippet", "") or ""
        raw_content = getattr(evidence, "raw_content", "") or ""
        items.append(
            {
                "evidence_id": getattr(evidence, "evidence_id", ""),
                "product_name": getattr(evidence, "product_name", ""),
                "topic": getattr(evidence, "topic", ""),
                "source_type": getattr(evidence, "source_type", ""),
                "title": truncate_summary_text(
                    getattr(evidence, "title", "")
                ),
                "has_url": bool(getattr(evidence, "url", None)),
                "evidence_preview_chars": len(str(snippet)),
                "source_content_chars": len(str(raw_content)),
            }
        )

    return build_limited_collection_summary(items, len(evidence_list))


def summarize_product_profiles(profiles: object) -> dict[str, object]:
    """摘要产品画像，只保留结构、数量和短名称。"""

    profile_items = list(profiles or [])
    items: list[dict[str, object]] = []
    for profile in profile_items[:SUMMARY_ITEM_LIMIT]:
        positioning = getattr(profile, "positioning", "") or ""
        items.append(
            {
                "product_name": getattr(profile, "product_name", ""),
                "positioning_chars": len(str(positioning)),
                "target_user_count": len(
                    list(getattr(profile, "target_users", []) or [])
                ),
                "features": summarize_feature_items(
                    getattr(profile, "features", [])
                ),
                "pricing": summarize_pricing_plans(
                    getattr(profile, "pricing", [])
                ),
                "limitation_count": len(
                    list(getattr(profile, "limitations", []) or [])
                ),
            }
        )

    return build_limited_collection_summary(items, len(profile_items))


def summarize_feature_items(features: object) -> dict[str, object]:
    """摘要功能项，不记录长描述。"""

    feature_items = list(features or [])
    items: list[dict[str, object]] = []
    for feature in feature_items[:SUMMARY_ITEM_LIMIT]:
        items.append(
            {
                "name": truncate_summary_text(getattr(feature, "name", "")),
                "description_chars": len(
                    str(getattr(feature, "description", "") or "")
                ),
                "evidence_ids": list(
                    getattr(feature, "evidence_ids", []) or []
                ),
            }
        )

    return build_limited_collection_summary(items, len(feature_items))


def summarize_pricing_plans(pricing_plans: object) -> dict[str, object]:
    """摘要价格项，保留短字段和 Evidence ID。"""

    pricing_items = list(pricing_plans or [])
    items: list[dict[str, object]] = []
    for pricing_plan in pricing_items[:SUMMARY_ITEM_LIMIT]:
        items.append(
            {
                "plan_name": truncate_summary_text(
                    getattr(pricing_plan, "plan_name", "")
                ),
                "has_price": getattr(pricing_plan, "price", None)
                is not None,
                "billing_cycle": getattr(
                    pricing_plan,
                    "billing_cycle",
                    None,
                ),
                "main_limit_count": len(
                    list(getattr(pricing_plan, "main_limits", []) or [])
                ),
                "evidence_ids": list(
                    getattr(pricing_plan, "evidence_ids", []) or []
                ),
            }
        )

    return build_limited_collection_summary(items, len(pricing_items))


def summarize_analysis(analysis: object | None) -> dict[str, object] | None:
    """摘要 Analyst 输出，只记录 claim 数量和引用规模。"""

    if analysis is None:
        return None

    return {
        "products": list(getattr(analysis, "products", []) or []),
        "positioning": summarize_claim_collection(
            getattr(analysis, "positioning", [])
        ),
        "features": summarize_claim_collection(
            getattr(analysis, "features", [])
        ),
        "pricing": summarize_claim_collection(
            getattr(analysis, "pricing", [])
        ),
        "opportunities": summarize_claim_collection(
            getattr(analysis, "opportunities", [])
        ),
        "conclusion": summarize_claim(getattr(analysis, "conclusion", None)),
    }


def summarize_claim_collection(claims: object) -> dict[str, object]:
    """摘要一组 claim，不记录 claim 原文。"""

    claim_items = list(claims or [])
    items = [
        summarize_claim(claim)
        for claim in claim_items[:SUMMARY_ITEM_LIMIT]
    ]
    return build_limited_collection_summary(items, len(claim_items))


def summarize_claim(claim: object | None) -> dict[str, object] | None:
    """摘要单条 claim，只保留类型、产品和引用信息。"""

    if claim is None:
        return None

    claim_text = getattr(claim, "claim", "") or ""
    return {
        "claim_chars": len(str(claim_text)),
        "claim_type": getattr(claim, "claim_type", ""),
        "product_names": list(getattr(claim, "product_names", []) or []),
        "evidence_ids": list(getattr(claim, "evidence_ids", []) or []),
    }


def summarize_verification_result(
    verification_result: object | None,
) -> dict[str, object] | None:
    """摘要 Verifier 输出，不记录 issue message 和 suggested_action 原文。"""

    if verification_result is None:
        return None

    issues = list(getattr(verification_result, "issues", []) or [])
    issue_items: list[dict[str, object]] = []
    for issue in issues[:SUMMARY_ITEM_LIMIT]:
        issue_items.append(
            {
                "claim_path": getattr(issue, "claim_path", ""),
                "issue_type": getattr(issue, "issue_type", ""),
                "evidence_ids": list(
                    getattr(issue, "evidence_ids", []) or []
                ),
                "message_chars": len(
                    str(getattr(issue, "message", "") or "")
                ),
                "suggested_action_chars": len(
                    str(getattr(issue, "suggested_action", "") or "")
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


def summarize_research_errors(errors: object) -> dict[str, object]:
    """摘要 Researcher 局部错误，不记录第三方异常文本。"""

    error_items = list(errors or [])
    items: list[dict[str, object]] = []
    for error in error_items[:SUMMARY_ITEM_LIMIT]:
        message = getattr(error, "message", "") or ""
        items.append(
            {
                "product_name": getattr(error, "product_name", ""),
                "topic": getattr(error, "topic", ""),
                "query_chars": len(str(getattr(error, "query", "") or "")),
                "code": getattr(error, "code", ""),
                "message_chars": len(str(message)),
            }
        )

    return build_limited_collection_summary(items, len(error_items))


def build_limited_collection_summary(
    items: list[object],
    total_count: int,
) -> dict[str, object]:
    """返回带总数和省略数量的列表摘要。"""

    return {
        "count": total_count,
        "items": items,
        "omitted_count": max(total_count - len(items), 0),
    }


def count_markdown_sections(markdown_text: object) -> int:
    """粗略统计 Markdown 标题数量，避免记录完整报告内容。"""

    text = str(markdown_text or "")
    return len(re.findall(r"(?m)^#{1,6}\s+", text))


def truncate_summary_text(value: object) -> str:
    """把日志摘要中的短文本压缩为单行，并限制最大长度。"""

    text = "" if value is None else str(value)
    compact_text = re.sub(r"\s+", " ", text).strip()
    if len(compact_text) <= SUMMARY_TEXT_LIMIT:
        return compact_text
    return compact_text[:SUMMARY_TEXT_LIMIT] + "...[truncated]"
