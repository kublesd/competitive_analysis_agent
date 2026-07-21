"""把已分析、已验证的数据确定性地渲染为 Markdown 报告。"""

from __future__ import annotations

from pathlib import Path
from decimal import Decimal

from pydantic import Field, model_validator

from competitive_analysis_agent.analyst import (
    ActionableRecommendation,
    AnalysisClaim,
    CompetitiveAnalysis,
    collect_analysis_claims,
)
from competitive_analysis_agent.pricing_utils import (
    should_include_billing_cycle,
    should_report_missing_billing_cycle,
)
from competitive_analysis_agent.researcher import ResearchError
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    MarketDefinition,
    ProductProfile,
)
from competitive_analysis_agent.cost_calculator import (
    ScenarioCost,
    calculate_scenario_costs,
)
from competitive_analysis_agent.verifier import VerificationResult
from competitive_analysis_agent.verifier import ClaimVerificationStatus


MISSING_VALUE = "资料不足"


class ReporterInput(ContractModel):
    """保存 Reporter 所需的分析、来源、画像和验证上下文。"""

    analysis: CompetitiveAnalysis
    market_definition: MarketDefinition
    product_profiles: list[ProductProfile] = Field(min_length=2)
    evidence: list[Evidence] = Field(min_length=1)
    verification_result: VerificationResult
    research_errors: list[ResearchError] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_sources(self) -> "ReporterInput":
        """保证报告中的产品顺序和全部引用都能映射到真实 Evidence。"""

        profile_names = [
            profile.product_name for profile in self.product_profiles
        ]
        if profile_names != self.analysis.products:
            raise ValueError(
                "Product profiles must match analysis product order."
            )

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique.")

        known_evidence_ids = set(evidence_ids)
        referenced_evidence_ids = collect_report_evidence_ids(
            analysis=self.analysis,
            product_profiles=self.product_profiles,
        )
        missing_evidence_ids = (
            referenced_evidence_ids - known_evidence_ids
        )
        if missing_evidence_ids:
            missing_text = ", ".join(sorted(missing_evidence_ids))
            raise ValueError(
                "Report data references unknown Evidence IDs: "
                f"{missing_text}"
            )

        return self


class Reporter:
    """将结构化竞品分析渲染并写出为稳定的 Markdown。"""

    def render(self, reporter_input: ReporterInput) -> str:
        """按固定章节顺序生成报告，不调用模型或补充新事实。"""

        evidence_by_id = {
            item.evidence_id: item for item in reporter_input.evidence
        }
        sections = [
            render_report_header(reporter_input),
            render_executive_summary(reporter_input),
            render_verification_section(
                reporter_input.verification_result,
                evidence_by_id,
            ),
            render_key_conclusions(
                analysis=reporter_input.analysis,
                verification_passed=reporter_input.verification_result.passed,
                evidence_by_id=evidence_by_id,
            ),
            render_capability_matrix(
                reporter_input.product_profiles,
                evidence_by_id,
            ),
            render_model_price_comparison(
                reporter_input.product_profiles,
                reporter_input.market_definition,
                evidence_by_id,
            ),
            render_scenario_cost_table(
                product_profiles=reporter_input.product_profiles,
                market_definition=reporter_input.market_definition,
            ),
            render_limits_and_enterprise(
                reporter_input.product_profiles,
                evidence_by_id=evidence_by_id,
            ),
            render_product_assessment_table(
                reporter_input.analysis,
                reporter_input.verification_result.passed,
                evidence_by_id=evidence_by_id,
            ),
            render_scenario_recommendation_table(
                reporter_input.analysis,
                reporter_input.verification_result.passed,
                evidence_by_id=evidence_by_id,
            ),
            render_market_opportunity_table(
                reporter_input.analysis,
                reporter_input.verification_result.passed,
                evidence_by_id=evidence_by_id,
            ),
            render_data_gap_section(
                product_profiles=reporter_input.product_profiles,
                market_definition=reporter_input.market_definition,
                research_errors=reporter_input.research_errors,
                verification_result=reporter_input.verification_result,
                evidence_by_id=evidence_by_id,
            ),
            render_evidence_sources(
                [
                    item
                    for item in reporter_input.evidence
                    if item.scope_status == "in_scope"
                ]
            ),
        ]
        return "\n\n".join(sections) + "\n"

    def write(
        self,
        reporter_input: ReporterInput,
        output_path: str | Path,
    ) -> Path:
        """生成 Markdown 并写入指定文件，必要时创建父目录。"""

        resolved_path = Path(output_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report = self.render(reporter_input)
        resolved_path.write_text(markdown_report, encoding="utf-8")
        return resolved_path


def collect_report_evidence_ids(
    analysis: CompetitiveAnalysis,
    product_profiles: list[ProductProfile],
) -> set[str]:
    """收集分析和画像中所有会进入报告的 Evidence ID。"""

    evidence_ids: set[str] = set()

    for claim in collect_analysis_claims(analysis):
        evidence_ids.update(claim.evidence_ids)

    for recommendation in analysis.recommendations:
        evidence_ids.update(recommendation.evidence_ids)

    for profile in product_profiles:
        for dimension_finding in profile.dimension_findings:
            evidence_ids.update(dimension_finding.evidence_ids)
        for feature in profile.features:
            evidence_ids.update(feature.evidence_ids)
        for pricing_plan in profile.pricing:
            evidence_ids.update(pricing_plan.evidence_ids)

    return evidence_ids


def render_report_header(reporter_input: ReporterInput) -> str:
    """报告标题只保留决策最需要的时间、范围和醒目验证状态。"""

    verification = reporter_input.verification_result
    if verification.passed:
        status = "✅ 已验证，可用于初步决策"
    elif verification.comparison_usable:
        status = "⚠️ 部分验证，仅供参考"
    else:
        status = "❌ 验证失败，不应作为正式决策依据"
    latest_time = max(item.collected_at for item in reporter_input.evidence)
    lines = [
        "# 竞品分析报告（决策版）",
        "",
        f"- 分析时间：{latest_time.isoformat()}",
        f"- 分析范围：{reporter_input.market_definition.market_name}；"
        f"{reporter_input.market_definition.comparison_level}",
        f"- 报告状态：{status}",
    ]
    if not verification.passed:
        lines.extend(["", "> **草稿状态：以下判断仍待验证，不可作为正式采购或技术决策依据。**"])
    return "\n".join(lines)


def render_executive_summary(reporter_input: ReporterInput) -> str:
    """生成 300–500 字的中文决策摘要，只复用结构化资料。"""

    market = reporter_input.market_definition
    products = "、".join(reporter_input.analysis.products)
    verification_text = (
        "已通过验证，可用于初步决策"
        if reporter_input.verification_result.passed
        else "尚未完成验证，只能作为待验证判断"
    )
    missing_products = [
        profile.product_name
        for profile in reporter_input.product_profiles
        if not profile.models or not profile.pricing
    ]
    missing_text = "、".join(missing_products) if missing_products else "无明显产品级缺口"
    summary = (
        f"本报告面向{market.target_buyer or '目标决策者'}，比较{products}在"
        f"{market.product_category}中的已收集资料，重点覆盖{market.comparison_level}、"
        f"统一能力、模型价格、场景成本、使用限制与企业能力。当前验证状态为“{verification_text}”。"
        "报告不会把未检索到的能力写成不支持，也不会只凭价格给出整体优劣结论；所有推荐均应同时"
        "查看关键证据、限制条件、置信度和待验证信息。对技术决策者，建议优先用能力矩阵和场景"
        "建议筛选候选，再以真实工作负载核验质量、延迟、限流与成本。对产品负责人，市场机会点"
        "聚焦成本可预测性、运行约束透明度和企业治理体验。"
        f"当前仍需补充或核验的产品资料主要涉及：{missing_text}。"
    )
    filler = "未覆盖字段均明确标记为资料不足，并应进入后续试用、采购和工程验证清单。"
    while len(summary) < 300:
        summary += filler
    return "## 执行摘要\n\n" + summary[:500]


def render_key_conclusions(
    analysis: CompetitiveAnalysis,
    verification_passed: bool,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """先回答场景选择，再给出结论置信度和可追溯依据。"""

    lines = ["## 关键结论", ""]
    if not verification_passed:
        lines.append("当前为待验证判断：验证未通过，未输出正式购买建议。")
        return "\n".join(lines)
    recommendations = [
        item
        for item in analysis.scenario_recommendations
        if item.recommended_product != "暂无可验证推荐"
    ]
    if not recommendations:
        lines.append("当前资料不足以将具体产品与业务场景建立可验证的一对一选择关系。")
        return "\n".join(lines)
    for item in recommendations[:3]:
        lines.append(
            f"- {item.scenario}：选择 {item.recommended_product}；"
            f"置信度为 {format_confidence(item.confidence)}。"
            f"依据：{format_citations(item.evidence_ids, evidence_by_id)}。"
        )
    lines.append(
        f"- 总体结论置信度：{format_confidence(analysis.conclusion_confidence)}。"
    )
    return "\n".join(lines)


def render_capability_matrix(
    product_profiles: list[ProductProfile],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """把能力集中到一张矩阵，避免重复列出原始事实。"""

    lines = ["## 统一能力矩阵", "", "| 产品 | 模型 | 上下文窗口 | 模态 | 工具调用 | 结构化输出 | 实时 | 缓存 | Batch | 证据 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    has_model = False
    for profile in product_profiles:
        if not profile.models:
            lines.append(f"| {escape_table_text(profile.product_name)} | 资料不足 | 资料不足 | 资料不足 | 资料不足 | 资料不足 | 资料不足 | 资料不足 | 资料不足 | 无直接引用 |")
            continue
        has_model = True
        for model in profile.models:
            lines.append(
                "| "
                f"{escape_table_text(profile.product_name)} | "
                f"{escape_table_text(model.model_name)} | "
                f"{model.context_window_tokens or '资料不足'} | "
                f"{format_modalities(model.supported_modalities)} | "
                f"{format_support_status(model.tool_calling)} | "
                f"{format_support_status(model.structured_output)} | "
                f"{format_support_status(model.realtime)} | "
                f"{format_support_status(model.prompt_caching)} | "
                f"{format_support_status(model.batch_api)} | "
                f"{format_source_references(model.source_evidence, evidence_by_id)} |"
            )
    if not has_model:
        lines.append("\n当前未收集可用于模型能力比较的资料。")
    return "\n".join(lines)


def render_model_price_comparison(
    product_profiles: list[ProductProfile],
    market_definition: MarketDefinition,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """合并显示模型费率，避免把每条价格复制成独立结论。"""

    lines = ["## 模型与价格对比", "", "| 产品 | 模型或套餐 | 输入 / 缓存输入 / 输出 | Batch | 适用条件与有效期 | 证据 |", "| --- | --- | --- | --- | --- | --- |"]
    if market_definition.pricing_scope == "api":
        for profile in product_profiles:
            if not profile.models:
                lines.append(
                    "| "
                    f"{escape_table_text(profile.product_name)} | 套餐汇总 | "
                    f"{format_subscription_prices(profile, evidence_by_id)} | "
                    "资料不足 | 资料不足 | "
                    f"{format_citations([evidence_id for plan in profile.pricing for evidence_id in plan.evidence_ids], evidence_by_id)} |"
                )
                continue
            for model in profile.models:
                pricing = model.pricing
                lines.append(
                    "| "
                    f"{escape_table_text(profile.product_name)} | {escape_table_text(model.model_name)} | "
                    f"{format_token_prices(pricing.input_price)} / {format_token_prices(pricing.cached_input_price)} / {format_token_prices(pricing.output_price)} | "
                    f"{format_token_prices(pricing.batch_input_price)} / {format_token_prices(pricing.batch_output_price)} | "
                    f"{format_price_conditions(pricing)} | "
                    f"{format_citations(sorted(pricing.evidence_ids()), evidence_by_id)} |"
                )
        return "\n".join(lines)
    for profile in product_profiles:
        lines.append(
            "| "
            f"{escape_table_text(profile.product_name)} | 套餐汇总 | "
            f"{format_subscription_prices(profile, evidence_by_id)} | 不适用 | 资料以套餐计费条件为准 | "
            f"{format_citations([evidence_id for plan in profile.pricing for evidence_id in plan.evidence_ids], evidence_by_id)} |"
        )
    return "\n".join(lines)


def render_limits_and_enterprise(
    product_profiles: list[ProductProfile],
    evidence_by_id: dict[str, Evidence],
) -> str:
    lines = ["## 使用限制和企业能力", "", "| 产品 | 已记录限流或限制 | 企业能力 | 数据状态 | 证据 |", "| --- | --- | --- | --- | --- |"]
    for profile in product_profiles:
        limits = [
                f"{format_rate_limit_metric(item.metric)}：{item.limit}"
            for item in (profile.rate_limits or [])
        ]
        limit_evidence = [
            evidence_id
            for item in (profile.rate_limits or [])
            for evidence_id in item.evidence_ids
        ]
        for model in profile.models:
            limits.extend(
                f"{model.model_name} {format_rate_limit_metric(item.metric)}：{item.limit}"
                for item in (model.rate_limits or [])
            )
            limit_evidence.extend(
                evidence_id
                for item in (model.rate_limits or [])
                for evidence_id in item.evidence_ids
            )
        enterprise = profile.enterprise_capabilities or ["资料不足"]
        status = "资料不足" if not limits or not profile.enterprise_capabilities else "已记录"
        lines.append(
            "| "
            f"{escape_table_text(profile.product_name)} | {format_text_items(limits)} | "
            f"{format_text_items(list(enterprise))} | {status} | "
            f"{format_citations(limit_evidence, evidence_by_id)} |"
        )
    return "\n".join(lines)


def format_support_status(status: object) -> str:
    labels = {"supported": "已记录", "not_supported": "不支持", "missing": "资料不足", "not_applicable": "不适用", "uncertain": "待核验"}
    return labels.get(getattr(status, "value", status), "资料不足")


def format_confidence(confidence: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(
        confidence, "待核验"
    )


def format_rate_limit_metric(metric: object) -> str:
    labels = {
        "requests_per_minute": "每分钟请求数",
        "tokens_per_minute": "每分钟 Token 数",
        "requests_per_day": "每日请求数",
        "tokens_per_day": "每日 Token 数",
        "concurrent_requests": "并发请求数",
        "other": "其他限制",
    }
    return labels.get(getattr(metric, "value", metric), "其他限制")


def format_modalities(modalities: object) -> str:
    values = [getattr(item, "value", str(item)) for item in modalities or []]
    return "、".join(values) if values else "资料不足"


def format_source_references(references: object, evidence_by_id: dict[str, Evidence]) -> str:
    return format_citations([item.evidence_id for item in references], evidence_by_id) if references else "无直接引用"


def format_token_prices(prices: object) -> str:
    rates = prices if isinstance(prices, list) else [prices]
    values = [f"${format_usd(rate.amount)}/1M" for rate in rates if rate is not None]
    return "；".join(values) if values else "资料不足"


def format_price_conditions(pricing: object) -> str:
    rates = [
        rate
        for value in (pricing.input_price, pricing.cached_input_price, pricing.output_price)
        for rate in (value if isinstance(value, list) else [value])
        if rate is not None
    ]
    conditions = [
        "；".join(
            item for item in [rate.condition, f"最大上下文 {rate.max_context_tokens}" if rate.max_context_tokens else None, rate.effective_from.isoformat() if rate.effective_from else None, rate.effective_to.isoformat() if rate.effective_to else None] if item
        )
        for rate in rates
    ]
    return " / ".join(item for item in conditions if item) or "资料不足"


def format_subscription_prices(profile: ProductProfile, evidence_by_id: dict[str, Evidence]) -> str:
    if not profile.pricing:
        return "资料不足"
    return "；".join(
        f"{escape_table_text(plan.plan_name)}：{escape_table_text(plan.price or '价格未提供')}"
        for plan in profile.pricing
    )


def render_market_definition(
    market_definition: MarketDefinition,
    product_names: list[str],
    evidence: list[Evidence],
) -> str:
    """在报告开头披露比较范围、数据时间和来源使用规则。"""

    latest_evidence_time = max(
        item.collected_at for item in evidence
    ).isoformat()
    target_buyer = market_definition.target_buyer or MISSING_VALUE
    exclusions = format_text_items(market_definition.exclusions)
    dimensions = format_text_items(market_definition.core_dimensions)
    products = "、".join(product_names)
    return "\n".join(
        [
            "## 分析范围",
            "",
            "| 市场 | 产品类别 | 目标购买者 | 比较层级 | 价格范围 | 比较产品 | 核心维度 | 排除项 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "| "
            f"{escape_table_text(market_definition.market_name)} | "
            f"{escape_table_text(market_definition.product_category)} | "
            f"{escape_table_text(target_buyer)} | "
            f"{escape_table_text(market_definition.comparison_level)} | "
            f"{escape_table_text(market_definition.pricing_scope)} | "
            f"{escape_table_text(products)} | "
            f"{dimensions} | {exclusions} |",
            "",
            f"- 分析时间（按最新资料采集时间）：{escape_plain_text(latest_evidence_time)}",
            "- 来源规则：官方来源优先；第三方来源仅作补充；只有范围内资料可支撑事实与结论。",
        ]
    )


def render_dimension_comparison(
    market_definition: MarketDefinition,
    product_profiles: list[ProductProfile],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """按用户选择的核心维度生成横向表，缺失项明确显示资料不足。"""

    product_headers = " | ".join(
        escape_table_text(profile.product_name)
        for profile in product_profiles
    )
    lines = [
        "## 核心维度对比",
        "",
        f"| 维度 | {product_headers} |",
        "| --- | " + " | ".join("---" for _ in product_profiles) + " |",
    ]
    for dimension in market_definition.core_dimensions:
        cells = [
            format_dimension_cell(
                profile=profile,
                dimension=dimension,
                evidence_by_id=evidence_by_id,
            )
            for profile in product_profiles
        ]
        lines.append(
            f"| {escape_table_text(dimension)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def format_dimension_cell(
    profile: ProductProfile,
    dimension: str,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """只展示带引用的 DimensionFinding，缺失时明确返回资料不足。"""

    normalized_dimension = dimension.casefold()
    for finding in profile.dimension_findings:
        if finding.dimension.casefold() != normalized_dimension:
            continue
        if not finding.facts:
            return MISSING_VALUE
        facts = "<br>".join(
            escape_table_text(fact) for fact in finding.facts
        )
        citations = format_citations(
            finding.evidence_ids,
            evidence_by_id,
        )
        return f"{facts} {citations}"

    return MISSING_VALUE


def render_verification_section(
    verification_result: VerificationResult,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """渲染验证状态；失败时完整保留问题和建议动作。"""

    lines = [
        "## 验证状态",
        "",
        "| 引用有效 | 范围一致 | 比较可用 |",
        "| --- | --- | --- |",
        "| "
        f"{format_verification_status(verification_result.citations_valid)} | "
        f"{format_verification_status(verification_result.scope_consistent)} | "
        f"{format_verification_status(verification_result.comparison_usable)} |",
    ]
    if verification_result.passed:
        lines.append("")
        lines.append("**状态：通过**")
        lines.append("")
        claim_count = len(verification_result.claim_verifications)
        detail = f"，已逐条验证 {claim_count} 条 Claim" if claim_count else ""
        lines.append(f"引用有效、范围一致且比较可用{detail}。")
        return "\n".join(lines)

    if not verification_result.passed:
        lines.append("")
        lines.append(
            "> **警告：本报告未通过最终验证。** "
            "以下内容只能作为待复核草稿，不能视为已确认结论。"
        )
    lines.append("")
    lines.append("| 字段路径 | 状态 | Claim | 证据 | 原因 | 建议修复动作 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    if verification_result.claim_verifications:
        for verification in verification_result.claim_verifications:
            evidence_text = format_verification_evidence(
                verification.evidence_ids,
                evidence_by_id,
            )
            lines.append(
                "| "
                f"{escape_table_text(verification.field_path)} | "
                f"{escape_table_text(format_claim_status(verification.status.value))} | "
                f"{escape_table_text(verification.claim)} | "
                f"{evidence_text} | "
                f"{escape_table_text(verification.reason)} | "
                f"{escape_table_text(verification.suggested_action)} |"
            )
        return "\n".join(lines)

    for issue in verification_result.issues:
        citations = format_issue_evidence(
            issue.evidence_ids,
            evidence_by_id,
        )
        lines.append(
            "| "
            f"{escape_table_text(issue.claim_path)} | "
            f"{escape_table_text(issue.issue_type)} | "
            "资料不足 | "
            f"{citations} | "
            f"{escape_table_text(issue.message)} | "
            f"{escape_table_text(issue.suggested_action)} |"
        )

    return "\n".join(lines)


def format_verification_status(passed: bool) -> str:
    """把验证布尔值转换成报告内可快速扫描的状态。"""

    return "通过" if passed else "未通过"


def format_claim_status(status: str) -> str:
    return {
        "supported": "已支持",
        "partially_supported": "部分支持",
        "conflicting": "证据冲突",
        "insufficient": "资料不足",
        "stale": "已过期",
        "invalid_scope": "范围无效",
    }.get(status, "待核验")


def build_supported_claims_by_section(
    analysis: CompetitiveAnalysis,
    verification_result: VerificationResult,
) -> dict[str, list[AnalysisClaim]]:
    """只让 supported claim 进入报告的正式分析章节。"""

    sections = {
        "positioning": analysis.positioning,
        "features": analysis.features,
        "pricing": analysis.pricing,
        "dimension_comparisons": analysis.dimension_comparisons,
        "opportunities": analysis.opportunities,
    }
    return {
        section_name: [
            claim
            for index, claim in enumerate(claims)
            if claim_path_is_supported(
                f"{section_name}[{index}]",
                verification_result,
            )
        ]
        for section_name, claims in sections.items()
    }


def claim_path_is_supported(
    claim_path: str,
    verification_result: VerificationResult,
) -> bool:
    """优先读取六态结果；旧结果则根据 issue 路径保守过滤。"""

    for verification in verification_result.claim_verifications:
        if verification.field_path == claim_path:
            return verification.status == ClaimVerificationStatus.SUPPORTED
    return not any(
        issue.claim_path == claim_path
        for issue in verification_result.issues
    )


def render_pending_claims(
    verification_result: VerificationResult,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """partially_supported 只进入待核验区，不混入正式分析。"""

    partial_items = [
        item
        for item in verification_result.claim_verifications
        if item.status == ClaimVerificationStatus.PARTIALLY_SUPPORTED
    ]
    lines = ["## 待核验 Claim", ""]
    if not partial_items:
        lines.append("没有 partially_supported 的 Claim。")
        return "\n".join(lines)

    lines.extend(
        [
            "| 字段路径 | Claim | 证据 | 原因 | 建议修复动作 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in partial_items:
        lines.append(
            "| "
            f"{escape_table_text(item.field_path)} | "
            f"{escape_table_text(item.claim)} | "
            f"{format_verification_evidence(item.evidence_ids, evidence_by_id)} | "
            f"{escape_table_text(item.reason)} | "
            f"{escape_table_text(item.suggested_action)} |"
        )
    return "\n".join(lines)


def render_blocked_recommendations() -> str:
    """验证未通过时硬性关闭正式购买建议。"""

    return "\n".join(
        [
            "## 购买建议状态",
            "",
            "验证未通过，未输出正式购买建议。",
        ]
    )


def render_blocked_conclusion() -> str:
    """验证未通过时不发布 Analyst 的正式结论文本。"""

    return "\n".join(
        [
            "## 结论状态",
            "",
            "验证未通过，未形成正式结论；请先处理验证报告中的问题。",
        ]
    )


def render_product_assessment_table(
    analysis: CompetitiveAnalysis,
    verification_passed: bool,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """以统一字段呈现优势、短板与资料缺口，不把缺失写成能力不足。"""

    lines = ["## 产品优势与短板", ""]
    if not verification_passed:
        lines.append("验证未通过，产品优势、短板与置信度仅供待核验，不输出确定性判断。")
        return "\n".join(lines)
    lines.extend(
        [
            "| 产品 | 优势（已记录） | 短板（已记录限制） | 数据缺口 | 结论置信度 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for assessment in analysis.product_assessments:
        lines.append(
            "| "
            f"{escape_table_text(assessment.product_name)} | "
            f"{format_assessment_claims(assessment.strengths, evidence_by_id)} | "
            f"{format_assessment_claims(assessment.shortcomings, evidence_by_id)} | "
            f"{format_text_items(list(assessment.data_gaps))} | "
            f"{format_confidence(assessment.confidence)} |"
        )
    return "\n".join(lines)


def format_assessment_claims(
    claims: list[AnalysisClaim],
    evidence_by_id: dict[str, Evidence],
) -> str:
    if not claims:
        return "资料不足"
    return "<br>".join(
        f"{escape_table_text(format_assessment_claim(claim.claim))} "
        f"{format_citations(claim.evidence_ids, evidence_by_id)}"
        for claim in claims
    )


def format_assessment_claim(claim: str) -> str:
    """把内部英文事实句压成中文标签，保留产品和技术名词。"""

    if " mentions " in claim:
        return "已记录能力：" + claim.split(" mentions ", 1)[1].rstrip(".")
    if " constraint: " in claim:
        return "已记录限制：" + claim.split(" constraint: ", 1)[1].rstrip(".")
    return "已记录事实：" + claim


def render_scenario_recommendation_table(
    analysis: CompetitiveAnalysis,
    verification_passed: bool,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """集中展示七类场景的推荐、取舍、置信度与待验证项。"""

    lines = ["## 场景化选择建议", ""]
    if not verification_passed:
        lines.append("验证未通过，未输出确定性场景购买建议。")
        return "\n".join(lines)
    recommendations = [
        item
        for item in analysis.scenario_recommendations
        if item.recommended_product != "暂无可验证推荐"
    ]
    unavailable_scenarios = [
        item.scenario
        for item in analysis.scenario_recommendations
        if item.recommended_product == "暂无可验证推荐"
    ]
    if not recommendations:
        scenarios = "、".join(unavailable_scenarios) or "全部预设场景"
        lines.append(
            f"{scenarios}均缺少形成唯一推荐所需的同范围证据；"
            "未输出正式购买建议。"
        )
        return "\n".join(lines)
    lines.extend(
        [
            "| 场景 | 推荐产品 | 推荐理由 | 成本与能力权衡 | 关键证据 | 主要限制 | 置信度 | 尚未验证 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for recommendation in recommendations:
        lines.append(
            "| "
            f"{escape_table_text(recommendation.scenario)} | "
            f"{escape_table_text(recommendation.recommended_product)} | "
            f"{escape_table_text(recommendation.recommendation_reason)} | "
            f"{escape_table_text(recommendation.cost_capability_tradeoff)} | "
            f"{format_citations(recommendation.evidence_ids, evidence_by_id)} | "
            f"{format_text_items(list(recommendation.primary_limitations))} | "
            f"{format_confidence(recommendation.confidence)} | "
            f"{format_text_items(list(recommendation.unverified_information))} |"
        )
    if unavailable_scenarios:
        lines.extend(
            [
                "",
                "资料不足、未形成唯一推荐的场景："
                + "、".join(unavailable_scenarios)
                + "。",
            ]
        )
    return "\n".join(lines)


def render_market_opportunity_table(
    analysis: CompetitiveAnalysis,
    verification_passed: bool,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """将机会点拆成可复核的痛点、现状、空白与产品方向。"""

    lines = ["## 市场机会点", ""]
    if not verification_passed:
        lines.append("验证未通过，机会点保留为待核验推断。")
        return "\n".join(lines)
    opportunities = [
        item for item in analysis.market_opportunities if item.evidence_ids
    ]
    if not opportunities:
        lines.append("资料不足，未形成有直接证据的市场机会点。")
        return "\n".join(lines)
    lines.extend(
        [
            "| 机会点 | 用户痛点 | 竞品现状 | 市场空白 | 可行产品方向 | 支撑证据 | 推断置信度 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for opportunity in opportunities:
        lines.append(
            "| "
            f"{escape_table_text(opportunity.title)} | "
            f"{escape_table_text(opportunity.user_pain)} | "
            f"{escape_table_text(opportunity.competitor_status)} | "
            f"{escape_table_text(opportunity.market_gap)} | "
            f"{escape_table_text(opportunity.product_direction)} | "
            f"{format_citations(opportunity.evidence_ids, evidence_by_id)} | "
            f"{format_confidence(opportunity.inference_confidence)} |"
        )
    return "\n".join(lines)


def render_scenario_cost_table(
    product_profiles: list[ProductProfile],
    market_definition: MarketDefinition,
) -> str:
    """渲染计算结果，不在报告层解释或重算任何价格。"""

    lines = ["## 场景成本估算", ""]
    if market_definition.pricing_scope != "api":
        lines.append("当前为订阅价格范围，未计算 Token 场景成本。")
        return "\n".join(lines)

    results = [
        item
        for profile in product_profiles
        for item in calculate_scenario_costs(
            product_name=profile.product_name,
            models=profile.models,
            monthly_call_count=market_definition.monthly_call_count,
        )
    ]
    if not results:
        lines.append("未提供可计算的模型 Token 价格。")
        return "\n".join(lines)

    calls = market_definition.monthly_call_count
    lines.extend(
        [
            "| 产品 | 模型 | 场景 | 实时（单次 / 月度） | Batch（单次 / 月度；节省） | Prompt caching（单次 / 月度；节省） |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            f"{escape_table_text(result.product_name)} | "
            f"{escape_table_text(result.model_name)} | "
            f"{escape_table_text(result.scenario_name)}（{calls} 次/月） | "
            f"{format_scenario_cost(result.realtime)} | "
            f"{format_scenario_cost(result.batch)} | "
            f"{format_scenario_cost(result.cached)} |"
        )
    return "\n".join(lines)


def format_scenario_cost(cost: ScenarioCost | None) -> str:
    """缺价时明确披露，避免把不完整的价格相加成假总价。"""

    if cost is None:
        return "—"
    if cost.cost is None or cost.monthly_cost is None:
        missing = "、".join(cost.missing_price_types)
        return f"资料不足（缺 {escape_table_text(missing)} 价格）"
    value = (
        f"${format_usd(cost.cost)} / ${format_usd(cost.monthly_cost)}"
    )
    if cost.savings_percent is not None:
        value += f"；节省 {format_usd(cost.savings_percent)}%"
    return value


def format_usd(value: Decimal) -> str:
    """避免科学计数法与多余尾随零，保留 Decimal 的精确结果。"""

    return format(value.normalize(), "f")


def render_product_overview(
    product_profiles: list[ProductProfile],
    market_definition: MarketDefinition,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """将产品画像渲染为便于横向浏览的概览表。"""

    lines = [
        "## 产品概览",
        "",
        "| 产品 | 定位 | 目标用户 | 功能 | 价格 | 已知限制 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for profile in product_profiles:
        positioning = profile.positioning or MISSING_VALUE
        target_users = format_text_items(profile.target_users)
        features = format_profile_features(
            profile,
            evidence_by_id,
        )
        pricing = format_profile_pricing(
            profile,
            evidence_by_id,
            market_definition.pricing_scope,
        )
        limitations = format_text_items(profile.limitations)
        lines.append(
            "| "
            f"{escape_table_text(profile.product_name)} | "
            f"{escape_table_text(positioning)} | "
            f"{target_users} | "
            f"{features} | "
            f"{pricing} | "
            f"{limitations} |"
        )

    return "\n".join(lines)


def render_claim_section(
    title: str,
    claims: list[AnalysisClaim],
    empty_message: str,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """把一类分析 claim 渲染成固定列的 Markdown 表格。"""

    lines = [f"## {title}", ""]
    if not claims:
        lines.append(empty_message)
        return "\n".join(lines)

    lines.extend(
        [
            "| 类型 | 涉及产品 | 结论 | 证据 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for claim in claims:
        claim_type = (
            "事实" if claim.claim_type == "fact" else "分析判断"
        )
        product_names = "<br>".join(
            escape_table_text(name) for name in claim.product_names
        )
        citations = format_citations(
            claim.evidence_ids,
            evidence_by_id,
        )
        lines.append(
            "| "
            f"{claim_type} | "
            f"{product_names} | "
            f"{escape_table_text(claim.claim)} | "
            f"{citations} |"
        )

    return "\n".join(lines)


def render_recommendations(
    recommendations: list[ActionableRecommendation],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """展示目标场景、取舍、动作、证据与适用边界。"""

    lines = ["## 面向购买者的建议", ""]
    if not recommendations:
        lines.append("资料不足，未生成可追溯建议。")
        return "\n".join(lines)

    lines.extend(
        [
            "| 目标场景 | 竞品取舍或空白 | 建议动作 | 证据 | 限制条件 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for recommendation in recommendations:
        citations = format_citations(
            recommendation.evidence_ids,
            evidence_by_id,
        )
        limitations = "<br>".join(
            escape_table_text(item)
            for item in recommendation.limitations
        )
        lines.append(
            "| "
            f"{escape_table_text(recommendation.target_scenario)} | "
            f"{escape_table_text(recommendation.tradeoff_or_gap)} | "
            f"{escape_table_text(recommendation.recommended_action)} | "
            f"{citations} | {limitations} |"
        )
    return "\n".join(lines)


def render_conclusion(
    conclusion: AnalysisClaim,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """渲染最终结论并保留产品范围和证据链接。"""

    product_names = "、".join(conclusion.product_names)
    citations = format_citations(
        conclusion.evidence_ids,
        evidence_by_id,
    )
    return "\n".join(
        [
            "## 结论",
            "",
            f"> {escape_plain_text(conclusion.claim)}",
            "",
            f"- 涉及产品：{escape_plain_text(product_names)}",
            f"- 证据：{citations}",
        ]
    )


def render_data_limitations(
    product_profiles: list[ProductProfile],
    market_definition: MarketDefinition,
    research_errors: list[ResearchError],
) -> str:
    """按产品和维度汇总资料不足，避免逐条套餐重复提示。"""

    limitations: list[str] = []

    for research_error in research_errors:
        limitations.append(
            f"{research_error.product_name} / "
            f"{research_error.topic}："
            f"{research_error.message}"
        )

    for profile in product_profiles:
        if profile.positioning is None:
            limitations.append(
                f"{profile.product_name}：未提供定位信息。"
            )
        if not profile.target_users:
            limitations.append(
                f"{profile.product_name}：未提供目标用户信息。"
            )
        if not profile.features:
            limitations.append(
                f"{profile.product_name}：未提供功能信息。"
            )
        if not profile.pricing:
            limitations.append(
                f"{profile.product_name}：未提供价格信息。"
            )

        missing_dimensions = [
            dimension
            for dimension in market_definition.core_dimensions
            if dimension_cell_is_missing(profile, dimension)
        ]
        if missing_dimensions:
            limitations.append(
                f"{profile.product_name}：核心维度资料不足："
                f"{'、'.join(missing_dimensions)}。"
            )

        missing_price_plans = [
            pricing_plan.plan_name
            for pricing_plan in profile.pricing
            if pricing_plan.price is None
        ]
        if missing_price_plans:
            limitations.append(
                f"{profile.product_name}：以下方案未提供公开价格："
                f"{'、'.join(missing_price_plans)}。"
            )
        missing_cycle_plans = []
        if market_definition.pricing_scope == "subscription":
            missing_cycle_plans = [
                pricing_plan.plan_name
                for pricing_plan in profile.pricing
                if should_report_missing_billing_cycle(
                    pricing_plan.price,
                    pricing_plan.billing_cycle,
                    pricing_plan.unit,
                )
            ]
        if missing_cycle_plans:
            limitations.append(
                f"{profile.product_name}：以下方案未提供计费周期："
                f"{'、'.join(missing_cycle_plans)}。"
            )

        for limitation in profile.limitations:
            limitations.append(
                f"{profile.product_name}：{limitation}"
            )

    unique_limitations = preserve_unique_order(limitations)
    lines = ["## 数据限制", ""]
    if not unique_limitations:
        lines.append("未记录明确的数据缺口或研究失败。")
        return "\n".join(lines)

    for limitation in unique_limitations:
        lines.append(f"- {escape_plain_text(limitation)}")

    return "\n".join(lines)


def render_data_gap_section(
    product_profiles: list[ProductProfile],
    market_definition: MarketDefinition,
    research_errors: list[ResearchError],
    verification_result: VerificationResult,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """合并资料缺口、待核验 Claim 与范围状态，供决策者一次查看。"""

    limitations = render_data_limitations(
        product_profiles,
        market_definition,
        research_errors,
    ).split("\n", 2)
    lines = ["## 数据缺口和待核验内容", ""]
    lines.extend(limitations[2:] if len(limitations) > 2 else ["资料不足。"])
    if any(
        item.status == ClaimVerificationStatus.PARTIALLY_SUPPORTED
        for item in verification_result.claim_verifications
    ):
        lines.extend(["", render_pending_claims(verification_result, evidence_by_id)])
    uncertain_count = sum(
        item.scope_status == "uncertain" for item in evidence_by_id.values()
    )
    excluded_count = sum(
        item.scope_status == "out_of_scope" for item in evidence_by_id.values()
    )
    if uncertain_count or excluded_count:
        lines.append(
            f"\n- 范围状态：待核验资料 {uncertain_count} 条；已排除资料 {excluded_count} 条。"
        )
    lines.extend(
        [
            "",
            render_scoped_evidence(
                title="已排除资料",
                evidence=list(evidence_by_id.values()),
                scope_status="out_of_scope",
                empty_message="没有已排除资料。",
            ),
            "",
            render_scoped_evidence(
                title="待核验资料",
                evidence=list(evidence_by_id.values()),
                scope_status="uncertain",
                empty_message="没有待核验资料。",
            ),
        ]
    )
    return "\n".join(lines)


def dimension_cell_is_missing(
    profile: ProductProfile,
    dimension: str,
) -> bool:
    """判断核心维度是否没有可展示事实。"""

    normalized_dimension = dimension.casefold()
    for finding in profile.dimension_findings:
        if finding.dimension.casefold() == normalized_dimension:
            return not (finding.facts and finding.evidence_ids)
    return True


def render_scoped_evidence(
    title: str,
    evidence: list[Evidence],
    scope_status: str,
    empty_message: str,
) -> str:
    """展示被排除或待核验资料及其范围原因。"""

    scoped_items = [
        item for item in evidence if item.scope_status == scope_status
    ]
    lines = [f"## {title}", ""]
    if not scoped_items:
        lines.append(empty_message)
        return "\n".join(lines)

    lines.extend(
        [
            "| 产品 | 主题 | 来源 | 原因 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in scoped_items:
        source_link = (
            f"[{escape_link_text(item.title)}]({str(item.url)})"
        )
        lines.append(
            "| "
            f"{escape_table_text(item.product_name)} | "
            f"{escape_table_text(item.topic)} | {source_link} | "
            f"{escape_table_text(item.scope_reason)} |"
        )
    return "\n".join(lines)


def render_evidence_sources(evidence: list[Evidence]) -> str:
    """按输入顺序渲染完整、可点击的 Evidence 来源表。"""

    lines = ["## 来源附录", ""]
    if not evidence:
        lines.append("没有可用于支撑结论的范围内资料。")
        return "\n".join(lines)
    lines.extend([
        "| ID | 产品 | 主题 | 来源类型 | 来源 | 完整 URL | 采集时间 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    source_type_labels = {
        "official": "官方",
        "third_party": "第三方",
    }

    for item in evidence:
        source_link = (
            f"[{escape_link_text(item.title)}]({str(item.url)})"
        )
        collected_at = item.collected_at.isoformat()
        lines.append(
            "| "
            f"{escape_table_text(item.evidence_id)} | "
            f"{escape_table_text(item.product_name)} | "
            f"{escape_table_text(item.topic)} | "
            f"{source_type_labels[item.source_type]} | "
            f"{source_link} | "
            f"{escape_table_text(str(item.url))} | "
            f"{escape_table_text(collected_at)} |"
        )

    return "\n".join(lines)


def format_profile_features(
    profile: ProductProfile,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """格式化产品功能及其来源链接。"""

    if not profile.features:
        return MISSING_VALUE

    feature_items: list[str] = []
    for feature in profile.features:
        citations = format_citations(
            feature.evidence_ids,
            evidence_by_id,
        )
        feature_items.append(
            f"{escape_table_text(feature.name)} {citations}"
        )
    return "<br>".join(feature_items)


def format_profile_pricing(
    profile: ProductProfile,
    evidence_by_id: dict[str, Evidence],
    pricing_scope: str,
) -> str:
    """格式化价格、单位、周期、服务等级、阈值和主要限制。"""

    if not profile.pricing:
        return MISSING_VALUE

    pricing_items: list[str] = []
    for pricing_plan in profile.pricing:
        price = pricing_plan.price or "价格未提供"
        include_billing_cycle = should_include_billing_cycle(
            pricing_plan.price,
            pricing_plan.billing_cycle,
        )
        report_missing_billing_cycle = (
            pricing_scope == "subscription"
            and should_report_missing_billing_cycle(
                pricing_plan.price,
                pricing_plan.billing_cycle,
                pricing_plan.unit,
            )
        )
        if report_missing_billing_cycle:
            include_billing_cycle = True
            billing_cycle = "计费周期未提供"
        else:
            billing_cycle = pricing_plan.billing_cycle or "计费周期未提供"
        citations = format_citations(
            pricing_plan.evidence_ids,
            evidence_by_id,
        )
        pricing_text = (
            f"{escape_table_text(pricing_plan.plan_name)}："
            f"{escape_table_text(price)}"
        )
        if include_billing_cycle:
            pricing_text += f" / {escape_table_text(billing_cycle)}"

        context_items: list[str] = []
        normalized_price = price.casefold()
        if (
            pricing_plan.unit
            and pricing_plan.unit.casefold() not in normalized_price
        ):
            context_items.append(f"单位：{pricing_plan.unit}")
        if pricing_plan.service_level:
            context_items.append(
                f"服务等级：{pricing_plan.service_level}"
            )
        if pricing_plan.threshold:
            context_items.append(f"阈值：{pricing_plan.threshold}")
        if pricing_plan.main_limits:
            context_items.append(
                "主要限制：" + "；".join(pricing_plan.main_limits)
            )
        if context_items:
            escaped_context = "；".join(
                escape_table_text(item) for item in context_items
            )
            pricing_text += f"（{escaped_context}）"

        pricing_items.append(f"{pricing_text} {citations}")
    return "<br>".join(pricing_items)


def format_text_items(items: list[str]) -> str:
    """把普通文本列表转换成表格内换行内容。"""

    if not items:
        return MISSING_VALUE
    return "<br>".join(escape_table_text(item) for item in items)


def format_citations(
    evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """把已校验的 Evidence ID 转成直接指向来源 URL 的链接。"""

    if not evidence_ids:
        return "无直接引用"

    citation_links: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id[evidence_id]
        citation_links.append(
            f"[{escape_link_text(evidence_id)}]({str(evidence.url)})"
        )
    return " ".join(citation_links)


def format_issue_evidence(
    evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """渲染 issue 证据；不存在的 ID 明确标为无法链接。"""

    if not evidence_ids:
        return "无直接引用"

    evidence_labels: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            evidence_labels.append(
                f"未找到来源：{escape_table_text(evidence_id)}"
            )
            continue
        evidence_labels.append(
            f"[{escape_link_text(evidence_id)}]({str(evidence.url)})"
        )
    return "<br>".join(evidence_labels)


def format_verification_evidence(
    evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """在验证表中同时展示来源链接和回答当前 claim 的证据文本。"""

    if not evidence_ids:
        return "无直接引用"

    evidence_items: list[str] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            evidence_items.append(
                f"未找到来源：{escape_table_text(evidence_id)}"
            )
            continue
        quote = evidence.snippet
        if len(quote) > 160:
            quote = quote[:157].rstrip() + "..."
        evidence_items.append(
            f"[{escape_link_text(evidence_id)}]({str(evidence.url)}) "
            f"{escape_table_text(quote)}"
        )
    return "<br>".join(evidence_items)


def preserve_unique_order(items: list[str]) -> list[str]:
    """按首次出现顺序去重，避免数据限制重复展示。"""

    unique_items: list[str] = []
    seen_items: set[str] = set()
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        unique_items.append(item)
    return unique_items


def escape_table_text(value: str) -> str:
    """转义 Markdown 表格中的分隔符和换行。"""

    return escape_plain_text(value).replace("|", "\\|").replace(
        "\n",
        "<br>",
    )


def escape_plain_text(value: str) -> str:
    """转义可能意外形成 Markdown 结构的反斜杠。"""

    return value.replace("\\", "\\\\")


def escape_link_text(value: str) -> str:
    """转义 Markdown 链接标签中的方括号。"""

    return escape_table_text(value).replace("[", "\\[").replace(
        "]",
        "\\]",
    )
