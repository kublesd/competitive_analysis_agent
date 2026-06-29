"""把已分析、已验证的数据确定性地渲染为 Markdown 报告。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from competitive_analysis_agent.analyst import (
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
    ProductProfile,
)
from competitive_analysis_agent.verifier import VerificationResult


MISSING_VALUE = "未提供"


class ReporterInput(ContractModel):
    """保存 Reporter 所需的分析、来源、画像和验证上下文。"""

    analysis: CompetitiveAnalysis
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
            "# 竞品分析报告",
            render_verification_section(
                reporter_input.verification_result,
                evidence_by_id,
            ),
            render_product_overview(
                reporter_input.product_profiles,
                evidence_by_id,
            ),
            render_claim_section(
                title="定位分析",
                claims=reporter_input.analysis.positioning,
                empty_message="未生成定位分析。",
                evidence_by_id=evidence_by_id,
            ),
            render_claim_section(
                title="功能对比",
                claims=reporter_input.analysis.features,
                empty_message="未生成有证据支持的功能对比。",
                evidence_by_id=evidence_by_id,
            ),
            render_claim_section(
                title="价格对比",
                claims=reporter_input.analysis.pricing,
                empty_message="未生成有证据支持的价格对比。",
                evidence_by_id=evidence_by_id,
            ),
            render_claim_section(
                title="机会点",
                claims=reporter_input.analysis.opportunities,
                empty_message="当前资料不足以生成可靠机会点。",
                evidence_by_id=evidence_by_id,
            ),
            render_conclusion(
                reporter_input.analysis.conclusion,
                evidence_by_id,
            ),
            render_data_limitations(
                product_profiles=reporter_input.product_profiles,
                research_errors=reporter_input.research_errors,
            ),
            render_evidence_sources(reporter_input.evidence),
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

    for profile in product_profiles:
        for feature in profile.features:
            evidence_ids.update(feature.evidence_ids)
        for pricing_plan in profile.pricing:
            evidence_ids.update(pricing_plan.evidence_ids)

    return evidence_ids


def render_verification_section(
    verification_result: VerificationResult,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """渲染验证状态；失败时完整保留问题和建议动作。"""

    lines = ["## 验证状态"]
    if verification_result.passed:
        lines.append("")
        lines.append("**状态：通过**")
        lines.append("")
        lines.append("未发现未解决的引用或证据支持问题。")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        "> **警告：本报告未通过最终验证。** "
        "以下内容只能作为待复核草稿，不能视为已确认结论。"
    )
    lines.append("")
    lines.append("| Claim 路径 | 问题类型 | 说明 | 相关证据 | 建议动作 |")
    lines.append("| --- | --- | --- | --- | --- |")

    for issue in verification_result.issues:
        citations = format_issue_evidence(
            issue.evidence_ids,
            evidence_by_id,
        )
        lines.append(
            "| "
            f"{escape_table_text(issue.claim_path)} | "
            f"{escape_table_text(issue.issue_type)} | "
            f"{escape_table_text(issue.message)} | "
            f"{citations} | "
            f"{escape_table_text(issue.suggested_action)} |"
        )

    return "\n".join(lines)


def render_product_overview(
    product_profiles: list[ProductProfile],
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
    research_errors: list[ResearchError],
) -> str:
    """汇总缺失画像字段、公开限制和未完成的研究任务。"""

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

        for pricing_plan in profile.pricing:
            if pricing_plan.price is None:
                limitations.append(
                    f"{profile.product_name} 的 "
                    f"{pricing_plan.plan_name} 方案未提供公开价格。"
                )
            if should_report_missing_billing_cycle(
                pricing_plan.price,
                pricing_plan.billing_cycle,
            ):
                limitations.append(
                    f"{profile.product_name} 的 "
                    f"{pricing_plan.plan_name} 方案未提供计费周期。"
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


def render_evidence_sources(evidence: list[Evidence]) -> str:
    """按输入顺序渲染完整、可点击的 Evidence 来源表。"""

    lines = [
        "## 资料来源",
        "",
        "| ID | 产品 | 主题 | 来源类型 | 来源 | 采集时间 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
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
) -> str:
    """格式化价格方案，并明确显示未知价格和计费周期。"""

    if not profile.pricing:
        return MISSING_VALUE

    pricing_items: list[str] = []
    for pricing_plan in profile.pricing:
        price = pricing_plan.price or "价格未提供"
        include_billing_cycle = should_include_billing_cycle(
            pricing_plan.price,
            pricing_plan.billing_cycle,
        )
        report_missing_billing_cycle = should_report_missing_billing_cycle(
            pricing_plan.price,
            pricing_plan.billing_cycle,
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
