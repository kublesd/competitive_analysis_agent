"""Verifier 节点：检查分析引用，并评审 claim 的证据支持情况。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import Enum
from typing import Literal, Protocol

from pydantic import Field, ValidationError, computed_field, model_validator

from competitive_analysis_agent.analyst import (
    AnalysisClaim,
    CompetitiveAnalysis,
    build_fallback_pricing_claims,
    market_dimension_is_selected,
)
from competitive_analysis_agent.model_io import (
    log_model_error,
    log_model_request,
    log_model_response,
)
from competitive_analysis_agent.pricing_utils import (
    billing_cycle_is_supported_by_text,
    detect_billing_cycle_category,
    is_free_price_text,
    is_custom_pricing_text,
    is_supported_billing_cycle_text,
    normalize_price_text,
    should_report_missing_billing_cycle,
)
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    EvidenceId,
    MarketDefinition,
    PricingPlan,
    ProductProfile,
    RequiredText,
)


VERIFIER_SYSTEM_PROMPT = """
你是竞品分析流程中的 Verifier。

你的唯一职责是逐条判断 claim 与给定 Evidence 的关系。
不得改写分析，不得搜索网页，不得使用外部知识。

用户消息会提供：
1. 本次 MarketDefinition；
2. 带稳定 claim_path ID 的 claims；
3. 全部 Evidence。

检查规则：
1. 每条 claim 必须且只能返回以下一个 status：
   - supported：证据明确支持完整 claim；
   - partially_supported：证据只支持 claim 的一部分；
   - conflicting：证据明确反驳 claim，或同范围证据存在实质冲突；
   - insufficient：证据不足以判断；
   - stale：证据明确标注已失效、已过有效期，或被更新官方资料替代；
   - invalid_scope：证据不属于 claim 的产品，或超出 MarketDefinition 范围。
2. 价格 claim 必须同时核对模型名、价格方向（input/output/cached/audio/Batch）、
   金额、币种、计量基数和单位。网页同时出现 $2 input 与 $10 output 时，两者是
   同一模型的不同价格字段，不能因为数字不同而判为冲突。
3. interpretation 可以进行合理归纳，不要求逐字出现在 Evidence 中；只有它引入
   未提供的新事实或与 Evidence 冲突时才报告。对输入范围的保守说明，例如
   “现有资料不足以比较某项能力”，不属于新增产品事实；当输入确实没有提供该项
   可比信息时，不要仅因为这类说明没有直接引用而报告 unsupported_claim。
   对 positioning 和 opportunities 这类解释型章节，除非出现明显新增硬事实、
   强评价或证据冲突，否则不要按事实句逐字要求 Evidence 支持。
4. 只报告真实问题，不要把措辞差异当成错误。
   Evidence 是搜索召回的补充上下文。应综合标题、snippet、raw_content 和表格上下文
   判断语义支持，不要因为网页被压平、字段分散或措辞不同就判定不支持。
   - 只审查用户消息中 claims 列表实际存在的 claim；不要根据 Evidence 或
     MarketDefinition 发明新的待审事实，也不要报告列表外的信息缺口。
   - 对 “Product mentions X” 这类标准化 claim，只要相关 Evidence 出现 X
     或覆盖 X 的关键词，就视为支持，不要求 Evidence 逐字写出 “mentions”。
   - 对价格 claim，只要相关 Evidence 同时出现套餐名和价格/Custom pricing/Free，
     就视为支持，不要求 Evidence 逐字写出 “lists the plan at”。
   - 对 “comparison is limited to supplied evidence/product profiles” 这类范围说明，
     不要当作产品事实要求 Evidence 逐字支持。
5. claim_path 是 `C001` 这类不含业务含义的 ID，必须逐字复制用户消息中的值；
   不要生成 `features[0]`、`pricing[0]` 等章节路径。
6. evidence_ids 只能使用用户消息中真实存在、与问题直接相关的 ID。
   如果问题正是 claim 没有任何直接支持证据，可以返回空列表。
7. suggested_action 应说明 Analyst 应如何修改或收窄 claim。
8. 必须为输入中的每条 claim 返回一条 verification，不得遗漏或重复。
   市场范围、核心维度覆盖和价格口径也会由普通代码检查。
9. 只输出 JSON，不要添加 Markdown 或解释。
10. reason 解释证据为何支持或不支持当前字段；suggested_action 给出修复动作，
    supported 时填写 "No action required."。
11. 输出格式：
{
  "verifications": [
    {
      "status": "supported",
      "claim_path": "C001",
      "evidence_ids": ["E1"],
      "reason": "...",
      "suggested_action": "..."
    }
  ]
}
""".strip()


SemanticIssueType = Literal[
    "unsupported_claim",
    "conflicting_evidence",
]
VerificationIssueType = Literal[
    "invalid_evidence_id",
    "wrong_product_evidence",
    "missing_product_evidence",
    "unsupported_claim",
    "conflicting_evidence",
    "out_of_scope_evidence",
    "scope_level_conflict",
    "incomplete_pricing_context",
    "incomparable_pricing",
    "missing_core_dimension",
    "partially_supported",
    "insufficient_evidence",
    "stale_evidence",
    "invalid_scope",
    "third_party_only_evidence",
]
CITATION_ISSUE_TYPES = {
    "invalid_evidence_id",
    "wrong_product_evidence",
    "missing_product_evidence",
    "unsupported_claim",
    "conflicting_evidence",
    "partially_supported",
    "insufficient_evidence",
    "stale_evidence",
    "third_party_only_evidence",
}
SCOPE_ISSUE_TYPES = {
    "out_of_scope_evidence",
    "scope_level_conflict",
    "invalid_scope",
    "third_party_only_evidence",
}
COMPARABILITY_ISSUE_TYPES = {
    "incomplete_pricing_context",
    "incomparable_pricing",
    "missing_core_dimension",
}
NON_RETRYABLE_ISSUE_TYPES = {
    "out_of_scope_evidence",
    "incomplete_pricing_context",
    "missing_core_dimension",
    "insufficient_evidence",
    "stale_evidence",
    "invalid_scope",
}
DIRECT_PRICE_COMPARISON_TERMS = (
    "cheaper",
    "costs less",
    "costs more",
    "higher price",
    "less expensive",
    "lower price",
    "more expensive",
    "价格更低",
    "价格更高",
    "成本更低",
    "更便宜",
)
PRICE_UNIT_MARKERS = (
    "per user",
    "per seat",
    "per workspace",
    "per token",
    "per request",
    "/user",
    "/seat",
    "/workspace",
    "/token",
    "/request",
)
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PRICE_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
PRICING_LIST_PATTERN = re.compile(
    r"\blists the (?P<plan>.+?) plan at (?P<price>.+?)"
    r"(?: with (?P<billing>.+?) billing)?\.$",
    re.IGNORECASE,
)
PRICING_WITHOUT_PRICE_PATTERN = re.compile(
    r"\bnames a[n]? (?P<plan>.+?) plan without a public price"
    r"(?: in the supplied profile)?"
    r"(?: with (?P<billing>.+?) billing)?\.$",
    re.IGNORECASE,
)
TEXT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "with",
}
TEXT_TOKEN_ALIASES = {
    "automate": "automation",
    "automated": "automation",
    "automates": "automation",
    "automating": "automation",
    "manage": "management",
    "managed": "management",
    "manages": "management",
    "managing": "management",
}
FEATURE_PHRASE_MIN_TOKEN_COUNT = 4
FEATURE_PHRASE_MIN_MATCH_COUNT = 3
FEATURE_PHRASE_MIN_COVERAGE = 0.75
SOFT_INTERPRETATION_PATH_PREFIXES = (
    "positioning[",
    "opportunities[",
    "product_assessments[",
    "scenario_recommendations[",
    "market_opportunities[",
)
STRONG_EVALUATION_TERMS = (
    "better",
    "best",
    "clearly stronger",
    "dominates",
    "leader",
    "leading",
    "superior",
    "wins",
    "更好",
    "更强",
    "领先",
    "优于",
)


class VerifierInput(ContractModel):
    """保存待验证分析和原始证据，并确保 Evidence ID 唯一。"""

    analysis: CompetitiveAnalysis
    evidence: list[Evidence] = Field(min_length=1)
    market_definition: MarketDefinition | None = None
    product_profiles: list[ProductProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "VerifierInput":
        """避免同一个 Evidence ID 指向多个来源。"""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique.")

        product_names = [
            profile.product_name for profile in self.product_profiles
        ]
        if len(product_names) != len(set(product_names)):
            raise ValueError("Product profile names must be unique.")

        return self


class VerificationIssue(ContractModel):
    """描述一个可定位、可修复的验证问题。"""

    issue_type: VerificationIssueType
    claim_path: RequiredText
    message: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    suggested_action: RequiredText


class ClaimVerificationStatus(str, Enum):
    """描述一条 claim 与其证据之间的完整验证关系。"""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    STALE = "stale"
    INVALID_SCOPE = "invalid_scope"


class ClaimVerification(ContractModel):
    """保存 Reporter 可直接消费的逐 claim 验证记录。"""

    field_path: RequiredText
    claim: RequiredText
    status: ClaimVerificationStatus
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    reason: RequiredText
    suggested_action: RequiredText


class VerificationResult(ContractModel):
    """返回验证是否通过、问题清单和后续重试意图。"""

    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    retry_recommended: bool

    @computed_field
    @property
    def citations_valid(self) -> bool:
        """只要存在引用或证据支持问题，就返回 False。"""

        return not any(
            issue.issue_type in CITATION_ISSUE_TYPES
            for issue in self.issues
        )

    @computed_field
    @property
    def scope_consistent(self) -> bool:
        """只要分析使用越界资料或跨范围 claim，就返回 False。"""

        return not any(
            issue.issue_type in SCOPE_ISSUE_TYPES
            for issue in self.issues
        )

    @computed_field
    @property
    def comparison_usable(self) -> bool:
        """价格口径或核心维度不足时返回 False。"""

        return not any(
            issue.issue_type in COMPARABILITY_ISSUE_TYPES
            for issue in self.issues
        )

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "VerificationResult":
        """确保布尔状态与问题列表保持一致。"""

        all_claims_supported = all(
            item.status == ClaimVerificationStatus.SUPPORTED
            for item in self.claim_verifications
        )
        expected_passed = not self.issues and all_claims_supported
        expected_retry = any(
            issue.issue_type not in NON_RETRYABLE_ISSUE_TYPES
            for issue in self.issues
        )
        if self.passed != expected_passed:
            raise ValueError("passed must be true only when issues is empty.")
        if self.retry_recommended != expected_retry:
            raise ValueError(
                "retry_recommended must match retryable issue types."
            )

        return self


class SemanticVerificationIssue(ContractModel):
    """约束模型只能报告语义支持或证据冲突问题。"""

    issue_type: SemanticIssueType
    claim_path: RequiredText
    message: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    suggested_action: RequiredText


class SemanticClaimVerification(ContractModel):
    """约束模型为每条输入 claim 返回一个六态判定。"""

    status: ClaimVerificationStatus
    claim_path: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    reason: RequiredText
    suggested_action: RequiredText


class VerifierModelOutput(ContractModel):
    """接收新逐 claim 输出，并暂时兼容旧版 issue 列表。"""

    verifications: list[SemanticClaimVerification] = Field(
        default_factory=list
    )
    issues: list[SemanticVerificationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_single_output_shape(self) -> "VerifierModelOutput":
        """非空的新旧输出不能同时出现，避免两套判定互相覆盖。"""

        if self.verifications and self.issues:
            raise ValueError(
                "Return verifications or legacy issues, not both."
            )
        return self


class VerifierModel(Protocol):
    """约定 Verifier 所需的最小结构化模型调用接口。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """根据 claims 和 Evidence 返回语义问题列表。"""


class StructuredChatModel(Protocol):
    """描述 LangChain ChatModel 的结构化输出能力。"""

    def with_structured_output(
        self,
        schema: type[VerifierModelOutput],
        *,
        method: Literal["json_mode"],
        include_raw: Literal[True],
    ) -> VerifierModel:
        """绑定 VerifierModelOutput，并返回可调用模型。"""


class LangChainVerifierModel:
    """把 LangChain ChatModel 包装成 Verifier 所需的模型接口。"""

    def __init__(self, chat_model: StructuredChatModel) -> None:
        # 当前模型供应商支持 json_object，因此沿用项目统一的 JSON mode。
        self._structured_model = chat_model.with_structured_output(
            VerifierModelOutput,
            method="json_mode",
            include_raw=True,
        )

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """执行模型调用，并在解析失败时返回原始文本供统一校验。"""

        call_id = log_model_request("Verifier", messages)
        try:
            structured_response = self._structured_model.invoke(messages)
        except Exception as error:
            log_model_error("Verifier", call_id, error)
            raise
        log_model_response("Verifier", call_id, structured_response)

        if not isinstance(structured_response, dict):
            return structured_response

        response_wrapper_keys = {"raw", "parsed", "parsing_error"}
        if not response_wrapper_keys.intersection(structured_response):
            return structured_response

        parsed_output = structured_response.get("parsed")
        if parsed_output is not None:
            return parsed_output

        raw_message = structured_response.get("raw")
        raw_content = getattr(raw_message, "content", raw_message)
        return raw_content


class FakeVerifierModel:
    """返回固定语义评审响应，并记录是否真的发生了模型调用。"""

    def __init__(self, response: object) -> None:
        self._response = response
        self.invocation_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回固定响应，并保存收到的消息。"""

        copied_messages = [message.copy() for message in messages]
        self.received_messages.append(copied_messages)
        self.invocation_count += 1
        return self._response


class VerifierError(RuntimeError):
    """表示语义评审调用或模型输出校验失败。"""

    def __init__(
        self,
        message: str,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        # public_detail 会显示到页面，因此只能保存脱敏后的定位信息。
        self.public_detail = public_detail or message


class Verifier:
    """先执行确定性检查，再对结构正确的分析进行一次语义评审。"""

    def __init__(self, model: VerifierModel) -> None:
        self._model = model

    def verify(
        self,
        verifier_input: VerifierInput,
    ) -> VerificationResult:
        """验证引用和语义；硬引用错误存在时跳过模型调用。"""

        deterministic_issues = find_deterministic_issues(verifier_input)
        if deterministic_issues:
            return build_verification_result(
                deterministic_issues,
                analysis=verifier_input.analysis,
            )

        messages = build_verifier_messages(verifier_input)
        raw_output = self._invoke_model(messages)
        try:
            semantic_issues, claim_verifications = validate_semantic_output(
                raw_output=raw_output,
                verifier_input=verifier_input,
            )
        except VerifierError:
            # 只修复 Verifier 自身的输出契约；分析中的语义问题仍交给图级回路。
            repair_messages = build_verifier_repair_messages(
                initial_messages=messages,
                verifier_input=verifier_input,
            )
            repaired_output = self._invoke_model(repair_messages)
            semantic_issues, claim_verifications = validate_semantic_output(
                raw_output=repaired_output,
                verifier_input=verifier_input,
            )
        return build_verification_result(
            semantic_issues,
            claim_verifications=claim_verifications,
        )

    def _invoke_model(
        self,
        messages: list[dict[str, str]],
    ) -> object:
        """调用语义评审模型，并把供应商异常转换成 VerifierError。"""

        try:
            return self._model.invoke(messages)
        except Exception as error:
            raise VerifierError(
                f"Verifier model call failed: {error}",
                public_detail=(
                    "Verifier 调用模型服务失败。"
                    "这通常是网络、超时、额度或模型服务临时不可用。"
                ),
            ) from error


def find_deterministic_issues(
    verifier_input: VerifierInput,
) -> list[VerificationIssue]:
    """用普通代码检查引用、范围、价格口径和核心维度覆盖。"""

    evidence_by_id = {
        item.evidence_id: item for item in verifier_input.evidence
    }
    issues: list[VerificationIssue] = []

    for claim_path, claim in iter_claims_with_paths(
        verifier_input.analysis
    ):
        referenced_ids = set(claim.evidence_ids)
        missing_ids = referenced_ids - set(evidence_by_id)
        if missing_ids:
            sorted_missing_ids = sorted(missing_ids)
            issues.append(
                VerificationIssue(
                    issue_type="invalid_evidence_id",
                    claim_path=claim_path,
                    message=(
                        "Claim references Evidence IDs that do not exist: "
                        f"{', '.join(sorted_missing_ids)}"
                    ),
                    evidence_ids=sorted_missing_ids,
                    suggested_action=(
                        "Remove the invalid IDs or cite existing Evidence."
                    ),
                )
            )

        known_ids = referenced_ids - missing_ids
        wrong_product_ids: list[str] = []
        for evidence_id in sorted(known_ids):
            evidence = evidence_by_id[evidence_id]
            if evidence.product_name not in claim.product_names:
                wrong_product_ids.append(evidence_id)

        if wrong_product_ids:
            issues.append(
                VerificationIssue(
                    issue_type="wrong_product_evidence",
                    claim_path=claim_path,
                    message=(
                        "Claim cites Evidence belonging to products outside "
                        f"product_names: {', '.join(wrong_product_ids)}"
                    ),
                    evidence_ids=wrong_product_ids,
                    suggested_action=(
                        "Use Evidence owned by the products named in the claim."
                    ),
                )
            )

        if claim.claim_type != "fact":
            continue

        # 多产品事实必须为每个产品提供至少一条自己的有效证据。
        unsupported_products: list[str] = []
        for product_name in claim.product_names:
            has_product_evidence = False
            for evidence_id in known_ids:
                evidence = evidence_by_id[evidence_id]
                if evidence.product_name == product_name:
                    has_product_evidence = True
                    break

            if not has_product_evidence:
                unsupported_products.append(product_name)

        if unsupported_products:
            issues.append(
                VerificationIssue(
                    issue_type="missing_product_evidence",
                    claim_path=claim_path,
                    message=(
                        "Factual claim lacks valid Evidence for products: "
                        f"{', '.join(unsupported_products)}"
                    ),
                    evidence_ids=sorted(known_ids),
                    suggested_action=(
                        "Add one valid citation for each product or narrow "
                        "the claim scope."
                    ),
                )
            )

    issues.extend(
        find_recommendation_reference_issues(
            verifier_input=verifier_input,
            evidence_by_id=evidence_by_id,
        )
    )
    issues.extend(find_primary_source_issues(verifier_input, evidence_by_id))
    issues.extend(
        find_scope_issues(
            verifier_input=verifier_input,
            evidence_by_id=evidence_by_id,
        )
    )
    issues.extend(find_pricing_context_issues(verifier_input))
    issues.extend(find_incomparable_pricing_issues(verifier_input))
    issues.extend(find_missing_dimension_issues(verifier_input))
    return issues


def find_primary_source_issues(
    verifier_input: VerifierInput,
    evidence_by_id: dict[str, Evidence],
) -> list[VerificationIssue]:
    """正式结论和建议至少需要一条官方主证据。"""

    critical_paths = (
        "conclusion",
        "recommendations[",
        "scenario_recommendations[",
        "market_opportunities[",
    )
    issues: list[VerificationIssue] = []
    for claim_path, claim in iter_claims_with_paths(verifier_input.analysis):
        if not claim_path.startswith(critical_paths) or not claim.evidence_ids:
            continue
        known_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        if known_evidence and all(
            getattr(item.source_type, "value", item.source_type) == "third_party"
            for item in known_evidence
        ):
            issues.append(
                VerificationIssue(
                    issue_type="third_party_only_evidence",
                    claim_path=claim_path,
                    message="Formal conclusions cannot rely only on third-party evidence.",
                    evidence_ids=list(claim.evidence_ids),
                    suggested_action=(
                        "Add an official source or keep the conclusion as unknown."
                    ),
                )
            )
    return issues


def find_recommendation_reference_issues(
    verifier_input: VerifierInput,
    evidence_by_id: dict[str, Evidence],
) -> list[VerificationIssue]:
    """独立复核建议引用，避免只验证普通 AnalysisClaim。"""

    issues: list[VerificationIssue] = []
    for index, recommendation in enumerate(
        verifier_input.analysis.recommendations
    ):
        claim_path = f"recommendations[{index}]"
        referenced_ids = set(recommendation.evidence_ids)
        missing_ids = sorted(referenced_ids - set(evidence_by_id))
        if missing_ids:
            issues.append(
                VerificationIssue(
                    issue_type="invalid_evidence_id",
                    claim_path=claim_path,
                    message=(
                        "Recommendation references Evidence IDs that do not "
                        f"exist: {', '.join(missing_ids)}"
                    ),
                    evidence_ids=missing_ids,
                    suggested_action=(
                        "Remove invalid IDs or cite existing Evidence."
                    ),
                )
            )

        wrong_product_ids: list[str] = []
        for evidence_id in sorted(referenced_ids - set(missing_ids)):
            evidence = evidence_by_id[evidence_id]
            if evidence.product_name not in recommendation.product_names:
                wrong_product_ids.append(evidence_id)
        if wrong_product_ids:
            issues.append(
                VerificationIssue(
                    issue_type="wrong_product_evidence",
                    claim_path=claim_path,
                    message=(
                        "Recommendation cites Evidence outside product_names: "
                        f"{', '.join(wrong_product_ids)}"
                    ),
                    evidence_ids=wrong_product_ids,
                    suggested_action=(
                        "Use Evidence owned by the products named in the "
                        "recommendation."
                    ),
                )
            )
    return issues


def find_scope_issues(
    verifier_input: VerifierInput,
    evidence_by_id: dict[str, Evidence],
) -> list[VerificationIssue]:
    """检查分析是否引用非范围内资料或直接写入排除项。"""

    issues: list[VerificationIssue] = []
    for claim_path, _, evidence_ids in iter_reference_records(
        verifier_input.analysis
    ):
        invalid_scope_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].scope_status != "in_scope"
        ]
        if invalid_scope_ids:
            issues.append(
                VerificationIssue(
                    issue_type="out_of_scope_evidence",
                    claim_path=claim_path,
                    message=(
                        "Analysis references Evidence that is not confirmed "
                        f"in scope: {', '.join(invalid_scope_ids)}"
                    ),
                    evidence_ids=invalid_scope_ids,
                    suggested_action=(
                        "Remove the affected claim and collect confirmed "
                        "in-scope Evidence before comparing it."
                    ),
                )
            )

    market_definition = verifier_input.market_definition
    if market_definition is None:
        return issues

    for claim_path, claim in iter_claims_with_paths(
        verifier_input.analysis
    ):
        conflicting_exclusions = [
            exclusion
            for exclusion in market_definition.exclusions
            if claim_contains_excluded_scope(claim.claim, exclusion)
        ]
        if not conflicting_exclusions:
            continue
        issues.append(
            VerificationIssue(
                issue_type="scope_level_conflict",
                claim_path=claim_path,
                message=(
                    "Claim includes excluded market scope: "
                    f"{', '.join(conflicting_exclusions)}"
                ),
                evidence_ids=list(claim.evidence_ids),
                suggested_action=(
                    "Remove the excluded product line or narrow the claim to "
                    f"{market_definition.comparison_level}."
                ),
            )
        )
    return issues


def claim_contains_excluded_scope(claim_text: str, exclusion: str) -> bool:
    """识别 claim 是否肯定写入排除项，保守排除说明不算冲突。"""

    normalized_claim = claim_text.casefold()
    normalized_exclusion = exclusion.casefold()
    if normalized_exclusion not in normalized_claim:
        return False
    conservative_markers = ("exclude", "excluding", "not included", "排除", "不包括")
    return not any(marker in normalized_claim for marker in conservative_markers)


def find_pricing_context_issues(
    verifier_input: VerifierInput,
) -> list[VerificationIssue]:
    """检查数值价格是否保留当前价格范围所需的上下文。"""

    market_definition = verifier_input.market_definition
    if not market_dimension_is_selected(market_definition, "pricing"):
        return []

    issues: list[VerificationIssue] = []
    for profile in verifier_input.product_profiles:
        for index, pricing_plan in enumerate(profile.pricing):
            if not pricing_plan_requires_context(pricing_plan):
                continue

            missing_fields: list[str] = []
            if not pricing_plan_has_unit(pricing_plan):
                missing_fields.append("unit")
            if (
                market_definition is None
                or market_definition.pricing_scope == "subscription"
            ) and should_report_missing_billing_cycle(
                price_text=pricing_plan.price,
                billing_cycle=pricing_plan.billing_cycle,
                unit_text=pricing_plan.unit,
            ):
                missing_fields.append("billing_cycle")
            if not missing_fields:
                continue

            issues.append(
                VerificationIssue(
                    issue_type="incomplete_pricing_context",
                    claim_path=(
                        f"product_profiles[{profile.product_name}]."
                        f"pricing[{index}]"
                    ),
                    message=(
                        "Pricing context is incomplete: "
                        f"{', '.join(missing_fields)}"
                    ),
                    evidence_ids=list(pricing_plan.evidence_ids),
                    suggested_action=(
                        "Show this price as not directly comparable until its "
                        "unit and billing conditions are confirmed."
                    ),
                )
            )
    return issues


def pricing_plan_requires_context(pricing_plan: PricingPlan) -> bool:
    """免费、自定义报价和无公开价格无需强制补齐数值价格口径。"""

    if pricing_plan.price is None:
        return False
    if is_free_price_text(pricing_plan.price):
        return False
    return not is_custom_pricing_text(pricing_plan.price)


def pricing_plan_has_unit(pricing_plan: PricingPlan) -> bool:
    """接受独立 unit 字段或价格原文中的常见单位。"""

    if pricing_plan.unit:
        return True
    if pricing_plan.price is None:
        return False
    normalized_price = normalize_price_text(pricing_plan.price)
    if any(marker in normalized_price for marker in PRICE_UNIT_MARKERS):
        return True
    return detect_billing_cycle_category(pricing_plan.price) is not None


def find_incomparable_pricing_issues(
    verifier_input: VerifierInput,
) -> list[VerificationIssue]:
    """阻止把不同单位或周期的价格写成直接高低结论。"""

    if not market_dimension_is_selected(
        verifier_input.market_definition, "pricing"
    ):
        return []

    profiles_by_name = {
        profile.product_name: profile
        for profile in verifier_input.product_profiles
    }
    issues: list[VerificationIssue] = []
    for index, claim in enumerate(verifier_input.analysis.pricing):
        if len(claim.product_names) < 2:
            continue
        if not contains_direct_price_comparison(claim.claim):
            continue

        signatures_by_product: list[
            frozenset[tuple[str | None, str | None]]
        ] = []
        for product_name in claim.product_names:
            profile = profiles_by_name.get(product_name)
            if profile is None:
                continue
            product_signatures: set[
                tuple[str | None, str | None]
            ] = set()
            for pricing_plan in profile.pricing:
                if not pricing_plan_requires_context(pricing_plan):
                    continue
                product_signatures.add(build_pricing_signature(pricing_plan))
            if product_signatures:
                signatures_by_product.append(frozenset(product_signatures))

        # ponytail: 先用单位+周期识别明显不可比；只有真实案例证明不足时再引入价格本体模型。
        if len(signatures_by_product) < 2:
            continue
        if len(set(signatures_by_product)) == 1:
            continue
        issues.append(
            VerificationIssue(
                issue_type="incomparable_pricing",
                claim_path=f"pricing[{index}]",
                message=(
                    "Claim directly compares prices with different units or "
                    "billing cycles."
                ),
                evidence_ids=list(claim.evidence_ids),
                suggested_action=(
                    "Remove the price ranking and present each original unit "
                    "and condition separately."
                ),
            )
        )
    return issues


def contains_direct_price_comparison(claim_text: str) -> bool:
    """识别明确的价格高低判断，透明度差异不算直接价格比较。"""

    normalized_claim = claim_text.casefold()
    return any(term in normalized_claim for term in DIRECT_PRICE_COMPARISON_TERMS)


def build_pricing_signature(
    pricing_plan: PricingPlan,
) -> tuple[str | None, str | None]:
    """提取价格单位与周期，供保守可比性判断。"""

    unit = normalize_pricing_unit(pricing_plan)
    billing_cycle = detect_billing_cycle_category(
        pricing_plan.billing_cycle or pricing_plan.price
    )
    if billing_cycle is None:
        billing_cycle = detect_billing_cycle_category(pricing_plan.unit)
    return unit, billing_cycle


def normalize_pricing_unit(pricing_plan: PricingPlan) -> str | None:
    """把 user/per user 等等价写法归一为同一价格单位。"""

    text_parts = [pricing_plan.unit or "", pricing_plan.price or ""]
    normalized_text = normalize_price_text(" ".join(text_parts))
    unit_aliases = {
        "user": ("per user", "/user", "user"),
        "seat": ("per seat", "/seat", "seat"),
        "workspace": ("per workspace", "/workspace", "workspace"),
        "token": ("per token", "/token", "token"),
        "request": ("per request", "/request", "request"),
    }
    for unit, markers in unit_aliases.items():
        if any(marker in normalized_text for marker in markers):
            return unit
    if pricing_plan.unit:
        return normalize_price_text(pricing_plan.unit)
    return None


def find_missing_dimension_issues(
    verifier_input: VerifierInput,
) -> list[VerificationIssue]:
    """按产品检查每个核心维度是否至少有一项范围内事实。"""

    market_definition = verifier_input.market_definition
    if market_definition is None or not verifier_input.product_profiles:
        return []

    issues: list[VerificationIssue] = []
    for dimension in market_definition.core_dimensions:
        missing_products = [
            profile.product_name
            for profile in verifier_input.product_profiles
            if not profile_has_dimension_data(profile, dimension)
        ]
        if not missing_products:
            continue
        issues.append(
            VerificationIssue(
                issue_type="missing_core_dimension",
                claim_path=f"core_dimensions[{dimension}]",
                message=(
                    f"Core dimension {dimension!r} has insufficient data for: "
                    f"{', '.join(missing_products)}"
                ),
                evidence_ids=[],
                suggested_action=(
                    "Display 资料不足 for these products and collect more "
                    "in-scope Evidence before comparison."
                ),
            )
        )
    return issues


def profile_has_dimension_data(
    profile: ProductProfile,
    dimension: str,
) -> bool:
    """只读取带引用的 DimensionFinding，避免消费者使用不同事实来源。"""

    normalized_dimension = dimension.casefold()
    for finding in profile.dimension_findings:
        if finding.dimension.casefold() == normalized_dimension:
            return bool(finding.facts and finding.evidence_ids)
    return False


def iter_reference_records(
    analysis: CompetitiveAnalysis,
) -> list[tuple[str, list[str], list[str]]]:
    """统一列出 claim 和 recommendation 的产品及引用。"""

    return [
        (path, list(claim.product_names), list(claim.evidence_ids))
        for path, claim in iter_claims_with_paths(analysis)
    ]


def build_verifier_messages(
    verifier_input: VerifierInput,
) -> list[dict[str, str]]:
    """把不透明 claim ID 和 Evidence 转换成语义评审消息。"""

    claim_records: list[dict[str, object]] = []
    for claim_id, _, claim in iter_model_claims(verifier_input.analysis):
        claim_record = claim.model_dump(mode="json")
        # 模型只看到短 ID，避免从 features/pricing 名称反推不存在的索引。
        claim_record["claim_path"] = claim_id
        claim_records.append(claim_record)

    payload = {
        "market_definition": (
            verifier_input.market_definition.model_dump(mode="json")
            if verifier_input.market_definition is not None
            else None
        ),
        "claims": claim_records,
        "evidence": [
            item.model_dump(mode="json")
            for item in verifier_input.evidence
        ],
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        "请检查以下 claims 是否被 Evidence 支持，或是否存在明确冲突。\n\n"
        f"{payload_json}"
    )

    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_verifier_repair_messages(
    initial_messages: list[dict[str, str]],
    verifier_input: VerifierInput,
) -> list[dict[str, str]]:
    """提供合法路径清单，让 Verifier 最多修复一次自身输出。"""

    valid_claim_paths = [
        claim_id
        for claim_id, _, _ in iter_model_claims(verifier_input.analysis)
    ]
    valid_paths_json = json.dumps(
        valid_claim_paths,
        ensure_ascii=False,
    )
    repair_instruction = (
        "上一次 Verifier 输出没有通过结构校验。请重新检查原始 claims 和 "
        "Evidence，并输出一份完整的 {\"verifications\": [...]} JSON。\n"
        "claim_path 必须逐字复制以下合法 claim_path 之一；不要猜测索引，"
        "也不要省略实际存在的语义问题。\n"
        "不要把输出格式错误报告为语义 issue 或验证结果；每条 claim 必须返回且"
        "只返回一个六态 status。\n"
        f"合法 claim_path：{valid_paths_json}"
    )
    repair_messages = [message.copy() for message in initial_messages]
    repair_messages.append(
        {"role": "user", "content": repair_instruction}
    )
    return repair_messages


def normalize_verifier_raw_output(raw_output: object) -> object:
    """兼容真实模型常见的 Verifier 输出外层形状。"""

    if isinstance(raw_output, str):
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
        return normalize_verifier_raw_output(parsed_output)

    if isinstance(raw_output, list):
        return {"issues": raw_output}

    if not isinstance(raw_output, dict):
        return raw_output

    raw_verifications = raw_output.get("verifications")
    if isinstance(raw_verifications, list):
        return {"verifications": raw_verifications}

    # 有些模型会返回完整 VerificationResult，这里只保留 VerifierModelOutput
    # 真正需要的 issues，避免因为 passed/retry_recommended 额外字段中断。
    raw_issues = raw_output.get("issues")
    if isinstance(raw_issues, list):
        return {"issues": raw_issues}

    nested_keys = [
        "verification_result",
        "verifier_result",
        "result",
        "output",
    ]
    for nested_key in nested_keys:
        nested_output = raw_output.get(nested_key)
        if isinstance(nested_output, dict):
            normalized_nested_output = normalize_verifier_raw_output(
                nested_output
            )
            if (
                isinstance(normalized_nested_output, dict)
                and (
                    "issues" in normalized_nested_output
                    or "verifications" in normalized_nested_output
                )
            ):
                return normalized_nested_output

    return raw_output


def build_verifier_schema_error_detail(
    validation_error: ValidationError,
) -> str:
    """把 Pydantic 错误整理成可展示、但不含原始响应的摘要。"""

    issue_summaries: list[str] = []
    for error_item in validation_error.errors(include_input=False):
        location_parts = [str(part) for part in error_item["loc"]]
        location = ".".join(location_parts) if location_parts else "root"
        message = error_item["msg"]
        issue_summaries.append(f"{location}: {message}")

    if issue_summaries:
        summary_text = "; ".join(issue_summaries[:3])
    else:
        summary_text = "无法解析模型返回结构"

    return (
        "Verifier 模型输出结构不符合要求。"
        f"结构问题：{summary_text}。"
        "期望顶层 JSON 为 {\"verifications\": [...]}；每项必须包含 "
        "status、claim_path、reason、evidence_ids、suggested_action。"
        "旧版 {\"issues\": [...]} 仍可兼容读取。"
    )


def validate_semantic_output(
    raw_output: object,
    verifier_input: VerifierInput,
) -> tuple[list[VerificationIssue], list[ClaimVerification]]:
    """校验六态结果，并把旧 issue 输出兼容转换为逐 claim 记录。"""

    normalized_output = normalize_verifier_raw_output(raw_output)
    try:
        if isinstance(normalized_output, str):
            model_output = VerifierModelOutput.model_validate_json(
                normalized_output
            )
        else:
            model_output = VerifierModelOutput.model_validate(
                normalized_output
            )
    except ValidationError as error:
        public_detail = build_verifier_schema_error_detail(error)
        raise VerifierError(
            f"Output does not match VerifierModelOutput: {error}",
            public_detail=public_detail,
        ) from error

    model_claims = iter_model_claims(verifier_input.analysis)
    claims_by_path = {
        claim_path: claim for _, claim_path, claim in model_claims
    }
    claim_paths_by_id = {
        claim_id: claim_path for claim_id, claim_path, _ in model_claims
    }
    valid_claim_paths = set(claims_by_path)
    evidence_by_id = {
        item.evidence_id: item for item in verifier_input.evidence
    }
    valid_evidence_ids = {
        item.evidence_id for item in verifier_input.evidence
    }
    legacy_verified_issues: list[VerificationIssue] | None = None
    if model_output.verifications:
        semantic_records = list(model_output.verifications)
        returned_ids = [item.claim_path for item in semantic_records]
        expected_ids = [item[0] for item in model_claims]
        if len(returned_ids) != len(set(returned_ids)):
            raise VerifierError(
                "Verifier model returned duplicate claim paths.",
                public_detail="Verifier 模型为同一 claim 返回了重复验证结果。",
            )
        if set(returned_ids) != set(expected_ids):
            raise VerifierError(
                "Verifier model did not return exactly one result per claim.",
                public_detail=(
                    "Verifier 模型没有为每条输入 claim 返回且只返回一条验证结果。"
                ),
            )
        claim_verifications = [
            map_semantic_verification(
                model_record=item,
                claim_paths_by_id=claim_paths_by_id,
                claims_by_path=claims_by_path,
                evidence_by_id=evidence_by_id,
                valid_evidence_ids=valid_evidence_ids,
                verifier_input=verifier_input,
            )
            for item in semantic_records
        ]
    else:
        # 旧 fixture/供应商响应只列问题：未列出的 claim 按旧契约视为 supported。
        if not model_output.issues:
            return [], []
        legacy_verified_issues = []
        legacy_by_path: dict[str, SemanticVerificationIssue] = {}
        for model_issue in model_output.issues:
            mapped_path = map_claim_path(
                model_issue.claim_path,
                claim_paths_by_id,
                valid_claim_paths,
            )
            validate_model_evidence_ids(
                model_issue.evidence_ids,
                valid_evidence_ids,
            )
            legacy_by_path[mapped_path] = model_issue.model_copy(
                update={"claim_path": mapped_path}
            )

        claim_verifications = []
        for _, claim_path, claim in model_claims:
            model_issue = legacy_by_path.get(claim_path)
            if model_issue is None:
                claim_verifications.append(
                    supported_claim_verification(claim_path, claim)
                )
                continue
            if should_ignore_semantic_issue(
                model_issue=model_issue,
                claims_by_path=claims_by_path,
                evidence_by_id=evidence_by_id,
                verifier_input=verifier_input,
            ):
                claim_verifications.append(
                    supported_claim_verification(claim_path, claim)
                )
                continue
            legacy_verified_issues.append(
                VerificationIssue(
                    issue_type=model_issue.issue_type,
                    claim_path=claim_path,
                    message=model_issue.message,
                    evidence_ids=list(model_issue.evidence_ids),
                    suggested_action=model_issue.suggested_action,
                )
            )
            status = (
                ClaimVerificationStatus.CONFLICTING
                if model_issue.issue_type == "conflicting_evidence"
                else ClaimVerificationStatus.INSUFFICIENT
            )
            claim_verifications.append(
                ClaimVerification(
                    field_path=claim_path,
                    claim=claim.claim,
                    status=status,
                    evidence_ids=list(model_issue.evidence_ids),
                    reason=model_issue.message,
                    suggested_action=model_issue.suggested_action,
                )
            )

    issues = legacy_verified_issues or [
        claim_verification_to_issue(item)
        for item in claim_verifications
        if item.status != ClaimVerificationStatus.SUPPORTED
    ]
    return issues, claim_verifications


def map_semantic_verification(
    model_record: SemanticClaimVerification,
    claim_paths_by_id: dict[str, str],
    claims_by_path: dict[str, AnalysisClaim],
    evidence_by_id: dict[str, Evidence],
    valid_evidence_ids: set[str],
    verifier_input: VerifierInput,
) -> ClaimVerification:
    """把模型短 ID 结果映射回稳定字段路径，并执行确定性纠偏。"""

    claim_path = map_claim_path(
        model_record.claim_path,
        claim_paths_by_id,
        set(claims_by_path),
    )
    validate_model_evidence_ids(
        model_record.evidence_ids,
        valid_evidence_ids,
    )
    claim = claims_by_path[claim_path]
    status = model_record.status

    synthetic_issue_type: SemanticIssueType = (
        "conflicting_evidence"
        if status == ClaimVerificationStatus.CONFLICTING
        else "unsupported_claim"
    )
    synthetic_issue = SemanticVerificationIssue(
        issue_type=synthetic_issue_type,
        claim_path=claim_path,
        message=model_record.reason,
        evidence_ids=list(model_record.evidence_ids),
        suggested_action=model_record.suggested_action,
    )
    if status in {
        ClaimVerificationStatus.CONFLICTING,
        ClaimVerificationStatus.INSUFFICIENT,
    } and should_ignore_semantic_issue(
        model_issue=synthetic_issue,
        claims_by_path=claims_by_path,
        evidence_by_id=evidence_by_id,
        verifier_input=verifier_input,
    ):
        return supported_claim_verification(
            claim_path,
            claim,
            reason=(
                "Structured product data and its cited evidence support the "
                "claim's complete field semantics."
            ),
        )

    return ClaimVerification(
        field_path=claim_path,
        claim=claim.claim,
        status=status,
        evidence_ids=list(model_record.evidence_ids),
        reason=model_record.reason,
        suggested_action=model_record.suggested_action,
    )


def map_claim_path(
    returned_identifier: str,
    claim_paths_by_id: dict[str, str],
    valid_claim_paths: set[str],
) -> str:
    """兼容短 ID 与旧内部路径，同时拒绝模型虚构路径。"""

    mapped_path = claim_paths_by_id.get(returned_identifier)
    if mapped_path is None and returned_identifier in valid_claim_paths:
        mapped_path = returned_identifier
    if mapped_path is None:
        raise VerifierError(
            "Verifier model returned an unknown claim_path: "
            f"{returned_identifier}",
            public_detail=(
                "Verifier 模型返回了不存在的 claim_path："
                f"{returned_identifier}。"
                "这通常表示模型没有逐字复制输入中的 claim_path。"
            ),
        )
    return mapped_path


def validate_model_evidence_ids(
    evidence_ids: Sequence[str],
    valid_evidence_ids: set[str],
) -> None:
    """拒绝未知或重复 Evidence ID。"""

    evidence_id_set = set(evidence_ids)
    unknown_evidence_ids = evidence_id_set - valid_evidence_ids
    if unknown_evidence_ids:
        unknown_text = ", ".join(sorted(unknown_evidence_ids))
        raise VerifierError(
            "Verifier model returned unknown Evidence IDs: " + unknown_text,
            public_detail=(
                "Verifier 模型返回了当前输入中不存在的 Evidence ID："
                f"{unknown_text}。"
            ),
        )
    if len(evidence_ids) != len(evidence_id_set):
        raise VerifierError(
            "Verifier model returned duplicate Evidence IDs.",
            public_detail="Verifier 模型返回了重复 Evidence ID。",
        )


def supported_claim_verification(
    claim_path: str,
    claim: AnalysisClaim,
    reason: str = "The cited evidence supports the complete claim.",
) -> ClaimVerification:
    """构造统一的 supported 记录。"""

    return ClaimVerification(
        field_path=claim_path,
        claim=claim.claim,
        status=ClaimVerificationStatus.SUPPORTED,
        evidence_ids=list(claim.evidence_ids),
        reason=reason,
        suggested_action="No action required.",
    )


def claim_verification_to_issue(
    verification: ClaimVerification,
) -> VerificationIssue:
    """为旧工作流重试接口保留 issue 视图。"""

    issue_types: dict[ClaimVerificationStatus, VerificationIssueType] = {
        ClaimVerificationStatus.PARTIALLY_SUPPORTED: "partially_supported",
        ClaimVerificationStatus.CONFLICTING: "conflicting_evidence",
        ClaimVerificationStatus.INSUFFICIENT: "insufficient_evidence",
        ClaimVerificationStatus.STALE: "stale_evidence",
        ClaimVerificationStatus.INVALID_SCOPE: "invalid_scope",
        ClaimVerificationStatus.SUPPORTED: "unsupported_claim",
    }
    return VerificationIssue(
        issue_type=issue_types[verification.status],
        claim_path=verification.field_path,
        message=verification.reason,
        evidence_ids=list(verification.evidence_ids),
        suggested_action=verification.suggested_action,
    )


def should_ignore_semantic_issue(
    model_issue: SemanticVerificationIssue,
    claims_by_path: dict[str, AnalysisClaim],
    evidence_by_id: dict[str, Evidence],
    verifier_input: VerifierInput,
) -> bool:
    """过滤模型对标准化 claim 的逐字误报，其他语义问题仍保留。"""

    claim = claims_by_path.get(model_issue.claim_path)
    if claim is None:
        return False

    # 如果模型 issue 自己带了 Evidence ID，但这些 ID 与当前 claim 的引用对不上，
    # 说明该 issue 可能已经和规范化后的 claim_path 错位，不能按误报过滤。
    if model_issue.evidence_ids:
        issue_evidence_ids = set(model_issue.evidence_ids)
        claim_evidence_ids = set(claim.evidence_ids)
        if not issue_evidence_ids.intersection(claim_evidence_ids):
            return False

    if is_supported_standard_pricing_claim(
        claim_path=model_issue.claim_path,
        claim=claim,
        evidence_by_id=evidence_by_id,
        verifier_input=verifier_input,
    ):
        # 完整模型+方向+费率已经与结构化画像一致时，不能因同页其他数字误判冲突。
        return True

    if model_issue.issue_type != "unsupported_claim":
        return False

    if is_conservative_scope_claim(model_issue.claim_path, claim):
        return True

    if is_soft_interpretation_issue(
        claim_path=model_issue.claim_path,
        claim=claim,
    ):
        return True

    if is_supported_standard_feature_claim(
        claim_path=model_issue.claim_path,
        claim=claim,
        evidence_by_id=evidence_by_id,
    ):
        return True

    return False


def is_soft_interpretation_issue(
    claim_path: str,
    claim: AnalysisClaim,
) -> bool:
    """个人项目中对解释型章节的 unsupported 误报更宽容。"""

    if claim.claim_type != "interpretation":
        return False
    if not claim.evidence_ids:
        return False
    if contains_strong_evaluation_language(claim.claim):
        return False

    for path_prefix in SOFT_INTERPRETATION_PATH_PREFIXES:
        if claim_path.startswith(path_prefix):
            return True

    normalized_claim = claim.claim.lower()
    if claim_path == "conclusion" and normalized_claim.startswith(
        "based on the supplied profiles"
    ):
        return True

    return False


def contains_strong_evaluation_language(claim_text: str) -> bool:
    """识别仍应交给 Verifier 阻断的强评价或胜负判断。"""

    normalized_claim = claim_text.lower()
    for term in STRONG_EVALUATION_TERMS:
        if term in normalized_claim:
            return True

    return False


def is_conservative_scope_claim(
    claim_path: str,
    claim: AnalysisClaim,
) -> bool:
    """判断结论是否只是系统输入范围说明，而不是产品事实。"""

    if claim_path != "conclusion":
        return False

    normalized_claim = claim.claim.lower()
    return normalized_claim.startswith(
        "the comparison is limited to the supplied evidence"
    ) or normalized_claim.startswith(
        "the comparison is limited to the supplied product profiles"
    )


def is_supported_standard_feature_claim(
    claim_path: str,
    claim: AnalysisClaim,
    evidence_by_id: dict[str, Evidence],
) -> bool:
    """识别 “Product mentions Feature” 是否已被引用证据支持。"""

    if not claim_path.startswith("features["):
        return False
    if claim.claim_type != "fact":
        return False
    if len(claim.product_names) != 1:
        return False

    feature_name = extract_mentions_feature_name(claim)
    if feature_name is None:
        return False

    evidence_text = build_claim_evidence_text(claim, evidence_by_id)
    return evidence_supports_feature_phrase(evidence_text, feature_name)


def evidence_supports_feature_phrase(
    evidence_text: str,
    feature_name: str,
) -> bool:
    """判断 Evidence 是否支持标准化后的功能名，允许轻微词形变化。"""

    if text_contains_phrase_or_tokens(evidence_text, feature_name):
        return True

    phrase_tokens = canonicalize_significant_tokens(feature_name)
    if len(phrase_tokens) < FEATURE_PHRASE_MIN_TOKEN_COUNT:
        return False

    evidence_tokens = set(canonicalize_significant_tokens(evidence_text))
    if not evidence_tokens:
        return False

    matched_tokens = [
        phrase_token
        for phrase_token in phrase_tokens
        if phrase_token in evidence_tokens
    ]
    match_count = len(matched_tokens)
    coverage = match_count / len(phrase_tokens)

    # 长功能名允许少量泛化词没逐字出现，例如 management/capability；
    # 但至少要覆盖三个关键信息词，避免 “Rovo AI features” 只因 AI/features 命中而通过。
    return (
        match_count >= FEATURE_PHRASE_MIN_MATCH_COUNT
        and coverage >= FEATURE_PHRASE_MIN_COVERAGE
    )


def extract_mentions_feature_name(claim: AnalysisClaim) -> str | None:
    """从标准化 feature claim 中取出功能名。"""

    product_name = claim.product_names[0]
    claim_text = claim.claim.strip().rstrip(".")
    normalized_claim_text = claim_text.lower()
    normalized_prefix = f"{product_name.lower()} mentions "
    if not normalized_claim_text.startswith(normalized_prefix):
        return None

    return claim_text[len(normalized_prefix) :]


def is_supported_standard_pricing_claim(
    claim_path: str,
    claim: AnalysisClaim,
    evidence_by_id: dict[str, Evidence],
    verifier_input: VerifierInput,
) -> bool:
    """识别标准化 pricing claim 是否已被套餐名和价格证据支持。"""

    if not claim_path.startswith("pricing["):
        return False
    if claim.claim_type != "fact":
        return False

    matching_profiles = [
        profile
        for profile in verifier_input.product_profiles
        if profile.product_name in claim.product_names
    ]
    canonical_claims = build_fallback_pricing_claims(matching_profiles)
    normalized_claim = normalize_text(claim.claim)
    for canonical_claim in canonical_claims:
        if normalize_text(canonical_claim.claim) != normalized_claim:
            continue
        if set(canonical_claim.evidence_ids).intersection(claim.evidence_ids):
            return True

    evidence_text = build_claim_evidence_text(claim, evidence_by_id)
    list_match = PRICING_LIST_PATTERN.search(claim.claim)
    if list_match is not None:
        plan_name = list_match.group("plan")
        price_text = list_match.group("price")
        billing_text = list_match.group("billing")
        has_plan = text_contains_phrase_or_tokens(evidence_text, plan_name)
        has_price = evidence_supports_price_text(evidence_text, price_text)
        has_billing = evidence_supports_billing_text(
            support_text=f"{price_text}\n{evidence_text}",
            billing_text=billing_text,
        )
        return has_plan and has_price and has_billing

    without_price_match = PRICING_WITHOUT_PRICE_PATTERN.search(claim.claim)
    if without_price_match is None:
        return False

    # “without a public price in the supplied profile” 是 Extractor 对缺失价格的
    # 结构化表达；只要套餐名存在，就不要求来源逐字声明“没有公开价格”。
    plan_name = without_price_match.group("plan")
    billing_text = without_price_match.group("billing")
    has_plan = text_contains_phrase_or_tokens(evidence_text, plan_name)
    has_billing = evidence_supports_billing_text(
        support_text=evidence_text,
        billing_text=billing_text,
    )
    return has_plan and has_billing


def evidence_supports_billing_text(
    support_text: str,
    billing_text: str | None,
) -> bool:
    """判断可选 billing 片段是否有证据支持。"""

    if billing_text is None:
        return True
    if not is_supported_billing_cycle_text(billing_text):
        return False

    return billing_cycle_is_supported_by_text(
        support_text=support_text,
        billing_cycle=billing_text,
    )


def evidence_supports_price_text(
    evidence_text: str,
    price_text: str,
) -> bool:
    """判断 Evidence 是否支持标准化价格文本。"""

    normalized_evidence_text = normalize_text(evidence_text)
    normalized_price_text = normalize_text(price_text)
    if is_free_price_text(price_text):
        return is_free_price_text(evidence_text)

    price_tokens = tokenize_text(price_text)
    if (
        normalized_price_text
        and len(price_tokens) > 1
        and normalized_price_text in normalized_evidence_text
    ):
        return True

    if "custom pricing" in normalized_price_text:
        return "custom pricing" in normalized_evidence_text

    evidence_numbers = {
        number.replace(",", "")
        for number in PRICE_NUMBER_PATTERN.findall(evidence_text)
    }
    price_numbers = PRICE_NUMBER_PATTERN.findall(price_text)
    for price_number in price_numbers:
        normalized_number = price_number.replace(",", "")
        if normalized_number in evidence_numbers:
            return True
        if normalized_number == "0" and "free" in normalized_evidence_text:
            return True

    return False


def build_claim_evidence_text(
    claim: AnalysisClaim,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """拼接 claim 自身引用的 Evidence 文本，供确定性兜底判断。"""

    text_parts: list[str] = []
    for evidence_id in claim.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue

        text_parts.append(evidence.title)
        text_parts.append(evidence.snippet)
        if evidence.raw_content:
            text_parts.append(evidence.raw_content)

    return "\n".join(text_parts)


def text_contains_phrase_or_tokens(text: str, phrase: str) -> bool:
    """判断文本是否包含短语，或覆盖短语中的全部关键信息词。"""

    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True

    phrase_tokens = canonicalize_significant_tokens(phrase)
    if not phrase_tokens:
        return False

    text_tokens = set(canonicalize_significant_tokens(text))
    for phrase_token in phrase_tokens:
        if phrase_token not in text_tokens:
            return False

    return True


def normalize_text(text: str) -> str:
    """把文本转成适合短语包含判断的英文小写形式。"""

    return " ".join(tokenize_text(text))


def tokenize_text(text: str) -> list[str]:
    """提取英文和数字 token，避免标点、链接格式影响匹配。"""

    return TEXT_TOKEN_PATTERN.findall(text.lower())


def canonicalize_significant_tokens(text: str) -> list[str]:
    """提取关键信息词，并做很轻量的英文词形归一。"""

    canonical_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in tokenize_text(text):
        if token in TEXT_STOP_WORDS:
            continue

        canonical_token = canonicalize_text_token(token)
        if not canonical_token or canonical_token in seen_tokens:
            continue

        seen_tokens.add(canonical_token)
        canonical_tokens.append(canonical_token)

    return canonical_tokens


def canonicalize_text_token(token: str) -> str:
    """把常见英文词形变体映射到同一 token。"""

    alias = TEXT_TOKEN_ALIASES.get(token)
    if alias is not None:
        return alias

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]

    return token


def iter_claims_with_paths(
    analysis: CompetitiveAnalysis,
) -> list[tuple[str, AnalysisClaim]]:
    """按章节顺序返回稳定 claim 路径和对应对象。"""

    claims_with_paths: list[tuple[str, AnalysisClaim]] = []

    sections = [
        ("positioning", analysis.positioning),
        ("features", analysis.features),
        ("pricing", analysis.pricing),
        ("dimension_comparisons", analysis.dimension_comparisons),
        ("opportunities", analysis.opportunities),
    ]
    for section_name, claims in sections:
        for index, claim in enumerate(claims):
            claim_path = f"{section_name}[{index}]"
            claims_with_paths.append((claim_path, claim))

    claims_with_paths.append(("conclusion", analysis.conclusion))
    for index, recommendation in enumerate(analysis.recommendations):
        claims_with_paths.append(
            (
                f"recommendations[{index}]",
                AnalysisClaim(
                    claim=(
                        f"{recommendation.target_scenario}: "
                        f"{recommendation.tradeoff_or_gap} "
                        f"Recommended action: "
                        f"{recommendation.recommended_action}"
                    ),
                    claim_type="interpretation",
                    product_names=list(recommendation.product_names),
                    evidence_ids=list(recommendation.evidence_ids),
                ),
            )
        )
    for assessment_index, assessment in enumerate(analysis.product_assessments):
        for finding_index, finding in enumerate(assessment.strengths):
            claims_with_paths.append(
                (
                    f"product_assessments[{assessment_index}].strengths[{finding_index}]",
                    finding,
                )
            )
        for finding_index, finding in enumerate(assessment.shortcomings):
            claims_with_paths.append(
                (
                    f"product_assessments[{assessment_index}].shortcomings[{finding_index}]",
                    finding,
                )
            )
    for index, recommendation in enumerate(analysis.scenario_recommendations):
        if not recommendation.evidence_ids:
            continue
        product_names = (
            [recommendation.recommended_product]
            if recommendation.recommended_product in analysis.products
            else list(analysis.products)
        )
        claims_with_paths.append(
            (
                f"scenario_recommendations[{index}]",
                AnalysisClaim(
                    claim=(
                        f"{recommendation.scenario}: "
                        f"{recommendation.recommendation_reason}"
                    ),
                    claim_type="interpretation",
                    product_names=product_names,
                    evidence_ids=list(recommendation.evidence_ids),
                ),
            )
        )
    for index, opportunity in enumerate(analysis.market_opportunities):
        if not opportunity.evidence_ids:
            continue
        claims_with_paths.append(
            (
                f"market_opportunities[{index}]",
                AnalysisClaim(
                    claim=(
                        f"{opportunity.title}: {opportunity.competitor_status} "
                        f"{opportunity.market_gap}"
                    ),
                    claim_type="interpretation",
                    product_names=list(opportunity.product_names),
                    evidence_ids=list(opportunity.evidence_ids),
                ),
            )
        )
    return claims_with_paths


def iter_model_claims(
    analysis: CompetitiveAnalysis,
) -> list[tuple[str, str, AnalysisClaim]]:
    """为模型生成短 ID，同时保留代码内部的真实 claim 路径。"""

    model_claims: list[tuple[str, str, AnalysisClaim]] = []
    for index, (claim_path, claim) in enumerate(
        iter_claims_with_paths(analysis),
        start=1,
    ):
        claim_id = f"C{index:03d}"
        model_claims.append((claim_id, claim_path, claim))

    return model_claims


def build_verification_result(
    issues: Sequence[VerificationIssue],
    analysis: CompetitiveAnalysis | None = None,
    claim_verifications: Sequence[ClaimVerification] = (),
) -> VerificationResult:
    """根据 issue 与逐 claim 状态生成一致的通过状态和重试意图。"""

    issue_list = list(issues)
    verification_list = list(claim_verifications)
    if analysis is not None and not verification_list:
        issues_by_path: dict[str, list[VerificationIssue]] = {}
        for issue in issue_list:
            issues_by_path.setdefault(issue.claim_path, []).append(issue)
        known_claim_paths: set[str] = set()
        for claim_path, claim in iter_claims_with_paths(analysis):
            known_claim_paths.add(claim_path)
            path_issues = issues_by_path.get(claim_path, [])
            if not path_issues:
                verification_list.append(
                    supported_claim_verification(claim_path, claim)
                )
                continue
            primary_issue = path_issues[0]
            status = issue_type_to_claim_status(primary_issue.issue_type)
            verification_list.append(
                ClaimVerification(
                    field_path=claim_path,
                    claim=claim.claim,
                    status=status,
                    evidence_ids=list(primary_issue.evidence_ids),
                    reason=primary_issue.message,
                    suggested_action=primary_issue.suggested_action,
                )
            )
        for issue in issue_list:
            if issue.claim_path in known_claim_paths:
                continue
            verification_list.append(
                ClaimVerification(
                    field_path=issue.claim_path,
                    claim=issue.message,
                    status=issue_type_to_claim_status(issue.issue_type),
                    evidence_ids=list(issue.evidence_ids),
                    reason=issue.message,
                    suggested_action=issue.suggested_action,
                )
            )

    has_non_supported_claim = any(
        item.status != ClaimVerificationStatus.SUPPORTED
        for item in verification_list
    )
    retry_recommended = any(
        issue.issue_type not in NON_RETRYABLE_ISSUE_TYPES
        for issue in issue_list
    )
    return VerificationResult(
        passed=not issue_list and not has_non_supported_claim,
        issues=issue_list,
        claim_verifications=verification_list,
        retry_recommended=retry_recommended,
    )


def issue_type_to_claim_status(
    issue_type: VerificationIssueType,
) -> ClaimVerificationStatus:
    """把旧确定性 issue 映射为新的六态验证。"""

    if issue_type == "conflicting_evidence":
        return ClaimVerificationStatus.CONFLICTING
    if issue_type == "partially_supported":
        return ClaimVerificationStatus.PARTIALLY_SUPPORTED
    if issue_type == "stale_evidence":
        return ClaimVerificationStatus.STALE
    if issue_type in {
        "wrong_product_evidence",
        "out_of_scope_evidence",
        "scope_level_conflict",
        "invalid_scope",
    }:
        return ClaimVerificationStatus.INVALID_SCOPE
    return ClaimVerificationStatus.INSUFFICIENT
