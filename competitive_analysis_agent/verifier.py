"""Verifier 节点：检查分析引用，并评审 claim 的证据支持情况。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from competitive_analysis_agent.analyst import (
    AnalysisClaim,
    CompetitiveAnalysis,
)
from competitive_analysis_agent.model_io import (
    log_model_error,
    log_model_request,
    log_model_response,
)
from competitive_analysis_agent.pricing_utils import (
    billing_cycle_is_supported_by_text,
    is_free_price_text,
    is_supported_billing_cycle_text,
)
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    EvidenceId,
    RequiredText,
)


VERIFIER_SYSTEM_PROMPT = """
你是竞品分析流程中的 Verifier。

你的唯一职责是检查每条 claim 是否被给定 Evidence 支持，或是否与 Evidence 冲突。
不得改写分析，不得搜索网页，不得使用外部知识。

用户消息会提供：
1. 带稳定 claim_path 的 claims；
2. 全部 Evidence。

检查规则：
1. unsupported_claim：claim 的核心事实无法从相关 Evidence 中得到支持。
2. conflicting_evidence：claim 与 Evidence 明确矛盾，或相关 Evidence 之间存在会影响
   该 claim 的冲突。
3. interpretation 可以进行合理归纳，不要求逐字出现在 Evidence 中；只有它引入
   未提供的新事实或与 Evidence 冲突时才报告。对输入范围的保守说明，例如
   “现有资料不足以比较某项能力”，不属于新增产品事实；当输入确实没有提供该项
   可比信息时，不要仅因为这类说明没有直接引用而报告 unsupported_claim。
   对 positioning 和 opportunities 这类解释型章节，除非出现明显新增硬事实、
   强评价或证据冲突，否则不要按事实句逐字要求 Evidence 支持。
4. 只报告真实问题，不要把措辞差异当成错误。
   - 对 “Product mentions X” 这类标准化 claim，只要相关 Evidence 出现 X
     或覆盖 X 的关键词，就视为支持，不要求 Evidence 逐字写出 “mentions”。
   - 对价格 claim，只要相关 Evidence 同时出现套餐名和价格/Custom pricing/Free，
     就视为支持，不要求 Evidence 逐字写出 “lists the plan at”。
   - 对 “comparison is limited to supplied evidence/product profiles” 这类范围说明，
     不要当作产品事实要求 Evidence 逐字支持。
5. claim_path 必须逐字复制用户消息中的路径。
6. evidence_ids 只能使用用户消息中真实存在、与问题直接相关的 ID。
   如果问题正是 claim 没有任何直接支持证据，可以返回空列表。
7. suggested_action 应说明 Analyst 应如何修改或收窄 claim。
8. 如果所有 claim 都有支持且没有冲突，返回空 issues。
9. 只输出 JSON，不要添加 Markdown 或解释。
10. 输出格式：
{
  "issues": [
    {
      "issue_type": "unsupported_claim",
      "claim_path": "features[0]",
      "message": "...",
      "evidence_ids": ["E1"],
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
]
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
SOFT_INTERPRETATION_PATH_PREFIXES = ("positioning[", "opportunities[")
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

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "VerifierInput":
        """避免同一个 Evidence ID 指向多个来源。"""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique.")

        return self


class VerificationIssue(ContractModel):
    """描述一个可定位、可修复的验证问题。"""

    issue_type: VerificationIssueType
    claim_path: RequiredText
    message: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    suggested_action: RequiredText


class VerificationResult(ContractModel):
    """返回验证是否通过、问题清单和后续重试意图。"""

    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    retry_recommended: bool

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "VerificationResult":
        """确保布尔状态与问题列表保持一致。"""

        expected_passed = not self.issues
        expected_retry = bool(self.issues)
        if self.passed != expected_passed:
            raise ValueError("passed must be true only when issues is empty.")
        if self.retry_recommended != expected_retry:
            raise ValueError(
                "retry_recommended must match whether issues exist."
            )

        return self


class SemanticVerificationIssue(ContractModel):
    """约束模型只能报告语义支持或证据冲突问题。"""

    issue_type: SemanticIssueType
    claim_path: RequiredText
    message: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    suggested_action: RequiredText


class VerifierModelOutput(ContractModel):
    """约束语义评审模型返回结构化 issue 列表。"""

    issues: list[SemanticVerificationIssue] = Field(default_factory=list)


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
            return build_verification_result(deterministic_issues)

        messages = build_verifier_messages(verifier_input)
        raw_output = self._invoke_model(messages)
        semantic_issues = validate_semantic_output(
            raw_output=raw_output,
            verifier_input=verifier_input,
        )
        return build_verification_result(semantic_issues)

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
    """用普通代码检查不存在、错产品和缺少逐产品支持的引用。"""

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

    return issues


def build_verifier_messages(
    verifier_input: VerifierInput,
) -> list[dict[str, str]]:
    """把稳定 claim 路径和 Evidence 转换成语义评审消息。"""

    claim_records: list[dict[str, object]] = []
    for claim_path, claim in iter_claims_with_paths(
        verifier_input.analysis
    ):
        claim_record = claim.model_dump(mode="json")
        claim_record["claim_path"] = claim_path
        claim_records.append(claim_record)

    payload = {
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
                and "issues" in normalized_nested_output
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
        "期望顶层 JSON 为 {\"issues\": [...]}；每个 issue 必须包含 "
        "issue_type、claim_path、message、evidence_ids、suggested_action。"
    )


def validate_semantic_output(
    raw_output: object,
    verifier_input: VerifierInput,
) -> list[VerificationIssue]:
    """校验模型 issue 的结构、claim 路径和 Evidence ID。"""

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

    claims_by_path = dict(iter_claims_with_paths(verifier_input.analysis))
    valid_claim_paths = set(claims_by_path)
    evidence_by_id = {
        item.evidence_id: item for item in verifier_input.evidence
    }
    valid_evidence_ids = {
        item.evidence_id for item in verifier_input.evidence
    }
    verified_issues: list[VerificationIssue] = []

    for model_issue in model_output.issues:
        if model_issue.claim_path not in valid_claim_paths:
            raise VerifierError(
                "Verifier model returned an unknown claim_path: "
                f"{model_issue.claim_path}",
                public_detail=(
                    "Verifier 模型返回了不存在的 claim_path："
                    f"{model_issue.claim_path}。"
                    "这通常表示模型没有逐字复制输入中的 claim_path。"
                ),
            )

        issue_evidence_ids = set(model_issue.evidence_ids)
        unknown_evidence_ids = issue_evidence_ids - valid_evidence_ids
        if unknown_evidence_ids:
            unknown_text = ", ".join(sorted(unknown_evidence_ids))
            raise VerifierError(
                "Verifier model returned unknown Evidence IDs: "
                f"{unknown_text}",
                public_detail=(
                    "Verifier 模型返回了当前输入中不存在的 Evidence ID："
                    f"{unknown_text}。"
                    "请检查 Verifier 输出是否引用了资料来源表以外的 ID。"
                ),
            )

        if len(model_issue.evidence_ids) != len(issue_evidence_ids):
            raise VerifierError(
                "Verifier model returned duplicate Evidence IDs.",
                public_detail=(
                    "Verifier 模型返回了重复 Evidence ID。"
                    "请检查同一个 issue 的 evidence_ids 是否有重复项。"
                ),
            )

        if should_ignore_semantic_issue(
            model_issue=model_issue,
            claims_by_path=claims_by_path,
            evidence_by_id=evidence_by_id,
        ):
            continue

        verified_issues.append(
            VerificationIssue(
                issue_type=model_issue.issue_type,
                claim_path=model_issue.claim_path,
                message=model_issue.message,
                evidence_ids=model_issue.evidence_ids,
                suggested_action=model_issue.suggested_action,
            )
        )

    return verified_issues


def should_ignore_semantic_issue(
    model_issue: SemanticVerificationIssue,
    claims_by_path: dict[str, AnalysisClaim],
    evidence_by_id: dict[str, Evidence],
) -> bool:
    """过滤模型对标准化 claim 的逐字误报，其他语义问题仍保留。"""

    if model_issue.issue_type != "unsupported_claim":
        return False

    claim = claims_by_path.get(model_issue.claim_path)
    if claim is None:
        return False

    if is_conservative_scope_claim(model_issue.claim_path, claim):
        return True

    # 如果模型 issue 自己带了 Evidence ID，但这些 ID 与当前 claim 的引用对不上，
    # 说明该 issue 可能已经和规范化后的 claim_path 错位，不能按误报过滤。
    if model_issue.evidence_ids:
        issue_evidence_ids = set(model_issue.evidence_ids)
        claim_evidence_ids = set(claim.evidence_ids)
        if not issue_evidence_ids.intersection(claim_evidence_ids):
            return False

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

    if is_supported_standard_pricing_claim(
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
) -> bool:
    """识别标准化 pricing claim 是否已被套餐名和价格证据支持。"""

    if not claim_path.startswith("pricing["):
        return False
    if claim.claim_type != "fact":
        return False

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
        ("opportunities", analysis.opportunities),
    ]
    for section_name, claims in sections:
        for index, claim in enumerate(claims):
            claim_path = f"{section_name}[{index}]"
            claims_with_paths.append((claim_path, claim))

    claims_with_paths.append(("conclusion", analysis.conclusion))
    return claims_with_paths


def build_verification_result(
    issues: Sequence[VerificationIssue],
) -> VerificationResult:
    """根据 issue 是否为空生成一致的通过状态和重试意图。"""

    issue_list = list(issues)
    has_issues = bool(issue_list)
    return VerificationResult(
        passed=not has_issues,
        issues=issue_list,
        retry_recommended=has_issues,
    )
