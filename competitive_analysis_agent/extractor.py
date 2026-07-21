"""Extractor 节点：只根据给定证据生成可追溯的产品画像。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from competitive_analysis_agent.model_io import (
    log_model_error,
    log_model_request,
    log_model_response,
)
from competitive_analysis_agent.pricing_utils import (
    detect_billing_cycle_category,
    is_custom_pricing_text,
    is_free_price_text,
    normalize_billing_cycle_value,
    normalize_price_value,
)
from competitive_analysis_agent.schemas import (
    ContractModel,
    DimensionFinding,
    Evidence,
    MarketDefinition,
    PricingPlan,
    ProductProfile,
)


EXTRACTOR_SYSTEM_PROMPT = """
你是竞品分析流程中的 Extractor，只能根据给定 Evidence 生成 ProductProfile。

通用要求：
1. 只能使用用户消息中的 Evidence，不得使用常识或外部资料；product_name 必须逐字复制。
2. Evidence 是搜索召回的补充上下文。只提取 scope_status=in_scope 且符合
   market_definition 的事实，并综合 title、snippet、raw_content 和相邻表格文本。
3. 官方来源优先于第三方来源。冲突时不猜选，字段返回 missing。
4. 资料没有明确说明时，文本和数值使用 null，列表使用 []，SupportStatus 使用
   missing；不得用其他字段的文本占位。
5. 每个已填字段都在 field_evidence 中返回 field_path、evidence_id、
   quote、rationale 和 confidence；缺失字段不得返回 field_evidence。quote 必须逐字来自对应 Evidence；
   quote 不得使用省略号或截断文本；无法逐字引用时整条 field_evidence 删除。
   rationale 说明原文为何回答该字段。positioning 和 target_users 的 confidence
   至少 0.85，其他字段至少 0.60。模型字段路径使用
   models.<model_name>.<field>，价格字段再加 pricing.<price_type>。

字段定义、正例和反例：
6. positioning 定义：产品是什么、所属市场和核心价值类别。
   正例：官网标题、产品标语或产品概览中的 "AI developer platform"、
   "workspace for knowledge and collaboration"。
   反例："Authenticate with access tokens"、使用步骤、页面导航、模型列表、
   context window、max output、RPM/TPM/RPD、价格、套餐额度或认证说明。
7. target_users 定义：来源明确写出的目标群体。target_users 可以来自 use case、
   customer、team、enterprise、small business 页面中的 "built for..."、
   "designed for..." 或 "used by..." 描述。
   正例："Built for enterprise engineering teams"。
   反例：看到营销功能后猜测 "marketing teams"；不得从功能名称反推用户。
8. features 定义：用户或开发者可实际使用的产品能力。
   正例：function calling、structured outputs、shared workspaces。
   反例：页面导航、认证步骤、价格、模型名称或限流数字。
9. context_window_tokens 只保存上下文窗口；max_output_tokens 只保存最大输出长度。
   "1M context / 128k max output" 必须分别写为 1000000 和 128000。
10. rate_limits 按 metric 分类：RPM -> requests_per_minute，
    TPM -> tokens_per_minute，RPD -> requests_per_day。限流不能写进定位或 Token 上限。
11. ModelPricing 分类：input -> input_price，output -> output_price，
    cached input -> cached_input_price，audio input -> audio_input_price，
    audio output -> audio_output_price，Batch -> batch_* 或
    batch_discount_percent。同一模型只能返回一个 ModelProfile，所有价格合并到该模型
    唯一的 pricing 对象，禁止创建 "model input"、"model output" 等模型名。
12. PriceRate 必须标准化为 USD、per_quantity=1000000、unit=token；可把原始的
    USD/K tokens 或 USD/token 作精确数量级换算，但不得猜测汇率。condition 保存
    适用条件，max_context_tokens 保存明确的上下文价格档位上限；effective_from、
    effective_to 和 evidence_ids 用于有效期。同一价格类型有多个有效期时返回 PriceRate 数组；
    同一时间的多个价格仅可在 condition 明确且互不相同时返回数组。没有明确日期时使用 null，
    禁止根据采集时间猜测有效期。

结构要求：
13. 每个 feature、PriceRate、RateLimit 和 field_evidence 必须引用真实且直接支持它的
    evidence_id，不得虚构或跨产品引用。
14. raw_content 是 Researcher 裁剪的价格页正文片段。可以读取明确的模型名、金额、
    单位、价格类型、条件和有效期，但不得换算或跨模型拼接。
15. dimension_findings 覆盖全部 core_dimensions；有事实时写 facts 和 evidence_ids，
    资料不足时两者均为 []。
16. API 价格只写入 models[].pricing；subscription 价格继续写入顶层 pricing。
17. features 项必须包含 name、description、evidence_ids。subscription pricing 项
    必须包含 plan_name、price、unit、billing_cycle、service_level、threshold、
    main_limits、evidence_ids；feature 缺少 description 时整条删除，未知值使用 null 或 []。
18. 只输出 JSON：
   {"profile": {"product_name": "...", "positioning": null,
   "target_users": [], "models": [], "features": [], "dimension_findings": [],
   "field_evidence": [], "pricing": [], "strengths": [], "limitations": []}}
19. models[].source_evidence 必须是从 Evidence 逐字段复制的对象数组，例如
    [{"evidence_id":"E1","title":"...","url":"https://...","source_type":"official",
    "collected_at":"2026-01-01T00:00:00+00:00"}]；不得只输出 ["E1"]。
""".strip()
POSITIONING_KEYWORDS = (
    "workspace",
    "platform",
    "source of truth",
    "one place",
    "collaboration",
    "collaborate",
    "knowledge",
)
PLAN_LEVEL_POSITIONING_MARKERS = (
    "daily users",
    "collaborate often",
    "most tasks",
    "plan",
    "pricing",
    "subscription",
    "billing",
    "billed",
    "per month",
    "per year",
    "seat",
    "seats",
    "usage",
    "credits",
)
PRICING_PAGE_MARKERS = (
    "pricing",
    "plans",
    "billing",
    "subscription",
)
POSITIONING_NEGATIVE_MARKERS = (
    "authenticate",
    "authentication",
    "access token",
    "api key",
    "sign in",
    "log in",
    "quickstart",
    "get started",
    "navigation",
    "available models",
    "model list",
    "rate limit",
    "rpm",
    "tpm",
    "rpd",
    "context window",
    "max output",
)
POSITIONING_TOPICS = {"positioning", "overview", "product_overview"}
FEATURE_NEGATIVE_MARKERS = (
    "authenticate with",
    "click ",
    "select ",
    "go to ",
    "sign in",
    "log in",
    "page navigation",
    "rate limit",
    "rpm",
    "tpm",
    "rpd",
)
TARGET_USER_RELATION_MARKERS = (
    "built for",
    "designed for",
    "intended for",
    "used by",
    "serves",
    "aimed at",
    "targeted at",
    "面向",
    "适用于",
    "专为",
    "目标用户",
)
DEFAULT_FIELD_CONFIDENCE = 0.60
HIGH_LEVEL_FIELD_CONFIDENCE = 0.85
EXTRACTOR_TITLE_MAX_CHARS = 160
EXTRACTOR_SNIPPET_MAX_CHARS = 700
EXTRACTOR_RAW_CONTENT_MAX_CHARS = 1400
EXTRACTOR_MAX_EVIDENCE_PER_PRODUCT = 6
EXTRACTOR_MAX_EVIDENCE_PER_TOPIC = 3
EXTRACTOR_DEEP_CONTEXT_TOPICS = ("api_pricing", "pricing")
EXTRACTOR_TOPIC_PRIORITY = (
    "api_pricing",
    "model_capabilities",
    "developer_platform",
    "usage_limits",
    "pricing",
    "features",
    "positioning",
    "target_users",
)
PricingScopeClassification = Literal[
    "api_pricing",
    "non_api_pricing",
    "ambiguous",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class PricingScopeRules:
    """保存 API 与订阅价格的确定性分类关键词。"""

    api_markers: tuple[str, ...]
    non_api_markers: tuple[str, ...]


GENERIC_PRICING_SCOPE_RULES = PricingScopeRules(
    api_markers=(
        "api pricing",
        "developer platform",
        "model pricing",
        "input token",
        "output token",
        "tokens",
        "1m tokens",
        "per million tokens",
    ),
    non_api_markers=(
        "subscription",
        "per user",
        "per member",
        "per seat",
        "per month",
        "billed monthly",
        "plus plan",
        "pro plan",
        "max plan",
        "standard plan",
        "starter plan",
        "basic plan",
        "premium plan",
        "team plan",
        "business plan",
        "enterprise plan",
    ),
)


API_PRICING_SCOPE_RULES_BY_PRODUCT_KEY = {
    "chatgpt": PricingScopeRules(
        api_markers=(
            "api pricing",
            "openai api",
            "developer platform",
            "platform.openai.com",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "chatgpt plus",
            "chatgpt pro",
            "chatgpt team",
            "chatgpt business",
            "chatgpt enterprise",
            "plus plan",
            "pro plan",
            "team plan",
            "business plan",
            "enterprise plan",
            "subscription",
            "per month",
            "per seat",
            "billed monthly",
        ),
    ),
    "openai": PricingScopeRules(
        api_markers=(
            "api pricing",
            "openai api",
            "developer platform",
            "platform.openai.com",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "chatgpt plus",
            "chatgpt pro",
            "chatgpt team",
            "chatgpt business",
            "chatgpt enterprise",
            "plus plan",
            "pro plan",
            "team plan",
            "business plan",
            "enterprise plan",
            "subscription",
            "per month",
            "per seat",
            "billed monthly",
        ),
    ),
    "claude": PricingScopeRules(
        api_markers=(
            "claude api",
            "anthropic api",
            "api pricing",
            "console",
            "console.anthropic.com",
            "docs.anthropic.com",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "claude pro",
            "claude max",
            "claude team",
            "pro plan",
            "max plan",
            "team plan",
            "daily users",
            "collaborate often",
            "monthly subscription",
            "subscription",
            "per month",
            "per seat",
            "billed monthly",
            "most tasks",
        ),
    ),
    "anthropic": PricingScopeRules(
        api_markers=(
            "claude api",
            "anthropic api",
            "api pricing",
            "console",
            "console.anthropic.com",
            "docs.anthropic.com",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "claude pro",
            "claude max",
            "claude team",
            "pro plan",
            "max plan",
            "team plan",
            "daily users",
            "collaborate often",
            "monthly subscription",
            "subscription",
            "per month",
            "per seat",
            "billed monthly",
            "most tasks",
        ),
    ),
    "gemini": PricingScopeRules(
        api_markers=(
            "gemini api",
            "google ai api",
            "api pricing",
            "ai.google.dev",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "gemini app",
            "workspace gemini",
            "google workspace",
            "gemini business",
            "gemini enterprise",
            "google home premium",
            "google home",
            "home premium",
            "google one ai premium",
            "veo",
            "nest",
            "consumer subscription",
            "subscription",
            "per user per month",
            "per month",
        ),
    ),
    "google ai": PricingScopeRules(
        api_markers=(
            "gemini api",
            "google ai api",
            "api pricing",
            "ai.google.dev",
            "model pricing",
            "input tokens",
            "output tokens",
            "token",
            "tokens",
            "1m tokens",
        ),
        non_api_markers=(
            "gemini app",
            "workspace gemini",
            "google workspace",
            "gemini business",
            "gemini enterprise",
            "google home premium",
            "google home",
            "home premium",
            "google one ai premium",
            "veo",
            "nest",
            "consumer subscription",
            "subscription",
            "per user per month",
            "per month",
        ),
    ),
}


class ExtractorInput(ContractModel):
    """保存待提取证据，并拒绝会破坏引用关系的重复证据 ID。"""

    evidence: list[Evidence] = Field(min_length=1)
    market_definition: MarketDefinition

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "ExtractorInput":
        """确保一个 Evidence ID 在本次提取中只对应一条证据。"""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique.")
        if any(item.scope_status != "in_scope" for item in self.evidence):
            raise ValueError("Extractor accepts only in-scope evidence.")

        return self


class ExtractorOutput(ContractModel):
    """约束模型每次只返回当前产品的一个画像。"""

    profile: ProductProfile


class ExtractorModel(Protocol):
    """约定 Extractor 所需的最小结构化模型调用接口。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """根据证据消息返回可被 ExtractorOutput 校验的对象。"""


class StructuredChatModel(Protocol):
    """描述 LangChain ChatModel 的结构化输出能力。"""

    def with_structured_output(
        self,
        schema: type[ExtractorOutput],
        *,
        method: Literal["json_mode"],
        include_raw: Literal[True],
    ) -> ExtractorModel:
        """绑定 ExtractorOutput，并返回可调用的结构化模型。"""


class LangChainExtractorModel:
    """把 LangChain ChatModel 包装成 Extractor 所需的模型接口。"""

    def __init__(self, chat_model: StructuredChatModel) -> None:
        # 硅基流动支持 json_object，因此与 Planner 一样显式使用 JSON mode。
        self._structured_model = chat_model.with_structured_output(
            ExtractorOutput,
            method="json_mode",
            include_raw=True,
        )

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """执行模型调用，并在解析失败时保留原始文本供修复。"""

        call_id = log_model_request("Extractor", messages)
        try:
            structured_response = self._structured_model.invoke(messages)
        except Exception as error:
            log_model_error("Extractor", call_id, error)
            raise
        log_model_response("Extractor", call_id, structured_response)

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


class FakeExtractorModel:
    """按顺序返回固定响应，用于无网络的确定性测试。"""

    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.invocation_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回下一条固定响应，并记录调用次数和消息。"""

        copied_messages = [message.copy() for message in messages]
        self.received_messages.append(copied_messages)

        if self.invocation_count >= len(self._responses):
            raise RuntimeError("Fake extractor has no response left.")

        response = self._responses[self.invocation_count]
        self.invocation_count += 1
        return response


class ExtractorError(RuntimeError):
    """表示 Extractor 调用失败或一次修复后仍无法生成有效画像。"""

    def __init__(
        self,
        message: str,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        # public_detail 会显示到页面，只保存脱敏后的定位信息。
        self.public_detail = public_detail or message


class ExtractorValidationError(ValueError):
    """表示模型输出结构、产品归属或证据引用无效。"""


class Extractor:
    """按产品提取、校验并在必要时修复一次产品画像。"""

    def __init__(self, model: ExtractorModel) -> None:
        self._model = model

    def extract(
        self,
        extractor_input: ExtractorInput,
    ) -> list[ProductProfile]:
        """按证据中的产品顺序生成画像，每个产品最多调用模型两次。"""

        evidence_groups = group_evidence_by_product(
            extractor_input.evidence
        )
        product_profiles: list[ProductProfile] = []

        for product_name, product_evidence in evidence_groups.items():
            selected_evidence = select_evidence_for_extraction(
                product_evidence
            )
            initial_messages = build_extractor_messages(
                product_name=product_name,
                evidence=selected_evidence,
                market_definition=extractor_input.market_definition,
            )
            raw_output = self._invoke_model(
                messages=initial_messages,
                product_name=product_name,
                evidence_count=len(product_evidence),
                selected_evidence_count=len(selected_evidence),
            )

            try:
                validated_output = validate_extractor_output(
                    raw_output=raw_output,
                    product_name=product_name,
                    evidence=selected_evidence,
                    market_definition=extractor_input.market_definition,
                )
                product_profiles.append(validated_output.profile)
                continue
            except ExtractorValidationError as first_error:
                # 把明确的校验错误反馈给模型，但只允许一次修复。
                repair_messages = build_repair_messages(
                    initial_messages=initial_messages,
                    raw_output=raw_output,
                    validation_error=str(first_error),
                )
                repaired_output = self._invoke_model(
                    messages=repair_messages,
                    product_name=product_name,
                    evidence_count=len(product_evidence),
                    selected_evidence_count=len(selected_evidence),
                )

            try:
                validated_repair = validate_extractor_output(
                    raw_output=repaired_output,
                    product_name=product_name,
                    evidence=selected_evidence,
                    market_definition=extractor_input.market_definition,
                )
                product_profiles.append(validated_repair.profile)
            except ExtractorValidationError as second_error:
                raise ExtractorError(
                    f"Extractor output for {product_name!r} remained "
                    f"invalid after one repair: {second_error}"
                ) from second_error

        return product_profiles

    def _invoke_model(
        self,
        messages: list[dict[str, str]],
        product_name: str,
        evidence_count: int,
        selected_evidence_count: int,
    ) -> object:
        """调用模型，并把供应商异常转换成统一的 ExtractorError。"""

        try:
            return self._model.invoke(messages)
        except Exception as error:
            public_detail = build_model_call_failure_detail(
                product_name=product_name,
                evidence_count=evidence_count,
                selected_evidence_count=selected_evidence_count,
                messages=messages,
                error=error,
            )
            raise ExtractorError(
                f"Extractor model call failed for {product_name!r}: {error}",
                public_detail=public_detail,
            ) from error


def group_evidence_by_product(
    evidence: Sequence[Evidence],
) -> dict[str, list[Evidence]]:
    """按 Evidence 首次出现的产品顺序分组。"""

    evidence_groups: dict[str, list[Evidence]] = {}

    for item in evidence:
        if item.product_name not in evidence_groups:
            evidence_groups[item.product_name] = []
        evidence_groups[item.product_name].append(item)

    return evidence_groups


def select_evidence_for_extraction(
    evidence: Sequence[Evidence],
) -> list[Evidence]:
    """为单个产品挑选少量代表性 Evidence，控制 Extractor 模型输入。"""

    evidence_by_topic: dict[str, list[Evidence]] = {}
    topic_order: list[str] = []
    for item in evidence:
        normalized_topic = item.topic.strip().lower()
        if normalized_topic not in evidence_by_topic:
            evidence_by_topic[normalized_topic] = []
            topic_order.append(normalized_topic)
        evidence_by_topic[normalized_topic].append(item)

    # 同一 topic 内优先把官方资料送入有限上下文，原始顺序作为次级顺序。
    for topic_evidence in evidence_by_topic.values():
        topic_evidence.sort(
            key=lambda item: 0 if item.source_type.value == "official" else 1
        )

    ordered_topics = build_extractor_topic_order(topic_order)
    selected_evidence: list[Evidence] = []
    selected_ids: set[str] = set()

    # 第一轮先保证不同 topic 都有代表证据，避免前几个搜索结果挤占全部空间。
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=ordered_topics,
        round_index=0,
    )

    # 价格表信息密度高，先保留它的第 2、3 条召回结果，再用其他 topic 补足名额。
    deep_context_topics = [
        topic
        for topic in ordered_topics
        if topic in EXTRACTOR_DEEP_CONTEXT_TOPICS
    ]
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=deep_context_topics,
        round_index=1,
    )
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=deep_context_topics,
        round_index=2,
    )

    # 价格资料不足时，继续按 topic 轮询，避免浪费剩余上下文预算。
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=ordered_topics,
        round_index=1,
    )
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=ordered_topics,
        round_index=2,
    )

    return selected_evidence


def build_extractor_topic_order(topic_order: list[str]) -> list[str]:
    """把价格、功能等重点 topic 排在前面，同时保留自定义 topic。"""

    ordered_topics: list[str] = []
    seen_topics: set[str] = set()
    for topic in EXTRACTOR_TOPIC_PRIORITY:
        if topic not in topic_order:
            continue
        ordered_topics.append(topic)
        seen_topics.add(topic)

    for topic in topic_order:
        if topic in seen_topics:
            continue
        ordered_topics.append(topic)
        seen_topics.add(topic)

    return ordered_topics


def add_evidence_round(
    selected_evidence: list[Evidence],
    selected_ids: set[str],
    evidence_by_topic: dict[str, list[Evidence]],
    ordered_topics: list[str],
    round_index: int,
) -> None:
    """按 topic 轮询补充 Evidence，达到上限后停止。"""

    for topic in ordered_topics:
        if len(selected_evidence) >= EXTRACTOR_MAX_EVIDENCE_PER_PRODUCT:
            return
        if round_index >= EXTRACTOR_MAX_EVIDENCE_PER_TOPIC:
            return

        topic_evidence = evidence_by_topic.get(topic, [])
        if round_index >= len(topic_evidence):
            continue

        candidate = topic_evidence[round_index]
        if candidate.evidence_id in selected_ids:
            continue

        selected_ids.add(candidate.evidence_id)
        selected_evidence.append(candidate)


def build_extractor_messages(
    product_name: str,
    evidence: Sequence[Evidence],
    market_definition: MarketDefinition,
) -> list[dict[str, str]]:
    """把单个产品的 Evidence 转换成模型可读取的消息。"""

    evidence_json = json.dumps(
        [build_extraction_evidence_item(item) for item in evidence],
        ensure_ascii=False,
        indent=2,
    )
    market_definition_json = json.dumps(
        market_definition.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        f"产品名：{product_name}\n"
        f"市场定义：\n{market_definition_json}\n\n"
        "请只根据以下 Evidence 生成该产品的 ProductProfile。\n\n"
        f"{evidence_json}"
    )

    return [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_extraction_evidence_item(item: Evidence) -> dict[str, object]:
    """把完整 Evidence 压缩成 Extractor 真正需要读取的字段。"""

    compact_item: dict[str, object] = {
        "evidence_id": item.evidence_id,
        "product_name": item.product_name,
        "topic": item.topic,
        "title": truncate_extractor_text(
            item.title,
            EXTRACTOR_TITLE_MAX_CHARS,
        ),
        "url": str(item.url),
        "snippet": truncate_extractor_text(
            item.snippet,
            EXTRACTOR_SNIPPET_MAX_CHARS,
        ),
        "source_type": item.source_type,
        "scope_status": item.scope_status,
        "scope_reason": item.scope_reason,
        "collected_at": item.collected_at.isoformat(),
    }

    # 只有价格页正文片段对提取价格有明显帮助，其他超长正文不塞进模型。
    if item.raw_content:
        compact_item["raw_content"] = truncate_extractor_text(
            item.raw_content,
            EXTRACTOR_RAW_CONTENT_MAX_CHARS,
        )

    return compact_item


def truncate_extractor_text(value: str, max_chars: int) -> str:
    """压缩 Evidence 文本，避免 Extractor 单次模型输入过大。"""

    compact_text = re.sub(r"\s+", " ", value).strip()
    if len(compact_text) <= max_chars:
        return compact_text
    return compact_text[:max_chars] + "...[truncated]"


def build_model_call_failure_detail(
    product_name: str,
    evidence_count: int,
    selected_evidence_count: int,
    messages: list[dict[str, str]],
    error: Exception,
) -> str:
    """构造可显示到页面的模型调用失败详情，不包含原始异常文本。"""

    input_chars = 0
    for message in messages:
        input_chars += len(message.get("content", ""))

    return (
        "Extractor 调用模型服务失败。"
        f"产品：{product_name}；"
        f"原始证据条数：{evidence_count}；"
        f"送入模型证据条数：{selected_evidence_count}；"
        f"模型输入约 {input_chars} 个字符；"
        f"底层异常类型：{type(error).__name__}。"
        "模型输入已自动限制；超时时请稍后重试，持续失败时再减少分析维度。"
    )


def build_repair_messages(
    initial_messages: list[dict[str, str]],
    raw_output: object,
    validation_error: str,
) -> list[dict[str, str]]:
    """把结构或引用错误反馈给模型，要求只修复当前画像。"""

    repair_messages = [message.copy() for message in initial_messages]
    repair_instruction = (
        "上一次产品画像没有通过校验，请只修复 JSON 输出，"
        "不要添加 Evidence 中不存在的事实。\n"
        "每个 feature 必须包含 name、description、evidence_ids；"
        "feature 的 description 不可为 null，缺少时整条删除；"
        "订阅 pricing 项必须包含 plan_name、price、unit、billing_cycle、"
        "service_level、threshold、main_limits、evidence_ids；"
        "API 价格必须写入 models[].pricing，同一模型只保留一个 ModelProfile；"
        "input、output、cached input、audio 和 Batch 必须写入各自字段；"
        "audio 必须明确为 audio_input_price 或 audio_output_price；"
        "每个 PriceRate 必须是 USD / 1000000 token，包含 amount、currency、"
        "per_quantity、unit、condition、max_context_tokens、effective_from、"
        "effective_to、evidence_ids；"
        "models[].source_evidence 必须是 Evidence 的完整对象，不能只写 evidence_id 字符串；"
        "无日期的多个同类 PriceRate 只能使用不同且非空的 condition；"
        "context_window_tokens、max_output_tokens 和 RPM/TPM/RPD 不得混写；"
        "每个已填字段提供 quote、rationale 和 confidence，"
        "quote 必须是 Evidence 中逐字连续的原文，不得使用省略号或截断；"
        "缺失字段不得返回 field_evidence；"
        "API pricing 中 price=null 的条目必须整条删除，"
        "不能生成没有公开价格的 API 事实；没有任何数字、Free 或 "
        "Custom pricing 的非价格文本也必须整条删除；"
        "1.1x、1.1× token pricing 这类倍率不是独立绝对费率，"
        "不得单独生成 pricing 项；"
        "dimension_findings 必须覆盖全部核心维度，未知值使用 null 或 []。\n"
        f"校验错误：{validation_error}\n"
        f"上一次输出：{raw_output!r}"
    )
    repair_messages.append(
        {"role": "user", "content": repair_instruction}
    )
    return repair_messages


def validate_extractor_output(
    raw_output: object,
    product_name: str,
    evidence: Sequence[Evidence],
    market_definition: MarketDefinition,
) -> ExtractorOutput:
    """校验输出结构、产品名和所有功能及价格的证据引用。"""

    normalized_output = normalize_extractor_raw_output(raw_output, evidence)
    try:
        if isinstance(normalized_output, str):
            extractor_output = ExtractorOutput.model_validate_json(
                normalized_output
            )
        else:
            extractor_output = ExtractorOutput.model_validate(
                normalized_output
            )
    except ValidationError as error:
        raise ExtractorValidationError(
            f"Output does not match ExtractorOutput: {error}"
        ) from error

    profile = extractor_output.profile
    if profile.product_name != product_name:
        raise ExtractorValidationError(
            "Profile product_name does not match the evidence group: "
            f"expected={product_name!r}, actual={profile.product_name!r}"
        )

    allowed_evidence_ids = {item.evidence_id for item in evidence}
    raw_referenced_evidence_ids = collect_profile_evidence_ids(profile)
    raw_unknown_evidence_ids = (
        raw_referenced_evidence_ids - allowed_evidence_ids
    )
    if raw_unknown_evidence_ids:
        unknown_text = ", ".join(sorted(raw_unknown_evidence_ids))
        raise ExtractorValidationError(
            f"Profile references unknown evidence IDs: {unknown_text}"
        )

    normalized_profile = normalize_profile_summary_fields(
        profile=profile,
        evidence=evidence,
    )
    validate_field_evidence(normalized_profile, evidence)
    validate_model_field_semantics(normalized_profile, evidence)
    if market_definition.pricing_scope == "api":
        normalized_profile = omit_unknown_api_pricing(normalized_profile)
    dimension_profile = ensure_dimension_findings(
        profile=normalized_profile,
        market_definition=market_definition,
    )

    referenced_evidence_ids = collect_profile_evidence_ids(dimension_profile)
    unknown_evidence_ids = (
        referenced_evidence_ids - allowed_evidence_ids
    )
    if unknown_evidence_ids:
        unknown_text = ", ".join(sorted(unknown_evidence_ids))
        raise ExtractorValidationError(
            f"Profile references unknown evidence IDs: {unknown_text}"
        )

    if dimension_profile is profile:
        return extractor_output

    return ExtractorOutput(profile=dimension_profile)


def normalize_extractor_raw_output(
    raw_output: object,
    evidence: Sequence[Evidence] = (),
) -> object:
    """在 Schema 校验前修正模型常见但可安全转换的嵌套形状。"""

    if isinstance(raw_output, str):
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
        return normalize_extractor_raw_output(parsed_output, evidence)

    if not isinstance(raw_output, dict):
        return raw_output

    normalized_output = raw_output.copy()
    raw_profile = normalized_output.get("profile")
    if not isinstance(raw_profile, dict):
        return normalized_output

    normalized_profile = raw_profile.copy()
    evidence_by_id = {item.evidence_id: item for item in evidence}
    if "source_evidence" in normalized_profile:
        normalized_profile["source_evidence"] = normalize_source_references(
            normalized_profile["source_evidence"], evidence_by_id
        )
    raw_field_evidence = normalized_profile.get("field_evidence")
    if isinstance(raw_field_evidence, list):
        normalized_field_evidence = [
            normalize_field_evidence_item(item)
            for item in raw_field_evidence
        ]
        if evidence_by_id:
            normalized_field_evidence = [
                item
                for item in normalized_field_evidence
                if not should_drop_unquoted_field_evidence(item, evidence_by_id)
            ]
        normalized_profile["field_evidence"] = normalized_field_evidence
    raw_features = normalized_profile.get("features")
    if isinstance(raw_features, list):
        normalized_profile["features"] = normalize_feature_items(
            raw_features, evidence_by_id
        )
    raw_dimension_findings = normalized_profile.get("dimension_findings")
    if isinstance(raw_dimension_findings, dict):
        normalized_profile["dimension_findings"] = [
            {"dimension": dimension, **finding}
            for dimension, finding in raw_dimension_findings.items()
            if isinstance(finding, dict)
        ]
    raw_pricing = normalized_profile.get("pricing")
    if isinstance(raw_pricing, list):
        normalized_profile["pricing"] = normalize_pricing_items(
            raw_pricing
        )
    raw_models = normalized_profile.get("models")
    if isinstance(raw_models, list):
        normalized_profile["models"] = merge_model_items(
            [
                normalize_model_source_evidence(item, evidence_by_id)
                for item in raw_models
            ]
        )

    normalized_output["profile"] = normalized_profile
    return normalized_output


def normalize_model_source_evidence(
    item: object,
    evidence_by_id: dict[str, Evidence],
) -> object:
    if not isinstance(item, dict):
        return item
    normalized_item = item.copy()
    if "source_evidence" in normalized_item:
        normalized_item["source_evidence"] = normalize_source_references(
            normalized_item["source_evidence"], evidence_by_id
        )
    return normalized_item


def normalize_feature_items(
    raw_features: list[object],
    evidence_by_id: dict[str, Evidence],
) -> list[object]:
    """只在同一来源存在原句时补齐模型漏掉的 feature 描述。"""

    normalized_features: list[object] = []
    for item in raw_features:
        if not isinstance(item, dict):
            normalized_features.append(item)
            continue
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            normalized_features.append(item)
            continue
        if description is not None:
            normalized_features.append(item)
            continue
        inferred_description = infer_feature_description(item, evidence_by_id)
        if inferred_description is not None:
            normalized_features.append({**item, "description": inferred_description})
    return normalized_features


def infer_feature_description(
    item: dict[object, object],
    evidence_by_id: dict[str, Evidence],
) -> str | None:
    feature_name = item.get("name")
    evidence_ids = item.get("evidence_ids")
    if not isinstance(feature_name, str) or not isinstance(evidence_ids, list):
        return None
    normalized_name = normalize_scope_text(feature_name)
    if not normalized_name:
        return None
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        for raw_text in (evidence.snippet, evidence.raw_content or ""):
            for sentence in split_sentences(clean_markdown_text(raw_text)):
                if (
                    normalized_name in normalize_scope_text(sentence)
                    and len(sentence) <= 180
                ):
                    return sentence
    return None


def normalize_source_references(
    value: object,
    evidence_by_id: dict[str, Evidence],
) -> object:
    """仅把本轮 Evidence 中可唯一解析的简写 ID 补成来源对象。"""

    if not isinstance(value, list):
        return value
    return [
        {
            "evidence_id": evidence.evidence_id,
            "title": evidence.title,
            "url": str(evidence.url),
            "source_type": evidence.source_type,
            "collected_at": evidence.collected_at.isoformat(),
        }
        if isinstance(item, str) and (evidence := evidence_by_id.get(item))
        else item
        for item in value
    ]


def normalize_field_evidence_item(item: object) -> object:
    """把模型常用别名映射到 FieldEvidence 的固定字段名。"""

    if not isinstance(item, dict):
        return item
    normalized_item = item.copy()
    if "quote" not in normalized_item and "evidence_quote" in normalized_item:
        normalized_item["quote"] = normalized_item["evidence_quote"]
    if "rationale" not in normalized_item and "field_rationale" in normalized_item:
        normalized_item["rationale"] = normalized_item["field_rationale"]
    normalized_item.pop("evidence_quote", None)
    normalized_item.pop("field_rationale", None)
    return normalized_item


def should_drop_unquoted_field_evidence(
    item: object,
    evidence_by_id: dict[str, Evidence],
) -> bool:
    """删除目录型引用及表格中无法独立表达字段语义的裸价格单元格。"""

    if not isinstance(item, dict):
        return False
    field_path = item.get("field_path")
    evidence_id = item.get("evidence_id")
    quote = item.get("quote")
    evidence = evidence_by_id.get(evidence_id) if isinstance(evidence_id, str) else None
    quote_in_evidence = (
        evidence is not None
        and isinstance(quote, str)
        and bool(quote.strip())
        and normalize_scope_text(quote)
        in normalize_scope_text(build_evidence_scope_text(evidence))
    )
    is_bare_table_price = (
        isinstance(field_path, str)
        and ".pricing." in field_path
        and isinstance(quote, str)
        and bool(re.fullmatch(r"\$\d+(?:\.\d+)?", quote.strip()))
        and evidence is not None
        and "|" in build_evidence_scope_text(evidence)
    )
    return (
        isinstance(field_path, str)
        and field_path in {"models", "features"}
        and evidence is not None
        and isinstance(quote, str)
        and bool(quote.strip())
        and not quote_in_evidence
    ) or (is_bare_table_price and quote_in_evidence)


def merge_model_items(raw_models: list[object]) -> list[object]:
    """合并同名模型被拆开的价格字段，并保留首次出现的其他值。"""

    merged_models: list[object] = []
    model_by_name: dict[str, dict[object, object]] = {}
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            merged_models.append(raw_model)
            continue

        model_name = raw_model.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            merged_models.append(raw_model)
            continue

        normalized_name = model_name.strip().casefold()
        current_model = model_by_name.get(normalized_name)
        if current_model is None:
            current_model = raw_model.copy()
            model_by_name[normalized_name] = current_model
            merged_models.append(current_model)
            continue

        current_pricing = current_model.get("pricing")
        incoming_pricing = raw_model.get("pricing")
        if isinstance(incoming_pricing, dict):
            if not isinstance(current_pricing, dict):
                current_pricing = {}
                current_model["pricing"] = current_pricing
            for price_type, price_value in incoming_pricing.items():
                current_price = current_pricing.get(price_type)
                if current_price is None:
                    current_pricing[price_type] = price_value
                elif price_value is not None and price_value != current_price:
                    if isinstance(current_price, list):
                        if isinstance(price_value, list):
                            current_price.extend(price_value)
                        else:
                            current_price.append(price_value)
                    else:
                        incoming_prices = (
                            price_value
                            if isinstance(price_value, list)
                            else [price_value]
                        )
                        current_pricing[price_type] = [current_price]
                        current_pricing[price_type].extend(incoming_prices)

        current_sources = current_model.get("source_evidence")
        incoming_sources = raw_model.get("source_evidence")
        if isinstance(current_sources, list) and isinstance(
            incoming_sources, list
        ):
            known_evidence_ids = {
                source.get("evidence_id")
                for source in current_sources
                if isinstance(source, dict)
            }
            for source in incoming_sources:
                evidence_id = (
                    source.get("evidence_id")
                    if isinstance(source, dict)
                    else None
                )
                if evidence_id in known_evidence_ids:
                    continue
                current_sources.append(source)
                known_evidence_ids.add(evidence_id)

    return merged_models


def normalize_pricing_items(raw_pricing: list[object]) -> list[object]:
    """丢弃无法识别的套餐，并规范化 pricing 中的 main_limits。"""

    normalized_pricing: list[object] = []
    for raw_plan in raw_pricing:
        if not isinstance(raw_plan, dict):
            normalized_pricing.append(raw_plan)
            continue

        raw_plan_name = raw_plan.get("plan_name")
        if raw_plan_name is None:
            continue
        if isinstance(raw_plan_name, str) and not raw_plan_name.strip():
            continue

        normalized_plan = raw_plan.copy()
        raw_main_limits = normalized_plan.get("main_limits")
        normalized_plan["main_limits"] = normalize_main_limits(
            raw_main_limits
        )
        normalized_pricing.append(normalized_plan)

    return normalized_pricing


def normalize_main_limits(raw_main_limits: object) -> object:
    """把模型返回的限制对象压平成字符串列表，避免可用事实因形状失败丢失。"""

    if raw_main_limits is None:
        return []
    if not isinstance(raw_main_limits, list):
        return raw_main_limits

    normalized_limits: list[object] = []
    for raw_limit in raw_main_limits:
        if isinstance(raw_limit, dict):
            normalized_limits.append(format_main_limit_object(raw_limit))
            continue
        normalized_limits.append(raw_limit)

    return normalized_limits


def format_main_limit_object(raw_limit: dict[object, object]) -> str:
    """把带 name/description 的限制对象转换成一句可读文本。"""

    preferred_keys = ["name", "description", "value", "limit"]
    text_parts: list[str] = []
    for key in preferred_keys:
        value = raw_limit.get(key)
        if value is None:
            continue

        text = str(value).strip()
        if text:
            text_parts.append(text)

    if text_parts:
        return ": ".join(text_parts)

    return json.dumps(raw_limit, ensure_ascii=False, sort_keys=True)


def normalize_profile_summary_fields(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """补齐可由证据或字段文本确定的画像摘要信息。"""

    scoped_profile = remove_plan_level_positioning(
        profile=profile,
        evidence=evidence,
    )
    positioned_profile = fill_missing_positioning_from_evidence(
        profile=scoped_profile,
        evidence=evidence,
    )
    targeted_profile = remove_unsupported_target_users(
        profile=positioned_profile,
        evidence=evidence,
    )
    featured_profile = remove_unsupported_features(
        profile=targeted_profile,
        evidence=evidence,
    )
    priced_profile = normalize_pricing_defaults(featured_profile)
    deduplicated_profile = remove_conflicting_pricing_duplicates(
        priced_profile
    )
    return deduplicated_profile


def validate_field_evidence(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> None:
    """校验字段级 quote、引用和分层置信度，失败时触发一次修复。"""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    for field_evidence in profile.field_evidence:
        evidence_item = evidence_by_id.get(field_evidence.evidence_id)
        if evidence_item is None:
            raise ExtractorValidationError(
                "Field evidence references unknown Evidence ID: "
                f"{field_evidence.evidence_id}"
            )

        quote = normalize_scope_text(field_evidence.quote)
        source_text = normalize_scope_text(
            build_evidence_scope_text(evidence_item)
        )
        if quote not in source_text:
            raise ExtractorValidationError(
                "Field evidence quote is not present in its Evidence: "
                f"{field_evidence.field_path}"
            )

        normalized_path = field_evidence.field_path.casefold()
        if normalized_path == "positioning" and profile.positioning is None:
            raise ExtractorValidationError(
                "Missing positioning cannot have field evidence."
            )
        if normalized_path.startswith("target_users") and not profile.target_users:
            raise ExtractorValidationError(
                "Missing target_users cannot have field evidence."
            )
        is_high_level = normalized_path == "positioning" or (
            normalized_path.startswith("target_users")
        )
        minimum_confidence = (
            HIGH_LEVEL_FIELD_CONFIDENCE
            if is_high_level
            else DEFAULT_FIELD_CONFIDENCE
        )
        if field_evidence.confidence < minimum_confidence:
            raise ExtractorValidationError(
                "Field evidence confidence is below the minimum for "
                f"{field_evidence.field_path}: {minimum_confidence:.2f}"
            )
        if not field_quote_matches_path(normalized_path, quote):
            raise ExtractorValidationError(
                "Field evidence quote does not answer its field: "
                f"{field_evidence.field_path}"
            )


def field_quote_matches_path(field_path: str, quote: str) -> bool:
    """用字段专属标记阻止 quote 被挂到错误语义层级。"""

    # ponytail: 当前只覆盖固定 Schema 的高风险错位；评测出现新字段再扩展标记。
    if field_path == "positioning":
        return is_valid_positioning_text(quote)
    if field_path.startswith("target_users"):
        return any(marker in quote for marker in TARGET_USER_RELATION_MARKERS)
    if field_path.endswith("context_window_tokens"):
        return "context" in quote and "window" in quote
    if field_path.endswith("max_output_tokens"):
        return "max output" in quote or "maximum output" in quote
    if ".rate_limits.requests_per_minute" in field_path:
        return "rpm" in quote or "requests per minute" in quote
    if ".rate_limits.tokens_per_minute" in field_path:
        return "tpm" in quote or "tokens per minute" in quote
    if ".rate_limits.requests_per_day" in field_path:
        return "rpd" in quote or "requests per day" in quote
    if ".pricing.cached_input_price" in field_path:
        return "cache" in quote or "cached" in quote
    if (
        ".pricing.audio_input_price" in field_path
        or ".pricing.audio_price" in field_path
    ):
        return "audio" in quote
    if ".pricing.batch_" in field_path:
        return "batch" in quote
    if ".pricing.input_price" in field_path:
        return "input" in quote and not any(
            marker in quote for marker in ("cache", "audio", "batch")
        )
    if ".pricing.output_price" in field_path:
        return "output" in quote and not any(
            marker in quote for marker in ("audio", "batch")
        )
    return True


def validate_model_field_semantics(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> None:
    """检查 Token 上限、限流和价格是否由同语义 Evidence 支持。"""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    for model in profile.models:
        model_evidence_ids = [
            item.evidence_id for item in model.source_evidence
        ]
        model_text = build_referenced_evidence_text(
            model_evidence_ids,
            evidence_by_id,
        )
        if model.context_window_tokens is not None and not all(
            marker in model_text for marker in ("context", "window")
        ):
            raise ExtractorValidationError(
                f"{model.model_name} context window lacks direct evidence."
            )
        if model.max_output_tokens is not None and not (
            "max output" in model_text
            or "maximum output" in model_text
        ):
            raise ExtractorValidationError(
                f"{model.model_name} max output lacks direct evidence."
            )

        rate_limit_markers = {
            "requests_per_minute": ("rpm", "requests per minute"),
            "tokens_per_minute": ("tpm", "tokens per minute"),
            "requests_per_day": ("rpd", "requests per day"),
            "tokens_per_day": ("tpd", "tokens per day"),
            "concurrent_requests": ("concurrent",),
            "other": ("limit",),
        }
        for rate_limit in model.rate_limits or []:
            rate_text = build_referenced_evidence_text(
                rate_limit.evidence_ids,
                evidence_by_id,
            )
            markers = rate_limit_markers[rate_limit.metric.value]
            if not any(marker in rate_text for marker in markers):
                raise ExtractorValidationError(
                    f"{model.model_name} rate limit is in the wrong field: "
                    f"{rate_limit.metric.value}"
                )

        price_markers = {
            "input_price": ("input",),
            "cached_input_price": ("cache", "cached"),
            "output_price": ("output",),
            "audio_input_price": ("audio",),
            "audio_output_price": ("audio",),
            "audio_price": ("audio",),
            "batch_input_price": ("batch",),
            "batch_cached_input_price": ("batch",),
            "batch_output_price": ("batch",),
        }
        for price_field, markers in price_markers.items():
            price_value = getattr(model.pricing, price_field)
            price_rates = (
                price_value if isinstance(price_value, list) else [price_value]
            )
            for price_rate in price_rates:
                if price_rate is None:
                    continue
                price_text = build_referenced_evidence_text(
                    price_rate.evidence_ids,
                    evidence_by_id,
                )
                if not any(marker in price_text for marker in markers):
                    raise ExtractorValidationError(
                        f"{model.model_name} price is in the wrong field: "
                        f"{price_field}"
                    )


def build_referenced_evidence_text(
    evidence_ids: Sequence[str],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """拼接指定 Evidence 的文本，供字段语义门禁复用。"""

    text_parts: list[str] = []
    for evidence_id in evidence_ids:
        evidence_item = evidence_by_id.get(evidence_id)
        if evidence_item is not None:
            text_parts.append(build_evidence_scope_text(evidence_item))
    return normalize_scope_text(" ".join(text_parts))


def ensure_dimension_findings(
    profile: ProductProfile,
    market_definition: MarketDefinition,
) -> ProductProfile:
    """按核心维度排序画像；缺失维度以空事实显式表示资料不足。"""

    findings_by_dimension: dict[str, DimensionFinding] = {}
    for finding in profile.dimension_findings:
        if finding.dimension in findings_by_dimension:
            raise ExtractorValidationError(
                f"Duplicate dimension finding: {finding.dimension}"
            )
        if finding.dimension not in market_definition.core_dimensions:
            raise ExtractorValidationError(
                f"Unexpected dimension finding: {finding.dimension}"
            )
        findings_by_dimension[finding.dimension] = finding

    ordered_findings: list[DimensionFinding] = []
    for dimension in market_definition.core_dimensions:
        normalized_dimension = normalize_scope_text(dimension)
        if normalized_dimension in {"features", "model_capabilities"} and (
            profile.features
            or any(model.model_capabilities for model in profile.models)
        ):
            # 已有结构化能力时，以它重建 finding，不能保留模型误写的空结果。
            finding = build_legacy_dimension_finding(dimension, profile)
        elif (
            market_definition.pricing_scope == "api"
            and normalized_dimension == "api_pricing"
        ):
            # API 价格由结构化列表重建事实；语义支持关系在 Verifier 中结合
            # 原始检索上下文判断，避免本地字符串规则提前丢失资料。
            finding = build_legacy_dimension_finding(dimension, profile)
        else:
            finding = findings_by_dimension.get(dimension)
            if finding is None:
                finding = build_legacy_dimension_finding(dimension, profile)
        ordered_findings.append(finding)

    if ordered_findings == profile.dimension_findings:
        return profile
    return profile.model_copy(update={"dimension_findings": ordered_findings})


def omit_unknown_api_pricing(profile: ProductProfile) -> ProductProfile:
    """删除未知或明显不是价格的 API 条目，不猜测缺失金额。"""

    pricing = [
        plan
        for plan in profile.pricing
        if plan.price is not None
        and (
            re.search(r"\d", plan.price) is not None
            and not is_api_price_multiplier(plan.price)
            or is_free_price_text(plan.price)
            or is_custom_pricing_text(plan.price)
        )
    ]
    if len(pricing) == len(profile.pricing):
        return profile
    return profile.model_copy(update={"pricing": pricing})


def is_api_price_multiplier(price_text: str) -> bool:
    """识别区域、批处理等相对倍率；倍率不能冒充独立绝对费率。"""

    normalized_text = normalize_scope_text(price_text)
    return re.fullmatch(
        r"\d+(?:[.,]\d+)?\s*[x×](?:\s+token\s+pricing)?",
        normalized_text,
    ) is not None


def build_legacy_dimension_finding(
    dimension: str,
    profile: ProductProfile,
) -> DimensionFinding:
    """把现有功能和价格字段映射到维度结构，兼容旧模型输出。"""

    normalized_dimension = normalize_scope_text(dimension)
    if normalized_dimension in {"features", "model_capabilities"}:
        facts = [
            f"{feature.name}: {feature.description}"
            for feature in profile.features
        ]
        evidence_ids = collect_item_evidence_ids(profile.features)
        for model in profile.models:
            for capability in model.model_capabilities or []:
                facts.append(f"{model.model_name}: {capability}")
            for source in model.source_evidence:
                if (
                    model.model_capabilities
                    and source.evidence_id not in evidence_ids
                ):
                    evidence_ids.append(source.evidence_id)
        return DimensionFinding(
            dimension=dimension,
            facts=facts,
            evidence_ids=evidence_ids,
        )

    if normalized_dimension in {"pricing", "api_pricing"}:
        facts = [format_pricing_fact(plan) for plan in profile.pricing]
        evidence_ids = collect_item_evidence_ids(profile.pricing)
        price_fields = (
            "input_price",
            "cached_input_price",
            "output_price",
            "audio_input_price",
            "audio_output_price",
            "audio_price",
            "batch_input_price",
            "batch_cached_input_price",
            "batch_output_price",
        )
        for model in profile.models:
            for price_field in price_fields:
                price_value = getattr(model.pricing, price_field)
                price_rates = (
                    price_value
                    if isinstance(price_value, list)
                    else [price_value]
                )
                for price_rate in price_rates:
                    if price_rate is None:
                        continue
                    facts.append(
                        f"{model.model_name} | {price_field} | "
                        f"{price_rate.amount} "
                        f"{price_rate.currency.value} per "
                        f"{price_rate.per_quantity} {price_rate.unit.value}"
                    )
                    for evidence_id in price_rate.evidence_ids:
                        if evidence_id not in evidence_ids:
                            evidence_ids.append(evidence_id)
            if model.pricing.batch_discount_percent is not None:
                facts.append(
                    f"{model.model_name} | batch_discount_percent | "
                    f"{model.pricing.batch_discount_percent}%"
                )
                for evidence_id in model.pricing.batch_evidence_ids:
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
        return DimensionFinding(
            dimension=dimension,
            facts=facts,
            evidence_ids=evidence_ids,
        )

    return DimensionFinding(dimension=dimension)


def collect_item_evidence_ids(items: Sequence[object]) -> list[str]:
    """按首次出现顺序汇总功能或价格项的 Evidence ID。"""

    evidence_ids: list[str] = []
    for item in items:
        for evidence_id in getattr(item, "evidence_ids", []):
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    return evidence_ids


def format_pricing_fact(pricing_plan: PricingPlan) -> str:
    """保留价格项原始上下文字段，不执行跨单位换算。"""

    context_parts = [pricing_plan.plan_name]
    for value in [
        pricing_plan.price,
        pricing_plan.unit,
        pricing_plan.billing_cycle,
        pricing_plan.service_level,
        pricing_plan.threshold,
    ]:
        if value:
            context_parts.append(value)
    context_parts.extend(pricing_plan.main_limits)
    return " | ".join(context_parts)


def remove_plan_level_positioning(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """删除认证、导航、套餐、模型清单和限流等伪定位。"""

    if profile.positioning is None:
        return profile

    if not is_valid_positioning_text(profile.positioning):
        return profile.model_copy(update={"positioning": None})

    if positioning_has_direct_product_support(
        profile.positioning,
        evidence,
    ):
        return profile

    return profile.model_copy(update={"positioning": None})


def is_valid_positioning_text(positioning: str) -> bool:
    """判断文本是否描述产品类别，而不是其他字段或操作说明。"""

    normalized_positioning = normalize_scope_text(positioning)
    if any(
        marker in normalized_positioning
        for marker in POSITIONING_NEGATIVE_MARKERS
    ):
        return False
    if looks_like_plan_level_positioning(positioning):
        return False
    return score_positioning_sentence(positioning) >= 2


def positioning_has_direct_product_support(
    positioning: str,
    evidence: Sequence[Evidence],
) -> bool:
    """检查定位文本是否被非价格页证据直接支持。"""

    normalized_positioning = normalize_scope_text(positioning)
    if not normalized_positioning:
        return False

    for item in evidence:
        if not is_positioning_evidence_candidate(item):
            continue

        support_text = normalize_scope_text(build_evidence_scope_text(item))
        if normalized_positioning in support_text:
            return True

    return False


def looks_like_plan_level_positioning(positioning: str) -> bool:
    """识别价格页常见的套餐适用人群，而不是产品级定位。"""

    normalized_positioning = normalize_scope_text(positioning)
    for marker in PLAN_LEVEL_POSITIONING_MARKERS:
        if marker in normalized_positioning:
            return True

    return False


def fill_missing_positioning_from_evidence(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """当模型漏掉定位时，从 Evidence 的官网标题或首句中保守补齐。"""

    if profile.positioning is not None:
        return profile

    positioning = infer_positioning_from_evidence(evidence)
    if positioning is None:
        return profile

    return profile.model_copy(update={"positioning": positioning})


def remove_unsupported_target_users(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """只保留来源明确以目标群体关系描述的用户类型。"""

    supported_users = [
        target_user
        for target_user in profile.target_users
        if target_user_has_direct_support(target_user, evidence)
    ]
    if supported_users == profile.target_users:
        return profile
    return profile.model_copy(update={"target_users": supported_users})


def target_user_has_direct_support(
    target_user: str,
    evidence: Sequence[Evidence],
) -> bool:
    """检查目标用户是否出现在明确的面向、适用或使用者陈述中。"""

    normalized_target = normalize_scope_text(target_user)
    if not normalized_target:
        return False

    for item in evidence:
        raw_texts = [item.title, item.snippet, item.raw_content or ""]
        for raw_text in raw_texts:
            for sentence in split_sentences(clean_markdown_text(raw_text)):
                normalized_sentence = normalize_scope_text(sentence)
                if normalized_target not in normalized_sentence:
                    continue
                if any(
                    marker in normalized_sentence
                    for marker in TARGET_USER_RELATION_MARKERS
                ):
                    return True
                if (
                    item.topic.strip().lower() == "target_users"
                    and normalized_sentence == normalized_target
                ):
                    return True
    return False


def remove_unsupported_features(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """删除操作步骤、导航、限流或没有直接能力词支持的伪功能。"""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    supported_features = []
    for feature in profile.features:
        feature_text = normalize_scope_text(
            f"{feature.name} {feature.description}"
        )
        if any(marker in feature_text for marker in FEATURE_NEGATIVE_MARKERS):
            continue

        source_text = build_referenced_evidence_text(
            feature.evidence_ids,
            evidence_by_id,
        )
        normalized_name = normalize_scope_text(feature.name)
        normalized_description = normalize_scope_text(feature.description)
        if (
            normalized_name in source_text
            or normalized_description in source_text
        ):
            supported_features.append(feature)
            continue

        meaningful_words = re.findall(r"[a-z0-9_]{4,}", normalized_name)
        if any(word in source_text for word in meaningful_words):
            supported_features.append(feature)

    if supported_features == profile.features:
        return profile
    return profile.model_copy(update={"features": supported_features})


def normalize_pricing_defaults(profile: ProductProfile) -> ProductProfile:
    """根据明确的 plan/price 文本补齐 Free 价格和月/年计费周期。"""

    normalized_pricing = []
    changed = False
    for pricing_plan in profile.pricing:
        update_fields: dict[str, str | None] = {}
        normalized_existing_price = normalize_price_value(
            pricing_plan.price
        )
        normalized_existing_billing_cycle = normalize_billing_cycle_value(
            pricing_plan.billing_cycle
        )

        if normalized_existing_price != pricing_plan.price:
            update_fields["price"] = normalized_existing_price
        if normalized_existing_billing_cycle != pricing_plan.billing_cycle:
            update_fields["billing_cycle"] = normalized_existing_billing_cycle

        if (
            normalized_existing_price is None
            and pricing_plan.plan_name.strip().lower() == "free"
        ):
            update_fields["price"] = "$0"

        normalized_price = update_fields.get("price", pricing_plan.price)
        if is_free_price_text(normalized_price):
            if normalized_existing_billing_cycle is not None:
                update_fields["billing_cycle"] = None
        elif is_custom_pricing_text(normalized_price):
            if normalized_existing_billing_cycle is not None:
                update_fields["billing_cycle"] = None
        elif normalized_existing_billing_cycle is None:
            inferred_cycle = infer_billing_cycle(normalized_price)
            if inferred_cycle is not None:
                update_fields["billing_cycle"] = inferred_cycle

        if update_fields:
            changed = True
            normalized_pricing.append(
                pricing_plan.model_copy(update=update_fields)
            )
            continue

        normalized_pricing.append(pricing_plan)

    if not changed:
        return profile

    return profile.model_copy(update={"pricing": normalized_pricing})


def classify_pricing_source_scope(
    product_name: str,
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> PricingScopeClassification:
    """把价格来源分成 API、订阅、混杂或无法判断四类。"""

    scope_rules = build_api_pricing_scope_rules(product_name)
    scope_text = build_pricing_plan_scope_text(
        pricing_plan=pricing_plan,
        evidence_by_id=evidence_by_id,
    )
    api_markers = GENERIC_PRICING_SCOPE_RULES.api_markers
    non_api_markers = GENERIC_PRICING_SCOPE_RULES.non_api_markers
    if scope_rules is not None:
        api_markers += scope_rules.api_markers
        non_api_markers += scope_rules.non_api_markers

    has_api_marker = scope_text_has_any_marker(
        scope_text,
        api_markers,
    )
    has_non_api_marker = scope_text_has_any_marker(
        scope_text,
        non_api_markers,
    )

    if has_api_marker and not has_non_api_marker:
        return "api_pricing"
    if has_non_api_marker and not has_api_marker:
        return "non_api_pricing"

    if has_api_marker and has_non_api_marker:
        return "ambiguous"
    return "unknown"


def build_api_pricing_scope_rules(
    product_name: str,
) -> PricingScopeRules | None:
    """返回产品专用补充词；价格范围仍由 MarketDefinition 决定。"""

    normalized_product = normalize_scope_text(product_name)
    for product_key, scope_rules in API_PRICING_SCOPE_RULES_BY_PRODUCT_KEY.items():
        if product_key == normalized_product:
            return scope_rules
        if scope_text_contains_marker(normalized_product, product_key):
            return scope_rules

    return None


def build_pricing_plan_scope_text(
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """拼接价格项和来源证据文本，用于判断 API pricing 范围。"""

    scope_text_parts = [
        pricing_plan.plan_name,
        pricing_plan.price or "",
        pricing_plan.unit or "",
        pricing_plan.billing_cycle or "",
        pricing_plan.service_level or "",
        pricing_plan.threshold or "",
        " ".join(pricing_plan.main_limits),
    ]

    for evidence_id in pricing_plan.evidence_ids:
        evidence_item = evidence_by_id.get(evidence_id)
        if evidence_item is None:
            continue
        scope_text_parts.append(build_evidence_scope_text(evidence_item))

    return normalize_scope_text(" ".join(scope_text_parts))


def scope_text_has_any_marker(
    scope_text: str,
    markers: Sequence[str],
) -> bool:
    """检查范围文本中是否出现任一关键词，单词关键词使用边界匹配。"""

    for marker in markers:
        if scope_text_contains_marker(scope_text, marker):
            return True

    return False


def scope_text_contains_marker(scope_text: str, marker: str) -> bool:
    """避免 `pro` 这类短词误命中 `product`，同时支持 URL 和短语。"""

    normalized_marker = normalize_scope_text(marker)
    if not normalized_marker:
        return False

    marker_has_separator = bool(re.search(r"[\s./_-]", normalized_marker))
    if marker_has_separator:
        return normalized_marker in scope_text

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_marker)}(?![a-z0-9])"
    return re.search(pattern, scope_text) is not None


def remove_conflicting_pricing_duplicates(
    profile: ProductProfile,
) -> ProductProfile:
    """删除同一套餐同一计费周期下出现多个价格的冲突项。"""

    pricing_groups: dict[tuple[str, str, str, str, str], list[object]] = {}
    for pricing_plan in profile.pricing:
        duplicate_key = build_pricing_duplicate_key(pricing_plan)
        if duplicate_key not in pricing_groups:
            pricing_groups[duplicate_key] = []
        pricing_groups[duplicate_key].append(pricing_plan)

    conflicting_keys: set[tuple[str, str, str, str, str]] = set()
    for duplicate_key, pricing_plans in pricing_groups.items():
        normalized_prices = {
            normalize_scope_text(pricing_plan.price or "")
            for pricing_plan in pricing_plans
        }
        if len(normalized_prices) > 1:
            conflicting_keys.add(duplicate_key)

    kept_pricing = []
    seen_duplicate_keys: set[tuple[str, str, str, str, str]] = set()
    changed = bool(conflicting_keys)
    for pricing_plan in profile.pricing:
        duplicate_key = build_pricing_duplicate_key(pricing_plan)
        if duplicate_key in conflicting_keys:
            changed = True
            continue
        if duplicate_key in seen_duplicate_keys:
            changed = True
            continue
        seen_duplicate_keys.add(duplicate_key)
        kept_pricing.append(pricing_plan)

    if not changed:
        return profile

    return profile.model_copy(update={"pricing": kept_pricing})


def build_pricing_duplicate_key(
    pricing_plan: object,
) -> tuple[str, str, str, str, str]:
    """用套餐、周期、单位、服务等级和阈值构造价格比较 key。"""

    plan_name = getattr(pricing_plan, "plan_name", "")
    price_text = getattr(pricing_plan, "price", None)
    billing_cycle = getattr(pricing_plan, "billing_cycle", None)
    billing_category = detect_billing_cycle_category(billing_cycle)
    if billing_category is None:
        billing_category = detect_billing_cycle_category(price_text)
    if billing_category is None:
        billing_category = "unknown"

    unit = normalize_scope_text(getattr(pricing_plan, "unit", None) or "")
    service_level = normalize_scope_text(
        getattr(pricing_plan, "service_level", None) or ""
    )
    threshold = normalize_scope_text(
        getattr(pricing_plan, "threshold", None) or ""
    )
    return (
        normalize_scope_text(plan_name),
        billing_category,
        unit,
        service_level,
        threshold,
    )


def infer_billing_cycle(price: str | None) -> str | None:
    """从价格文本中识别常见计费周期，无法确定时保持为空。"""

    return detect_billing_cycle_category(price)


def infer_positioning_from_evidence(
    evidence: Sequence[Evidence],
) -> str | None:
    """从定位证据中选择最像产品定位的一句短文本。"""

    best_candidate: str | None = None
    best_score = 0
    for item in sorted_positioning_evidence(evidence):
        candidate_sentences = extract_positioning_sentences(item)
        for sentence in candidate_sentences:
            score = score_positioning_sentence(sentence)
            if score > best_score:
                best_score = score
                best_candidate = sentence

    if best_score < 2:
        return None
    return best_candidate


def sorted_positioning_evidence(
    evidence: Sequence[Evidence],
) -> list[Evidence]:
    """把定位候选证据排在前面，并排除明显的价格页。"""

    candidate_evidence = [
        item for item in evidence if is_positioning_evidence_candidate(item)
    ]

    return sorted(
        candidate_evidence,
        key=lambda item: 0 if item.topic.lower() == "positioning" else 1,
    )


def is_positioning_evidence_candidate(item: Evidence) -> bool:
    """判断 Evidence 是否适合作为产品级定位来源。"""

    normalized_topic = item.topic.strip().lower()
    if normalized_topic not in POSITIONING_TOPICS:
        return False

    evidence_text = normalize_scope_text(build_evidence_scope_text(item))
    for marker in PRICING_PAGE_MARKERS:
        if marker in evidence_text:
            return False

    return True


def extract_positioning_sentences(item: Evidence) -> list[str]:
    """从标题和摘要中拆出可作为定位候选的短句。"""

    raw_texts = [item.snippet, item.title]
    sentences: list[str] = []
    for raw_text in raw_texts:
        cleaned_text = clean_markdown_text(raw_text)
        for sentence in split_sentences(cleaned_text):
            if len(sentence) > 180:
                continue
            sentences.append(sentence)

    return sentences


def build_evidence_scope_text(item: Evidence) -> str:
    """拼接 Evidence 的可见文本，用于范围判断，不改变原始证据。"""

    text_parts = [
        item.title,
        str(item.url),
        item.snippet,
    ]
    if item.raw_content:
        text_parts.append(item.raw_content)

    return " ".join(text_parts)


def normalize_scope_text(text: str) -> str:
    """把范围判断文本统一成小写单空格，便于做保守关键词匹配。"""

    lowered_text = text.lower()
    compact_text = re.sub(r"\s+", " ", lowered_text)
    return compact_text.strip()


def clean_markdown_text(raw_text: str) -> str:
    """去掉常见 Markdown 标记，保留用户可读的网页文字。"""

    without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", raw_text)
    without_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", without_images)
    without_heading_marks = re.sub(r"#+\s*", " ", without_links)
    compact_text = re.sub(r"\s+", " ", without_heading_marks)
    return compact_text.strip(" -*|")


def split_sentences(text: str) -> list[str]:
    """按中英文句号和常见分隔符拆句，并删除空句。"""

    raw_sentences = re.split(r"(?<=[.!?。！？])\s+|\s+-\s+|\s+\|\s+", text)
    sentences: list[str] = []
    for raw_sentence in raw_sentences:
        sentence = raw_sentence.strip(" -*|")
        if sentence:
            sentences.append(sentence)
    return sentences


def score_positioning_sentence(sentence: str) -> int:
    """用关键词给候选定位句打分，分数太低的不自动补齐。"""

    lowered_sentence = sentence.lower()
    if any(
        marker in lowered_sentence for marker in POSITIONING_NEGATIVE_MARKERS
    ):
        return 0
    score = 0
    for keyword in POSITIONING_KEYWORDS:
        if keyword in lowered_sentence:
            score += 2

    if "workspace" in lowered_sentence:
        score += 3
    if "one place" in lowered_sentence:
        score += 2

    return score


def collect_profile_evidence_ids(
    profile: ProductProfile,
) -> set[str]:
    """收集画像中所有功能和价格项引用的 Evidence ID。"""

    evidence_ids: set[str] = set()

    for feature in profile.features:
        evidence_ids.update(feature.evidence_ids)

    for pricing_plan in profile.pricing:
        evidence_ids.update(pricing_plan.evidence_ids)

    for source in profile.source_evidence:
        evidence_ids.add(source.evidence_id)

    for field_evidence in profile.field_evidence:
        evidence_ids.add(field_evidence.evidence_id)

    for model in profile.models:
        for source in model.source_evidence:
            evidence_ids.add(source.evidence_id)

    for finding in profile.dimension_findings:
        evidence_ids.update(finding.evidence_ids)

    return evidence_ids
