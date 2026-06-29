"""Extractor 节点：只根据给定证据生成可追溯的产品画像。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from competitive_analysis_agent.pricing_utils import (
    detect_billing_cycle_category,
    is_custom_pricing_text,
    is_free_price_text,
    normalize_billing_cycle_value,
    normalize_price_value,
)
from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    PricingPlan,
    ProductProfile,
)


EXTRACTOR_SYSTEM_PROMPT = """
你是竞品分析流程中的 Extractor。

你的唯一职责是把当前产品的 Evidence 提取成一个 ProductProfile。

要求：
1. 只能使用用户消息中提供的 Evidence，不得使用训练记忆、常识或外部资料补充事实。
2. product_name 必须逐字复制用户消息中的产品名。
3. 每个 feature 和 pricing 项必须引用至少一个当前 Evidence 中真实存在的 evidence_id。
4. evidence_ids 只能引用直接支持该条目的证据，不得虚构或引用其他产品的证据。
5. Evidence 没有明确提供的信息必须保持为空：
   positioning 使用 null，列表字段使用 []，价格和计费周期使用 null。
6. 不得把没有明确写出的目标用户、优势、限制、价格或计费周期推测出来。
7. positioning 可以来自官网标题、产品标语或产品概览中直接出现的产品类别，
   例如 "AI workspace"、"workspace for knowledge and collaboration"。
   不得把价格页里的套餐适用人群、额度说明或订阅说明当作产品定位。
8. target_users 可以来自 use case、customer、team、enterprise、small business
   等页面中直接写出的用户群，例如 "enterprise teams"、"marketing teams"、
   "small and medium sized teams"；不得从功能名称反推用户。
9. 如果证据只写出了方案名称但没有价格，可以保留该方案，并把 price 和 billing_cycle 设为 null。
10. 如果 Evidence 包含 raw_content，它是 Researcher 从网页正文中裁剪出的
    价格页正文片段。pricing 提取可以同时使用 snippet 和 raw_content 中明确出现的
    plan_name、price、billing_cycle 和 main_limits；仍然不得使用 Evidence 外的信息。
    如果证据页明显属于另一个产品线，例如当前产品是 Gemini 但价格项来自
    Google Home Premium，则不要放入当前产品 pricing。
    默认产品范围：
    - ChatGPT 表示 OpenAI / ChatGPT API 价格，只提取 developer platform、
      model pricing、token、input/output 等 API 价格。
    - Claude 表示 Anthropic Claude API 价格，只提取 API、console、token、
      model pricing、input/output 等 API 价格。
    - Gemini 表示 Gemini API / Google AI API 价格，只提取 ai.google.dev、
      token、model pricing、input/output 等 API 价格。
    不要把 ChatGPT Plus/Pro/Team/Business、Claude Pro/Max/Team、
    Gemini App、Workspace Gemini、Google Home Premium、Veo 等订阅或
    非 API 产品价格放入 pricing。
11. 只输出 JSON 对象，不要添加 Markdown 或解释。
12. features 中每一项必须同时包含以下三个字段，description 不得省略：
   {"name": "...", "description": "...", "evidence_ids": ["E1"]}
13. pricing 中每一项必须包含 plan_name、price、billing_cycle、
    main_limits 和 evidence_ids；未知值使用 null 或 []，不得省略字段。
14. 顶层格式必须是：
   {"profile": {"product_name": "...", "positioning": null,
   "target_users": [], "features": [], "pricing": [],
   "strengths": [], "limitations": []}}
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
PRODUCT_SCOPE_EXCLUSION_MARKERS = {
    "gemini": (
        "google home premium",
        "google home",
        "home premium",
        "nest aware",
    )
}
EXTRACTOR_TITLE_MAX_CHARS = 160
EXTRACTOR_SNIPPET_MAX_CHARS = 700
EXTRACTOR_RAW_CONTENT_MAX_CHARS = 1400
EXTRACTOR_MAX_EVIDENCE_PER_PRODUCT = 6
EXTRACTOR_MAX_EVIDENCE_PER_TOPIC = 2
EXTRACTOR_TOPIC_PRIORITY = (
    "pricing",
    "features",
    "positioning",
    "target_users",
)
PricingScopeClassification = Literal[
    "api_pricing",
    "non_api_pricing",
    "ambiguous",
    "not_applicable",
]


@dataclass(frozen=True, slots=True)
class ApiPricingScopeRules:
    """保存默认模型产品的 API 价格范围关键词。"""

    api_markers: tuple[str, ...]
    non_api_markers: tuple[str, ...]


API_PRICING_SCOPE_RULES_BY_PRODUCT_KEY = {
    "chatgpt": ApiPricingScopeRules(
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
    "openai": ApiPricingScopeRules(
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
    "claude": ApiPricingScopeRules(
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
    "anthropic": ApiPricingScopeRules(
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
    "gemini": ApiPricingScopeRules(
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
    "google ai": ApiPricingScopeRules(
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

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "ExtractorInput":
        """确保一个 Evidence ID 在本次提取中只对应一条证据。"""

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique.")

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

        structured_response = self._structured_model.invoke(messages)
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

    # 第二轮再补充每个 topic 的第二条证据，价格和功能通常最值得保留。
    add_evidence_round(
        selected_evidence=selected_evidence,
        selected_ids=selected_ids,
        evidence_by_topic=evidence_by_topic,
        ordered_topics=ordered_topics,
        round_index=1,
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
) -> list[dict[str, str]]:
    """把单个产品的 Evidence 转换成模型可读取的消息。"""

    evidence_json = json.dumps(
        [build_extraction_evidence_item(item) for item in evidence],
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        f"产品名：{product_name}\n"
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
        "如果维度很多，可以减少自定义维度或减少每个产品的官方域名后重试。"
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
        "每个 pricing 项必须包含 plan_name、price、billing_cycle、"
        "main_limits、evidence_ids，未知值使用 null 或 []。\n"
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
) -> ExtractorOutput:
    """校验输出结构、产品名和所有功能及价格的证据引用。"""

    normalized_output = normalize_extractor_raw_output(raw_output)
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
    referenced_evidence_ids = collect_profile_evidence_ids(profile)
    unknown_evidence_ids = (
        referenced_evidence_ids - allowed_evidence_ids
    )
    if unknown_evidence_ids:
        unknown_text = ", ".join(sorted(unknown_evidence_ids))
        raise ExtractorValidationError(
            f"Profile references unknown evidence IDs: {unknown_text}"
        )

    enriched_profile = normalize_profile_summary_fields(
        profile=profile,
        evidence=evidence,
    )
    if enriched_profile is profile:
        return extractor_output

    return ExtractorOutput(profile=enriched_profile)


def normalize_extractor_raw_output(raw_output: object) -> object:
    """在 Schema 校验前修正模型常见但可安全转换的嵌套形状。"""

    if isinstance(raw_output, str):
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
        return normalize_extractor_raw_output(parsed_output)

    if not isinstance(raw_output, dict):
        return raw_output

    normalized_output = raw_output.copy()
    raw_profile = normalized_output.get("profile")
    if not isinstance(raw_profile, dict):
        return normalized_output

    normalized_profile = raw_profile.copy()
    raw_pricing = normalized_profile.get("pricing")
    if isinstance(raw_pricing, list):
        normalized_profile["pricing"] = normalize_pricing_items(
            raw_pricing
        )

    normalized_output["profile"] = normalized_profile
    return normalized_output


def normalize_pricing_items(raw_pricing: list[object]) -> list[object]:
    """只规范化 pricing 中的 main_limits，保留其他字段给 Pydantic 校验。"""

    normalized_pricing: list[object] = []
    for raw_plan in raw_pricing:
        if not isinstance(raw_plan, dict):
            normalized_pricing.append(raw_plan)
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
    priced_profile = normalize_pricing_defaults(positioned_profile)
    scoped_pricing_profile = filter_pricing_plans_by_product_scope(
        profile=priced_profile,
        evidence=evidence,
    )
    deduplicated_profile = remove_conflicting_pricing_duplicates(
        scoped_pricing_profile
    )
    return deduplicated_profile


def remove_plan_level_positioning(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """删除模型从套餐页误提取出来的产品定位。"""

    if profile.positioning is None:
        return profile

    if positioning_has_direct_product_support(profile.positioning, evidence):
        return profile

    if not looks_like_plan_level_positioning(profile.positioning):
        return profile

    return profile.model_copy(update={"positioning": None})


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


def filter_pricing_plans_by_product_scope(
    profile: ProductProfile,
    evidence: Sequence[Evidence],
) -> ProductProfile:
    """删除明显来自其他产品线的价格项，避免官方域名里的旁支页面污染画像。"""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    kept_pricing = []
    changed = False

    for pricing_plan in profile.pricing:
        if pricing_plan_matches_requested_scope(
            product_name=profile.product_name,
            pricing_plan=pricing_plan,
            evidence_by_id=evidence_by_id,
        ):
            kept_pricing.append(pricing_plan)
            continue

        changed = True

    if not changed:
        return profile

    return profile.model_copy(update={"pricing": kept_pricing})


def pricing_plan_matches_requested_scope(
    product_name: str,
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> bool:
    """判断价格项是否符合当前默认请求范围。"""

    scope_classification = classify_pricing_source_scope(
        product_name=product_name,
        pricing_plan=pricing_plan,
        evidence_by_id=evidence_by_id,
    )
    if scope_classification == "not_applicable":
        return pricing_plan_matches_product_scope(
            product_name=product_name,
            pricing_plan=pricing_plan,
            evidence_by_id=evidence_by_id,
        )

    # 默认模型产品只接受明确 API pricing；非 API 和无法判断的价格都先删除。
    return scope_classification == "api_pricing"


def classify_pricing_source_scope(
    product_name: str,
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> PricingScopeClassification:
    """把价格来源分成 API、非 API、模糊或不适用四类。"""

    scope_rules = build_api_pricing_scope_rules(product_name)
    if scope_rules is None:
        return "not_applicable"

    scope_text = build_pricing_plan_scope_text(
        product_name=product_name,
        pricing_plan=pricing_plan,
        evidence_by_id=evidence_by_id,
    )
    has_api_marker = scope_text_has_any_marker(
        scope_text,
        scope_rules.api_markers,
    )
    has_non_api_marker = scope_text_has_any_marker(
        scope_text,
        scope_rules.non_api_markers,
    )

    if has_api_marker and not has_non_api_marker:
        return "api_pricing"
    if has_non_api_marker and not has_api_marker:
        return "non_api_pricing"

    # 同时命中 API 与订阅/旁支产品时，宁可认为该价格来源语义混杂。
    return "ambiguous"


def build_api_pricing_scope_rules(
    product_name: str,
) -> ApiPricingScopeRules | None:
    """为默认模型产品返回 API 价格范围规则；其他产品不套用该规则。"""

    normalized_product = normalize_scope_text(product_name)
    for product_key, scope_rules in API_PRICING_SCOPE_RULES_BY_PRODUCT_KEY.items():
        if product_key == normalized_product:
            return scope_rules
        if scope_text_contains_marker(normalized_product, product_key):
            return scope_rules

    return None


def build_pricing_plan_scope_text(
    product_name: str,
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> str:
    """拼接价格项和来源证据文本，用于判断 API pricing 范围。"""

    scope_text_parts = [
        product_name,
        pricing_plan.plan_name,
        pricing_plan.price or "",
        pricing_plan.billing_cycle or "",
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


def pricing_plan_matches_product_scope(
    product_name: str,
    pricing_plan: PricingPlan,
    evidence_by_id: dict[str, Evidence],
) -> bool:
    """判断一个价格项是否明显偏离当前产品范围。"""

    scope_text_parts = [product_name]
    scope_text_parts.append(pricing_plan.plan_name)

    for evidence_id in pricing_plan.evidence_ids:
        evidence_item = evidence_by_id.get(evidence_id)
        if evidence_item is None:
            continue
        scope_text_parts.append(build_evidence_scope_text(evidence_item))

    scope_text = normalize_scope_text(" ".join(scope_text_parts))
    product_key = normalize_scope_text(product_name)
    exclusion_markers = PRODUCT_SCOPE_EXCLUSION_MARKERS.get(product_key, ())

    for marker in exclusion_markers:
        if marker in scope_text:
            return False

    return True


def remove_conflicting_pricing_duplicates(
    profile: ProductProfile,
) -> ProductProfile:
    """删除同一套餐同一计费周期下出现多个价格的冲突项。"""

    pricing_groups: dict[tuple[str, str], list[object]] = {}
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

    kept_pricing = []
    seen_duplicate_keys: set[tuple[str, str]] = set()
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


def build_pricing_duplicate_key(pricing_plan: object) -> tuple[str, str]:
    """把套餐名和计费周期归一化成用于冲突检测的 key。"""

    plan_name = getattr(pricing_plan, "plan_name", "")
    price_text = getattr(pricing_plan, "price", None)
    billing_cycle = getattr(pricing_plan, "billing_cycle", None)
    billing_category = detect_billing_cycle_category(billing_cycle)
    if billing_category is None:
        billing_category = detect_billing_cycle_category(price_text)
    if billing_category is None:
        billing_category = "unknown"

    return normalize_scope_text(plan_name), billing_category


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
    if normalized_topic == "pricing":
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

    return evidence_ids
