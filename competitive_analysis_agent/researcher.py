"""把调研任务转换成带上下文、可追溯的搜索证据。"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from competitive_analysis_agent.schemas import (
    ContractModel,
    Evidence,
    MarketDefinition,
    RequiredText,
    ResearchTask,
)
from competitive_analysis_agent.search import (
    SearchAdapter,
    SearchRequest,
    SearchResult,
)


ResearchErrorCode = Literal[
    "timeout",
    "provider_error",
    "no_results",
    "profile_validation",
]
PRICING_TOPICS = {"pricing", "api_pricing"}
API_TOPIC_SEARCH_TERMS = {
    "model_capabilities": "models multimodal capabilities",
    "api_pricing": "API pricing input output tokens",
    "developer_platform": "API documentation SDK tools",
    "usage_limits": "context window rate limits",
}
API_TOPIC_SCOPE_MARKERS = {
    "model_capabilities": (
        "model",
        "multimodal",
        "reasoning",
        "tool use",
    ),
    "api_pricing": (
        "api pricing",
        "input token",
        "output token",
        "tokens",
    ),
    "developer_platform": (
        "api documentation",
        "api docs",
        "sdk",
        "function calling",
        "developer platform",
    ),
    "usage_limits": (
        "context window",
        "rate limit",
        "quota",
        "usage limit",
    ),
}
PRICING_RAW_CONTENT_MAX_CHARS = 6000
PRICING_RAW_CONTENT_MAX_LINES = 80
PRICING_CONTEXT_WINDOW = 1
PRICING_PLAN_TERMS = {
    "free",
    "plus",
    "standard",
    "business",
    "enterprise",
    "premium",
    "team",
    "starter",
    "pro",
    "basic",
}
PRICING_KEYWORDS = {
    "price",
    "pricing",
    "monthly",
    "yearly",
    "annual",
    "annually",
    "per month",
    "per year",
    "per user",
    "per member",
    "per seat",
    "custom pricing",
    "contact sales",
    "user limit",
    "storage",
}
PRICE_TEXT_PATTERN = re.compile(
    r"(?i)(?:[$€£¥￥]\s?\d+|\b\d+(?:\.\d+)?\s?(?:usd|eur|gbp|rmb|cny)\b)"
)


class ResearcherInput(ContractModel):
    """描述 Researcher 运行所需的任务和搜索配置。"""

    tasks: list[ResearchTask] = Field(min_length=1)
    market_definition: MarketDefinition
    official_domains_by_product: dict[str, list[RequiredText]] = Field(
        default_factory=dict
    )
    max_results_per_task: int = Field(default=3, ge=1, le=10)


class ResearchError(ContractModel):
    """记录单个调研任务的失败，不中断其他任务。"""

    product_name: RequiredText
    topic: RequiredText
    query: RequiredText
    code: ResearchErrorCode
    message: RequiredText


class ResearchResult(ContractModel):
    """返回范围内、排除、待核验证据和未完成任务错误。"""

    evidence: list[Evidence] = Field(default_factory=list)
    excluded_evidence: list[Evidence] = Field(default_factory=list)
    uncertain_evidence: list[Evidence] = Field(default_factory=list)
    errors: list[ResearchError] = Field(default_factory=list)


def utc_now() -> datetime:
    """返回带时区的 UTC 时间，作为生产环境默认时钟。"""

    return datetime.now(timezone.utc)


class Researcher:
    """按输入顺序执行搜索任务，并生成稳定的证据列表。"""

    def __init__(
        self,
        search_adapter: SearchAdapter,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._search_adapter = search_adapter
        self._clock = clock

    def research(self, researcher_input: ResearcherInput) -> ResearchResult:
        """执行全部任务；单个任务失败时记录错误并继续。"""

        evidence: list[Evidence] = []
        excluded_evidence: list[Evidence] = []
        uncertain_evidence: list[Evidence] = []
        errors: list[ResearchError] = []
        seen_evidence_keys: set[tuple[str, str, str]] = set()
        collected_at = self._clock()
        next_evidence_number = 1

        for task in researcher_input.tasks:
            effective_query = build_focused_search_query(
                task,
                researcher_input.market_definition,
            )
            request = SearchRequest(
                query=effective_query,
                official_domains=researcher_input.official_domains_by_product.get(
                    task.product_name,
                    [],
                ),
                max_results=researcher_input.max_results_per_task,
                include_raw_content=should_request_raw_content(task),
                search_depth=(
                    "advanced"
                    if should_request_raw_content(task)
                    else "basic"
                ),
                chunks_per_source=3,
                extract_query=(
                    effective_query
                    if should_request_raw_content(task)
                    else None
                ),
                extract_top_results=(
                    min(2, researcher_input.max_results_per_task)
                    if should_request_raw_content(task)
                    else 0
                ),
            )
            response = self._search_adapter.search(request)

            if response.status == "error":
                error_code: ResearchErrorCode = "provider_error"
                error_message = "Search failed without error details."
                if response.error is not None:
                    error_code = response.error.code
                    error_message = response.error.message

                errors.append(
                    ResearchError(
                        product_name=task.product_name,
                        topic=task.topic,
                        query=effective_query,
                        code=error_code,
                        message=error_message,
                    )
                )
                continue

            if not response.results:
                errors.append(
                    ResearchError(
                        product_name=task.product_name,
                        topic=task.topic,
                        query=effective_query,
                        code="no_results",
                        message="Search completed but returned no results.",
                    )
                )
                continue

            extraction_errors = {
                result.extraction_error
                for result in response.results
                if result.extraction_error is not None
            }
            for extraction_error in sorted(extraction_errors):
                errors.append(
                    ResearchError(
                        product_name=task.product_name,
                        topic=task.topic,
                        query=effective_query,
                        code="provider_error",
                        message=(
                            "Tavily Extract failed; Search content was kept: "
                            f"{extraction_error}"
                        ),
                    )
                )

            for search_result in response.results:
                normalized_url = str(search_result.url)
                deduplication_key = build_evidence_deduplication_key(
                    task=task,
                    normalized_url=normalized_url,
                )
                if deduplication_key in seen_evidence_keys:
                    continue

                raw_content_excerpt = build_relevant_raw_content_excerpt(
                    raw_content=search_result.raw_content,
                    topic=task.topic,
                    content_is_query_focused=(
                        search_result.extracted_content
                    ),
                )
                snippet = build_evidence_snippet(
                    snippet=search_result.snippet,
                    raw_content_excerpt=raw_content_excerpt,
                    topic=task.topic,
                )
                scope_status, scope_reason = classify_evidence_scope(
                    search_result=search_result,
                    task=task,
                    market_definition=researcher_input.market_definition,
                )
                evidence_item = Evidence(
                    evidence_id=f"E{next_evidence_number}",
                    product_name=task.product_name,
                    topic=task.topic,
                    title=search_result.title,
                    url=search_result.url,
                    snippet=snippet,
                    raw_content=raw_content_excerpt,
                    source_type=search_result.source_type,
                    scope_status=scope_status,
                    scope_reason=scope_reason,
                    collected_at=collected_at,
                )
                next_evidence_number += 1

                # 只有范围内资料进入 Extractor；另外两类仍保留完整追溯信息。
                if scope_status == "in_scope":
                    evidence.append(evidence_item)
                elif scope_status == "out_of_scope":
                    excluded_evidence.append(evidence_item)
                else:
                    uncertain_evidence.append(evidence_item)
                seen_evidence_keys.add(deduplication_key)

        return ResearchResult(
            evidence=evidence,
            excluded_evidence=excluded_evidence,
            uncertain_evidence=uncertain_evidence,
            errors=errors,
        )


def build_focused_search_query(
    task: ResearchTask,
    market_definition: MarketDefinition,
) -> str:
    """把审计用范围任务转换成供应商容易召回资料的聚焦查询。"""

    focused_query = task.query
    for exclusion in market_definition.exclusions:
        exclusion_clause = rf"\bexclude\s+{re.escape(exclusion)}"
        focused_query = re.sub(
            exclusion_clause,
            " ",
            focused_query,
            flags=re.IGNORECASE,
        )

    # 产品类别和比较层级用于约束后续范围，不适合作为搜索关键词反复发送。
    removed_scope_term = False
    for scope_term in [
        market_definition.product_category,
        market_definition.comparison_level,
    ]:
        updated_query = re.sub(
            re.escape(scope_term),
            " ",
            focused_query,
            flags=re.IGNORECASE,
        )
        if updated_query != focused_query:
            removed_scope_term = True
        focused_query = updated_query

    if removed_scope_term:
        normalized_topic = task.topic.strip().lower()
        if normalized_topic in API_TOPIC_SEARCH_TERMS:
            search_topic = API_TOPIC_SEARCH_TERMS[normalized_topic]
        elif normalized_topic == "pricing":
            search_topic = "pricing plans price"
        elif normalized_topic == "features":
            search_topic = "product features collaboration"
        else:
            search_topic = task.topic
        return f"{task.product_name} official {search_topic}"

    # 已经聚焦的手写任务只移除 exclude 子句，避免无谓改写调用方查询。
    return re.sub(r"\s+", " ", focused_query).strip()


def classify_evidence_scope(
    search_result: SearchResult,
    task: ResearchTask,
    market_definition: MarketDefinition,
) -> tuple[Literal["in_scope", "out_of_scope", "uncertain"], str]:
    """按显式排除项和可见文本确定性判断证据范围。"""

    source_text = " ".join(
        [
            search_result.title,
            search_result.snippet,
            search_result.raw_content or "",
        ]
    )
    normalized_source_text = normalize_scope_text(source_text)
    normalized_topic = task.topic.strip().lower()

    # 厂商托管的社区仍是用户内容；API 定价只能把它保留为待核验资料。
    source_host = (search_result.url.host or "").casefold()
    if normalized_topic == "api_pricing" and source_host.startswith(
        ("community.", "forum.", "discuss.")
    ):
        return "uncertain", "社区讨论不能作为权威 API 价表。"

    # 显式排除项优先级最高，同品牌其他产品线不能因来源官方而通过。
    for exclusion in market_definition.exclusions:
        if normalize_scope_text(exclusion) in normalized_source_text:
            return "out_of_scope", f"命中排除项：{exclusion}"

    product_matches = product_name_matches_scope_text(
        product_name=task.product_name,
        normalized_source_text=normalized_source_text,
    )

    if normalized_topic in API_TOPIC_SCOPE_MARKERS:
        topic_matches = any(
            marker in normalized_source_text
            for marker in API_TOPIC_SCOPE_MARKERS[normalized_topic]
        )
        if product_matches and topic_matches:
            return "in_scope", "文本同时匹配 API 产品与当前分析主题。"
        if search_result.source_type == "official" and product_matches:
            return "uncertain", "官方资料包含目标产品，但没有回答当前 API 主题。"
        return "uncertain", "现有文本不足以确认 API 产品与主题边界。"

    if search_result.source_type == "official" and product_matches:
        return "in_scope", "官方来源且文本包含目标产品名称。"

    category_matches = (
        normalize_scope_text(market_definition.product_category)
        in normalized_source_text
    )
    level_matches = (
        normalize_scope_text(market_definition.comparison_level)
        in normalized_source_text
    )
    if product_matches and (category_matches or level_matches):
        return "in_scope", "文本包含目标产品和市场范围描述。"

    return "uncertain", "现有文本不足以确认产品边界。"


def normalize_scope_text(value: str) -> str:
    """压缩空白并统一大小写，供范围关键词稳定匹配。"""

    return re.sub(r"\s+", " ", value).strip().casefold()


def product_name_matches_scope_text(
    product_name: str,
    normalized_source_text: str,
) -> bool:
    """匹配完整 API 产品名，或同时出现品牌名与 API 标记。"""

    normalized_product = normalize_scope_text(product_name)
    if normalized_product in normalized_source_text:
        return True

    if not normalized_product.endswith(" api"):
        return False

    brand_name = normalized_product.removesuffix(" api").strip()
    return brand_name in normalized_source_text and bool(
        re.search(r"\bapi\b", normalized_source_text)
    )


def normalize_product_name_for_scope(product_name: str) -> str:
    """把产品名压缩成 scope 查询使用的稳定小写 key。"""

    compact_name = re.sub(r"\s+", " ", product_name.lower())
    return compact_name.strip()


def build_evidence_deduplication_key(
    task: ResearchTask,
    normalized_url: str,
) -> tuple[str, str, str]:
    """用产品、主题和 URL 去重，保留同一页面在不同 topic 下的上下文。"""

    normalized_product_name = normalize_product_name_for_scope(
        task.product_name
    )
    normalized_topic = task.topic.strip().lower()
    return normalized_product_name, normalized_topic, normalized_url


def should_request_raw_content(task: ResearchTask) -> bool:
    """只有订阅或 API 价格任务请求网页正文。"""

    return task.topic.strip().lower() in PRICING_TOPICS


def build_relevant_raw_content_excerpt(
    raw_content: str | None,
    topic: str,
    content_is_query_focused: bool = False,
) -> str | None:
    """从网页正文中裁剪和当前 topic 相关的片段。"""

    if raw_content is None:
        return None

    if topic.strip().lower() not in PRICING_TOPICS:
        return None

    if content_is_query_focused:
        # Tavily Extract 已按 query 选出相关 chunks，只保留统一长度上限。
        return trim_excerpt_lines(normalize_raw_content_lines(raw_content))

    return extract_pricing_excerpt(raw_content)


def extract_pricing_excerpt(raw_content: str) -> str | None:
    """保留价格页正文里含套餐、价格、周期和限制的短片段。"""

    normalized_lines = normalize_raw_content_lines(raw_content)
    if not normalized_lines:
        return None

    selected_indexes: set[int] = set()
    for line_index, line in enumerate(normalized_lines):
        if not is_pricing_related_line(line):
            continue

        start_index = max(0, line_index - PRICING_CONTEXT_WINDOW)
        end_index = min(
            len(normalized_lines),
            line_index + PRICING_CONTEXT_WINDOW + 1,
        )
        for selected_index in range(start_index, end_index):
            selected_indexes.add(selected_index)

    if not selected_indexes:
        return None

    selected_lines = [
        normalized_lines[index]
        for index in sorted(selected_indexes)
    ]
    return trim_excerpt_lines(selected_lines)


def normalize_raw_content_lines(raw_content: str) -> list[str]:
    """把网页正文拆成干净短行，方便后续按价格关键词筛选。"""

    lines: list[str] = []
    for raw_line in raw_content.replace("\r", "\n").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lines.append(line)
    return lines


def is_pricing_related_line(line: str) -> bool:
    """判断一行正文是否像价格、套餐、计费周期或限制。"""

    normalized_line = line.lower()
    if PRICE_TEXT_PATTERN.search(line):
        return True

    for keyword in PRICING_KEYWORDS:
        if keyword in normalized_line:
            return True

    line_words = {
        word.strip(" .,:;()[]{}")
        for word in normalized_line.split()
    }
    if line_words.intersection(PRICING_PLAN_TERMS):
        return True

    return False


def trim_excerpt_lines(lines: list[str]) -> str | None:
    """限制价格片段行数和长度，避免把整页正文塞进模型。"""

    excerpt_lines: list[str] = []
    current_length = 0
    for line in lines:
        if len(excerpt_lines) >= PRICING_RAW_CONTENT_MAX_LINES:
            break

        next_length = current_length + len(line) + 1
        if next_length > PRICING_RAW_CONTENT_MAX_CHARS:
            break

        excerpt_lines.append(line)
        current_length = next_length

    if not excerpt_lines:
        return None
    return "\n".join(excerpt_lines)


def build_evidence_snippet(
    snippet: str,
    raw_content_excerpt: str | None,
    topic: str,
) -> str:
    """把搜索摘要和可控正文片段合并成 Extractor 可读的 Evidence 文本。"""

    if raw_content_excerpt is None:
        return snippet

    if topic.strip().lower() not in PRICING_TOPICS:
        return snippet

    return (
        f"{snippet}\n\n"
        "Pricing page excerpt:\n"
        f"{raw_content_excerpt}"
    )
