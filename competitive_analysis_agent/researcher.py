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
    RequiredText,
    ResearchTask,
)
from competitive_analysis_agent.search import SearchAdapter, SearchRequest


ResearchErrorCode = Literal[
    "timeout",
    "provider_error",
    "no_results",
    "profile_validation",
]
TOPIC_QUERY_TERMS = {
    "features": "official product features capabilities",
    "pricing": (
        "official pricing plans price Free Plus Standard Business Enterprise"
    ),
    "positioning": "official product overview workspace teams business",
    "target_users": (
        "official use cases customers teams enterprise small business"
    ),
    "limitations": "official limits storage users plan restrictions",
}
API_PRICING_QUERY_TERMS_BY_PRODUCT = {
    "chatgpt": (
        "official API pricing developer platform token model pricing "
        "input output"
    ),
    "openai": (
        "official API pricing developer platform token model pricing "
        "input output"
    ),
    "claude": (
        "official Anthropic Claude API pricing console token model pricing "
        "input output"
    ),
    "anthropic": (
        "official Anthropic Claude API pricing console token model pricing "
        "input output"
    ),
    "gemini": (
        "official Gemini API Google AI API pricing ai.google.dev token "
        "model pricing input output"
    ),
    "google ai": (
        "official Gemini API Google AI API pricing ai.google.dev token "
        "model pricing input output"
    ),
}
PRICING_TOPIC = "pricing"
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
    """同时返回成功证据和未完成任务的结构化错误。"""

    evidence: list[Evidence] = Field(default_factory=list)
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
        errors: list[ResearchError] = []
        seen_evidence_keys: set[tuple[str, str, str]] = set()
        collected_at = self._clock()

        for task in researcher_input.tasks:
            effective_query = build_focused_search_query(task)
            request = SearchRequest(
                query=effective_query,
                official_domains=researcher_input.official_domains_by_product.get(
                    task.product_name,
                    [],
                ),
                max_results=researcher_input.max_results_per_task,
                include_raw_content=should_request_raw_content(task),
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
                )
                snippet = build_evidence_snippet(
                    snippet=search_result.snippet,
                    raw_content_excerpt=raw_content_excerpt,
                    topic=task.topic,
                )
                evidence_id = f"E{len(evidence) + 1}"
                evidence.append(
                    Evidence(
                        evidence_id=evidence_id,
                        product_name=task.product_name,
                        topic=task.topic,
                        title=search_result.title,
                        url=search_result.url,
                        snippet=snippet,
                        raw_content=raw_content_excerpt,
                        source_type=search_result.source_type,
                        collected_at=collected_at,
                    )
                )
                seen_evidence_keys.add(deduplication_key)

        return ResearchResult(evidence=evidence, errors=errors)


def build_focused_search_query(task: ResearchTask) -> str:
    """按分析维度生成更适合搜索引擎的查询语句。"""

    normalized_topic = task.topic.strip().lower()
    if normalized_topic == PRICING_TOPIC:
        pricing_terms = build_pricing_search_terms(task.product_name)
        return f"{task.product_name} {pricing_terms}"

    topic_terms = TOPIC_QUERY_TERMS.get(normalized_topic)
    if topic_terms is None:
        return task.query

    # 不直接使用 positioning / target_users 这类内部字段名，避免命中模板、
    # API 类名或开发者文档，而是换成用户会在官网页面看到的表达。
    return f"{task.product_name} {topic_terms}"


def build_pricing_search_terms(product_name: str) -> str:
    """为价格任务选择查询词；默认模型产品优先搜索 API/token 价格。"""

    normalized_product = normalize_product_name_for_scope(product_name)
    api_pricing_terms = API_PRICING_QUERY_TERMS_BY_PRODUCT.get(
        normalized_product
    )
    if api_pricing_terms is not None:
        return api_pricing_terms

    return TOPIC_QUERY_TERMS[PRICING_TOPIC]


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
    """只有 pricing 任务请求网页正文，控制成本和后续模型输入长度。"""

    return task.topic.strip().lower() == PRICING_TOPIC


def build_relevant_raw_content_excerpt(
    raw_content: str | None,
    topic: str,
) -> str | None:
    """从网页正文中裁剪和当前 topic 相关的片段。"""

    if raw_content is None:
        return None

    if topic.strip().lower() != PRICING_TOPIC:
        return None

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

    if topic.strip().lower() != PRICING_TOPIC:
        return snippet

    return (
        f"{snippet}\n\n"
        "Pricing page excerpt:\n"
        f"{raw_content_excerpt}"
    )
