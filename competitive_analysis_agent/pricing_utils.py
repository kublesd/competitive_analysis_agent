"""共享的价格文本归一化工具。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from competitive_analysis_agent.schemas import Evidence, PricingPlan


ZERO_PRICE_PATTERN = re.compile(
    r"(?:^|[^\d])(?:us\$|usd\s*)?\$?0(?:[.,]00)?(?=$|[^\d.,])",
    re.IGNORECASE,
)
POSITIVE_CURRENCY_PRICE_PATTERN = re.compile(
    r"(?:\$|usd\s*)(?P<amount>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
TRAILING_CURRENCY_PRICE_PATTERN = re.compile(
    r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?:usd|us\$)",
    re.IGNORECASE,
)
API_METERING_PATTERN = re.compile(
    r"(?:"
    r"(?:\bper\b|/)\s*"
    r"(?:(?:\d+(?:[.,]\d+)?\s*[km]?|one|thousand|million|billion)\s+)?"
    r"(?:input\s+|output\s+|cached(?:\s+input)?\s+|"
    r"cache(?:d)?\s+(?:read|write)s?\s+)?"
    r"(?:tokens?|requests?|images?|characters?|audio|seconds?|minutes?)\b"
    r"|(?:\bper\b|/)\s*(?:token\s+)?units?\s*/\s*day\b"
    r")",
    re.IGNORECASE,
)
MILLION_TOKEN_ALIAS_PATTERN = re.compile(
    r"\b(?:1\s*)?m\s*tok(?:ens?)?\b",
    re.IGNORECASE,
)
THOUSAND_TOKEN_ALIAS_PATTERN = re.compile(
    r"\b(?:1\s*)?k\s*tok(?:ens?)?\b",
    re.IGNORECASE,
)
API_RATE_FRAGMENT_SEPARATOR = re.compile(r"[|;\n]+")
API_METER_KINDS = (
    "token",
    "request",
    "image",
    "audio",
    "second",
    "minute",
    "character",
    "unit",
)
API_RATE_QUALIFIERS = (
    "cached input",
    "cache write",
    "cache read",
    "input",
    "output",
)
MONTHLY_BILLING_MARKERS = [
    "/month",
    "/ month",
    "/mo",
    "/ mo",
    "per month",
    "per user/month",
    "per user / month",
    "per seat/month",
    "per seat / month",
    "each month",
    "billed monthly",
    "monthly billing",
]
YEARLY_BILLING_MARKERS = [
    "/year",
    "/ year",
    "/yr",
    "/ yr",
    "per year",
    "per user/year",
    "per user / year",
    "per seat/year",
    "per seat / year",
    "each year",
    "billed annually",
    "billed yearly",
    "annual billing",
    "yearly billing",
]
MONTHLY_BILLING_VALUES = {
    "month",
    "monthly",
    "per month",
    "per user/month",
    "per user / month",
    "per seat/month",
    "per seat / month",
}
YEARLY_BILLING_VALUES = {
    "annual",
    "annually",
    "year",
    "yearly",
    "per year",
    "per user/year",
    "per user / year",
    "per seat/year",
    "per seat / year",
}


def normalize_price_text(price_text: str) -> str:
    """把价格文本压成适合做保守匹配的小写字符串。"""

    return re.sub(r"\s+", " ", price_text.strip().lower())


def _normalize_api_metering_text(price_text: str) -> str:
    """把 MTok 等常见 API Token 单位转换成校验使用的标准文本。"""

    normalized_text = normalize_price_text(price_text)
    normalized_text = MILLION_TOKEN_ALIAS_PATTERN.sub(
        "million tokens",
        normalized_text,
    )
    return THOUSAND_TOKEN_ALIAS_PATTERN.sub(
        "thousand tokens",
        normalized_text,
    )


def complete_api_price_text(pricing_plan: PricingPlan) -> str | None:
    """把分散在 price 与 unit 的 API 金额和计量依据合成完整费率。"""

    price_text = pricing_plan.price
    if price_text is None or is_custom_pricing_text(price_text):
        return price_text
    if API_METERING_PATTERN.search(
        _normalize_api_metering_text(price_text)
    ):
        return price_text

    unit = pricing_plan.unit
    if unit is None:
        return price_text
    completed_price = f"{price_text.strip()} {unit.strip()}"
    if API_METERING_PATTERN.search(
        _normalize_api_metering_text(completed_price)
    ):
        return completed_price
    return price_text


def api_pricing_evidence_error(
    pricing_plan: PricingPlan,
    evidence_by_id: Mapping[str, Evidence],
) -> str | None:
    """返回 API 价格与同条证据不一致的原因；有效时返回 None。"""

    price_text = complete_api_price_text(pricing_plan)
    if price_text is None:
        return "price 不能为 null；未知 API 价格不得生成价格事实"

    cited_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in pricing_plan.evidence_ids
        if evidence_id in evidence_by_id
    ]
    if not cited_evidence:
        return "没有可用的引用 Evidence"

    if is_custom_pricing_text(price_text):
        if normalize_price_text(price_text) != "custom pricing":
            return "自定义报价必须明确保存为 Custom pricing"
        if any(
            _evidence_supports_custom_pricing(pricing_plan, item)
            for item in cited_evidence
        ):
            return None
        return "同一 Evidence 未同时支持套餐名和 Custom pricing"

    normalized_price = _normalize_api_metering_text(price_text)
    if API_METERING_PATTERN.search(normalized_price) is None:
        return (
            "price 必须保留金额和 API 计量依据，"
            "例如 $2.50 per million input tokens 或 $750 per unit/day；"
            "USD 只是币种，不是计量单位"
        )

    price_amounts = _currency_amounts(price_text)
    if not price_amounts:
        return "price 缺少带币种的明确金额"

    if any(
        _evidence_supports_api_rate(
            pricing_plan=pricing_plan,
            evidence=item,
            price_amounts=price_amounts,
            price_text=price_text,
        )
        for item in cited_evidence
    ):
        return None

    return "同一 Evidence 未同时支持套餐名、金额和 API 计量依据"


def api_pricing_plan_is_evidence_supported(
    pricing_plan: PricingPlan,
    evidence_by_id: Mapping[str, Evidence],
) -> bool:
    """判断一个 API 价格条目是否通过共享证据校验。"""

    return api_pricing_evidence_error(pricing_plan, evidence_by_id) is None


def _evidence_supports_custom_pricing(
    pricing_plan: PricingPlan,
    evidence: Evidence,
) -> bool:
    """要求单条证据同时出现套餐名和自定义报价。"""

    evidence_text = _build_evidence_price_text(evidence)
    return _text_supports_plan_name(
        evidence_text,
        pricing_plan.plan_name,
    ) and is_custom_pricing_text(evidence_text)


def _evidence_supports_api_rate(
    pricing_plan: PricingPlan,
    evidence: Evidence,
    price_amounts: set[Decimal],
    price_text: str,
) -> bool:
    """保守确认单条证据包含同一套餐的金额、费率单位和方向。"""

    evidence_text = _build_evidence_price_text(evidence)
    if not _text_supports_plan_name(evidence_text, pricing_plan.plan_name):
        return False
    normalized_price = _normalize_api_metering_text(price_text)
    required_meter_kinds = {
        marker for marker in API_METER_KINDS if marker in normalized_price
    }
    required_qualifiers = {
        marker for marker in API_RATE_QUALIFIERS if marker in normalized_price
    }
    if "cached input" in required_qualifiers:
        required_qualifiers.discard("input")

    # 同页可能同时列出输入和输出价格，只在金额所在费率片段内核对方向。
    rate_contexts = _build_api_rate_contexts(
        evidence_text,
        price_amounts,
    )
    for rate_context in rate_contexts:
        normalized_context = _normalize_api_metering_text(rate_context)
        if not _text_supports_plan_name(
            rate_context,
            pricing_plan.plan_name,
        ):
            continue
        if API_METERING_PATTERN.search(normalized_context) is None:
            continue
        if not all(
            marker in normalized_context
            for marker in required_meter_kinds
        ):
            continue
        if (
            "input" in required_qualifiers
            and "cached input" in normalized_context
        ):
            continue
        if not all(
            marker in normalized_context
            for marker in required_qualifiers
        ):
            continue
        return True

    return False


def _build_api_rate_contexts(
    evidence_text: str,
    price_amounts: set[Decimal],
) -> list[str]:
    """按表格列或短行定位金额，避免借用同页其他费率的方向。"""

    rate_contexts = _build_markdown_table_rate_contexts(
        evidence_text,
        price_amounts,
    )
    fragments = [
        fragment.strip()
        for fragment in API_RATE_FRAGMENT_SEPARATOR.split(evidence_text)
        if fragment.strip()
    ]

    for fragment_index, fragment in enumerate(fragments):
        if not price_amounts.intersection(_currency_amounts(fragment)):
            continue

        normalized_fragment = _normalize_api_metering_text(fragment)
        context_parts = [fragment]

        # 行内价表可能先写一次模型名，再连续列出输入和输出费率。
        previous_context_count = 0
        for previous_fragment in reversed(fragments[:fragment_index]):
            if _currency_amounts(previous_fragment):
                continue
            context_parts.insert(0, previous_fragment)
            previous_context_count += 1
            if previous_context_count >= 2:
                break
        if (
            API_METERING_PATTERN.search(normalized_fragment) is None
            and fragment_index + 1 < len(fragments)
        ):
            context_parts.append(fragments[fragment_index + 1])

        rate_contexts.append(" ".join(context_parts))

    return rate_contexts


def _build_markdown_table_rate_contexts(
    evidence_text: str,
    price_amounts: set[Decimal],
) -> list[str]:
    """从压平的 Markdown 表格恢复模型、费率列和全表计量单位。"""

    normalized_evidence = _normalize_api_metering_text(evidence_text)
    metering_match = API_METERING_PATTERN.search(normalized_evidence)
    if metering_match is None:
        return []

    # 搜索正文用空单元格分隔表头和数据行，普通竖线只分隔同一行的列。
    rows: list[list[str]] = []
    for line in evidence_text.splitlines():
        current_row: list[str] = []
        for raw_cell in line.split("|"):
            cell = raw_cell.strip()
            if cell:
                current_row.append(cell)
                continue
            if current_row:
                rows.append(current_row)
                current_row = []
        if current_row:
            rows.append(current_row)

    rate_contexts: list[str] = []
    metering_text = metering_match.group(0)
    for row_index, header_row in enumerate(rows):
        if not header_row or normalize_price_text(header_row[0]) != "model":
            continue

        for data_row in rows[row_index + 1 :]:
            if len(data_row) != len(header_row):
                continue
            for column_index in range(1, len(header_row)):
                price_cell = data_row[column_index]
                if not price_amounts.intersection(
                    _currency_amounts(price_cell)
                ):
                    continue
                rate_contexts.append(
                    " ".join(
                        [
                            data_row[0],
                            header_row[column_index],
                            price_cell,
                            metering_text,
                        ]
                    )
                )

    return rate_contexts


def _build_evidence_price_text(evidence: Evidence) -> str:
    """合并一条 Evidence 内可用于价格核对的全部文本。"""

    return "\n".join(
        value
        for value in [evidence.title, evidence.snippet, evidence.raw_content]
        if value
    )


def _text_supports_plan_name(text: str, plan_name: str) -> bool:
    """忽略标点差异匹配套餐名，同时拒绝只命中少量泛词。"""

    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    normalized_plan = re.sub(
        r"[^a-z0-9]+",
        " ",
        plan_name.casefold(),
    ).strip()
    if not normalized_plan:
        return False
    if normalized_plan in normalized_text:
        return True

    # 产品名中的 API 常被模型省略，但其余套餐词仍必须连续匹配。
    text_without_api = re.sub(r"\bapi\b", " ", normalized_text)
    plan_without_api = re.sub(r"\bapi\b", " ", normalized_plan)
    text_without_api = re.sub(r"\s+", " ", text_without_api).strip()
    plan_without_api = re.sub(r"\s+", " ", plan_without_api).strip()
    if (
        len(plan_without_api.split()) >= 2
        and plan_without_api in text_without_api
    ):
        return True

    # 表格通常只在行首写一次模型名，输入/输出方向位于后续列。
    for qualifier in API_RATE_QUALIFIERS:
        qualifier_suffix = f" {qualifier}"
        if not normalized_plan.endswith(qualifier_suffix):
            continue
        base_plan = normalized_plan.removesuffix(qualifier_suffix).strip()
        return len(base_plan.split()) >= 2 and base_plan in normalized_text

    return False


def _currency_amounts(text: str) -> set[Decimal]:
    """提取明确带美元币种的金额，避免把模型版本号当成价格。"""

    amounts: set[Decimal] = set()
    patterns = [
        POSITIVE_CURRENCY_PRICE_PATTERN,
        TRAILING_CURRENCY_PRICE_PATTERN,
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                amounts.add(Decimal(match.group("amount").replace(",", "")))
            except InvalidOperation:
                continue
    return amounts


def normalize_price_value(price_text: str | None) -> str | None:
    """把明显冗余的价格文本收窄为可展示的价格值。"""

    if price_text is None:
        return None

    normalized_text = normalize_price_text(price_text)
    if not normalized_text:
        return price_text

    # 免费方案常被模型写成 "$0 per seat/month"。这里保留价格本身，
    # 把计费周期交给 billing_cycle 字段或展示逻辑处理。
    if ZERO_PRICE_PATTERN.search(normalized_text):
        return "$0"

    return price_text


def is_free_price_text(price_text: str | None) -> bool:
    """判断价格文本是否明确表示免费或 0 价格。"""

    if price_text is None:
        return False

    normalized_text = normalize_price_text(price_text)
    if not normalized_text:
        return False

    if contains_positive_currency_price(normalized_text):
        return False

    if "free" in normalized_text:
        return True

    return ZERO_PRICE_PATTERN.search(normalized_text) is not None


def contains_positive_currency_price(normalized_text: str) -> bool:
    """判断文本中是否出现大于 0 的货币价格。"""

    for match in POSITIVE_CURRENCY_PRICE_PATTERN.finditer(normalized_text):
        amount_text = match.group("amount").replace(",", "")
        try:
            amount = float(amount_text)
        except ValueError:
            continue
        if amount > 0:
            return True

    return False


def is_custom_pricing_text(price_text: str | None) -> bool:
    """判断价格文本是否是联系销售或自定义报价。"""

    if price_text is None:
        return False

    normalized_text = normalize_price_text(price_text)
    custom_markers = [
        "custom pricing",
        "contact sales",
        "contact us",
        "talk to sales",
    ]
    for marker in custom_markers:
        if marker in normalized_text:
            return True

    return False


def normalize_billing_cycle_value(
    billing_cycle: str | None,
) -> str | None:
    """只保留明确的计费周期，丢弃 Beta 等状态词。"""

    if billing_cycle is None:
        return None

    if not is_supported_billing_cycle_text(billing_cycle):
        return None

    return billing_cycle


def is_supported_billing_cycle_text(billing_cycle: str | None) -> bool:
    """判断文本是否可以作为 billing_cycle 字段。"""

    if billing_cycle is None:
        return False

    normalized_text = normalize_price_text(billing_cycle)
    if normalized_text in MONTHLY_BILLING_VALUES:
        return True
    if normalized_text in YEARLY_BILLING_VALUES:
        return True

    return text_contains_billing_cycle(normalized_text)


def text_contains_billing_cycle(text: str | None) -> bool:
    """判断任意文本中是否包含明确的月付或年付周期。"""

    return detect_billing_cycle_category(text) is not None


def detect_billing_cycle_category(text: str | None) -> str | None:
    """把文本中的计费周期归类为 monthly/yearly，无法确定时返回 None。"""

    if text is None:
        return None

    normalized_text = normalize_price_text(text)
    if normalized_text in MONTHLY_BILLING_VALUES:
        return "monthly"
    if normalized_text in YEARLY_BILLING_VALUES:
        return "yearly"

    for marker in MONTHLY_BILLING_MARKERS:
        if marker in normalized_text:
            return "monthly"

    for marker in YEARLY_BILLING_MARKERS:
        if marker in normalized_text:
            return "yearly"

    return None


def billing_cycle_is_supported_by_text(
    support_text: str | None,
    billing_cycle: str | None,
) -> bool:
    """判断一段证据文本是否支持某个 billing_cycle。"""

    billing_category = detect_billing_cycle_category(billing_cycle)
    if billing_category is None:
        return False

    support_category = detect_billing_cycle_category(support_text)
    return support_category == billing_category


def should_include_billing_cycle(
    price_text: str | None,
    billing_cycle: str | None,
) -> bool:
    """判断价格展示或 fallback claim 是否应该带计费周期。"""

    if not is_supported_billing_cycle_text(billing_cycle):
        return False
    if is_free_price_text(price_text):
        return False
    if is_custom_pricing_text(price_text):
        return False
    if billing_cycle_is_supported_by_text(price_text, billing_cycle):
        return False

    return True


def should_report_missing_billing_cycle(
    price_text: str | None,
    billing_cycle: str | None,
    unit_text: str | None = None,
) -> bool:
    """判断价格、周期和单位文本是否都没有提供计费周期。"""

    if is_supported_billing_cycle_text(billing_cycle):
        return False
    if is_free_price_text(price_text):
        return False
    if is_custom_pricing_text(price_text):
        return False
    if text_contains_billing_cycle(price_text):
        return False
    if text_contains_billing_cycle(unit_text):
        return False

    return True
