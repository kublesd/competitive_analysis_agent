"""共享的价格文本归一化工具。"""

from __future__ import annotations

import re


ZERO_PRICE_PATTERN = re.compile(
    r"(?:^|[^\d])(?:us\$|usd\s*)?\$?0(?:[.,]00)?(?=$|[^\d.,])",
    re.IGNORECASE,
)
POSITIVE_CURRENCY_PRICE_PATTERN = re.compile(
    r"(?:\$|usd\s*)(?P<amount>\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
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
) -> bool:
    """判断报告里是否需要把计费周期缺失列为数据限制。"""

    if is_supported_billing_cycle_text(billing_cycle):
        return False
    if is_free_price_text(price_text):
        return False
    if is_custom_pricing_text(price_text):
        return False
    if text_contains_billing_cycle(price_text):
        return False

    return True
