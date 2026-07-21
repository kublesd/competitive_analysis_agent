"""确定性计算标准化模型价格的场景成本，不负责报告展示。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from competitive_analysis_agent.schemas import ModelProfile, PriceRate


ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class CostScenario:
    """一次调用的 token 用量；cached_input_tokens 是 input_tokens 的子集。"""

    name: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    audio_input_tokens: int = 0

    def __post_init__(self) -> None:
        if min(
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.audio_input_tokens,
        ) < 0:
            raise ValueError("Scenario token counts cannot be negative.")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens.")


DEFAULT_COST_SCENARIOS = (
    CostScenario("普通聊天", input_tokens=2_000, output_tokens=500),
    CostScenario("RAG 问答", input_tokens=20_000, output_tokens=1_000),
    CostScenario("长文档分析", input_tokens=200_000, output_tokens=5_000),
)


@dataclass(frozen=True)
class ScenarioCost:
    """单个模式的结果；缺少任一必需价格时 cost 为 None。"""

    mode: Literal["realtime", "batch", "cached"]
    cost: Decimal | None
    monthly_cost: Decimal | None
    savings_percent: Decimal | None
    missing_price_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelScenarioCosts:
    product_name: str
    model_name: str
    scenario_name: str
    realtime: ScenarioCost
    batch: ScenarioCost | None
    cached: ScenarioCost | None


def calculate_scenario_costs(
    product_name: str,
    models: list[ModelProfile],
    monthly_call_count: int,
    scenarios: tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS,
    price_as_of: date | None = None,
) -> list[ModelScenarioCosts]:
    """计算每个模型、每个场景的实时、Batch 与缓存成本。"""

    if monthly_call_count < 0:
        raise ValueError("monthly_call_count cannot be negative.")
    as_of = price_as_of or date.today()
    results: list[ModelScenarioCosts] = []
    for model in models:
        for scenario in scenarios:
            realtime = _calculate_mode(
                model, scenario, monthly_call_count, "realtime", as_of
            )
            batch = _calculate_mode(
                model, scenario, monthly_call_count, "batch", as_of
            )
            cached = (
                _calculate_mode(
                    model, scenario, monthly_call_count, "cached", as_of
                )
                if scenario.cached_input_tokens
                else None
            )
            results.append(
                ModelScenarioCosts(
                    product_name=product_name,
                    model_name=model.model_name,
                    scenario_name=scenario.name,
                    realtime=realtime,
                    batch=batch,
                    cached=cached,
                )
            )
    return results


def _calculate_mode(
    model: ModelProfile,
    scenario: CostScenario,
    monthly_call_count: int,
    mode: Literal["realtime", "batch", "cached"],
    as_of: date,
) -> ScenarioCost:
    prices = model.pricing
    batch_discount = prices.batch_discount_percent
    input_field = "batch_input_price" if mode == "batch" else "input_price"
    output_field = "batch_output_price" if mode == "batch" else "output_price"
    cached_field = (
        "batch_cached_input_price"
        if mode == "batch"
        else "cached_input_price"
    )
    input_rate = _active_rate(getattr(prices, input_field), scenario, as_of)
    output_rate = _active_rate(getattr(prices, output_field), scenario, as_of)
    cached_rate = _active_rate(
        getattr(prices, cached_field), scenario, as_of
    )
    input_factor = output_factor = cached_factor = audio_factor = Decimal("1")

    # Batch 可只公开折扣；此时用同一有效期的实时价格派生，而非猜测费率。
    if mode == "batch" and batch_discount is not None:
        discount_factor = (
            Decimal("100") - batch_discount
        ) / Decimal("100")
        if input_rate is None:
            input_rate = _active_rate(prices.input_price, scenario, as_of)
            input_factor = discount_factor
        if output_rate is None:
            output_rate = _active_rate(prices.output_price, scenario, as_of)
            output_factor = discount_factor
        if cached_rate is None:
            cached_rate = _active_rate(
                prices.cached_input_price, scenario, as_of
            )
            cached_factor = discount_factor
        audio_factor = discount_factor

    if mode == "cached":
        input_rate = _active_rate(prices.input_price, scenario, as_of)
        output_rate = _active_rate(prices.output_price, scenario, as_of)
        cached_rate = _active_rate(
            prices.cached_input_price, scenario, as_of
        )

    audio_rate = _active_rate(
        prices.audio_input_price or prices.audio_price,
        scenario,
        as_of,
    )
    cached_tokens = scenario.cached_input_tokens if mode == "cached" else 0
    required_rates = {
        "input": input_rate if scenario.input_tokens else True,
        "output": output_rate if scenario.output_tokens else True,
        "cached_input": cached_rate if cached_tokens else True,
        "audio_input": audio_rate if scenario.audio_input_tokens else True,
    }
    missing = tuple(name for name, rate in required_rates.items() if not rate)
    if missing:
        return ScenarioCost(mode, None, None, None, missing)

    cost = (
        _cost(scenario.input_tokens - cached_tokens, input_rate) * input_factor
        + _cost(cached_tokens, cached_rate) * cached_factor
        + _cost(scenario.output_tokens, output_rate) * output_factor
        + _cost(scenario.audio_input_tokens, audio_rate) * audio_factor
    )
    baseline = (
        _calculate_mode(model, scenario, monthly_call_count, "realtime", as_of)
        if mode != "realtime"
        else None
    )
    savings = (
        _saving_percent(baseline.cost, cost)
        if baseline is not None and baseline.cost is not None
        else None
    )
    return ScenarioCost(
        mode,
        cost,
        cost * monthly_call_count,
        savings,
    )


def _active_rate(
    value: PriceRate | list[PriceRate] | None,
    scenario: CostScenario,
    as_of: date,
) -> PriceRate | None:
    rates = value if isinstance(value, list) else [value]
    candidates = [
        rate
        for rate in rates
        if rate is not None
        and (rate.effective_from is None or rate.effective_from <= as_of)
        and (rate.effective_to is None or as_of <= rate.effective_to)
        and (
            rate.max_context_tokens is None
            or scenario.input_tokens <= rate.max_context_tokens
        )
    ]
    if len(candidates) > 1 and all(
        rate.effective_from is None and rate.effective_to is None
        for rate in candidates
    ):
        return None
    return max(candidates, key=lambda rate: rate.effective_from or date.min, default=None)


def _cost(tokens: int, rate: PriceRate | bool | None) -> Decimal:
    if not tokens:
        return Decimal("0")
    assert isinstance(rate, PriceRate)
    return Decimal(tokens) / ONE_MILLION * rate.amount


def _saving_percent(baseline: Decimal, cost: Decimal) -> Decimal | None:
    if not baseline:
        return None
    return (baseline - cost) / baseline * Decimal("100")
