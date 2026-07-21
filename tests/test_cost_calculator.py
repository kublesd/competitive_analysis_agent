import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from competitive_analysis_agent.cost_calculator import CostScenario, calculate_scenario_costs
from competitive_analysis_agent.schemas import (
    Currency,
    ModelPricing,
    ModelProfile,
    PriceRate,
    PricingUnit,
    SourceReference,
    SupportStatus,
)


def _rate(amount: str, **extra: object) -> PriceRate:
    return PriceRate(
        amount=amount,
        currency=Currency.USD,
        per_quantity=1_000_000,
        unit=PricingUnit.TOKEN,
        evidence_ids=["E1"],
        **extra,
    )


def _model(pricing: ModelPricing) -> ModelProfile:
    return ModelProfile(
        model_name="Example 1",
        batch_api=(
            SupportStatus.SUPPORTED
            if pricing.has_batch_pricing
            else SupportStatus.MISSING
        ),
        pricing=pricing,
        source_evidence=[
            SourceReference(
                evidence_id="E1",
                title="Official pricing",
                url="https://example.com/pricing",
                source_type="official",
                collected_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            )
        ],
        extraction_confidence=1,
    )


class CostCalculatorTest(unittest.TestCase):
    def test_calculates_realtime_batch_and_cache_with_decimal_savings(self) -> None:
        model = _model(
            ModelPricing(
                input_price=_rate("2"),
                cached_input_price=_rate("0.5"),
                output_price=_rate("8"),
                batch_discount_percent=Decimal("50"),
                batch_evidence_ids=["E1"],
            )
        )
        scenario = CostScenario(
            "cached chat", input_tokens=2_000, cached_input_tokens=1_000, output_tokens=500
        )

        result = calculate_scenario_costs(
            "Example API", [model], monthly_call_count=100, scenarios=(scenario,)
        )[0]

        self.assertEqual(result.realtime.cost, Decimal("0.008"))
        self.assertEqual(result.realtime.monthly_cost, Decimal("0.800"))
        self.assertEqual(result.cached.cost, Decimal("0.0065"))
        self.assertEqual(result.cached.savings_percent, Decimal("18.75"))
        self.assertEqual(result.batch.cost, Decimal("0.004"))
        self.assertEqual(result.batch.savings_percent, Decimal("50"))

    def test_selects_active_context_tier(self) -> None:
        model = _model(
            ModelPricing(
                input_price=[
                    _rate(
                        "1",
                        max_context_tokens=100_000,
                        effective_from=date(2025, 1, 1),
                    ),
                    _rate(
                        "2",
                        max_context_tokens=1_000_000,
                        effective_from=date(2026, 1, 1),
                    ),
                ],
                output_price=_rate("3"),
            )
        )

        result = calculate_scenario_costs(
            "Example API",
            [model],
            monthly_call_count=1,
            scenarios=(CostScenario("long", 200_000, 1),),
            price_as_of=date(2026, 7, 20),
        )[0]

        self.assertEqual(result.realtime.cost, Decimal("0.400003"))

    def test_explicit_batch_rates_are_not_discounted_twice(self) -> None:
        model = _model(
            ModelPricing(
                input_price=_rate("2"),
                output_price=_rate("8"),
                batch_input_price=_rate("1"),
                batch_output_price=_rate("4"),
                batch_discount_percent=Decimal("50"),
                batch_evidence_ids=["E1"],
            )
        )

        result = calculate_scenario_costs(
            "Example API",
            [model],
            monthly_call_count=1,
            scenarios=(CostScenario("chat", 2_000, 500),),
        )[0]

        self.assertEqual(result.batch.cost, Decimal("0.004"))
        self.assertEqual(result.batch.savings_percent, Decimal("50"))

    def test_missing_output_price_never_produces_total(self) -> None:
        model = _model(ModelPricing(input_price=_rate("2")))

        result = calculate_scenario_costs(
            "Example API", [model], 1, scenarios=(CostScenario("chat", 2_000, 1),)
        )[0]

        self.assertIsNone(result.realtime.cost)
        self.assertEqual(result.realtime.missing_price_types, ("output",))

    def test_condition_tier_prices_are_not_arbitrarily_costed(self) -> None:
        model = _model(
            ModelPricing(
                input_price=[
                    _rate("2", condition="Standard"),
                    _rate("4", condition="Priority"),
                ],
                output_price=[
                    _rate("8", condition="Standard"),
                    _rate("16", condition="Priority"),
                ],
            )
        )

        result = calculate_scenario_costs(
            "Example API", [model], 1, scenarios=(CostScenario("chat", 2_000, 1),)
        )[0]

        self.assertIsNone(result.realtime.cost)
        self.assertEqual(result.realtime.missing_price_types, ("input", "output"))

    def test_model_prices_must_be_normalized_to_usd_per_million_tokens(self) -> None:
        with self.assertRaises(ValueError):
            ModelPricing(
                input_price=PriceRate(
                    amount="2",
                    currency=Currency.USD,
                    per_quantity=1_000,
                    unit=PricingUnit.TOKEN,
                    evidence_ids=["E1"],
                )
            )


if __name__ == "__main__":
    unittest.main()
