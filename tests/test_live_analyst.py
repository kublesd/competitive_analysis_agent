"""真实 Analyst 集成测试；普通 pytest 默认排除此用例。"""

from __future__ import annotations

import pytest

from competitive_analysis_agent.analyst import (
    AnalystInput,
    collect_analysis_claims,
    contains_pricing_language,
)
from competitive_analysis_agent.live_analyst import (
    build_live_sample_profiles,
    create_live_analyst,
)
from competitive_analysis_agent.live_config import load_live_settings


@pytest.mark.live_llm
def test_analyst_with_real_llm_returns_grounded_comparison() -> None:
    """验证真实模型覆盖全部产品，并为每条事实保留有效引用。"""

    profiles = build_live_sample_profiles()
    analyst = create_live_analyst(load_live_settings())

    analysis = analyst.analyze(AnalystInput(profiles=profiles))

    assert analysis.products == ["Atlas Notes", "Beacon Docs"]
    assert analysis.positioning
    assert analysis.features
    assert analysis.pricing
    assert analysis.opportunities

    claims = collect_analysis_claims(analysis)
    factual_claims = [
        claim for claim in claims if claim.claim_type == "fact"
    ]
    interpretation_claims = [
        claim for claim in claims if claim.claim_type == "interpretation"
    ]

    assert factual_claims
    assert interpretation_claims
    assert all(claim.evidence_ids for claim in factual_claims)
    assert not any(
        contains_pricing_language(claim.claim)
        for claim in analysis.features
    )
    assert all(
        claim.claim_type == "interpretation"
        for claim in analysis.opportunities
    )
    assert analysis.conclusion.claim_type == "interpretation"
    assert set(analysis.conclusion.product_names) == {
        "Atlas Notes",
        "Beacon Docs",
    }
