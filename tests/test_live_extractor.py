"""真实 Extractor 集成测试；默认 pytest 会排除本文件中的 live 用例。"""

from __future__ import annotations

import pytest

from competitive_analysis_agent.extractor import ExtractorInput
from competitive_analysis_agent.live_extractor import (
    build_live_sample_evidence,
    create_live_extractor,
    load_live_settings,
)


@pytest.mark.live_llm
def test_extractor_with_real_llm_uses_only_supplied_evidence() -> None:
    """验证真实模型返回两个画像，且引用和空值符合 Extractor 契约。"""

    evidence = build_live_sample_evidence()
    extractor = create_live_extractor(load_live_settings())

    profiles = extractor.extract(ExtractorInput(evidence=evidence))

    evidence_ids_by_product: dict[str, set[str]] = {}
    evidence_text_by_product: dict[str, str] = {}
    for item in evidence:
        evidence_ids_by_product.setdefault(item.product_name, set()).add(
            item.evidence_id
        )
        evidence_text_by_product.setdefault(item.product_name, "")
        evidence_text_by_product[item.product_name] += (
            f" {item.title} {item.snippet} {item.raw_content or ''}"
        )

    assert [profile.product_name for profile in profiles] == [
        "Atlas Notes",
        "Beacon Docs",
    ]

    for profile in profiles:
        allowed_ids = evidence_ids_by_product[profile.product_name]
        referenced_ids: set[str] = set()
        for feature in profile.features:
            referenced_ids.update(feature.evidence_ids)
        for pricing_plan in profile.pricing:
            referenced_ids.update(pricing_plan.evidence_ids)

        assert referenced_ids
        assert referenced_ids <= allowed_ids
        if profile.positioning is not None:
            supplied_text = evidence_text_by_product[profile.product_name]
            assert profile.positioning.lower() in supplied_text.lower()
        assert profile.target_users == []
        assert profile.strengths == []
        assert profile.limitations == []

    beacon_profile = profiles[1]
    for pricing_plan in beacon_profile.pricing:
        assert pricing_plan.price is None
        assert pricing_plan.billing_cycle is None
