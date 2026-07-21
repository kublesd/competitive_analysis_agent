"""真实 Extractor 集成测试；默认 pytest 会排除本文件中的 live 用例。"""

from __future__ import annotations

from os import environ
from pathlib import Path
from unittest.mock import patch

import pytest

from competitive_analysis_agent.extractor import ExtractorInput
from competitive_analysis_agent.live_extractor import (
    build_live_market_definition,
    build_live_sample_evidence,
    create_live_extractor,
    load_live_settings,
)


@pytest.mark.live_llm
def test_extractor_with_real_llm_uses_only_supplied_evidence() -> None:
    """验证真实模型返回三个 API 画像和四维有效引用。"""

    evidence = build_live_sample_evidence()
    env_file_name = environ.get("LIVE_TEST_ENV_FILE", ".env")
    env_file = Path(__file__).parents[1] / env_file_name
    # pytest 插件可能预加载 .env；先清空，确保本测试只读取指定文件。
    with patch.dict(environ, {}, clear=True):
        settings = load_live_settings(env_file)
    extractor = create_live_extractor(settings)

    profiles = extractor.extract(
        ExtractorInput(
            evidence=evidence,
            market_definition=build_live_market_definition(),
        )
    )

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
        "OpenAI API",
        "Claude API",
        "Gemini API",
    ]

    for profile in profiles:
        assert [finding.dimension for finding in profile.dimension_findings] == [
            "model_capabilities",
            "api_pricing",
            "developer_platform",
            "usage_limits",
        ]
        allowed_ids = evidence_ids_by_product[profile.product_name]
        referenced_ids: set[str] = set()
        for finding in profile.dimension_findings:
            if finding.facts:
                assert finding.evidence_ids
            referenced_ids.update(finding.evidence_ids)
        for feature in profile.features:
            referenced_ids.update(feature.evidence_ids)
        for field_evidence in profile.field_evidence:
            referenced_ids.add(field_evidence.evidence_id)
        for model in profile.models:
            referenced_ids.update(
                source.evidence_id for source in model.source_evidence
            )
            referenced_ids.update(model.pricing.evidence_ids())
            for rate_limit in model.rate_limits or []:
                referenced_ids.update(rate_limit.evidence_ids)

        assert referenced_ids
        assert referenced_ids <= allowed_ids
        assert profile.models
        assert profile.pricing == []
        if profile.positioning is not None:
            supplied_text = evidence_text_by_product[profile.product_name]
            assert profile.positioning.lower() in supplied_text.lower()
        assert profile.target_users == []
        assert profile.strengths == []
        assert profile.limitations == []
