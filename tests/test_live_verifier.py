"""真实 Verifier 集成测试；普通 pytest 默认排除此用例。"""

from __future__ import annotations

from os import environ
from pathlib import Path
from unittest.mock import patch

import pytest

from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_verifier import (
    build_live_verifier_input,
    create_live_verifier,
)


@pytest.mark.live_llm
def test_verifier_with_real_llm_flags_conflicting_claim() -> None:
    """验证真实模型能发现引用有效但与证据冲突的价格 claim。"""

    env_file = Path(__file__).parents[1] / environ.get(
        "LIVE_TEST_ENV_FILE", ".env"
    )
    # 防止 pytest 预加载 .env，真实验收只能读取规定的配置文件。
    with patch.dict(environ, {}, clear=True):
        settings = load_live_settings(env_file)
    verifier = create_live_verifier(settings)

    result = verifier.verify(build_live_verifier_input())

    assert not result.passed
    assert result.retry_recommended
    assert not result.citations_valid
    assert result.scope_consistent
    assert result.comparison_usable
    assert result.issues
    assert result.issues[0].claim_path == "pricing[0]"
    assert result.issues[0].issue_type in {
        "unsupported_claim",
        "conflicting_evidence",
        "insufficient_evidence",
    }
    assert result.issues[0].suggested_action
    assert set(result.issues[0].evidence_ids).issubset({"E1", "E2"})
    assert result.claim_verifications
    assert any(
        item.status.value in {"conflicting", "insufficient"}
        for item in result.claim_verifications
    )
