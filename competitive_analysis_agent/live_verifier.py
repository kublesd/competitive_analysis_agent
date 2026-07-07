"""使用 OpenAI 兼容模型运行 Verifier 的真实验收样例。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from competitive_analysis_agent.analyst import (
    AnalysisClaim,
    CompetitiveAnalysis,
)
from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_config import (
    LIVE_MODEL_MAX_RETRIES,
    build_provider_request_options,
    load_live_settings,
)
from competitive_analysis_agent.schemas import Evidence
from competitive_analysis_agent.verifier import (
    LangChainVerifierModel,
    Verifier,
    VerifierInput,
)


class LiveVerifierConfigurationError(ValueError):
    """表示真实 Verifier 调用缺少必要配置或模型依赖。"""


def create_live_verifier(settings: Settings) -> Verifier:
    """根据应用配置创建真实 Verifier，不在代码中保存 API Key。"""

    missing_variables: list[str] = []
    if settings.llm_api_key is None:
        missing_variables.append("LLM_API_KEY")
    if settings.llm_base_url is None:
        missing_variables.append("LLM_BASE_URL")
    if settings.llm_model is None:
        missing_variables.append("LLM_MODEL")

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise LiveVerifierConfigurationError(
            f"Missing environment variables: {missing_text}"
        )

    # 延迟导入让普通离线测试不依赖 LangChain。
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise LiveVerifierConfigurationError(
            'Install model dependencies with: python -m pip install -e ".[llm]"'
        ) from error

    chat_model = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        max_tokens=700,
        timeout=60,
        max_retries=LIVE_MODEL_MAX_RETRIES,
        **build_provider_request_options(settings),
    )
    return Verifier(LangChainVerifierModel(chat_model))


def build_live_verifier_input() -> VerifierInput:
    """创建引用有效但与证据冲突的价格 claim，验证语义检查能力。"""

    collected_at = datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc)
    evidence = [
        Evidence(
            evidence_id="E1",
            product_name="Atlas Notes",
            topic="pricing",
            title="Atlas Notes Pricing",
            url="https://example.com/atlas/pricing",
            snippet="The Team plan costs 12 USD per user each month.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E2",
            product_name="Beacon Docs",
            topic="pricing",
            title="Beacon Docs Pricing",
            url="https://example.com/beacon/pricing",
            snippet="The Business plan is named, but no public price is listed.",
            source_type="official",
            collected_at=collected_at,
        ),
    ]
    analysis = CompetitiveAnalysis(
        products=["Atlas Notes", "Beacon Docs"],
        pricing=[
            AnalysisClaim(
                claim="Atlas Notes offers its Team plan for free.",
                claim_type="fact",
                product_names=["Atlas Notes"],
                evidence_ids=["E1"],
            )
        ],
        conclusion=AnalysisClaim(
            claim="The supplied pricing information differs between products.",
            claim_type="interpretation",
            product_names=["Atlas Notes", "Beacon Docs"],
            evidence_ids=["E1", "E2"],
        ),
    )
    return VerifierInput(analysis=analysis, evidence=evidence)


def run_smoke_test(
    settings: Settings | None = None,
) -> dict[str, object]:
    """调用真实模型验证冲突 claim，并返回普通字典。"""

    current_settings = settings or load_live_settings()
    verifier = create_live_verifier(current_settings)
    result = verifier.verify(build_live_verifier_input())
    return result.model_dump(mode="json")


def main() -> None:
    """运行真实 Verifier smoke test，并输出结构化 JSON。"""

    result = run_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
