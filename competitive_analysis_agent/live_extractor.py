"""使用 OpenAI 兼容模型运行 Extractor 的真实验收样例。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.extractor import (
    Extractor,
    ExtractorInput,
    LangChainExtractorModel,
)
from competitive_analysis_agent.live_config import (
    LIVE_MODEL_MAX_RETRIES,
    load_live_settings,
)
from competitive_analysis_agent.schemas import Evidence


class LiveExtractorConfigurationError(ValueError):
    """表示真实 Extractor 调用缺少文件、配置或模型依赖。"""


def create_live_extractor(settings: Settings) -> Extractor:
    """根据应用配置创建真实 Extractor，不在代码中保存 API Key。"""

    missing_variables: list[str] = []
    if settings.llm_api_key is None:
        missing_variables.append("LLM_API_KEY")
    if settings.llm_base_url is None:
        missing_variables.append("LLM_BASE_URL")
    if settings.llm_model is None:
        missing_variables.append("LLM_MODEL")

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise LiveExtractorConfigurationError(
            f"Missing environment variables: {missing_text}"
        )

    # 延迟导入让普通离线测试不依赖 LangChain。
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise LiveExtractorConfigurationError(
            'Install model dependencies with: python -m pip install -e ".[llm]"'
        ) from error

    chat_model = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        max_tokens=900,
        extra_body={"enable_thinking": False},
        timeout=45,
        max_retries=LIVE_MODEL_MAX_RETRIES,
    )
    return Extractor(LangChainExtractorModel(chat_model))


def build_live_sample_evidence() -> list[Evidence]:
    """创建两个虚构产品的短证据，避免真实验收依赖搜索服务。"""

    collected_at = datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc)
    return [
        Evidence(
            evidence_id="E1",
            product_name="Atlas Notes",
            topic="features",
            title="Atlas Notes Features",
            url="https://example.com/atlas/features",
            snippet="Atlas Notes supports shared workspaces and reusable templates.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E2",
            product_name="Atlas Notes",
            topic="pricing",
            title="Atlas Notes Pricing",
            url="https://example.com/atlas/pricing",
            snippet="The Team plan costs 12 USD per user each month.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E3",
            product_name="Beacon Docs",
            topic="features",
            title="Beacon Docs Features",
            url="https://example.com/beacon/features",
            snippet="Beacon Docs supports collaborative pages and comments.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E4",
            product_name="Beacon Docs",
            topic="pricing",
            title="Beacon Docs Pricing",
            url="https://example.com/beacon/pricing",
            snippet="The page names a Business plan but does not list its price.",
            source_type="official",
            collected_at=collected_at,
        ),
    ]


def run_smoke_test(
    settings: Settings | None = None,
) -> list[dict[str, object]]:
    """调用真实模型提取两个画像，并返回可检查的普通字典。"""

    current_settings = settings or load_live_settings()
    extractor = create_live_extractor(current_settings)
    extractor_input = ExtractorInput(evidence=build_live_sample_evidence())
    profiles = extractor.extract(extractor_input)
    return [profile.model_dump(mode="json") for profile in profiles]


def main() -> None:
    """运行真实 Extractor smoke test，并输出脱离模型对象的 JSON。"""

    profiles = run_smoke_test()
    print(json.dumps(profiles, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
