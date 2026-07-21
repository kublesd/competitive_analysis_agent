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
    build_provider_request_options,
    load_live_settings,
)
from competitive_analysis_agent.schemas import Evidence, MarketDefinition


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
        # 生产结构化响应曾超过 45 秒；留出尾延迟空间，仍只重试一次。
        timeout=90,
        max_retries=LIVE_MODEL_MAX_RETRIES,
        **build_provider_request_options(settings),
    )
    return Extractor(LangChainExtractorModel(chat_model))


def build_live_sample_evidence() -> list[Evidence]:
    """创建三个 API 产品的四维短证据，避免真实验收依赖搜索服务。"""

    collected_at = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)
    return [
        Evidence(
            evidence_id="E1",
            product_name="OpenAI API",
            topic="model_capabilities",
            title="OpenAI API Models",
            url="https://platform.openai.com/docs/models",
            snippet="OpenAI API models support text, image, reasoning, and tool use.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E2",
            product_name="OpenAI API",
            topic="api_pricing",
            title="OpenAI API Pricing",
            url="https://openai.com/api/pricing",
            snippet="OpenAI API pricing lists GPT-5 input and output token prices.",
            raw_content="GPT-5 input tokens cost $1.25 per 1M input tokens.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E3",
            product_name="OpenAI API",
            topic="developer_platform",
            title="OpenAI API Documentation",
            url="https://platform.openai.com/docs",
            snippet="OpenAI API documentation provides SDK guides and function calling tools.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E4",
            product_name="OpenAI API",
            topic="usage_limits",
            title="OpenAI API Usage Limits",
            url="https://platform.openai.com/docs/guides/rate-limits",
            snippet="OpenAI API documents context windows and rate-limit tiers.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E5",
            product_name="Claude API",
            topic="model_capabilities",
            title="Claude API Models",
            url="https://docs.anthropic.com/en/docs/about-claude/models",
            snippet="Claude API models support multimodal reasoning and tool use.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E6",
            product_name="Claude API",
            topic="api_pricing",
            title="Claude API Pricing",
            url="https://docs.anthropic.com/en/docs/about-claude/pricing",
            snippet="Claude API pricing lists Sonnet input and output token prices.",
            raw_content="Claude Sonnet input tokens cost $3 per 1M input tokens.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E7",
            product_name="Claude API",
            topic="developer_platform",
            title="Claude API Documentation",
            url="https://docs.anthropic.com/en/api/client-sdks",
            snippet="Claude API documentation provides Python and TypeScript SDK tools.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E8",
            product_name="Claude API",
            topic="usage_limits",
            title="Claude API Usage Limits",
            url="https://docs.anthropic.com/en/api/rate-limits",
            snippet="Claude API documents context-window constraints, quotas, and rate limits.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E9",
            product_name="Gemini API",
            topic="model_capabilities",
            title="Gemini API Models",
            url="https://ai.google.dev/gemini-api/docs/models",
            snippet="Gemini API models support multimodal reasoning and tool use.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E10",
            product_name="Gemini API",
            topic="api_pricing",
            title="Gemini API Pricing",
            url="https://ai.google.dev/gemini-api/docs/pricing",
            snippet="Gemini API pricing lists input and output token prices.",
            raw_content="Gemini input tokens cost $1.25 per 1M input tokens.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E11",
            product_name="Gemini API",
            topic="developer_platform",
            title="Gemini API Documentation",
            url="https://ai.google.dev/gemini-api/docs",
            snippet="Gemini API documentation provides SDK guides and function calling tools.",
            source_type="official",
            collected_at=collected_at,
        ),
        Evidence(
            evidence_id="E12",
            product_name="Gemini API",
            topic="usage_limits",
            title="Gemini API Usage Limits",
            url="https://ai.google.dev/gemini-api/docs/rate-limits",
            snippet="Gemini API documents context-window sizes, quotas, and rate limits.",
            source_type="official",
            collected_at=collected_at,
        ),
    ]


def build_live_market_definition() -> MarketDefinition:
    """创建真实 Extractor 验收使用的固定市场定义。"""

    return MarketDefinition(
        market_name="生成式 AI API",
        product_category="大语言模型 API",
        target_buyer="开发团队、AI 产品负责人、企业技术团队",
        comparison_level="模型 API 服务",
        pricing_scope="api",
        core_dimensions=[
            "model_capabilities",
            "api_pricing",
            "developer_platform",
            "usage_limits",
        ],
        exclusions=["消费端订阅套餐", "按席位企业套餐"],
    )


def run_smoke_test(
    settings: Settings | None = None,
) -> list[dict[str, object]]:
    """调用真实模型提取三个 API 画像，并返回可检查的普通字典。"""

    current_settings = settings or load_live_settings()
    extractor = create_live_extractor(current_settings)
    extractor_input = ExtractorInput(
        evidence=build_live_sample_evidence(),
        market_definition=build_live_market_definition(),
    )
    profiles = extractor.extract(extractor_input)
    return [profile.model_dump(mode="json") for profile in profiles]


def main() -> None:
    """运行真实 Extractor smoke test，并输出脱离模型对象的 JSON。"""

    profiles = run_smoke_test()
    print(json.dumps(profiles, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
