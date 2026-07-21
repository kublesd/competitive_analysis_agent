"""使用 OpenAI 兼容模型运行 Analyst 的真实验收样例。"""

from __future__ import annotations

import json

from competitive_analysis_agent.analyst import (
    Analyst,
    AnalystInput,
    LangChainAnalystModel,
)
from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_config import (
    LIVE_MODEL_MAX_RETRIES,
    build_provider_request_options,
    load_live_settings,
)
from competitive_analysis_agent.schemas import (
    DimensionFinding,
    FeatureItem,
    MarketDefinition,
    PricingPlan,
    ProductProfile,
)


class LiveAnalystConfigurationError(ValueError):
    """表示真实 Analyst 调用缺少必要配置或模型依赖。"""


def create_live_analyst(settings: Settings) -> Analyst:
    """根据应用配置创建真实 Analyst，不在代码中保存 API Key。"""

    missing_variables: list[str] = []
    if settings.llm_api_key is None:
        missing_variables.append("LLM_API_KEY")
    if settings.llm_base_url is None:
        missing_variables.append("LLM_BASE_URL")
    if settings.llm_model is None:
        missing_variables.append("LLM_MODEL")

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise LiveAnalystConfigurationError(
            f"Missing environment variables: {missing_text}"
        )

    # 延迟导入让普通离线测试不依赖 LangChain。
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise LiveAnalystConfigurationError(
            'Install model dependencies with: python -m pip install -e ".[llm]"'
        ) from error

    chat_model = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        max_tokens=1400,
        timeout=60,
        max_retries=LIVE_MODEL_MAX_RETRIES,
        **build_provider_request_options(settings),
    )
    return Analyst(LangChainAnalystModel(chat_model))


def build_live_sample_profiles() -> list[ProductProfile]:
    """创建两个短产品画像，真实验收无需重新调用搜索和 Extractor。"""

    return [
        ProductProfile(
            product_name="Atlas Notes",
            positioning="A collaborative workspace for small teams.",
            features=[
                FeatureItem(
                    name="Reusable templates",
                    description="Teams can create pages from reusable templates.",
                    evidence_ids=["E1"],
                )
            ],
            dimension_findings=[
                DimensionFinding(
                    dimension="features",
                    facts=["Reusable templates are documented."],
                    evidence_ids=["E1"],
                ),
                DimensionFinding(
                    dimension="pricing",
                    facts=["The Team plan lists 12 USD per user monthly."],
                    evidence_ids=["E2"],
                ),
            ],
            pricing=[
                PricingPlan(
                    plan_name="Team",
                    price="12 USD per user",
                    billing_cycle="monthly",
                    evidence_ids=["E2"],
                )
            ],
        ),
        ProductProfile(
            product_name="Beacon Docs",
            features=[
                FeatureItem(
                    name="Collaborative pages",
                    description="Users can edit pages together and leave comments.",
                    evidence_ids=["E3"],
                )
            ],
            dimension_findings=[
                DimensionFinding(
                    dimension="features",
                    facts=["Collaborative pages are documented."],
                    evidence_ids=["E3"],
                ),
                DimensionFinding(
                    dimension="pricing",
                    facts=["The Business plan has no public price."],
                    evidence_ids=["E4"],
                ),
            ],
            pricing=[
                PricingPlan(
                    plan_name="Business",
                    price=None,
                    billing_cycle=None,
                    evidence_ids=["E4"],
                )
            ],
            limitations=[
                "Public pricing is unavailable in the supplied profile."
            ],
        ),
    ]


def build_live_market_definition() -> MarketDefinition:
    """创建真实 Analyst 验收使用的同层级市场范围。"""

    return MarketDefinition(
        market_name="Team knowledge workspace",
        product_category="SaaS collaboration software",
        target_buyer="Mid-sized company IT and business leaders",
        comparison_level="Team subscription product",
        core_dimensions=["features", "pricing"],
        exclusions=["consumer plans", "API usage pricing"],
    )


def run_smoke_test(
    settings: Settings | None = None,
) -> dict[str, object]:
    """调用真实模型生成比较，并返回可检查的普通字典。"""

    current_settings = settings or load_live_settings()
    analyst = create_live_analyst(current_settings)
    analyst_input = AnalystInput(
        profiles=build_live_sample_profiles(),
        market_definition=build_live_market_definition(),
    )
    analysis = analyst.analyze(analyst_input)
    return analysis.model_dump(mode="json")


def main() -> None:
    """运行真实 Analyst smoke test，并输出结构化 JSON。"""

    analysis = run_smoke_test()
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
