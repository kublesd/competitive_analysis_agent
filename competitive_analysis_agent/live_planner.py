"""使用 OpenAI 兼容模型运行 Planner 的手动 smoke test。"""

from __future__ import annotations

import json

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_config import LIVE_MODEL_MAX_RETRIES
from competitive_analysis_agent.planner import (
    LangChainPlannerModel,
    Planner,
    PlannerInput,
)


class LivePlannerConfigurationError(ValueError):
    """表示真实 Planner 调用缺少必要环境配置。"""


def create_live_planner(settings: Settings) -> Planner:
    """根据环境配置创建真实 Planner，不在代码中保存 API Key。"""

    missing_variables: list[str] = []
    if settings.llm_api_key is None:
        missing_variables.append("LLM_API_KEY")
    if settings.llm_base_url is None:
        missing_variables.append("LLM_BASE_URL")
    if settings.llm_model is None:
        missing_variables.append("LLM_MODEL")

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise LivePlannerConfigurationError(
            f"Missing environment variables: {missing_text}"
        )

    # 延迟导入让不安装 llm 可选依赖的离线测试仍可运行。
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise LivePlannerConfigurationError(
            'Install model dependencies with: python -m pip install -e ".[llm]"'
        ) from error

    chat_model = ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        max_tokens=512,
        extra_body={"enable_thinking": False},
        timeout=45,
        # UI 路径允许一次供应商级重试，降低临时网络断开对整次分析的影响。
        max_retries=LIVE_MODEL_MAX_RETRIES,
    )
    return Planner(LangChainPlannerModel(chat_model))


def run_smoke_test(settings: Settings | None = None) -> list[dict[str, str]]:
    """调用真实模型生成固定样例任务，并返回可打印的普通字典。"""

    current_settings = settings or Settings.from_env()
    planner = create_live_planner(current_settings)
    planner_input = PlannerInput(
        target_product="Notion",
        competitors=["飞书"],
        dimensions=["features", "pricing"],
    )

    tasks = planner.plan(planner_input)
    return [task.model_dump() for task in tasks]


def main() -> None:
    """运行真实 Planner smoke test，并将任务输出为 JSON。"""

    tasks = run_smoke_test()
    print(json.dumps(tasks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
