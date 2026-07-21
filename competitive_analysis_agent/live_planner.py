"""运行确定性 Planner 的手动 smoke test。"""

from __future__ import annotations

import json

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.planner import (
    Planner,
    PlannerInput,
)
from competitive_analysis_agent.schemas import MarketDefinition


class LivePlannerConfigurationError(ValueError):
    """表示真实 Planner 调用缺少必要环境配置。"""


def create_live_planner(settings: Settings) -> Planner:
    """创建无需模型配置的确定性 Planner。"""

    del settings
    return Planner()


def run_smoke_test(settings: Settings | None = None) -> list[dict[str, str]]:
    """生成固定样例任务，并返回可打印的普通字典。"""

    current_settings = settings or Settings.from_env()
    planner = create_live_planner(current_settings)
    planner_input = PlannerInput(
        target_product="Notion",
        competitors=["飞书"],
        market_definition=MarketDefinition(
            market_name="企业知识管理工具",
            product_category="SaaS 协作软件",
            target_buyer="中型企业 IT 与业务负责人",
            comparison_level="企业订阅产品",
            core_dimensions=["features", "pricing"],
            exclusions=["消费端套餐", "API 用量价格"],
        ),
    )

    tasks = planner.plan(planner_input)
    return [task.model_dump() for task in tasks]


def main() -> None:
    """运行 Planner smoke test，并将任务输出为 JSON。"""

    tasks = run_smoke_test()
    print(json.dumps(tasks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
