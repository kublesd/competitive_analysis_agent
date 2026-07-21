import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

import pytest

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_planner import (
    create_live_planner,
)
from competitive_analysis_agent.planner import PlannerInput
from competitive_analysis_agent.schemas import MarketDefinition


LIVE_ENV_FILE = Path(
    r"F:\大模型应用开发学习\competitive-analysis-agent\.env"
)


@pytest.mark.live_llm
def test_planner_with_real_llm_preserves_market_scope() -> None:
    """验证真实 Planner 覆盖产品维度，且查询保留市场范围。"""

    # 清空 pytest 插件预加载的值，证明配置由这里指定的 .env 重新读取。
    with patch.dict(environ, {}, clear=True):
        settings = load_live_settings(LIVE_ENV_FILE)
    planner = create_live_planner(settings)
    planner_input = PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        market_definition=MarketDefinition(
            market_name="团队知识管理工具",
            product_category="SaaS 协作软件",
            target_buyer="中型企业 IT 与业务负责人",
            comparison_level="企业订阅产品",
            core_dimensions=["features"],
            exclusions=["消费端套餐"],
        ),
    )

    tasks = planner.plan(planner_input)

    assert {
        (task.product_name, task.topic) for task in tasks
    } == {
        ("Atlas Notes", "features"),
        ("Beacon Docs", "features"),
    }


class LivePlannerConfigurationTest(unittest.TestCase):
    def test_planner_does_not_require_live_model_configuration(self) -> None:
        # 确定性 Planner 在模型配置缺失时仍能生成完整任务矩阵。
        settings = Settings.from_env({})

        planner = create_live_planner(settings)
        tasks = planner.plan(
            PlannerInput(
                target_product="Atlas Notes",
                competitors=["Beacon Docs"],
                market_definition=MarketDefinition(
                    market_name="团队知识管理工具",
                    product_category="SaaS 协作软件",
                    comparison_level="企业订阅产品",
                    core_dimensions=["features"],
                ),
            )
        )

        self.assertEqual(len(tasks), 2)


if __name__ == "__main__":
    unittest.main()
