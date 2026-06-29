import unittest

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_planner import (
    LivePlannerConfigurationError,
    create_live_planner,
)


class LivePlannerConfigurationTest(unittest.TestCase):
    def test_missing_live_model_configuration_fails_clearly(self) -> None:
        # 真实模型入口缺少配置时应立即失败，不能发出不完整请求。
        settings = Settings.from_env({})

        with self.assertRaises(LivePlannerConfigurationError) as raised:
            create_live_planner(settings)

        error_message = str(raised.exception)
        self.assertIn("LLM_API_KEY", error_message)
        self.assertIn("LLM_BASE_URL", error_message)
        self.assertIn("LLM_MODEL", error_message)


if __name__ == "__main__":
    unittest.main()
