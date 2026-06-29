import json
import unittest

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.health import build_health_report


class HealthCheckSmokeTest(unittest.TestCase):
    def test_settings_repr_hides_secret_values(self) -> None:
        # 测试失败或日志记录 Settings 时不能顺带泄露 API Key。
        settings = Settings(
            llm_api_key="private-llm-key",
            llm_base_url="https://example.com/v1",
            llm_model="example-model",
            tavily_api_key="private-search-key",
        )

        settings_text = repr(settings)

        self.assertNotIn("private-llm-key", settings_text)
        self.assertNotIn("private-search-key", settings_text)

    def test_health_check_runs_without_credentials(self) -> None:
        settings = Settings.from_env({})

        report = build_health_report(settings)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["configuration"],
            {"llm": "not_configured", "search": "not_configured"},
        )
        self.assertNotIn("api_key", json.dumps(report).lower())


if __name__ == "__main__":
    unittest.main()
