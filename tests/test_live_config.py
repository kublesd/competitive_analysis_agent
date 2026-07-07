"""验证真实模型配置文件使用可移动的项目相对路径。"""

from pathlib import Path

from competitive_analysis_agent.live_config import (
    LIVE_ENV_FILE,
    PROJECT_ROOT,
    build_provider_request_options,
)
from competitive_analysis_agent.config import Settings


def test_live_environment_file_is_relative_to_project_root() -> None:
    """项目移动后，默认配置仍应从当前代码仓库根目录读取。"""

    expected_root = Path(__file__).resolve().parents[1]

    assert PROJECT_ROOT == expected_root
    assert LIVE_ENV_FILE == expected_root / ".env"


def test_gemini_25_flash_uses_openai_reasoning_effort() -> None:
    """Gemini 2.5 Flash 应用官方 OpenAI 兼容 thinking 关闭参数。"""

    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        llm_model="gemini-2.5-flash",
        tavily_api_key=None,
    )

    assert build_provider_request_options(settings) == {
        "reasoning_effort": "none"
    }


def test_non_gemini_endpoint_does_not_receive_provider_options() -> None:
    """其他 OpenAI 兼容供应商不应收到 Gemini 专用参数。"""

    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1/",
        llm_model="qwen-test-model",
        tavily_api_key=None,
    )

    assert build_provider_request_options(settings) == {}
