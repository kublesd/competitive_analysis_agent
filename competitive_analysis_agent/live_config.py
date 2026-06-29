"""为真实模型验收加载共享环境配置，不暴露敏感值。"""

from __future__ import annotations

from os import environ
from pathlib import Path

from competitive_analysis_agent.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_ENV_FILE = PROJECT_ROOT / ".env.example"
LIVE_VARIABLES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "TAVILY_API_KEY",
)
LIVE_MODEL_MAX_RETRIES = 1


class LiveModelConfigurationError(ValueError):
    """表示真实模型环境文件缺失或无法提供必要配置。"""


def load_live_settings(env_file: Path = LIVE_ENV_FILE) -> Settings:
    """从指定环境文件加载模型配置，不打印或返回任何密钥文本。"""

    if not env_file.is_file():
        raise LiveModelConfigurationError(
            f"Live environment file does not exist: {env_file}"
        )

    file_values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if "=" not in stripped_line:
            continue

        variable_name, variable_value = stripped_line.split("=", maxsplit=1)
        normalized_name = variable_name.strip()
        normalized_value = variable_value.strip().strip("\"'")
        if normalized_name in LIVE_VARIABLES:
            file_values[normalized_name] = normalized_value

    # 与 python-dotenv override=False 一致：调用者已设置的环境变量优先。
    for variable_name, variable_value in file_values.items():
        environ.setdefault(variable_name, variable_value)

    return Settings.from_env()
