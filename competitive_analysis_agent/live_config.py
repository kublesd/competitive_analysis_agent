"""为真实模型验收加载共享环境配置，不暴露敏感值。"""

from __future__ import annotations

from os import environ
from pathlib import Path
from urllib.parse import urlparse

from competitive_analysis_agent.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_ENV_FILE = PROJECT_ROOT / ".env"
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


def build_provider_request_options(settings: Settings) -> dict[str, object]:
    """返回 OpenAI 兼容模型的供应商专用请求选项。"""

    request_options: dict[str, object] = {}
    if should_disable_siliconflow_qwen3_thinking(settings):
        # Qwen3 默认会先生成思考 token；结构化节点不需要这个开销。
        request_options["extra_body"] = {"enable_thinking": False}
    if should_disable_gemini_flash_thinking(settings):
        request_options["reasoning_effort"] = "none"

    return request_options


def should_disable_siliconflow_qwen3_thinking(settings: Settings) -> bool:
    """判断当前配置是否为 SiliconFlow 的 Qwen3 文本模型。"""

    base_url = settings.llm_base_url or ""
    model_name = settings.llm_model or ""
    hostname = urlparse(base_url).hostname or ""

    return (
        hostname.endswith("siliconflow.cn")
        and model_name.startswith("Qwen/Qwen3-")
    )


def should_disable_gemini_flash_thinking(settings: Settings) -> bool:
    """判断当前配置是否应按 Gemini 2.5 Flash 方式关闭 thinking。"""

    base_url = settings.llm_base_url or ""
    model_name = (settings.llm_model or "").lower()
    hostname = urlparse(base_url).hostname or ""

    is_gemini_openai_endpoint = hostname.endswith(
        "generativelanguage.googleapis.com"
    )
    is_gemini_25_model = model_name.startswith("gemini-2.5")
    is_pro_model = "pro" in model_name

    # Google 文档要求 Gemini 2.5 Flash/Lite 使用 reasoning_effort="none"
    # 关闭思考；2.5 Pro 和 Gemini 3 不能关闭，因此不注入该参数。
    return (
        is_gemini_openai_endpoint
        and is_gemini_25_model
        and not is_pro_model
    )
