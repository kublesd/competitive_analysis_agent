"""验证 UI 真实工作流的搜索配置边界。"""

import pytest

from competitive_analysis_agent.application_workflow import (
    ApplicationSearchConfigurationError,
    create_application_workflow_components,
)
from competitive_analysis_agent.config import Settings


def test_application_workflow_requires_tavily_api_key() -> None:
    """缺少真实搜索 Key 时应在创建组件阶段明确失败。"""

    settings = Settings(
        llm_api_key="llm-key",
        llm_base_url="https://example.com/v1",
        llm_model="example-model",
        tavily_api_key=None,
    )

    with pytest.raises(
        ApplicationSearchConfigurationError,
        match="TAVILY_API_KEY",
    ):
        create_application_workflow_components(settings)
