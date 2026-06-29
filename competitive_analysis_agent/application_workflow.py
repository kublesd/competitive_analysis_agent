"""创建 UI 使用的真实模型与真实搜索工作流组件。"""

from __future__ import annotations

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_analyst import create_live_analyst
from competitive_analysis_agent.live_extractor import create_live_extractor
from competitive_analysis_agent.live_planner import create_live_planner
from competitive_analysis_agent.live_verifier import create_live_verifier
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.search import SearchAdapter, TavilySearchProvider
from competitive_analysis_agent.workflow import WorkflowComponents


class ApplicationSearchConfigurationError(ValueError):
    """表示 UI 缺少真实搜索所需的 Tavily 配置。"""


def create_application_workflow_components(
    settings: Settings,
) -> WorkflowComponents:
    """创建生产入口组件；模型和搜索都使用应用真实配置。"""

    if settings.tavily_api_key is None:
        raise ApplicationSearchConfigurationError(
            "Missing environment variable: TAVILY_API_KEY"
        )

    researcher = Researcher(
        search_adapter=SearchAdapter(
            TavilySearchProvider(settings.tavily_api_key)
        )
    )
    return WorkflowComponents(
        planner=create_live_planner(settings),
        researcher=researcher,
        extractor=create_live_extractor(settings),
        analyst=create_live_analyst(settings),
        verifier=create_live_verifier(settings),
        reporter=Reporter(),
    )
