"""使用真实模型和固定搜索结果运行完整 LangGraph 工作流。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_analyst import create_live_analyst
from competitive_analysis_agent.live_config import load_live_settings
from competitive_analysis_agent.live_extractor import create_live_extractor
from competitive_analysis_agent.live_planner import create_live_planner
from competitive_analysis_agent.live_verifier import create_live_verifier
from competitive_analysis_agent.planner import PlannerInput
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.search import (
    ProviderSearchResult,
    SearchAdapter,
    SearchRequest,
)
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    WorkflowGraphState,
    create_initial_state,
    create_workflow_graph,
)


FIXED_COLLECTED_AT = datetime(
    2026,
    6,
    13,
    8,
    0,
    tzinfo=timezone.utc,
)


class WorkflowFixtureSearchProvider:
    """为真实 LLM 图测试提供稳定证据，不依赖外部搜索服务。"""

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        """根据查询中的产品名返回对应功能证据。"""

        normalized_query = request.query.casefold()
        if "atlas notes" in normalized_query:
            return [
                ProviderSearchResult(
                    title="Atlas Notes Features",
                    url="https://example.com/atlas/features",
                    snippet=(
                        "Atlas Notes supports shared workspaces and "
                        "reusable page templates."
                    ),
                )
            ]

        if "beacon docs" in normalized_query:
            return [
                ProviderSearchResult(
                    title="Beacon Docs Features",
                    url="https://example.com/beacon/features",
                    snippet=(
                        "Beacon Docs supports collaborative pages and "
                        "inline comments."
                    ),
                )
            ]

        return []


def create_live_workflow_components(
    settings: Settings,
) -> WorkflowComponents:
    """创建真实模型节点和确定性 Researcher。"""

    researcher = Researcher(
        search_adapter=SearchAdapter(WorkflowFixtureSearchProvider()),
        clock=lambda: FIXED_COLLECTED_AT,
    )
    return WorkflowComponents(
        planner=create_live_planner(settings),
        researcher=researcher,
        extractor=create_live_extractor(settings),
        analyst=create_live_analyst(settings),
        verifier=create_live_verifier(settings),
        reporter=Reporter(),
    )


def run_smoke_test(
    settings: Settings | None = None,
) -> WorkflowGraphState:
    """运行完整真实模型工作流，并返回最终共享 State。"""

    current_settings = settings or load_live_settings()
    components = create_live_workflow_components(current_settings)
    graph = create_workflow_graph(components)
    planner_input = PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        dimensions=["features"],
    )
    initial_state = create_initial_state(
        planner_input=planner_input,
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
        max_results_per_task=1,
    )
    return graph.invoke(initial_state)


def to_json_compatible(value: Any) -> Any:
    """递归转换 Pydantic 对象，供命令行安全打印最终 State。"""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            key: to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]
    return value


def main() -> None:
    """运行真实整图 smoke test，并输出最终共享 State。"""

    final_state = run_smoke_test()
    print(
        json.dumps(
            to_json_compatible(final_state),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
