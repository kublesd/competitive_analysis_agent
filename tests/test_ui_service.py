import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from competitive_analysis_agent.analyst import Analyst, FakeAnalystModel
from competitive_analysis_agent.extractor import (
    Extractor,
    FakeExtractorModel,
)
from competitive_analysis_agent.planner import FakePlannerModel, Planner
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.search import (
    FakeSearchProvider,
    SearchAdapter,
)
from competitive_analysis_agent.ui_service import (
    AnalysisRequest,
    build_analysis_dimensions,
    build_stage_summary,
    choose_max_results_per_task,
    create_analysis_request,
    describe_user_error,
    infer_failed_stage_from_state,
    parse_competitors,
    parse_official_domains,
    run_analysis,
)
from competitive_analysis_agent.verifier import (
    FakeVerifierModel,
    Verifier,
    VerifierError,
)
from competitive_analysis_agent.workflow import WorkflowComponents


FIXED_TIME = datetime(2026, 6, 14, 8, 0, tzinfo=timezone.utc)


def _build_demo_components() -> WorkflowComponents:
    """创建与 UI 演示输入对应的完整离线工作流组件。"""

    planner_output = {
        "tasks": [
            {
                "product_name": "Atlas Notes",
                "topic": "features",
                "query": "Atlas Notes official features",
            },
            {
                "product_name": "Beacon Docs",
                "topic": "features",
                "query": "Beacon Docs official features",
            },
        ]
    }
    search_results = {
        "Atlas Notes official product features capabilities": [
            {
                "title": "Atlas Notes Features",
                "url": "https://example.com/atlas/features",
                "snippet": (
                    "Atlas Notes supports shared workspaces and "
                    "reusable page templates."
                ),
            }
        ],
        "Beacon Docs official product features capabilities": [
            {
                "title": "Beacon Docs Features",
                "url": "https://example.com/beacon/features",
                "snippet": (
                    "Beacon Docs supports collaborative pages and "
                    "inline comments."
                ),
            }
        ],
    }
    extractor_outputs = [
        {
            "profile": {
                "product_name": "Atlas Notes",
                "positioning": None,
                "target_users": [],
                "features": [
                    {
                        "name": "Shared workspaces",
                        "description": (
                            "Supports shared workspaces and templates."
                        ),
                        "evidence_ids": ["E1"],
                    }
                ],
                "pricing": [],
                "strengths": [],
                "limitations": [],
            }
        },
        {
            "profile": {
                "product_name": "Beacon Docs",
                "positioning": None,
                "target_users": [],
                "features": [
                    {
                        "name": "Collaborative pages",
                        "description": (
                            "Supports collaborative pages and comments."
                        ),
                        "evidence_ids": ["E2"],
                    }
                ],
                "pricing": [],
                "strengths": [],
                "limitations": [],
            }
        },
    ]
    analyst_output = {
        "analysis": {
            "products": ["Atlas Notes", "Beacon Docs"],
            "positioning": [],
            "features": [
                {
                    "claim": "Atlas Notes supports shared workspaces.",
                    "claim_type": "fact",
                    "product_names": ["Atlas Notes"],
                    "evidence_ids": ["E1"],
                },
                {
                    "claim": "Beacon Docs supports collaborative pages.",
                    "claim_type": "fact",
                    "product_names": ["Beacon Docs"],
                    "evidence_ids": ["E2"],
                },
            ],
            "pricing": [],
            "opportunities": [],
            "conclusion": {
                "claim": (
                    "Both supplied products provide collaboration "
                    "features."
                ),
                "claim_type": "interpretation",
                "product_names": ["Atlas Notes", "Beacon Docs"],
                "evidence_ids": ["E1", "E2"],
            },
        }
    }

    return WorkflowComponents(
        planner=Planner(FakePlannerModel([planner_output])),
        researcher=Researcher(
            SearchAdapter(FakeSearchProvider(search_results)),
            clock=lambda: FIXED_TIME,
        ),
        extractor=Extractor(FakeExtractorModel(extractor_outputs)),
        analyst=Analyst(FakeAnalystModel([analyst_output])),
        verifier=Verifier(FakeVerifierModel({"issues": []})),
        reporter=Reporter(),
    )


class UiServiceTest(unittest.TestCase):
    def test_competitors_accept_newlines_and_commas(self) -> None:
        # 页面允许常见分隔方式，但 service 统一返回干净列表。
        competitors = parse_competitors(
            "Beacon Docs\nNorth Star, Cloud Page，Team Wiki"
        )

        self.assertEqual(
            competitors,
            [
                "Beacon Docs",
                "North Star",
                "Cloud Page",
                "Team Wiki",
            ],
        )

    def test_custom_dimensions_are_merged_with_selected_dimensions(self) -> None:
        # 自定义维度应支持常见分隔符，并避免和已勾选维度重复。
        dimensions = build_analysis_dimensions(
            selected_dimensions=["features", "pricing"],
            custom_dimensions_text=(
                "coding\nresearch, enterprise_security；Features"
            ),
        )

        self.assertEqual(
            dimensions,
            [
                "features",
                "pricing",
                "coding",
                "research",
                "enterprise_security",
            ],
        )

    def test_many_dimensions_reduce_search_results_per_task(self) -> None:
        # 维度越多，单任务搜索结果数越少，避免 Extractor 输入膨胀。
        self.assertEqual(
            choose_max_results_per_task(
                ["features", "pricing", "positioning", "target_users"]
            ),
            3,
        )
        self.assertEqual(
            choose_max_results_per_task(
                ["a", "b", "c", "d", "e"]
            ),
            2,
        )
        self.assertEqual(
            choose_max_results_per_task(
                ["a", "b", "c", "d", "e", "f", "g"]
            ),
            1,
        )

    def test_empty_dimensions_are_rejected_before_workflow(self) -> None:
        # 缺少分析维度时应在 UI 输入边界失败，不调用模型。
        with self.assertRaises(ValueError):
            create_analysis_request(
                target_product="Atlas Notes",
                competitors_text="Beacon Docs",
                dimensions=[],
            )

    def test_official_domains_are_parsed_for_known_products(self) -> None:
        # 用户显式域名会进入 State，供搜索限定和官方来源分类使用。
        domains = parse_official_domains(
            "Notion=notion.so\nConfluence=atlassian.com,confluence.com",
            ["Notion", "Confluence"],
        )

        self.assertEqual(domains["Notion"], ["notion.so"])
        self.assertEqual(
            domains["Confluence"],
            ["atlassian.com", "confluence.com"],
        )

    def test_unknown_official_domain_product_is_rejected(self) -> None:
        # 域名配置不能悄悄绑定到本次请求以外的产品。
        with self.assertRaises(ValueError):
            parse_official_domains(
                "Unknown Product=example.com",
                ["Notion", "Confluence"],
            )

    def test_fixture_workflow_reports_progress_and_returns_report(self) -> None:
        # service 应运行真实图结构，并按节点顺序回调页面状态。
        request = AnalysisRequest(
            target_product="Atlas Notes",
            competitors=["Beacon Docs"],
            dimensions=["features"],
            official_domains_by_product={
                "Atlas Notes": ["example.com"],
                "Beacon Docs": ["example.com"],
            },
        )
        reported_stages: list[str] = []

        with self.assertLogs(
            "competitive_analysis_agent.ui_service",
            level="INFO",
        ) as captured_logs:
            result = run_analysis(
                request,
                progress_callback=reported_stages.append,
                components=_build_demo_components(),
            )

        self.assertTrue(result.verification_result.passed)
        self.assertIn("# 竞品分析报告", result.final_report)
        self.assertEqual(
            reported_stages,
            [
                "planner",
                "researcher",
                "extractor",
                "analyst",
                "verifier",
                "reporter",
            ],
        )
        self.assertEqual(result.stage_history, reported_stages)
        self.assertEqual(len(result.evidence), 2)
        log_text = "\n".join(captured_logs.output)
        self.assertIn("analysis_started analysis_id=", log_text)
        self.assertIn("analysis_stage_io analysis_id=", log_text)
        self.assertIn("stage=planner direction=input", log_text)
        self.assertIn('"target_product":"Atlas Notes"', log_text)
        self.assertIn("stage=planner direction=output", log_text)
        self.assertIn('"research_tasks"', log_text)
        self.assertIn("stage=extractor direction=input", log_text)
        self.assertIn("stage=analyst direction=output", log_text)
        self.assertIn("stage=reporter direction=output", log_text)
        self.assertIn('"final_report_preview"', log_text)
        self.assertIn(
            "analysis_stage_completed analysis_id=",
            log_text,
        )
        self.assertIn("stage=researcher", log_text)
        self.assertIn("evidence_count=2", log_text)
        self.assertIn("analysis_completed analysis_id=", log_text)

    def test_stage_summary_uses_user_facing_labels(self) -> None:
        summary = build_stage_summary(["planner", "researcher", "reporter"])

        self.assertEqual(
            summary,
            "规划调研任务 → 收集并整理证据 → 生成 Markdown 报告",
        )

    def test_failure_log_excludes_exception_message(self) -> None:
        # 第三方异常文本可能包含敏感请求信息，后台只记录类型和代码位置。
        request = AnalysisRequest(
            target_product="Atlas Notes",
            competitors=["Beacon Docs"],
            dimensions=["features"],
        )
        with patch(
            "competitive_analysis_agent.ui_service._run_analysis_workflow",
            side_effect=RuntimeError("secret-token-must-not-be-logged"),
        ):
            with self.assertLogs(
                "competitive_analysis_agent.ui_service",
                level="ERROR",
            ) as captured_logs:
                with self.assertRaises(RuntimeError):
                    run_analysis(
                        request,
                        components=_build_demo_components(),
                    )

        log_text = "\n".join(captured_logs.output)
        self.assertIn("error_type=RuntimeError", log_text)
        self.assertIn("failure_function=", log_text)
        self.assertNotIn("secret-token-must-not-be-logged", log_text)

    def test_run_analysis_attaches_failure_context(self) -> None:
        # 工作流中断时，异常对象应携带可和日志互相定位的上下文。
        request = AnalysisRequest(
            target_product="Atlas Notes",
            competitors=["Beacon Docs"],
            dimensions=["features"],
            official_domains_by_product={
                "Atlas Notes": ["example.com"],
                "Beacon Docs": ["example.com"],
            },
        )
        components = _build_demo_components()
        failing_components = WorkflowComponents(
            planner=components.planner,
            researcher=components.researcher,
            extractor=components.extractor,
            analyst=components.analyst,
            verifier=Verifier(
                FakeVerifierModel(
                    {"unexpected": "secret-token-must-not-be-shown"}
                )
            ),
            reporter=components.reporter,
        )

        with self.assertRaises(VerifierError) as captured_error:
            run_analysis(request, components=failing_components)

        error = captured_error.exception
        self.assertTrue(error.analysis_id)
        self.assertEqual(error.workflow_failed_stage, "verifier")
        self.assertEqual(
            error.workflow_stage_history,
            ["planner", "researcher", "extractor", "analyst"],
        )
        self.assertEqual(error.failure_function, "validate_semantic_output")
        self.assertNotIn(
            "secret-token-must-not-be-shown",
            error.public_detail,
        )

    def test_describe_user_error_includes_safe_details(self) -> None:
        # 页面错误应说明失败阶段和安全详情，但不暴露内部敏感文本。
        error = VerifierError(
            "internal secret-token-must-not-be-shown",
            public_detail=(
                "Verifier 模型输出结构不符合要求。"
                "结构问题：issues: Field required。"
            ),
        )
        error.analysis_id = "abc123"
        error.workflow_failed_stage = "verifier"
        error.workflow_stage_history = [
            "planner",
            "researcher",
            "extractor",
            "analyst",
        ]
        error.failure_function = "validate_semantic_output"
        error.failure_line = "391"

        message = describe_user_error(error)

        self.assertIn("错误类别：VerifierError", message)
        self.assertIn("分析编号：abc123", message)
        self.assertIn("失败阶段：验证结论与引用", message)
        self.assertIn("已完成阶段：规划调研任务", message)
        self.assertIn("validate_semantic_output:391", message)
        self.assertIn("Verifier 模型输出结构不符合要求", message)
        self.assertNotIn("secret-token-must-not-be-shown", message)

    def test_retry_pending_state_reports_analyst_as_next_failed_stage(self) -> None:
        # Verifier 完成后如果要重试，下一步是 analyst，不是 reporter。
        state = {
            "target_product": "Atlas Notes",
            "competitors": ["Beacon Docs"],
            "dimensions": ["features"],
            "official_domains_by_product": {},
            "max_results_per_task": 3,
            "research_tasks": [],
            "evidence": [],
            "research_errors": [],
            "product_profiles": [],
            "analysis_result": None,
            "verification_result": None,
            "final_report": None,
            "retry_count": 1,
            "retry_pending": True,
            "stage_history": [
                "planner",
                "researcher",
                "extractor",
                "analyst",
                "verifier",
            ],
        }

        self.assertEqual(infer_failed_stage_from_state(state), "analyst")


if __name__ == "__main__":
    unittest.main()
