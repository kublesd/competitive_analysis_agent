import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from competitive_analysis_agent.researcher import ResearchError
from competitive_analysis_agent.schemas import Evidence, MarketDefinition
from competitive_analysis_agent.ui_service import AnalysisRunResult
from competitive_analysis_agent.verifier import (
    VerificationIssue,
    VerificationResult,
)


APP_PATH = (
    Path(__file__).parents[1]
    / "competitive_analysis_agent"
    / "streamlit_app.py"
)


def _build_ui_result(
    *,
    verification_passed: bool = True,
    has_research_error: bool = False,
) -> AnalysisRunResult:
    """创建 AppTest 使用的短报告结果，避免离线 UI 测试调用模型。"""

    evidence = [
        Evidence(
            evidence_id="E1",
            product_name="Atlas Notes",
            topic="features",
            title="Atlas Notes Features",
            url="https://example.com/atlas/features",
            snippet="Atlas Notes supports shared workspaces.",
            source_type="official",
            collected_at=datetime(
                2026,
                6,
                14,
                8,
                0,
                tzinfo=timezone.utc,
            ),
        )
    ]
    issues: list[VerificationIssue] = []
    if not verification_passed:
        issues.append(
            VerificationIssue(
                issue_type="unsupported_claim",
                claim_path="features[0]",
                message="The claim is unsupported.",
                evidence_ids=["E1"],
                suggested_action="Remove the claim.",
            )
        )

    research_errors: list[ResearchError] = []
    if has_research_error:
        research_errors.append(
            ResearchError(
                product_name="Beacon Docs",
                topic="features",
                query="Beacon Docs official features",
                code="no_results",
                message="Search completed but returned no results.",
            )
        )

    return AnalysisRunResult(
        final_report=(
            "# 竞品分析报告\n\n"
            "## 资料来源\n\n"
            "[E1](https://example.com/atlas/features)\n"
        ),
        market_definition=MarketDefinition(
            market_name="团队知识管理工具",
            product_category="SaaS 协作软件",
            target_buyer="中型企业 IT 与业务负责人",
            comparison_level="企业订阅产品",
            core_dimensions=["features"],
            exclusions=["消费端套餐"],
        ),
        stage_history=[
            "planner",
            "researcher",
            "extractor",
            "analyst",
            "verifier",
            "reporter",
        ],
        evidence=evidence,
        verification_result=VerificationResult(
            passed=verification_passed,
            issues=issues,
            retry_recommended=not verification_passed,
        ),
        research_errors=research_errors,
    )


class StreamlitAppTest(unittest.TestCase):
    def setUp(self) -> None:
        """隔离真实日志文件，避免本地运行中的应用锁住 UI 测试。"""

        logging_patcher = patch(
            "competitive_analysis_agent.logging_config."
            "configure_application_logging"
        )
        logging_patcher.start()
        self.addCleanup(logging_patcher.stop)

    def test_page_renders_real_search_form(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run()

        self.assertEqual(app.title[0].value, "AI 竞品分析 Agent")
        self.assertEqual(app.text_input[0].value, "OpenAI API")
        self.assertEqual(app.text_input[1].value, "生成式 AI API")
        self.assertEqual(app.text_input[2].value, "大语言模型 API")
        self.assertEqual(app.text_input[4].value, "模型 API 服务")
        self.assertEqual(app.text_area[0].value, "Claude API\nGemini API")
        self.assertEqual(app.text_area[1].value, "")
        self.assertIn(
            "OpenAI API=openai.com,platform.openai.com",
            app.text_area[3].value,
        )
        self.assertEqual(
            app.multiselect[0].value,
            [
                "model_capabilities",
                "api_pricing",
                "developer_platform",
                "usage_limits",
            ],
        )
        self.assertEqual(app.selectbox[0].value, "api")
        self.assertEqual(app.exception, [])

    def test_custom_dimensions_are_submitted_to_service(self) -> None:
        # 自定义维度要进入 AnalysisRequest，而不是只停留在页面显示层。
        fake_result = _build_ui_result()
        with patch(
            "competitive_analysis_agent.ui_service.run_analysis",
            return_value=fake_result,
        ) as mocked_run:
            app = AppTest.from_file(str(APP_PATH)).run()
            app.multiselect[0].set_value(["model_capabilities"])
            app.text_area[1].set_value(
                "coding\nresearch, enterprise_security；model_capabilities"
            )
            app.button[0].click().run()

        submitted_request = mocked_run.call_args.args[0]
        self.assertEqual(
            submitted_request.dimensions,
            [
                "model_capabilities",
                "coding",
                "research",
                "enterprise_security",
            ],
        )
        self.assertEqual(
            submitted_request.market_definition.exclusions,
            ["消费端订阅套餐", "按席位企业套餐"],
        )
        self.assertEqual(
            submitted_request.market_definition.pricing_scope,
            "api",
        )
        self.assertEqual(app.exception, [])

    def test_submit_saves_report_and_survives_rerun(self) -> None:
        # 点击按钮后报告进入 Session State，页面重跑不再次调用模型。
        fake_result = _build_ui_result()
        with patch(
            "competitive_analysis_agent.ui_service.run_analysis",
            return_value=fake_result,
        ) as mocked_run:
            app = AppTest.from_file(str(APP_PATH)).run()
            app.button[0].click().run()

            self.assertEqual(mocked_run.call_count, 1)
            self.assertIn(
                "# 竞品分析报告",
                app.session_state["final_report"],
            )
            self.assertEqual(app.error, [])
            self.assertEqual(app.exception, [])
            self.assertEqual(len(app.get("download_button")), 1)

            app.run()
            self.assertEqual(mocked_run.call_count, 1)
            self.assertIn(
                "竞品分析报告",
                "\n".join(item.value for item in app.markdown),
            )

    def test_invalid_official_domain_shows_error_without_traceback(self) -> None:
        # 未知产品的域名配置应显示普通输入错误，不展示 traceback。
        app = AppTest.from_file(str(APP_PATH)).run()
        app.text_area[3].set_value("Unknown Product=example.com")
        app.button[0].click().run()

        self.assertEqual(len(app.error), 1)
        self.assertIn("输入格式不正确", app.error[0].value)
        self.assertEqual(app.exception, [])
        self.assertIsNone(app.session_state["final_report"])

    def test_partial_failure_and_verification_warning_are_visible(self) -> None:
        # 部分研究失败和最终验证失败都必须在报告外额外提示。
        fake_result = _build_ui_result(
            verification_passed=False,
            has_research_error=True,
        )
        with patch(
            "competitive_analysis_agent.ui_service.run_analysis",
            return_value=fake_result,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.button[0].click().run()

        warning_text = "\n".join(item.value for item in app.warning)
        self.assertIn("部分研究任务未完成", warning_text)
        self.assertIn("最终验证未通过", warning_text)
        self.assertEqual(app.exception, [])

    def test_result_summary_shows_scope_counts_and_verification_states(
        self,
    ) -> None:
        # 页面摘要必须与服务结果一致，不在 Streamlit 内重新推导业务口径。
        fake_result = _build_ui_result()
        excluded = fake_result.evidence[0].model_copy(
            update={
                "evidence_id": "E2",
                "scope_status": "out_of_scope",
                "scope_reason": "Consumer plan is excluded.",
            }
        )
        uncertain = fake_result.evidence[0].model_copy(
            update={
                "evidence_id": "E3",
                "scope_status": "uncertain",
                "scope_reason": "Product level is not confirmed.",
            }
        )
        fake_result = fake_result.model_copy(
            update={"evidence": [*fake_result.evidence, excluded, uncertain]}
        )

        with patch(
            "competitive_analysis_agent.ui_service.run_analysis",
            return_value=fake_result,
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.button[0].click().run()

        caption_text = "\n".join(item.value for item in app.caption)
        source_text = "\n".join(item.value for item in app.markdown)
        self.assertIn("市场：团队知识管理工具", caption_text)
        self.assertIn(
            "范围内资料：1；已排除：1；待核验：1",
            caption_text,
        )
        self.assertIn(
            "引用有效：通过；范围一致：通过；比较可用：通过",
            caption_text,
        )
        self.assertIn("已排除；原因：Consumer plan is excluded.", source_text)
        self.assertIn("待核验；原因：Product level is not confirmed.", source_text)
        self.assertEqual(app.exception, [])


if __name__ == "__main__":
    unittest.main()
