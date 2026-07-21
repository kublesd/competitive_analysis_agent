import json
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from competitive_analysis_agent.api_app import create_app
from competitive_analysis_agent.application_workflow import (
    ApplicationSearchConfigurationError,
)
from competitive_analysis_agent.researcher import ResearchError
from competitive_analysis_agent.schemas import Evidence, MarketDefinition
from competitive_analysis_agent.ui_service import AnalysisRunResult
from competitive_analysis_agent.verifier import VerificationResult


FIXED_TIME = datetime(2026, 6, 30, 8, 0, tzinfo=timezone.utc)
MARKET_DEFINITION_PAYLOAD = {
    "market_name": "团队知识管理工具",
    "product_category": "SaaS 协作软件",
    "target_buyer": "中型企业 IT 与业务负责人",
    "comparison_level": "企业订阅产品",
    "pricing_scope": "subscription",
    "core_dimensions": ["features"],
    "exclusions": ["消费端套餐"],
}


def build_api_run_result(
    research_errors: list[ResearchError] | None = None,
) -> AnalysisRunResult:
    """创建 API 测试用的最小完整分析结果。"""

    evidence = Evidence(
        evidence_id="E1",
        product_name="Atlas Notes",
        topic="features",
        title="Atlas Notes Features",
        url="https://example.com/atlas/features",
        snippet="Atlas Notes supports shared workspaces.",
        raw_content="raw content should not be returned by the API",
        source_type="official",
        collected_at=FIXED_TIME,
    )
    verification_result = VerificationResult(
        passed=True,
        issues=[],
        retry_recommended=False,
    )
    return AnalysisRunResult(
        final_report="# 竞品分析报告\n\nAtlas Notes has shared workspaces.",
        market_definition=MarketDefinition.model_validate(
            MARKET_DEFINITION_PAYLOAD
        ),
        stage_history=[
            "planner",
            "researcher",
            "extractor",
            "analyst",
            "verifier",
            "reporter",
        ],
        evidence=[evidence],
        verification_result=verification_result,
        research_errors=research_errors or [],
    )


class ApiAppTest(unittest.TestCase):
    def test_health_endpoint_returns_sanitized_configuration(self) -> None:
        # 健康检查只返回配置是否存在，不返回任何密钥字段。
        client = TestClient(create_app(configure_logging=False))

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("configuration", payload)
        self.assertNotIn("api_key", json.dumps(payload).lower())

    def test_analysis_endpoint_runs_shared_service_contract(self) -> None:
        # API 层只负责 HTTP 转换，真正的业务请求仍交给 ui_service 契约。
        received_requests = []

        def fake_runner(request):
            received_requests.append(request)
            return build_api_run_result()

        client = TestClient(
            create_app(
                analysis_runner=fake_runner,
                configure_logging=False,
            )
        )

        response = client.post(
            "/analyses",
            json={
                "target_product": "Atlas Notes",
                "competitors": ["Beacon Docs"],
                "market_definition": MARKET_DEFINITION_PAYLOAD,
                "official_domains_by_product": {
                    "Atlas Notes": ["example.com"],
                    "Beacon Docs": ["example.com"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(received_requests), 1)
        service_request = received_requests[0]
        self.assertEqual(service_request.target_product, "Atlas Notes")
        self.assertEqual(service_request.competitors, ["Beacon Docs"])
        self.assertEqual(service_request.dimensions, ["features"])
        self.assertEqual(
            service_request.market_definition.comparison_level,
            "企业订阅产品",
        )
        self.assertEqual(
            service_request.official_domains_by_product["Beacon Docs"],
            ["example.com"],
        )

        payload = response.json()
        self.assertIn("# 竞品分析报告", payload["final_report"])
        self.assertEqual(
            payload["market_definition"],
            {**MARKET_DEFINITION_PAYLOAD, "monthly_call_count": 1_000},
        )
        self.assertEqual(payload["verification_passed"], True)
        self.assertEqual(payload["citations_valid"], True)
        self.assertEqual(payload["scope_consistent"], True)
        self.assertEqual(payload["comparison_usable"], True)
        self.assertEqual(
            payload["evidence_scope_counts"],
            {"in_scope": 1, "out_of_scope": 0, "uncertain": 0},
        )
        self.assertEqual(payload["stage_history"][-1], "reporter")
        self.assertEqual(payload["evidence"][0]["evidence_id"], "E1")
        self.assertEqual(
            payload["evidence"][0]["scope_status"],
            "in_scope",
        )
        self.assertIn("scope_reason", payload["evidence"][0])
        self.assertEqual(
            payload["evidence"][0]["snippet_preview"],
            "Atlas Notes supports shared workspaces.",
        )
        self.assertNotIn("raw content", json.dumps(payload))

    def test_analysis_endpoint_returns_validation_error(self) -> None:
        # 未知官方域名产品应在应用请求边界被拒绝，不进入真实工作流。
        client = TestClient(
            create_app(
                analysis_runner=lambda request: build_api_run_result(),
                configure_logging=False,
            )
        )

        response = client.post(
            "/analyses",
            json={
                "target_product": "Atlas Notes",
                "competitors": ["Beacon Docs"],
                "market_definition": MARKET_DEFINITION_PAYLOAD,
                "official_domains_by_product": {
                    "Unknown Product": ["example.com"],
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["detail"]["error_type"], "ValidationError")
        self.assertIn("输入格式不正确", payload["detail"]["message"])

    def test_analysis_endpoint_requires_market_definition(self) -> None:
        # API 缺少市场范围时应由共享 Pydantic 契约直接拒绝。
        client = TestClient(
            create_app(
                analysis_runner=lambda request: build_api_run_result(),
                configure_logging=False,
            )
        )

        response = client.post(
            "/analyses",
            json={
                "target_product": "Atlas Notes",
                "competitors": ["Beacon Docs"],
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn(
            "market_definition",
            json.dumps(response.json()),
        )

    def test_analysis_endpoint_returns_configuration_error(self) -> None:
        # 缺少外部服务配置时返回 503，并继续隐藏内部异常原文。
        def failing_runner(request):
            raise ApplicationSearchConfigurationError(
                "missing private-search-key"
            )

        client = TestClient(
            create_app(
                analysis_runner=failing_runner,
                configure_logging=False,
            )
        )

        response = client.post(
            "/analyses",
            json={
                "target_product": "Atlas Notes",
                "competitors": ["Beacon Docs"],
                "market_definition": MARKET_DEFINITION_PAYLOAD,
            },
        )

        self.assertEqual(response.status_code, 503)
        payload_text = json.dumps(response.json(), ensure_ascii=False)
        self.assertIn("搜索配置不完整", payload_text)
        self.assertIn("ApplicationSearchConfigurationError", payload_text)
        self.assertNotIn("private-search-key", payload_text)


if __name__ == "__main__":
    unittest.main()
