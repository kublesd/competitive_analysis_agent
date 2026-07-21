import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
from openai import APITimeoutError

from competitive_analysis_agent.agent_hooks import AgentRunContext, HookManager
from competitive_analysis_agent.analyst import (
    Analyst,
    AnalystOutput,
    FakeAnalystModel,
)
from competitive_analysis_agent.extractor import (
    Extractor,
    FakeExtractorModel,
)
from competitive_analysis_agent.planner import (
    FakePlannerModel,
    Planner,
    PlannerInput,
)
from competitive_analysis_agent.researcher import Researcher
from competitive_analysis_agent.reporter import Reporter
from competitive_analysis_agent.search import (
    FakeSearchProvider,
    SearchAdapter,
)
from competitive_analysis_agent.schemas import (
    Evidence,
    MarketDefinition,
    ProductProfile,
    ResearchTask,
    WorkflowState,
)
from competitive_analysis_agent.verifier import (
    ClaimVerification,
    ClaimVerificationStatus,
    VerificationIssue,
    VerificationResult,
    Verifier,
)
from competitive_analysis_agent.workflow import (
    WorkflowComponents,
    build_revision_feedback,
    create_initial_state,
    create_workflow_graph,
    run_planner_node,
    run_researcher_node,
    run_extractor_node,
    run_analyst_node,
    run_reporter_node,
    run_verifier_node,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
FIXED_TIME = datetime(2026, 6, 13, 8, 0, tzinfo=timezone.utc)


def _load_json(file_name: str) -> dict:
    """读取 Stage 8 固定数据，确保图测试不访问网络。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


class SequenceVerifierModel:
    """按顺序返回多个 Verifier 响应，用于测试图级循环。"""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.invocation_count = 0

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回下一条固定响应，并拒绝超出测试预设次数。"""

        if self.invocation_count >= len(self._responses):
            raise RuntimeError("No verifier response left.")

        response = self._responses[self.invocation_count]
        self.invocation_count += 1
        return response


class StaticExtractor:
    """返回预置画像，用于测试 Workflow 的画像入场校验。"""

    def __init__(self, profiles: list[ProductProfile]) -> None:
        self.profiles = profiles

    def extract(self, extractor_input: object) -> list[ProductProfile]:
        """忽略输入并返回测试画像，模拟上游漏掉的污染情况。"""

        return self.profiles


class TimeoutAnalystModel:
    """模拟真实供应商 APITimeoutError，确认 Analyst 使用确定性 fallback。"""

    def __init__(self) -> None:
        self.invocation_count = 0

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """每次都抛出 OpenAI SDK 的真实超时异常类型。"""

        self.invocation_count += 1
        request = httpx.Request("POST", "https://example.com/v1/chat")
        raise APITimeoutError(request=request)


class StageStartRecorder:
    """记录工作流阶段开始事件，用于验证重试 attempt_index。"""

    def __init__(self) -> None:
        self.started_stages: list[tuple[str, int]] = []

    def on_run_started(self, context) -> None:
        """本测试只关心阶段事件。"""

    def on_stage_started(
        self,
        run_context,
        stage_context,
        input_summary,
    ) -> None:
        """记录阶段名称和尝试次数。"""

        self.started_stages.append(
            (stage_context.stage_name, stage_context.attempt_index)
        )

    def on_stage_completed(
        self,
        run_context,
        stage_context,
        output_summary,
    ) -> None:
        """本测试只关心阶段开始事件。"""

    def on_stage_failed(
        self,
        run_context,
        stage_context,
        error_summary,
    ) -> None:
        """本测试不触发失败事件。"""

    def on_run_completed(self, context, result_summary) -> None:
        """本测试只关心阶段事件。"""

    def on_run_failed(self, context, error_summary) -> None:
        """本测试不触发失败事件。"""


def _build_components(
    analyst_responses: list[object],
    verifier_responses: list[object],
) -> tuple[WorkflowComponents, FakeAnalystModel, SequenceVerifierModel]:
    """创建完整的 fixture-backed 工作流依赖。"""

    planner_outputs = _load_json("planner_outputs.json")
    extractor_outputs = _load_json("extractor_outputs.json")
    search_results = _load_json("workflow_search_results.json")

    planner = Planner(FakePlannerModel([planner_outputs["valid"]]))
    researcher = Researcher(
        search_adapter=SearchAdapter(FakeSearchProvider(search_results)),
        clock=lambda: FIXED_TIME,
    )
    extractor = Extractor(
        FakeExtractorModel(
            [
                extractor_outputs["valid_atlas"],
                extractor_outputs["valid_beacon"],
            ]
        )
    )
    analyst_model = FakeAnalystModel(analyst_responses)
    verifier_model = SequenceVerifierModel(verifier_responses)
    components = WorkflowComponents(
        planner=planner,
        researcher=researcher,
        extractor=extractor,
        analyst=Analyst(analyst_model),
        verifier=Verifier(verifier_model),
        reporter=Reporter(),
    )
    return components, analyst_model, verifier_model


def _build_initial_state():
    """创建两个产品、两个维度的固定图输入。"""

    market_definition = MarketDefinition(
        market_name="团队知识管理工具",
        product_category="SaaS 协作软件",
        target_buyer="中型企业 IT 与业务负责人",
        comparison_level="企业订阅产品",
        core_dimensions=["features", "pricing"],
        exclusions=["消费端套餐", "API 用量价格"],
    )
    planner_input = PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        market_definition=market_definition,
    )
    return create_initial_state(
        planner_input,
        market_definition=market_definition,
        official_domains_by_product={
            "Atlas Notes": ["example.com"],
            "Beacon Docs": ["example.com"],
        },
    )


def _build_evidence(
    evidence_id: str,
    product_name: str,
    topic: str,
    title: str,
    snippet: str,
    raw_content: str | None = None,
) -> Evidence:
    """创建 Workflow 校验测试需要的最小 Evidence。"""

    return Evidence(
        evidence_id=evidence_id,
        product_name=product_name,
        topic=topic,
        title=title,
        url=f"https://example.com/{evidence_id.lower()}",
        snippet=snippet,
        raw_content=raw_content,
        source_type="official",
        collected_at=FIXED_TIME,
    )


class WorkflowTest(unittest.TestCase):
    def test_reporter_node_enforces_claim_verification_statuses(self) -> None:
        # 六态结果必须穿过 Workflow State 并实际控制最终 Markdown。
        sample_state = WorkflowState.model_validate(
            _load_json("sample_case.json")
        )
        analysis = AnalystOutput.model_validate(
            _load_json("analyst_outputs.json")["valid"]
        ).analysis
        partial_claim = analysis.features[0]
        state = _build_initial_state()
        state["evidence"] = list(sample_state.evidence)
        state["product_profiles"] = list(sample_state.product_profiles)
        state["analysis_result"] = analysis
        state["verification_result"] = VerificationResult(
            passed=False,
            issues=[
                VerificationIssue(
                    issue_type="partially_supported",
                    claim_path="features[0]",
                    message="Only the named template is supported.",
                    evidence_ids=list(partial_claim.evidence_ids),
                    suggested_action="Narrow the claim.",
                )
            ],
            claim_verifications=[
                ClaimVerification(
                    field_path="features[0]",
                    claim=partial_claim.claim,
                    status=ClaimVerificationStatus.PARTIALLY_SUPPORTED,
                    evidence_ids=list(partial_claim.evidence_ids),
                    reason="Only the named template is supported.",
                    suggested_action="Narrow the claim.",
                )
            ],
            retry_recommended=True,
        )

        update = run_reporter_node(state, Reporter())
        report = update["final_report"]

        self.assertIn("## 待核验 Claim", report)
        self.assertIn(partial_claim.claim, report)
        self.assertIn("验证未通过，未输出正式购买建议", report)
        self.assertNotIn("## 面向购买者的建议", report)

    def test_researcher_node_preserves_all_scope_buckets(self) -> None:
        # Extractor 只消费范围内资料，但 State 必须保留排除和待核验资料供报告展示。
        state = _build_initial_state()
        state["research_tasks"] = [
            ResearchTask(
                product_name="Atlas Notes",
                topic="features",
                query="Atlas Notes scoped features",
            )
        ]
        search_results = {
            "Atlas Notes scoped features": [
                {
                    "title": "Atlas Notes Features",
                    "url": "https://example.com/atlas/features",
                    "snippet": "Atlas Notes documents team features.",
                },
                {
                    "title": "Atlas Notes Consumer Plan",
                    "url": "https://example.com/atlas/consumer",
                    "snippet": "消费端套餐 for Atlas Notes.",
                },
                {
                    "title": "General collaboration article",
                    "url": "https://third-party.example/collaboration",
                    "snippet": "A general market overview.",
                },
            ]
        }
        researcher = Researcher(
            SearchAdapter(FakeSearchProvider(search_results)),
            clock=lambda: FIXED_TIME,
        )

        update = run_researcher_node(state, researcher)

        self.assertEqual(len(update["evidence"]), 1)
        self.assertEqual(len(update["excluded_evidence"]), 1)
        self.assertEqual(len(update["uncertain_evidence"]), 1)
        self.assertEqual(
            update["excluded_evidence"][0].scope_status,
            "out_of_scope",
        )
        self.assertEqual(
            update["uncertain_evidence"][0].scope_status,
            "uncertain",
        )

    def test_fixture_graph_reaches_verified_terminal_state(self) -> None:
        # happy path 应按线性顺序执行一次并通过验证。
        analyst_outputs = _load_json("analyst_outputs.json")
        verifier_outputs = _load_json("verifier_outputs.json")
        components, analyst_model, verifier_model = _build_components(
            [analyst_outputs["valid"]],
            [verifier_outputs["supported"]],
        )
        graph = create_workflow_graph(components)

        final_state = graph.invoke(_build_initial_state())

        self.assertEqual(len(final_state["research_tasks"]), 4)
        self.assertEqual(len(final_state["evidence"]), 4)
        self.assertEqual(len(final_state["product_profiles"]), 2)
        self.assertIsNotNone(final_state["analysis_result"])
        self.assertTrue(final_state["verification_result"].passed)
        self.assertIn("# 竞品分析报告", final_state["final_report"])
        self.assertEqual(final_state["retry_count"], 0)
        self.assertEqual(
            final_state["stage_history"],
            [
                "planner",
                "researcher",
                "extractor",
                "analyst",
                "verifier",
                "reporter",
            ],
        )
        self.assertEqual(analyst_model.invocation_count, 1)
        self.assertEqual(verifier_model.invocation_count, 1)

    def test_failed_verification_retries_analyst_once(self) -> None:
        # 第一轮语义失败后，issues 应进入 Analyst，第二轮通过后结束。
        analyst_outputs = _load_json("analyst_outputs.json")
        verifier_outputs = _load_json("verifier_outputs.json")
        components, analyst_model, verifier_model = _build_components(
            [
                analyst_outputs["semantic_unsupported"],
                analyst_outputs["valid"],
            ],
            [
                verifier_outputs["unsupported"],
                verifier_outputs["supported"],
            ],
        )
        stage_recorder = StageStartRecorder()
        hook_manager = HookManager(
            run_context=AgentRunContext(
                analysis_id="retry-test",
                entrypoint="test",
                started_at=perf_counter(),
            ),
            hooks=[stage_recorder],
        )
        graph = create_workflow_graph(
            components,
            hook_manager=hook_manager,
        )

        final_state = graph.invoke(_build_initial_state())

        self.assertTrue(final_state["verification_result"].passed)
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(analyst_model.invocation_count, 2)
        self.assertEqual(verifier_model.invocation_count, 2)
        self.assertEqual(
            final_state["stage_history"],
            [
                "planner",
                "researcher",
                "extractor",
                "analyst",
                "verifier",
                "analyst",
                "verifier",
                "reporter",
            ],
        )
        revision_message = analyst_model.received_messages[1][1]["content"]
        self.assertIn("features[0]", revision_message)
        self.assertIn("unsupported_claim", revision_message)
        analyst_attempts = [
            attempt
            for stage_name, attempt in stage_recorder.started_stages
            if stage_name == "analyst"
        ]
        verifier_attempts = [
            attempt
            for stage_name, attempt in stage_recorder.started_stages
            if stage_name == "verifier"
        ]
        self.assertEqual(analyst_attempts, [1, 2])
        self.assertEqual(verifier_attempts, [1, 2])

    def test_api_pricing_fallback_passes_without_analysis_retry(self) -> None:
        # Analyst 超时后，正确价格画像的 fallback 应在第一次 Verifier 中通过。
        evidence = [
            _build_evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="pricing",
                title="OpenAI API pricing",
                snippet=(
                    "GPT-5.6-sol input costs $5.00 per million input tokens."
                ),
            ),
            _build_evidence(
                evidence_id="E2",
                product_name="Gemini API",
                topic="pricing",
                title="Gemini API pricing",
                snippet=(
                    "Gemini 2.0 Flash input costs $0.15 per million input tokens."
                ),
            ),
        ]
        profiles = [
            ProductProfile(
                product_name="OpenAI API",
                dimension_findings=[
                    {
                        "dimension": "api_pricing",
                        "facts": [
                            "GPT-5.6-sol input | "
                            "$5.00 per million input tokens | "
                            "per million input tokens"
                        ],
                        "evidence_ids": ["E1"],
                    }
                ],
                pricing=[
                    {
                        "plan_name": "GPT-5.6-sol input",
                        "price": "$5.00 per million input tokens",
                        "unit": "per million input tokens",
                        "evidence_ids": ["E1"],
                    }
                ],
            ),
            ProductProfile(
                product_name="Gemini API",
                dimension_findings=[
                    {
                        "dimension": "api_pricing",
                        "facts": [
                            "Gemini 2.0 Flash input | "
                            "$0.15 per million input tokens | "
                            "per million input tokens"
                        ],
                        "evidence_ids": ["E2"],
                    }
                ],
                pricing=[
                    {
                        "plan_name": "Gemini 2.0 Flash input",
                        "price": "$0.15 per million input tokens",
                        "unit": "per million input tokens",
                        "evidence_ids": ["E2"],
                    }
                ],
            ),
        ]
        state = _build_initial_state()
        state["market_definition"] = MarketDefinition(
            market_name="大模型 API",
            product_category="模型 API",
            target_buyer="开发者",
            comparison_level="开发者 API",
            pricing_scope="api",
            core_dimensions=["api_pricing"],
        )
        state["dimensions"] = ["api_pricing"]
        state["evidence"] = evidence
        state["product_profiles"] = profiles
        timeout_model = TimeoutAnalystModel()

        state.update(run_analyst_node(state, Analyst(timeout_model)))
        state.update(
            run_verifier_node(
                state,
                Verifier(SequenceVerifierModel([{"issues": []}])),
            )
        )
        state.update(run_reporter_node(state, Reporter()))

        self.assertEqual(timeout_model.invocation_count, 1)
        self.assertTrue(state["verification_result"].passed)
        self.assertEqual(state["retry_count"], 0)
        self.assertIn("$5.00 per million input tokens", state["final_report"])

    def test_revision_feedback_does_not_copy_suggested_action(self) -> None:
        # Verifier 的 suggested_action 可能带偏模型，Workflow 只传保守修订规则。
        verification_result = VerificationResult(
            passed=False,
            retry_recommended=True,
            issues=[
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[0]",
                    "message": "The broad feature claim is unsupported.",
                    "evidence_ids": ["E1"],
                    "suggested_action": (
                        "Rephrase as includes features like Search and "
                        "Deep research."
                    ),
                }
            ],
        )

        feedback = build_revision_feedback(verification_result)

        self.assertEqual(len(feedback), 1)
        self.assertIn("features[0]", feedback[0])
        self.assertIn("Product mentions Feature", feedback[0])
        self.assertNotIn("includes features like Search", feedback[0])

    def test_retry_fallback_removes_verifier_rejected_feature(self) -> None:
        # 第二轮 Analyst 服务不可用时，fallback 也必须吸收 Verifier 反馈。
        analyst_outputs = _load_json("analyst_outputs.json")
        verifier_outputs = _load_json("verifier_outputs.json")
        unsupported_beacon_feature = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[1]",
                    "message": (
                        "The claim that Beacon Docs mentions Collaborative "
                        "pages is not supported by the provided evidence."
                    ),
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove this claim.",
                }
            ]
        }
        components, analyst_model, verifier_model = _build_components(
            [analyst_outputs["valid"]],
            [unsupported_beacon_feature, verifier_outputs["supported"]],
        )
        graph = create_workflow_graph(components)

        final_state = graph.invoke(_build_initial_state())

        feature_claims = [
            claim.claim for claim in final_state["analysis_result"].features
        ]
        self.assertTrue(final_state["verification_result"].passed)
        self.assertEqual(analyst_model.invocation_count, 1)
        self.assertEqual(verifier_model.invocation_count, 2)
        self.assertNotIn(
            "Beacon Docs mentions Collaborative pages.",
            feature_claims,
        )

    def test_retry_limit_stops_second_failed_verification(self) -> None:
        # 连续两次失败后必须结束，不能形成无限 LangGraph 循环。
        analyst_outputs = _load_json("analyst_outputs.json")
        conflicting_feature = {
            "issues": [
                {
                    "issue_type": "conflicting_evidence",
                    "claim_path": "features[0]",
                    "message": "The cited evidence contradicts this feature.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the conflicting feature.",
                }
            ]
        }
        components, analyst_model, verifier_model = _build_components(
            [
                analyst_outputs["valid"],
                analyst_outputs["valid"],
            ],
            [
                conflicting_feature,
                conflicting_feature,
            ],
        )
        graph = create_workflow_graph(components)

        final_state = graph.invoke(_build_initial_state())

        self.assertFalse(final_state["verification_result"].passed)
        self.assertTrue(
            final_state["verification_result"].retry_recommended
        )
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(analyst_model.invocation_count, 2)
        self.assertEqual(verifier_model.invocation_count, 2)
        self.assertIn(
            "本报告未通过最终验证",
            final_state["final_report"],
        )
        self.assertNotIn(
            "## 面向购买者的建议",
            final_state["final_report"],
        )
        self.assertIn(
            "验证未通过，未输出正式购买建议",
            final_state["final_report"],
        )

    def test_node_function_runs_without_compiled_graph(self) -> None:
        # 节点包装仍是普通函数，可以脱离 LangGraph 独立测试。
        planner_outputs = _load_json("planner_outputs.json")
        planner = Planner(FakePlannerModel([planner_outputs["valid"]]))
        initial_state = _build_initial_state()

        update = run_planner_node(initial_state, planner)

        self.assertEqual(len(update["research_tasks"]), 4)
        self.assertEqual(update["stage_history"], ["planner"])

    def test_extractor_node_validates_profile_before_analyst(self) -> None:
        # Workflow 应在 Analyst 前挡住订阅价格和套餐级定位，并记录数据限制。
        evidence = [
            _build_evidence(
                evidence_id="E1",
                product_name="ChatGPT",
                topic="pricing",
                title="OpenAI API pricing",
                snippet="OpenAI API pricing lists token-based model prices.",
                raw_content="GPT-4.1 input tokens $2.00 / 1M tokens.",
            ),
            _build_evidence(
                evidence_id="E2",
                product_name="ChatGPT",
                topic="pricing",
                title="ChatGPT plans",
                snippet=(
                    "ChatGPT Plus subscription plan is billed per month."
                ),
                raw_content="ChatGPT Plus plan $20 per month.",
            ),
        ]
        polluted_profile = ProductProfile(
            product_name="ChatGPT",
            positioning="ChatGPT Plus is for daily users on a monthly plan.",
            dimension_findings=[
                {
                    "dimension": "api_pricing",
                    "facts": [
                        "GPT-4.1 input tokens：$2.00 / 1M tokens",
                        "ChatGPT Plus：$20 / monthly",
                    ],
                    "evidence_ids": ["E1", "E2"],
                }
            ],
            pricing=[
                {
                    "plan_name": "GPT-4.1 input tokens",
                    "price": "$2.00 / 1M tokens",
                    "billing_cycle": None,
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
                {
                    "plan_name": "ChatGPT Plus",
                    "price": "$20",
                    "billing_cycle": "monthly",
                    "main_limits": [],
                    "evidence_ids": ["E2"],
                },
            ],
        )
        state = _build_initial_state()
        state["market_definition"] = state["market_definition"].model_copy(
            update={
                "pricing_scope": "api",
                "core_dimensions": ["api_pricing"],
            }
        )
        state["dimensions"] = ["api_pricing"]
        state["evidence"] = evidence
        state["research_errors"] = []

        update = run_extractor_node(
            state,
            StaticExtractor([polluted_profile]),
        )

        validated_profile = update["product_profiles"][0]
        validation_errors = update["research_errors"]
        self.assertIsNone(validated_profile.positioning)
        self.assertEqual(len(validated_profile.pricing), 1)
        self.assertEqual(
            validated_profile.pricing[0].plan_name,
            "GPT-4.1 input tokens",
        )
        price_finding = validated_profile.dimension_findings[0]
        self.assertEqual(price_finding.evidence_ids, ["E1"])
        self.assertNotIn("ChatGPT Plus", " ".join(price_finding.facts))
        error_messages = [error.message for error in validation_errors]
        self.assertTrue(
            any("ChatGPT Plus" in message for message in error_messages)
        )
        self.assertTrue(
            any("positioning" in message for message in error_messages)
        )

        # 同一输入改为订阅范围时，保留结果应反转，证明产品名不再决定价格范围。
        subscription_state = _build_initial_state()
        subscription_state["evidence"] = evidence
        subscription_state["research_errors"] = []
        subscription_update = run_extractor_node(
            subscription_state,
            StaticExtractor([polluted_profile]),
        )
        self.assertEqual(
            [
                plan.plan_name
                for plan in subscription_update["product_profiles"][0].pricing
            ],
            ["ChatGPT Plus"],
        )

    def test_extractor_node_removes_conflicting_profile_prices(self) -> None:
        # 同名套餐同一计费周期出现多个价格时，Workflow 不应让它进入 Analyst。
        evidence = [
            _build_evidence(
                evidence_id="E1",
                product_name="Atlas Notes",
                topic="pricing",
                title="Atlas pricing",
                snippet="Standard plan $4.99/month and $19.99/month.",
            )
        ]
        polluted_profile = ProductProfile(
            product_name="Atlas Notes",
            pricing=[
                {
                    "plan_name": "Standard",
                    "price": "$4.99",
                    "billing_cycle": "monthly",
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
                {
                    "plan_name": "Standard",
                    "price": "$19.99",
                    "billing_cycle": "monthly",
                    "main_limits": [],
                    "evidence_ids": ["E1"],
                },
            ],
        )
        state = _build_initial_state()
        state["evidence"] = evidence
        state["research_errors"] = []

        update = run_extractor_node(
            state,
            StaticExtractor([polluted_profile]),
        )

        self.assertEqual(update["product_profiles"][0].pricing, [])
        self.assertTrue(
            any(
                "conflicting prices" in error.message
                for error in update["research_errors"]
            )
        )

    def test_extractor_node_keeps_retrieved_api_rates_for_verifier(self) -> None:
        # Workflow 不再用字符串规则删价格，语义支持关系留给 Verifier 模型判断。
        evidence = [
            _build_evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="api_pricing",
                title="OpenAI API pricing",
                snippet=(
                    "GPT-5.6-sol input costs $5.00 per million input tokens. "
                    "GPT-5.6-sol output costs $30.00 per million output tokens."
                ),
            )
        ]
        profile = ProductProfile(
            product_name="OpenAI API",
            dimension_findings=[
                {
                    "dimension": "api_pricing",
                    "facts": ["input: $5.00", "output: $30.00"],
                    "evidence_ids": ["E1"],
                }
            ],
            pricing=[
                {
                    "plan_name": "GPT-5.6-sol input",
                    "price": "$5.00",
                    "unit": "per million input tokens",
                    "evidence_ids": ["E1"],
                },
                {
                    "plan_name": "GPT-5.6-sol output",
                    "price": "$30.00",
                    "unit": "USD",
                    "evidence_ids": ["E1"],
                },
            ],
        )
        state = _build_initial_state()
        state["market_definition"] = state["market_definition"].model_copy(
            update={
                "pricing_scope": "api",
                "core_dimensions": ["api_pricing"],
            }
        )
        state["dimensions"] = ["api_pricing"]
        state["evidence"] = evidence
        state["research_errors"] = []

        update = run_extractor_node(state, StaticExtractor([profile]))

        validated_profile = update["product_profiles"][0]
        self.assertEqual(
            [plan.plan_name for plan in validated_profile.pricing],
            ["GPT-5.6-sol input", "GPT-5.6-sol output"],
        )
        self.assertEqual(
            validated_profile.pricing[0].price,
            "$5.00",
        )
        self.assertEqual(
            validated_profile.pricing[0].unit,
            "per million input tokens",
        )
        self.assertEqual(
            validated_profile.dimension_findings[0].evidence_ids,
            ["E1"],
        )
        self.assertIn(
            "GPT-5.6-sol output | $30.00 | USD",
            validated_profile.dimension_findings[0].facts,
        )
        self.assertEqual(update["research_errors"], [])

    def test_stream_exposes_node_level_state_updates(self) -> None:
        # stream 应暴露节点更新，证明运行状态可以被观察。
        analyst_outputs = _load_json("analyst_outputs.json")
        verifier_outputs = _load_json("verifier_outputs.json")
        components, _, _ = _build_components(
            [analyst_outputs["valid"]],
            [verifier_outputs["supported"]],
        )
        graph = create_workflow_graph(components)

        updates = list(
            graph.stream(
                _build_initial_state(),
                stream_mode="updates",
            )
        )

        node_names = [next(iter(update)) for update in updates]
        self.assertEqual(
            node_names,
            [
                "planner",
                "researcher",
                "extractor",
                "analyst",
                "verifier",
                "reporter",
            ],
        )


if __name__ == "__main__":
    unittest.main()
