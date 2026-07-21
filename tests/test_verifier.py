import json
import unittest
from pathlib import Path

from competitive_analysis_agent.analyst import (
    AnalysisClaim,
    AnalystOutput,
    CompetitiveAnalysis,
)
from competitive_analysis_agent.schemas import (
    Evidence,
    MarketDefinition,
    PricingPlan,
    ProductProfile,
    WorkflowState,
)
from competitive_analysis_agent.verifier import (
    ClaimVerificationStatus,
    FakeVerifierModel,
    LangChainVerifierModel,
    Verifier,
    VerifierError,
    VerifierInput,
    VerifierModelOutput,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


def _load_json(file_name: str) -> dict:
    """读取固定 JSON，确保 Verifier 单元测试不调用真实模型。"""

    fixture_path = FIXTURE_DIRECTORY / file_name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _load_sample_evidence() -> list[Evidence]:
    """读取 Stage 1 固定 Evidence。"""

    sample_case = _load_json("sample_case.json")
    workflow_state = WorkflowState.model_validate(sample_case)
    return workflow_state.evidence


def _load_valid_analysis() -> CompetitiveAnalysis:
    """读取 Stage 6 的有效固定分析。"""

    fixture = _load_json("analyst_outputs.json")
    analyst_output = AnalystOutput.model_validate(fixture["valid"])
    return analyst_output.analysis


def _load_sample_state() -> WorkflowState:
    """读取包含市场定义和产品画像的固定工作流状态。"""

    return WorkflowState.model_validate(_load_json("sample_case.json"))


class FakeChatModel:
    """模拟 LangChain ChatModel 的 with_structured_output 接口。"""

    def __init__(self, structured_model: FakeVerifierModel) -> None:
        self.structured_model = structured_model
        self.received_schema: type[VerifierModelOutput] | None = None
        self.received_method: str | None = None
        self.received_include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type[VerifierModelOutput],
        *,
        method: str,
        include_raw: bool,
    ) -> FakeVerifierModel:
        self.received_schema = schema
        self.received_method = method
        self.received_include_raw = include_raw
        return self.structured_model


class SequencedVerifierModel:
    """按顺序返回固定响应，用于验证一次受限的输出修复。"""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.invocation_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回当前响应，并保存每轮收到的消息。"""

        copied_messages = [message.copy() for message in messages]
        self.received_messages.append(copied_messages)
        response = self._responses[self.invocation_count]
        self.invocation_count += 1
        return response


class VerifierTest(unittest.TestCase):
    def test_formal_conclusion_cannot_rely_only_on_third_party_evidence(self) -> None:
        evidence = [
            item.model_copy(update={"source_type": "third_party"})
            for item in _load_sample_evidence()
        ]

        result = Verifier(FakeVerifierModel({"issues": []})).verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=evidence,
            )
        )

        self.assertFalse(result.passed)
        self.assertTrue(any(
            issue.issue_type == "third_party_only_evidence"
            for issue in result.issues
        ))

    def test_semantic_verifier_preserves_all_six_claim_statuses(self) -> None:
        analysis = CompetitiveAnalysis(
            products=["Atlas Notes", "Beacon Docs"],
            conclusion=AnalysisClaim(
                claim="The supplied products require further comparison.",
                claim_type="interpretation",
                product_names=["Atlas Notes", "Beacon Docs"],
                evidence_ids=["E1", "E3"],
            ),
        )
        statuses = list(ClaimVerificationStatus)

        for status in statuses:
            with self.subTest(status=status.value):
                output = {
                    "verifications": [
                        {
                            "claim_path": "C001",
                            "status": status.value,
                            "evidence_ids": ["E1", "E3"],
                            "reason": f"Result is {status.value}.",
                            "suggested_action": "Review this claim.",
                        }
                    ]
                }
                result = Verifier(FakeVerifierModel(output)).verify(
                    VerifierInput(
                        analysis=analysis,
                        evidence=_load_sample_evidence(),
                    )
                )

                self.assertEqual(
                    result.claim_verifications[0].status,
                    status,
                )
                self.assertEqual(
                    result.claim_verifications[0].field_path,
                    "conclusion",
                )
                self.assertEqual(
                    result.claim_verifications[0].claim,
                    analysis.conclusion.claim,
                )
                self.assertEqual(
                    result.passed,
                    status == ClaimVerificationStatus.SUPPORTED,
                )

    def test_input_and_output_price_claims_do_not_conflict_by_number(self) -> None:
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Claude API",
                topic="api_pricing",
                title="Claude pricing",
                url="https://docs.anthropic.com/pricing",
                snippet=(
                    "Claude Sonnet 5 input price is $2 per million tokens; "
                    "output price is $10 per million tokens."
                ),
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        profile = ProductProfile(
            product_name="Claude API",
            dimension_findings=[
                {
                    "dimension": "api_pricing",
                    "facts": [
                        "Claude Sonnet 5 input $2 and output $10 per million tokens."
                    ],
                    "evidence_ids": ["E1"],
                }
            ],
            pricing=[
                PricingPlan(
                    plan_name="Claude Sonnet 5 input",
                    price="$2 per million tokens",
                    unit="per million input tokens",
                    evidence_ids=["E1"],
                ),
                PricingPlan(
                    plan_name="Claude Sonnet 5 output",
                    price="$10 per million tokens",
                    unit="per million output tokens",
                    evidence_ids=["E1"],
                ),
            ],
        )
        analysis = CompetitiveAnalysis(
            products=["Claude API", "Other API"],
            pricing=[
                AnalysisClaim(
                    claim=(
                        "Claude Sonnet 5 input price is $2 per million tokens."
                    ),
                    claim_type="fact",
                    product_names=["Claude API"],
                    evidence_ids=["E1"],
                ),
                AnalysisClaim(
                    claim=(
                        "Claude Sonnet 5 output price is $10 per million tokens."
                    ),
                    claim_type="fact",
                    product_names=["Claude API"],
                    evidence_ids=["E1"],
                ),
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Claude API", "Other API"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "verifications": [
                {
                    "claim_path": "C001",
                    "status": "conflicting",
                    "evidence_ids": ["E1"],
                    "reason": "The page also contains the output amount.",
                    "suggested_action": "Check the price direction.",
                },
                {
                    "claim_path": "C002",
                    "status": "conflicting",
                    "evidence_ids": ["E1"],
                    "reason": "The page also contains the input amount.",
                    "suggested_action": "Check the price direction.",
                },
                {
                    "claim_path": "C003",
                    "status": "supported",
                    "evidence_ids": ["E1"],
                    "reason": "This is a scope statement.",
                    "suggested_action": "No action required.",
                },
            ]
        }

        result = Verifier(FakeVerifierModel(model_output)).verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
                product_profiles=[profile],
                market_definition=MarketDefinition(
                    market_name="Model API",
                    product_category="Model API",
                    comparison_level="developer API",
                    pricing_scope="api",
                    core_dimensions=["api_pricing"],
                ),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(
            [item.status for item in result.claim_verifications[:2]],
            [
                ClaimVerificationStatus.SUPPORTED,
                ClaimVerificationStatus.SUPPORTED,
            ],
        )

    def test_supported_analysis_passes_semantic_review(self) -> None:
        # 引用结构正确且模型没有发现语义问题时，应通过验证。
        fixture = _load_json("verifier_outputs.json")
        model = FakeVerifierModel(fixture["supported"])
        verifier = Verifier(model)

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])
        self.assertFalse(result.retry_recommended)
        self.assertEqual(model.invocation_count, 1)

    def test_scoped_supported_analysis_returns_three_passed_statuses(
        self,
    ) -> None:
        # 引用、范围和比较口径均无问题时，三类状态应全部通过。
        sample_state = _load_sample_state()
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )
        verifier = Verifier(model)

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=sample_state.evidence,
                market_definition=sample_state.market_definition,
                product_profiles=sample_state.product_profiles,
            )
        )

        self.assertTrue(result.citations_valid)
        self.assertTrue(result.scope_consistent)
        self.assertTrue(result.comparison_usable)
        self.assertTrue(result.passed)

    def test_out_of_scope_reference_fails_scope_without_retry(self) -> None:
        # 已标记为 uncertain 的资料不能支撑最终分析，也不能靠 Analyst 重写修复。
        sample_state = _load_sample_state()
        evidence = list(sample_state.evidence)
        evidence[0] = evidence[0].model_copy(
            update={
                "scope_status": "uncertain",
                "scope_reason": "Product level could not be confirmed.",
            }
        )
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )

        result = Verifier(model).verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=evidence,
                market_definition=sample_state.market_definition,
                product_profiles=sample_state.product_profiles,
            )
        )

        self.assertFalse(result.scope_consistent)
        self.assertTrue(result.citations_valid)
        self.assertTrue(result.comparison_usable)
        self.assertFalse(result.retry_recommended)
        self.assertEqual(model.invocation_count, 0)

    def test_claim_using_excluded_product_line_fails_scope(self) -> None:
        # 即使引用有效，明确写入市场排除项也不能通过范围一致性检查。
        sample_state = _load_sample_state()
        analysis = _load_valid_analysis().model_copy(
            update={
                "conclusion": AnalysisClaim(
                    claim=(
                        "Consumer plans make Atlas Notes better for this "
                        "enterprise subscription comparison."
                    ),
                    claim_type="interpretation",
                    product_names=["Atlas Notes", "Beacon Docs"],
                    evidence_ids=["E1", "E3"],
                )
            }
        )
        market_definition = sample_state.market_definition.model_copy(
            update={"exclusions": ["consumer plans"]}
        )

        result = Verifier(
            FakeVerifierModel(
                _load_json("verifier_outputs.json")["supported"]
            )
        ).verify(
            VerifierInput(
                analysis=analysis,
                evidence=sample_state.evidence,
                market_definition=market_definition,
                product_profiles=sample_state.product_profiles,
            )
        )

        self.assertFalse(result.scope_consistent)
        self.assertEqual(result.issues[0].issue_type, "scope_level_conflict")
        self.assertTrue(result.retry_recommended)

    def test_incomplete_paid_price_fails_comparison_usability(self) -> None:
        # 数值价格缺少单位和周期时，引用正确也不能标为可比较。
        sample_state = _load_sample_state()
        profiles = list(sample_state.product_profiles)
        profiles[0] = profiles[0].model_copy(
            update={
                "pricing": [
                    PricingPlan(
                        plan_name="Team",
                        price="12 USD",
                        evidence_ids=["E2"],
                    )
                ]
            }
        )
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )

        result = Verifier(model).verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=sample_state.evidence,
                market_definition=sample_state.market_definition,
                product_profiles=profiles,
            )
        )

        self.assertFalse(result.comparison_usable)
        self.assertTrue(result.citations_valid)
        self.assertTrue(result.scope_consistent)
        self.assertFalse(result.retry_recommended)
        self.assertEqual(
            result.issues[0].issue_type,
            "incomplete_pricing_context",
        )

    def test_billing_cycle_in_unit_keeps_comparison_usable(self) -> None:
        # unit 已明确写出每月计费时，不应重复要求独立 billing_cycle 字段。
        sample_state = _load_sample_state()
        profiles = list(sample_state.product_profiles)
        profiles[0] = profiles[0].model_copy(
            update={
                "pricing": [
                    PricingPlan(
                        plan_name="Team",
                        price="10",
                        unit="per member / month",
                        evidence_ids=["E2"],
                    )
                ]
            }
        )
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )

        result = Verifier(model).verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=sample_state.evidence,
                market_definition=sample_state.market_definition,
                product_profiles=profiles,
            )
        )

        self.assertTrue(result.comparison_usable)
        self.assertEqual(model.invocation_count, 1)

    def test_different_price_units_cannot_be_ranked_directly(self) -> None:
        # 每用户月付和每工作区年付不能直接写成更便宜或更贵。
        sample_state = _load_sample_state()
        profiles = list(sample_state.product_profiles)
        atlas_plan = profiles[0].pricing[0].model_copy(
            update={"unit": "user"}
        )
        beacon_plan = PricingPlan(
            plan_name="Business",
            price="120 USD per workspace",
            unit="workspace",
            billing_cycle="yearly",
            evidence_ids=["E4"],
        )
        profiles[0] = profiles[0].model_copy(
            update={"pricing": [atlas_plan]}
        )
        profiles[1] = profiles[1].model_copy(
            update={"pricing": [beacon_plan]}
        )
        analysis = _load_valid_analysis().model_copy(
            update={
                "pricing": [
                    AnalysisClaim(
                        claim="Atlas Notes has a lower price than Beacon Docs.",
                        claim_type="interpretation",
                        product_names=["Atlas Notes", "Beacon Docs"],
                        evidence_ids=["E2", "E4"],
                    )
                ]
            }
        )

        result = Verifier(
            FakeVerifierModel(
                _load_json("verifier_outputs.json")["supported"]
            )
        ).verify(
            VerifierInput(
                analysis=analysis,
                evidence=sample_state.evidence,
                market_definition=sample_state.market_definition,
                product_profiles=profiles,
            )
        )

        self.assertFalse(result.comparison_usable)
        self.assertEqual(result.issues[0].issue_type, "incomparable_pricing")
        self.assertTrue(result.retry_recommended)

    def test_same_price_units_can_continue_to_semantic_review(self) -> None:
        # 两个产品都有每用户月付口径时，不应被可比性规则误拦截。
        sample_state = _load_sample_state()
        profiles = list(sample_state.product_profiles)
        profiles[0] = profiles[0].model_copy(
            update={
                "pricing": [
                    profiles[0].pricing[0].model_copy(
                        update={"unit": "user"}
                    )
                ]
            }
        )
        profiles[1] = profiles[1].model_copy(
            update={
                "pricing": [
                    PricingPlan(
                        plan_name="Business",
                        price="15 USD per user",
                        unit="per user",
                        billing_cycle="monthly",
                        evidence_ids=["E4"],
                    )
                ]
            }
        )
        analysis = _load_valid_analysis().model_copy(
            update={
                "pricing": [
                    AnalysisClaim(
                        claim="Atlas Notes has a lower price than Beacon Docs.",
                        claim_type="interpretation",
                        product_names=["Atlas Notes", "Beacon Docs"],
                        evidence_ids=["E2", "E4"],
                    )
                ]
            }
        )
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )

        result = Verifier(model).verify(
            VerifierInput(
                analysis=analysis,
                evidence=sample_state.evidence,
                market_definition=sample_state.market_definition,
                product_profiles=profiles,
            )
        )

        self.assertTrue(result.comparison_usable)
        self.assertEqual(model.invocation_count, 1)

    def test_missing_custom_dimension_is_explicit_and_not_retried(self) -> None:
        # 缺少 governance 资料属于研究输入不足，不能让 Analyst 凭空补写。
        sample_state = _load_sample_state()
        market_definition = sample_state.market_definition.model_copy(
            update={
                "core_dimensions": ["features", "pricing", "governance"]
            }
        )

        result = Verifier(
            FakeVerifierModel(
                _load_json("verifier_outputs.json")["supported"]
            )
        ).verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=sample_state.evidence,
                market_definition=market_definition,
                product_profiles=sample_state.product_profiles,
            )
        )

        self.assertFalse(result.comparison_usable)
        self.assertFalse(result.retry_recommended)
        self.assertEqual(result.issues[0].issue_type, "missing_core_dimension")
        self.assertIn("资料不足", result.issues[0].suggested_action)

    def test_invalid_evidence_id_fails_without_model_call(self) -> None:
        # E99 是确定性错误，Verifier 应直接返回并跳过模型调用。
        fixture = _load_json("analyst_outputs.json")
        invalid_analysis = AnalystOutput.model_validate(
            fixture["unknown_reference"]
        ).analysis
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )
        verifier = Verifier(model)

        result = verifier.verify(
            VerifierInput(
                analysis=invalid_analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.retry_recommended)
        self.assertEqual(result.issues[0].issue_type, "invalid_evidence_id")
        self.assertEqual(result.issues[0].claim_path, "features[0]")
        self.assertEqual(model.invocation_count, 0)

    def test_wrong_product_evidence_fails_without_model_call(self) -> None:
        # E1 虽然存在，但属于 Atlas Notes，不能支持 Beacon Docs claim。
        fixture = _load_json("analyst_outputs.json")
        invalid_analysis = AnalystOutput.model_validate(
            fixture["cross_product_reference"]
        ).analysis
        model = FakeVerifierModel(
            _load_json("verifier_outputs.json")["supported"]
        )
        verifier = Verifier(model)

        result = verifier.verify(
            VerifierInput(
                analysis=invalid_analysis,
                evidence=_load_sample_evidence(),
            )
        )

        issue_types = {issue.issue_type for issue in result.issues}
        self.assertIn("wrong_product_evidence", issue_types)
        self.assertIn("missing_product_evidence", issue_types)
        self.assertEqual(model.invocation_count, 0)

    def test_semantic_unsupported_claim_returns_actionable_issue(self) -> None:
        # 引用存在但文字不受支持时，应保留模型返回的定位和修复建议。
        analysis = CompetitiveAnalysis(
            products=["Atlas Notes", "Beacon Docs"],
            features=[
                AnalysisClaim(
                    claim="Atlas Notes includes video conferencing.",
                    claim_type="fact",
                    product_names=["Atlas Notes"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The supplied products have different features.",
                claim_type="interpretation",
                product_names=["Atlas Notes", "Beacon Docs"],
                evidence_ids=["E1", "E3"],
            ),
        )
        fixture = _load_json("verifier_outputs.json")
        verifier = Verifier(FakeVerifierModel(fixture["unsupported"]))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].issue_type, "unsupported_claim")
        self.assertEqual(result.issues[0].claim_path, "features[0]")
        self.assertTrue(result.issues[0].suggested_action)

    def test_opaque_model_claim_id_maps_back_to_internal_path(self) -> None:
        # 模型只返回短 ID，Python 必须把它还原成 Reporter/Analyst 使用的路径。
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "C002",
                    "message": "The feature claim is unsupported.",
                    "evidence_ids": [],
                    "suggested_action": "Remove the feature claim.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].claim_path, "features[0]")

    def test_conflicting_claim_recommends_retry(self) -> None:
        # claim 与定价证据相反时，应报告冲突并建议重新分析。
        analysis = CompetitiveAnalysis(
            products=["Atlas Notes", "Beacon Docs"],
            pricing=[
                AnalysisClaim(
                    claim="Atlas Notes Team plan is free.",
                    claim_type="fact",
                    product_names=["Atlas Notes"],
                    evidence_ids=["E2"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The supplied pricing differs.",
                claim_type="interpretation",
                product_names=["Atlas Notes", "Beacon Docs"],
                evidence_ids=["E2", "E4"],
            ),
        )
        fixture = _load_json("verifier_outputs.json")
        verifier = Verifier(FakeVerifierModel(fixture["conflicting"]))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.retry_recommended)
        self.assertEqual(
            result.issues[0].issue_type,
            "conflicting_evidence",
        )

    def test_unknown_model_claim_path_is_rejected(self) -> None:
        # 评审模型不能虚构 claim 路径，否则 Verifier 自身结果也不可信。
        fixture = _load_json("verifier_outputs.json")
        verifier = Verifier(
            FakeVerifierModel(fixture["unknown_claim_path"])
        )

        with self.assertRaises(VerifierError):
            verifier.verify(
                VerifierInput(
                    analysis=_load_valid_analysis(),
                    evidence=_load_sample_evidence(),
                )
            )

    def test_unknown_claim_path_is_repaired_once_with_valid_paths(self) -> None:
        # 真实模型可能把 pricing[4] 抄成 pricing[5]；第二轮必须看到合法清单。
        fixture = _load_json("verifier_outputs.json")
        model = SequencedVerifierModel(
            [
                fixture["unknown_claim_path"],
                fixture["supported"],
            ]
        )
        verifier = Verifier(model)

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(model.invocation_count, 2)
        initial_message = model.received_messages[0][-1]["content"]
        self.assertIn('"claim_path": "C001"', initial_message)
        self.assertNotIn('"claim_path": "features[0]"', initial_message)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn("合法 claim_path", repair_message)
        self.assertIn("C001", repair_message)
        self.assertNotIn("features[0]", repair_message)
        self.assertNotIn("features[99]", repair_message)
        self.assertIn("不要把输出格式错误报告为语义 issue", repair_message)

    def test_second_unknown_claim_path_is_still_rejected(self) -> None:
        # 修复输出仍虚构路径时要明确失败，不能猜测映射或丢弃 issue。
        fixture = _load_json("verifier_outputs.json")
        model = SequencedVerifierModel(
            [
                fixture["unknown_claim_path"],
                fixture["unknown_claim_path"],
            ]
        )
        verifier = Verifier(model)

        with self.assertRaises(VerifierError):
            verifier.verify(
                VerifierInput(
                    analysis=_load_valid_analysis(),
                    evidence=_load_sample_evidence(),
                )
            )

        self.assertEqual(model.invocation_count, 2)

    def test_unknown_model_evidence_id_is_rejected(self) -> None:
        # 模型 issue 中的 Evidence ID 也必须来自当前输入。
        fixture = _load_json("verifier_outputs.json")
        verifier = Verifier(
            FakeVerifierModel(fixture["unknown_evidence"])
        )

        with self.assertRaises(VerifierError):
            verifier.verify(
                VerifierInput(
                    analysis=_load_valid_analysis(),
                    evidence=_load_sample_evidence(),
                )
            )

    def test_full_verification_result_shape_is_normalized(self) -> None:
        # 真实模型有时会返回完整验证结果，Verifier 只取其中 issues。
        model_output = {
            "passed": True,
            "issues": [],
            "retry_recommended": False,
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)

    def test_nested_verification_result_shape_is_normalized(self) -> None:
        # 有些模型会把结构包在 verification_result 里，也可以安全展开。
        model_output = {
            "verification_result": {
                "passed": True,
                "issues": [],
                "retry_recommended": False,
            }
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)

    def test_malformed_model_output_exposes_safe_detail(self) -> None:
        # 结构错误要给出定位信息，但不能把原始模型内容完整展示给页面。
        model_output = {"unexpected": "secret-token-must-not-be-shown"}
        verifier = Verifier(FakeVerifierModel(model_output))

        with self.assertRaises(VerifierError) as captured_error:
            verifier.verify(
                VerifierInput(
                    analysis=_load_valid_analysis(),
                    evidence=_load_sample_evidence(),
                )
            )

        public_detail = captured_error.exception.public_detail
        self.assertIn("Verifier 模型输出结构不符合要求", public_detail)
        self.assertIn("issues", public_detail)
        self.assertNotIn("secret-token-must-not-be-shown", public_detail)

    def test_unsupported_claim_can_report_no_direct_evidence(self) -> None:
        # 完全没有直接支持来源时，空列表比虚构 Evidence ID 更准确。
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "conclusion",
                    "message": "The conclusion adds an unsupported claim.",
                    "evidence_ids": [],
                    "suggested_action": "Remove the unsupported detail.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].evidence_ids, [])

    def test_conservative_conclusion_scope_issue_is_ignored(self) -> None:
        # 保守范围说明不是产品事实，模型误报时不应阻断报告。
        analysis = _load_valid_analysis().model_copy(
            update={
                "conclusion": AnalysisClaim(
                    claim=(
                        "The comparison is limited to the supplied "
                        "evidence for Atlas Notes and Beacon Docs."
                    ),
                    claim_type="interpretation",
                    product_names=["Atlas Notes", "Beacon Docs"],
                    evidence_ids=["E1", "E2", "E3", "E4"],
                )
            }
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "conclusion",
                    "message": "The evidence does not state this scope note.",
                    "evidence_ids": [],
                    "suggested_action": "Remove the conclusion.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_product_profile_scope_conclusion_issue_is_ignored(self) -> None:
        # fallback 结论描述的是系统输入范围，不需要 Evidence 逐字支持。
        analysis = _load_valid_analysis().model_copy(
            update={
                "conclusion": AnalysisClaim(
                    claim=(
                        "The comparison is limited to the supplied product "
                        "profiles for Atlas Notes and Beacon Docs."
                    ),
                    claim_type="interpretation",
                    product_names=["Atlas Notes", "Beacon Docs"],
                    evidence_ids=["E1", "E2", "E3", "E4"],
                )
            }
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "conclusion",
                    "message": "The evidence does not state this scope note.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the conclusion.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_positioning_interpretation_wording_issue_is_ignored(self) -> None:
        # 个人项目中，定位分析是解释型章节，不应因措辞不是证据原句而阻断报告。
        analysis = _load_valid_analysis()
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "positioning[0]",
                    "message": "The positioning wording is not exact.",
                    "evidence_ids": [],
                    "suggested_action": "Remove the positioning analysis.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_strong_positioning_evaluation_is_not_ignored(self) -> None:
        # 强胜负判断仍然属于需要人工修正的 unsupported claim。
        analysis = _load_valid_analysis().model_copy(
            update={
                "positioning": [
                    AnalysisClaim(
                        claim=(
                            "Atlas Notes is clearly stronger than "
                            "Beacon Docs."
                        ),
                        claim_type="interpretation",
                        product_names=["Atlas Notes", "Beacon Docs"],
                        evidence_ids=["E1", "E3"],
                    )
                ]
            }
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "positioning[0]",
                    "message": "The evidence does not support stronger.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the strength comparison.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].claim_path, "positioning[0]")

    def test_fallback_summary_conclusion_wording_issue_is_ignored(
        self,
    ) -> None:
        # fallback 总结由多条事实拼成，Verifier 不应要求它逐字出现在 Evidence 中。
        analysis = _load_valid_analysis().model_copy(
            update={
                "conclusion": AnalysisClaim(
                    claim=(
                        "Based on the supplied profiles, Atlas Notes mentions "
                        "Reusable templates and lists the Team plan at "
                        "12 USD per user; Beacon Docs mentions Collaborative "
                        "pages and names a Business plan without a public "
                        "price in the supplied profile."
                    ),
                    claim_type="interpretation",
                    product_names=["Atlas Notes", "Beacon Docs"],
                    evidence_ids=["E1", "E2", "E3", "E4"],
                )
            }
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "conclusion",
                    "message": "The summary wording is not exact.",
                    "evidence_ids": ["E3"],
                    "suggested_action": "Make the conclusion shorter.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_mentions_feature_issue_is_ignored_when_phrase_exists(self) -> None:
        # 标准化的 “mentions X” 句式只要证据出现 X，就不应被逐字洁癖误杀。
        analysis = CompetitiveAnalysis(
            products=["Atlas Notes", "Beacon Docs"],
            features=[
                AnalysisClaim(
                    claim="Atlas Notes mentions Reusable templates.",
                    claim_type="fact",
                    product_names=["Atlas Notes"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Atlas Notes", "Beacon Docs"],
                evidence_ids=["E1", "E3"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[0]",
                    "message": (
                        "The evidence mentions reusable templates but does "
                        "not explicitly state this exact sentence."
                    ),
                    "evidence_ids": [],
                    "suggested_action": "Remove the claim.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_feature_issue_is_ignored_for_inflection_variants(self) -> None:
        # workflow automation 可由 automated processes and workflows 支持。
        analysis = CompetitiveAnalysis(
            products=["Confluence", "Notion"],
            features=[
                AnalysisClaim(
                    claim="Confluence mentions Workflow automation.",
                    claim_type="fact",
                    product_names=["Confluence"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Confluence", "Notion"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[0]",
                    "message": "The evidence uses automated workflows.",
                    "evidence_ids": [],
                    "suggested_action": "Rephrase the claim.",
                }
            ]
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Confluence",
                topic="features",
                title="Confluence Pricing",
                url="https://example.com/confluence/pricing",
                snippet=(
                    "Increase efficiency by replacing manual, "
                    "time-consuming work with automated processes and "
                    "workflows."
                ),
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_feature_issue_is_ignored_for_high_token_coverage(self) -> None:
        # 长功能名允许少量泛化词没逐字出现，但核心词要被覆盖。
        analysis = CompetitiveAnalysis(
            products=["Confluence", "Notion"],
            features=[
                AnalysisClaim(
                    claim=(
                        "Confluence mentions AI-powered knowledge "
                        "management."
                    ),
                    claim_type="fact",
                    product_names=["Confluence"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Confluence", "Notion"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[0]",
                    "message": "The evidence uses nearby AI wording.",
                    "evidence_ids": [],
                    "suggested_action": "Rephrase the claim.",
                }
            ]
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Confluence",
                topic="features",
                title="Confluence AI",
                url="https://example.com/confluence/ai",
                snippet=(
                    "AI that knows your business. AI-powered apps driven "
                    "by your team's knowledge."
                ),
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_feature_issue_is_not_ignored_when_proper_noun_is_missing(
        self,
    ) -> None:
        # Rovo 是专有词；证据只说 AI features 时不能被高覆盖率规则放过。
        analysis = CompetitiveAnalysis(
            products=["Confluence", "Notion"],
            features=[
                AnalysisClaim(
                    claim="Confluence mentions Rovo AI features.",
                    claim_type="fact",
                    product_names=["Confluence"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Confluence", "Notion"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "features[0]",
                    "message": "The evidence mentions AI features generally.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the Rovo-specific wording.",
                }
            ]
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Confluence",
                topic="features",
                title="Confluence AI",
                url="https://example.com/confluence/ai",
                snippet="Confluence includes AI features for teams.",
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].claim_path, "features[0]")

    def test_pricing_issue_is_ignored_when_plan_and_price_exist(self) -> None:
        # 价格 claim 不要求 Evidence 逐字出现 “lists the plan at” 的句式。
        analysis = CompetitiveAnalysis(
            products=["Atlas Notes", "Beacon Docs"],
            pricing=[
                AnalysisClaim(
                    claim=(
                        "Atlas Notes lists the Team plan at 12 USD per user "
                        "with monthly billing."
                    ),
                    claim_type="fact",
                    product_names=["Atlas Notes"],
                    evidence_ids=["E2"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Atlas Notes", "Beacon Docs"],
                evidence_ids=["E2", "E4"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "pricing[0]",
                    "message": (
                        "The evidence lists the plan and amount but not this "
                        "exact phrasing."
                    ),
                    "evidence_ids": ["E2"],
                    "suggested_action": "Remove the pricing claim.",
                }
            ]
        }
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=_load_sample_evidence(),
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_semantic_api_pricing_issue_is_preserved(self) -> None:
        # API 价格语义由读取检索上下文的 Verifier 决定，代码不再覆盖模型 issue。
        collected_at = _load_sample_evidence()[0].collected_at
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="pricing",
                title="OpenAI API pricing",
                url="https://example.com/openai/pricing",
                snippet=(
                    "GPT-5.6-sol input costs $5.00 per million input tokens."
                ),
                source_type="official",
                collected_at=collected_at,
            ),
            Evidence(
                evidence_id="E2",
                product_name="Gemini API",
                topic="pricing",
                title="Gemini API pricing",
                url="https://example.com/gemini/pricing",
                snippet=(
                    "Gemini 2.0 Flash input costs $0.15 per million input tokens."
                ),
                source_type="official",
                collected_at=collected_at,
            ),
        ]
        openai_fact = (
            "GPT-5.6-sol input | $5.00 per million input tokens | "
            "per million input tokens"
        )
        profiles = [
            ProductProfile(
                product_name="OpenAI API",
                dimension_findings=[
                    {
                        "dimension": "api_pricing",
                        "facts": [openai_fact],
                        "evidence_ids": ["E1"],
                    }
                ],
                pricing=[
                    PricingPlan(
                        plan_name="GPT-5.6-sol input",
                        price="$5.00 per million input tokens",
                        unit="per million input tokens",
                        evidence_ids=["E1"],
                    )
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
                    PricingPlan(
                        plan_name="Gemini 2.0 Flash input",
                        price="$0.15 per million input tokens",
                        unit="per million input tokens",
                        evidence_ids=["E2"],
                    )
                ],
            ),
        ]
        market_definition = MarketDefinition(
            market_name="大模型 API",
            product_category="模型 API",
            target_buyer="开发者",
            comparison_level="开发者 API",
            pricing_scope="api",
            core_dimensions=["api_pricing"],
        )
        correct_claim = AnalysisClaim(
            claim=f"OpenAI API — api_pricing: {openai_fact}",
            claim_type="fact",
            product_names=["OpenAI API"],
            evidence_ids=["E1"],
        )

        for issue_type in ["unsupported_claim", "conflicting_evidence"]:
            with self.subTest(issue_type=issue_type):
                analysis = CompetitiveAnalysis(
                    products=["OpenAI API", "Gemini API"],
                    dimension_comparisons=[correct_claim],
                    conclusion=AnalysisClaim(
                        claim=(
                            "The comparison is limited to the supplied "
                            "product profiles for OpenAI API and Gemini API."
                        ),
                        claim_type="interpretation",
                        product_names=["OpenAI API", "Gemini API"],
                        evidence_ids=["E1", "E2"],
                    ),
                )
                model_output = {
                    "issues": [
                        {
                            "issue_type": issue_type,
                            "claim_path": "dimension_comparisons[0]",
                            "message": "The evidence includes other rates.",
                            "evidence_ids": ["E1"],
                            "suggested_action": "Remove the rate.",
                        }
                    ]
                }

                result = Verifier(FakeVerifierModel(model_output)).verify(
                    VerifierInput(
                        analysis=analysis,
                        evidence=evidence,
                        market_definition=market_definition,
                        product_profiles=profiles,
                    )
                )

                self.assertFalse(result.passed)
                self.assertEqual(len(result.issues), 1)
                self.assertEqual(result.issues[0].issue_type, issue_type)

    def test_wrong_api_dimension_amount_is_not_ignored(self) -> None:
        # claim 金额只要偏离 ProductProfile，就必须保留模型 issue。
        collected_at = _load_sample_evidence()[0].collected_at
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="OpenAI API",
                topic="pricing",
                title="OpenAI API pricing",
                url="https://example.com/openai/pricing",
                snippet=(
                    "GPT-5.6-sol input costs $5.00 per million input tokens."
                ),
                source_type="official",
                collected_at=collected_at,
            )
        ]
        fact = (
            "GPT-5.6-sol input | $5.00 per million input tokens | "
            "per million input tokens"
        )
        profile = ProductProfile(
            product_name="OpenAI API",
            dimension_findings=[
                {
                    "dimension": "api_pricing",
                    "facts": [fact],
                    "evidence_ids": ["E1"],
                }
            ],
            pricing=[
                PricingPlan(
                    plan_name="GPT-5.6-sol input",
                    price="$5.00 per million input tokens",
                    unit="per million input tokens",
                    evidence_ids=["E1"],
                )
            ],
        )
        analysis = CompetitiveAnalysis(
            products=["OpenAI API", "Gemini API"],
            dimension_comparisons=[
                AnalysisClaim(
                    claim=(
                        "OpenAI API — api_pricing: GPT-5.6-sol input | "
                        "$6.00 per million input tokens | "
                        "per million input tokens"
                    ),
                    claim_type="fact",
                    product_names=["OpenAI API"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["OpenAI API", "Gemini API"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "conflicting_evidence",
                    "claim_path": "dimension_comparisons[0]",
                    "message": "The amount conflicts with the evidence.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Use $5.00.",
                }
            ]
        }
        market_definition = MarketDefinition(
            market_name="大模型 API",
            product_category="模型 API",
            comparison_level="开发者 API",
            pricing_scope="api",
            core_dimensions=["api_pricing"],
        )

        result = Verifier(FakeVerifierModel(model_output)).verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
                market_definition=market_definition,
                product_profiles=[profile],
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].claim_path, "dimension_comparisons[0]")

    def test_free_pricing_issue_is_ignored_when_free_evidence_exists(
        self,
    ) -> None:
        # Evidence 里的 “Free forever” 足以支持标准化的 Free 价格 claim。
        analysis = CompetitiveAnalysis(
            products=["Confluence", "Notion"],
            pricing=[
                AnalysisClaim(
                    claim="Confluence lists the Free plan at Free.",
                    claim_type="fact",
                    product_names=["Confluence"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Confluence", "Notion"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "pricing[0]",
                    "message": "The evidence does not use this exact phrasing.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the pricing claim.",
                }
            ]
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Confluence",
                topic="pricing",
                title="Confluence Pricing",
                url="https://example.com/confluence/pricing",
                snippet="Free forever for 10 users.",
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
            )
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])

    def test_invalid_billing_issue_is_not_ignored_for_missing_price(
        self,
    ) -> None:
        # 套餐名存在不代表 “Beta billing” 也被证据支持。
        analysis = CompetitiveAnalysis(
            products=["Notion", "Confluence"],
            pricing=[
                AnalysisClaim(
                    claim=(
                        "Notion names a Workers plan without a public price "
                        "in the supplied profile with Beta billing."
                    ),
                    claim_type="fact",
                    product_names=["Notion"],
                    evidence_ids=["E1"],
                )
            ],
            conclusion=AnalysisClaim(
                claim="The comparison is limited to the supplied evidence.",
                claim_type="interpretation",
                product_names=["Notion", "Confluence"],
                evidence_ids=["E1"],
            ),
        )
        model_output = {
            "issues": [
                {
                    "issue_type": "unsupported_claim",
                    "claim_path": "pricing[0]",
                    "message": "Beta is a status, not supported as billing.",
                    "evidence_ids": ["E1"],
                    "suggested_action": "Remove the billing phrase.",
                }
            ]
        }
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Notion",
                topic="pricing",
                title="Notion Pricing",
                url="https://example.com/notion/pricing",
                snippet="Workers is listed as Beta.",
                source_type="official",
                collected_at=_load_sample_evidence()[0].collected_at,
            )
        ]
        verifier = Verifier(FakeVerifierModel(model_output))

        result = verifier.verify(
            VerifierInput(
                analysis=analysis,
                evidence=evidence,
            )
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].claim_path, "pricing[0]")

    def test_langchain_wrapper_binds_verifier_schema(self) -> None:
        # 真实模型边界必须绑定 VerifierModelOutput 和 JSON mode。
        fixture = _load_json("verifier_outputs.json")
        structured_model = FakeVerifierModel(fixture["supported"])
        chat_model = FakeChatModel(structured_model)
        verifier_model = LangChainVerifierModel(chat_model)
        verifier = Verifier(verifier_model)

        result = verifier.verify(
            VerifierInput(
                analysis=_load_valid_analysis(),
                evidence=_load_sample_evidence(),
            )
        )

        self.assertIs(chat_model.received_schema, VerifierModelOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
