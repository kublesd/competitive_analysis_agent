import json
import unittest
from pathlib import Path

from competitive_analysis_agent.analyst import (
    AnalysisClaim,
    AnalystOutput,
    CompetitiveAnalysis,
)
from competitive_analysis_agent.schemas import Evidence, WorkflowState
from competitive_analysis_agent.verifier import (
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


class VerifierTest(unittest.TestCase):
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
