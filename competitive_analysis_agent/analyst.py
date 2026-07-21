"""Analyst 节点：比较产品画像，并区分事实与分析判断。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from competitive_analysis_agent.model_io import (
    log_model_error,
    log_model_request,
    log_model_response,
)
from competitive_analysis_agent.pricing_utils import should_include_billing_cycle
from competitive_analysis_agent.schemas import (
    ContractModel,
    EvidenceId,
    FeatureItem,
    MarketDefinition,
    ModelProfile,
    PriceRate,
    ProductProfile,
    PricingPlan,
    RequiredText,
    SupportStatus,
)


ANALYST_SYSTEM_PROMPT = """
你是竞品分析流程中的 Analyst。

你的唯一职责是比较用户提供的 ProductProfile。不得搜索网页，不得使用外部知识，
不得补充 ProductProfile 中不存在的产品事实。

要求：
1. products 必须按输入顺序逐字复制所有产品名，不得遗漏或新增产品。
2. 每条 claim 必须标记 claim_type：
   - fact：ProductProfile 直接提供的功能、价格或其他可引用事实；
   - interpretation：基于一个或多个事实形成的比较、机会点或结论。
3. fact 必须引用 evidence_ids，且涉及的每个产品都必须有属于自己的证据支持。
4. 除 conclusion 外，interpretation 可以没有 evidence_ids；如果填写，也只能使用
   相关产品已有的 ID。conclusion 的引用要求见第 7 条。
5. ProductProfile 的 positioning、target_users、strengths、limitations 当前没有独立
   Evidence ID，不能直接写成 fact，但可以用于形成轻量 interpretation。
6. opportunities 中的所有 claim 必须是 interpretation。只要输入画像里能看到
   定位、功能、目标用户、价格透明度或公开信息缺口的差异，就写 1 到 3 条
   实用机会点；不要因为证据简短就全部留空。机会点可以使用“could / may /
   opportunity / follow-up”这类谨慎措辞，但不得编造市场规模、胜率或用户偏好。
   如果机会点涉及具体产品，必须带上来自对应 ProductProfile 的 evidence_ids；
   没有证据支撑时宁可少写。
7. conclusion 必须是 interpretation，并且 product_names 必须包含全部产品。
   结论应限制为输入中可直接观察的共同点、差异或信息不足。
   只要输入画像中存在 Evidence，结论就必须引用用于形成结论的 evidence_ids。
   优先直接总结各产品已观察到的事实，不要使用“更好”“独特”等证据未定义的评价词。
8. 写 fact 时尽量贴近 ProductProfile 中 feature.name、feature.description、
   models[].pricing 或旧版 pricing.plan_name、price 和 main_limits 的原文。
   模型费率必须写成“模型名 + input/output/cached/audio/Batch 方向 + 完整费率”，
   不能写成两个笼统的 plan 数字。不得把短标签扩写成更大的能力，
   不得使用“has a section for / 提供某某完整能力”这类证据未直接支持的说法。
   如果 pricing.price 为 null，只能写“未提供公开价格”，绝不能写成 $0 或 Free。
   如果 pricing.price 是普通数字，也不得擅自补美元符号或其他货币单位。
9. features 章节只能描述 ProductProfile.features 中的功能。价格、套餐、计费周期、
   每用户/每席位费用、免费/标准/企业套餐、用户数限制和存储限制只能写入 pricing，
   不得放入 features。
10. 如果功能证据只是简短标签，应使用原功能名或“X mentions Y”这类最小事实；
    不要添加原文没有的形容词、效果承诺或营销扩写。
    单产品功能 claim 不要写 “includes / offers / provides features like ...”，
    除非 ProductProfile 原文已经直接表达这个完整关系。
11. 每个输入产品必须至少出现在一条比较 claim 的 product_names 中。
12. 信息不足时应明确限制，不得猜测未知价格、用户或市场数据；但普通个人项目
    更需要可读分析，不要把定位分析、机会点和结论全部写成空白或兜底句。
13. 收到 Verifier 修订反馈时，优先删除不受支持的 claim；只有画像中存在直接依据
    时才把它收窄重写，不得用另一个推测替换。
    Verifier 的 suggested_action 只是诊断参考，不是可以照抄的最终文案。
14. 如果 Verifier 指出多个 features 或 conclusion 不受支持，下一轮应显著减少
    features claim，只保留最直接、最短、证据措辞最接近的事实。
15. 如果 Verifier 指出 conclusion 不受支持，不要继续总结具体功能优劣；
    改为只总结画像中直接可见的定位、功能和价格差异。
16. 只比较 MarketDefinition.core_dimensions 中选择的维度，并遵守 product_category、
    comparison_level 和 exclusions。不得把不同产品层级或不同计费口径强行排序。
17. 每个产品都要在 product_assessments 中给出 strengths、shortcomings、data_gaps 和
    confidence。strengths/shortcomings 只能引用画像中的已记录功能、限制、价格或模型字段；
    未检索到资料只能写 data_gaps，绝不能写成“不支持”或“能力较弱”。
18. scenario_recommendations 必须覆盖：编程 Agent、RAG 与知识库问答、长文档分析、
    实时语音应用、多模态理解、低成本批处理、企业级 API 集成。每项都要写推荐产品、理由、
    成本与能力权衡、关键 Evidence ID、主要限制、confidence 与尚未验证的信息。没有足够
    直接证据时 recommended_product 写“暂无可验证推荐”，并把原因写入未验证信息；不得只按
    价格判定整体优劣。
19. market_opportunities 最多写三项共同未解决或体验不统一的问题，例如跨模型成本难预测、
    定价结构复杂、限流规则不透明、多模型路由困难、企业监控与预算治理困难。每项必须包含
    用户痛点、竞品现状、市场空白、可行产品方向、支撑 Evidence ID 和推断置信度。资料仅
    未覆盖某项时，明确说“尚未验证”，不得声称任何产品不支持。
20. recommendations 与事实和 interpretation 分开。每项建议必须包含目标场景、
    竞品取舍或资料空白、建议动作、相关产品、Evidence ID 和限制条件；目标场景应面向
    target_buyer。资料不足时明确写“资料不足”，不得猜测或强行给出产品排名。
21. conclusion 要回答适合谁、适合什么场景、依据和待验证风险，不要重复功能清单；同时填写
    conclusion_confidence。
22. 只输出 JSON 对象，不要添加 Markdown 或解释。
23. 顶层格式必须是：
{
  "analysis": {
    "products": ["..."],
    "positioning": [],
    "features": [],
    "pricing": [],
    "dimension_comparisons": [],
    "opportunities": [],
    "recommendations": [
      {
        "target_scenario": "...",
        "tradeoff_or_gap": "...",
        "recommended_action": "...",
        "product_names": ["..."],
        "evidence_ids": ["..."],
        "limitations": ["..."]
      }
    ],
    "product_assessments": [],
    "scenario_recommendations": [],
    "market_opportunities": [],
    "conclusion_confidence": "low",
    "conclusion": {
      "claim": "...",
      "claim_type": "interpretation",
      "product_names": ["..."],
      "evidence_ids": []
    }
  }
}
""".strip()


PRICING_CURRENCY_SYMBOLS = ("$", "¥", "￥", "€", "£")
PRICING_TEXT_PATTERNS = [
    re.compile(r"\b(?:price|priced|pricing|cost|billing|paid)\b"),
    re.compile(r"\b(?:usd|eur|gbp|rmb|cny)\b"),
    re.compile(r"\bper\s+(?:user|seat|month|year)\b"),
    re.compile(r"/\s*(?:user|seat|month|year)\b"),
    re.compile(r"\b(?:monthly|yearly|annual|annually)\b"),
    re.compile(
        r"\b(?:free|plus|standard|business|enterprise|premium|team|"
        r"starter|pro|basic)\s+(?:plan|tier)\b"
    ),
    re.compile(
        r"\b(?:plan|tier)\s+(?:free|plus|standard|business|enterprise|"
        r"premium|team|starter|pro|basic)\b"
    ),
]
CHINESE_PRICING_TERMS = (
    "价格",
    "定价",
    "计费",
    "费用",
    "每用户",
    "每席位",
    "每月",
    "每年",
    "月付",
    "年付",
    "年度",
    "免费套餐",
    "免费方案",
    "标准套餐",
    "标准方案",
    "企业套餐",
    "企业方案",
)
USER_LIMIT_PATTERN = re.compile(r"\b\d+[\s,]*(?:user|users|seat|seats)\b")
STORAGE_LIMIT_PATTERN = re.compile(r"\b\d+[\s,]*(?:gb|mb|tb)\b")
FEATURE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FEATURE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
}


class AnalysisClaim(ContractModel):
    """表示一条事实或解释，并保存涉及产品和证据引用。"""

    claim: RequiredText
    claim_type: Literal["fact", "interpretation"]
    product_names: list[RequiredText] = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_shape(self) -> "AnalysisClaim":
        """拒绝重复产品、重复引用和没有证据的事实。"""

        if len(self.product_names) != len(set(self.product_names)):
            raise ValueError("Claim product_names must be unique.")

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Claim evidence_ids must be unique.")

        if self.claim_type == "fact" and not self.evidence_ids:
            raise ValueError("Factual claims must reference evidence IDs.")

        return self


class ActionableRecommendation(ContractModel):
    """面向目标购买者、带证据和适用边界的建议。"""

    target_scenario: RequiredText
    tradeoff_or_gap: RequiredText
    recommended_action: RequiredText
    product_names: list[RequiredText] = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(min_length=1)
    limitations: list[RequiredText] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_references(self) -> "ActionableRecommendation":
        """拒绝重复产品、证据和限制，保持建议可追溯。"""

        if len(self.product_names) != len(set(self.product_names)):
            raise ValueError("Recommendation product_names must be unique.")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Recommendation evidence_ids must be unique.")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("Recommendation limitations must be unique.")
        return self


ConfidenceLevel = Literal["high", "medium", "low"]


class ProductAssessment(ContractModel):
    """一个产品的证据化优势、已知限制、数据缺口与置信度。"""

    product_name: RequiredText
    strengths: list[AnalysisClaim] = Field(default_factory=list)
    shortcomings: list[AnalysisClaim] = Field(default_factory=list)
    data_gaps: list[RequiredText] = Field(default_factory=list)
    confidence: ConfidenceLevel


class ScenarioRecommendation(ContractModel):
    """一个业务场景的谨慎选型结论；无足够证据时不虚构推荐。"""

    scenario: RequiredText
    recommended_product: RequiredText
    recommendation_reason: RequiredText
    cost_capability_tradeoff: RequiredText
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    primary_limitations: list[RequiredText] = Field(default_factory=list)
    confidence: ConfidenceLevel
    unverified_information: list[RequiredText] = Field(default_factory=list)


class MarketOpportunity(ContractModel):
    """由共同资料缺口或已记录差异推导出的市场机会。"""

    title: RequiredText
    user_pain: RequiredText
    competitor_status: RequiredText
    market_gap: RequiredText
    product_direction: RequiredText
    product_names: list[RequiredText] = Field(min_length=1)
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    inference_confidence: ConfidenceLevel


class CompetitiveAnalysis(ContractModel):
    """保存定位、功能、价格、机会点和结论的结构化比较。"""

    products: list[RequiredText] = Field(min_length=2)
    positioning: list[AnalysisClaim] = Field(default_factory=list)
    features: list[AnalysisClaim] = Field(default_factory=list)
    pricing: list[AnalysisClaim] = Field(default_factory=list)
    dimension_comparisons: list[AnalysisClaim] = Field(default_factory=list)
    opportunities: list[AnalysisClaim] = Field(default_factory=list)
    recommendations: list[ActionableRecommendation] = Field(
        default_factory=list
    )
    product_assessments: list[ProductAssessment] = Field(
        default_factory=list
    )
    scenario_recommendations: list[ScenarioRecommendation] = Field(
        default_factory=list
    )
    market_opportunities: list[MarketOpportunity] = Field(
        default_factory=list
    )
    conclusion_confidence: ConfidenceLevel = "low"
    conclusion: AnalysisClaim

    @model_validator(mode="after")
    def validate_analysis_types(self) -> "CompetitiveAnalysis":
        """确保产品唯一，并把机会点和结论限定为分析判断。"""

        if len(self.products) != len(set(self.products)):
            raise ValueError("Analysis products must be unique.")

        for opportunity in self.opportunities:
            if opportunity.claim_type != "interpretation":
                raise ValueError(
                    "Opportunity claims must be interpretations."
                )

        if self.conclusion.claim_type != "interpretation":
            raise ValueError("Conclusion must be an interpretation.")

        return self


class AnalystInput(ContractModel):
    """保存至少两个产品画像，并拒绝重复产品和跨产品重复 ID。"""

    profiles: list[ProductProfile] = Field(min_length=2)
    market_definition: MarketDefinition | None = None
    revision_feedback: list[RequiredText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile_identity(self) -> "AnalystInput":
        """确保产品名唯一，Evidence ID 不会同时归属于不同产品。"""

        product_names = [profile.product_name for profile in self.profiles]
        if len(product_names) != len(set(product_names)):
            raise ValueError("Profile product names must be unique.")

        evidence_owner: dict[str, str] = {}
        for profile in self.profiles:
            profile_evidence_ids = collect_profile_evidence_ids(profile)
            for evidence_id in profile_evidence_ids:
                existing_owner = evidence_owner.get(evidence_id)
                if (
                    existing_owner is not None
                    and existing_owner != profile.product_name
                ):
                    raise ValueError(
                        "An Evidence ID cannot belong to multiple products."
                    )
                evidence_owner[evidence_id] = profile.product_name

        return self


class AnalystOutput(ContractModel):
    """约束模型必须返回一个结构化竞品分析对象。"""

    analysis: CompetitiveAnalysis


class AnalystModel(Protocol):
    """约定 Analyst 所需的最小结构化模型调用接口。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """根据产品画像返回可被 AnalystOutput 校验的对象。"""


class StructuredChatModel(Protocol):
    """描述 LangChain ChatModel 的结构化输出能力。"""

    def with_structured_output(
        self,
        schema: type[AnalystOutput],
        *,
        method: Literal["json_mode"],
        include_raw: Literal[True],
    ) -> AnalystModel:
        """绑定 AnalystOutput，并返回可调用的结构化模型。"""


class LangChainAnalystModel:
    """把 LangChain ChatModel 包装成 Analyst 所需的模型接口。"""

    def __init__(self, chat_model: StructuredChatModel) -> None:
        # 当前模型供应商支持 json_object，因此沿用项目统一的 JSON mode。
        self._structured_model = chat_model.with_structured_output(
            AnalystOutput,
            method="json_mode",
            include_raw=True,
        )

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """执行模型调用，并在解析失败时保留原始文本供修复。"""

        call_id = log_model_request("Analyst", messages)
        try:
            structured_response = self._structured_model.invoke(messages)
        except Exception as error:
            log_model_error("Analyst", call_id, error)
            raise
        log_model_response("Analyst", call_id, structured_response)

        if not isinstance(structured_response, dict):
            return structured_response

        response_wrapper_keys = {"raw", "parsed", "parsing_error"}
        if not response_wrapper_keys.intersection(structured_response):
            return structured_response

        parsed_output = structured_response.get("parsed")
        if parsed_output is not None:
            return parsed_output

        raw_message = structured_response.get("raw")
        raw_content = getattr(raw_message, "content", raw_message)
        return raw_content


class FakeAnalystModel:
    """按顺序返回固定响应，用于无网络的确定性测试。"""

    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.invocation_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回下一条固定响应，并记录调用次数和消息。"""

        copied_messages = [message.copy() for message in messages]
        self.received_messages.append(copied_messages)

        if self.invocation_count >= len(self._responses):
            raise RuntimeError("Fake analyst has no response left.")

        response = self._responses[self.invocation_count]
        self.invocation_count += 1
        return response


class AnalystError(RuntimeError):
    """表示 Analyst 调用失败或一次修复后仍无法生成有效分析。"""

    def __init__(
        self,
        message: str,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        # public_detail 可显示到页面，因此只能保存脱敏后的定位信息。
        self.public_detail = public_detail or message


class AnalystValidationError(ValueError):
    """表示分析结构、产品覆盖或证据引用无效。"""


class Analyst:
    """生成、校验并在必要时修复一次结构化竞品分析。"""

    def __init__(self, model: AnalystModel) -> None:
        self._model = model

    def analyze(
        self,
        analyst_input: AnalystInput,
    ) -> CompetitiveAnalysis:
        """比较全部产品画像；首次校验失败时最多请求一次修复。"""

        initial_messages = build_analyst_messages(analyst_input)
        try:
            raw_output = self._invoke_model(initial_messages)
        except AnalystError:
            # Analyst 是横向组织者；模型服务不可用时，可以从已提取画像
            # 生成保守分析，避免整条工作流丢失已完成的研究结果。
            return build_fallback_analysis(analyst_input)

        try:
            validated_output = validate_analyst_output(
                raw_output=raw_output,
                analyst_input=analyst_input,
            )
            return validated_output.analysis
        except AnalystValidationError as first_error:
            # 有限修复可以处理漏产品和错引，同时避免模型无限调用。
            repair_messages = build_repair_messages(
                initial_messages=initial_messages,
                raw_output=raw_output,
                validation_error=str(first_error),
            )
            try:
                repaired_output = self._invoke_model(repair_messages)
            except AnalystError:
                return build_fallback_analysis(analyst_input)

        try:
            validated_repair = validate_analyst_output(
                raw_output=repaired_output,
                analyst_input=analyst_input,
            )
            return validated_repair.analysis
        except AnalystValidationError:
            # 修复后仍无效时，用确定性 fallback 保住已完成的研究和提取结果。
            return build_fallback_analysis(analyst_input)

    def _invoke_model(
        self,
        messages: list[dict[str, str]],
    ) -> object:
        """调用模型，并把供应商异常转换成统一的 AnalystError。"""

        try:
            return self._model.invoke(messages)
        except Exception as error:
            raise AnalystError(
                f"Analyst model call failed: {error}",
                public_detail=(
                    "Analyst 调用模型服务失败，已尝试使用产品画像生成保守分析。"
                    "如果仍然中断，通常是后续 Verifier 或模型服务继续不可用。"
                ),
            ) from error


def build_analyst_messages(
    analyst_input: AnalystInput,
) -> list[dict[str, str]]:
    """把产品画像转换成模型可读取的 system 和 user 消息。"""

    profiles_json = json.dumps(
        [
            profile.model_dump(mode="json")
            for profile in analyst_input.profiles
        ],
        ensure_ascii=False,
        indent=2,
    )
    market_definition_json = json.dumps(
        analyst_input.market_definition.model_dump(mode="json")
        if analyst_input.market_definition is not None
        else None,
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        "请只根据以下 MarketDefinition 和 ProductProfile 生成结构化竞品比较。\n"
        "输入顺序就是 products 必须使用的顺序。\n\n"
        f"MarketDefinition:\n{market_definition_json}\n\n"
        f"ProductProfile:\n{profiles_json}"
    )
    if analyst_input.revision_feedback:
        feedback_json = json.dumps(
            analyst_input.revision_feedback,
            ensure_ascii=False,
            indent=2,
        )
        user_message += (
            "\n\n上一次分析未通过 Verifier。请根据以下反馈修正，"
            "不要引入新的产品事实。Verifier 的 suggested_action 只用于定位问题，"
            "不要照抄其中的改写句。features 被点名时，只保留 "
            "`产品 mentions 功能名.` 这类最小事实，或删除该 claim；"
            "不要写 includes/offers/provides features like。结论应退回 "
            "`Based on the supplied profiles` 开头的保守摘要：\n"
            f"{feedback_json}"
        )

    return [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_repair_messages(
    initial_messages: list[dict[str, str]],
    raw_output: object,
    validation_error: str,
) -> list[dict[str, str]]:
    """把覆盖或引用错误反馈给模型，要求只修复结构化分析。"""

    repair_messages = [message.copy() for message in initial_messages]
    repair_instruction = (
        "上一次竞品分析没有通过校验，请只修复 JSON 输出，"
        "不要搜索或添加 ProductProfile 中不存在的事实。\n"
        "不要照抄 Verifier suggested_action 中的改写句；"
        "features 只能用 `产品 mentions 功能名.` 这类最小事实，"
        "或直接删除不受支持的 claim。\n"
        "如果输入画像中存在 Evidence，conclusion 必须引用用于形成结论的 "
        "evidence_ids。\n"
        f"校验错误：{validation_error}\n"
        f"上一次输出：{raw_output!r}"
    )
    repair_messages.append(
        {"role": "user", "content": repair_instruction}
    )
    return repair_messages


def build_fallback_analysis(
    analyst_input: AnalystInput,
) -> CompetitiveAnalysis:
    """模型不可用时，根据 ProductProfile 生成轻量、可追溯的实用分析。"""

    product_names = [
        profile.product_name for profile in analyst_input.profiles
    ]
    positioning_claims = build_fallback_positioning_claims(
        analyst_input.profiles
    )
    feature_claims = build_fallback_feature_claims(
        profiles=analyst_input.profiles,
        revision_feedback=analyst_input.revision_feedback,
    ) if market_dimension_is_selected(
        analyst_input.market_definition, "features"
    ) else []
    pricing_claims = build_fallback_pricing_claims(
        analyst_input.profiles
    ) if market_dimension_is_selected(
        analyst_input.market_definition, "pricing"
    ) else []
    dimension_claims = build_fallback_dimension_claims(analyst_input)
    opportunity_claims = build_fallback_opportunity_claims(
        analyst_input.profiles,
        market_definition=analyst_input.market_definition,
    )
    recommendations = build_fallback_recommendations(
        analyst_input=analyst_input,
        opportunity_claims=opportunity_claims,
    )
    product_assessments = build_product_assessments(analyst_input.profiles)
    scenario_recommendations = build_scenario_recommendations(
        analyst_input.profiles
    )
    market_opportunities = build_market_opportunities(analyst_input.profiles)
    conclusion = build_fallback_conclusion(
        analyst_input.profiles,
        market_definition=analyst_input.market_definition,
    )

    return CompetitiveAnalysis(
        products=product_names,
        positioning=positioning_claims,
        features=feature_claims,
        pricing=pricing_claims,
        dimension_comparisons=dimension_claims,
        opportunities=opportunity_claims,
        recommendations=recommendations,
        product_assessments=product_assessments,
        scenario_recommendations=scenario_recommendations,
        market_opportunities=market_opportunities,
        conclusion_confidence=confidence_from_evidence(
            collect_profile_evidence_ids_in_order(analyst_input.profiles)
        ),
        conclusion=conclusion,
    )


def market_dimension_is_selected(
    market_definition: MarketDefinition | None,
    section: Literal["positioning", "features", "pricing"],
) -> bool:
    """用少量通用别名判断固定章节是否属于用户选择维度。"""

    if market_definition is None:
        return True
    aliases = {
        "positioning": {"positioning", "定位"},
        "features": {
            "feature",
            "features",
            "function",
            "functions",
            "功能",
            "model_capabilities",
        },
        "pricing": {
            "price",
            "pricing",
            "cost",
            "价格",
            "定价",
            "成本",
            "api_pricing",
        },
    }
    selected = {item.casefold() for item in market_definition.core_dimensions}
    return bool(selected.intersection(aliases[section]))


def build_fallback_dimension_claims(
    analyst_input: AnalystInput,
) -> list[AnalysisClaim]:
    """按用户维度顺序把 DimensionFinding 转成带引用的事实。"""

    market = analyst_input.market_definition
    if market is None:
        return []
    selected_dimensions = {
        dimension.casefold() for dimension in market.core_dimensions
    }
    claims: list[AnalysisClaim] = []
    for profile in analyst_input.profiles:
        for finding in profile.dimension_findings:
            if finding.dimension.casefold() not in selected_dimensions:
                continue
            for fact in finding.facts:
                claims.append(
                    AnalysisClaim(
                        claim=(
                            f"{profile.product_name} — "
                            f"{finding.dimension}: {fact}"
                        ),
                        claim_type="fact",
                        product_names=[profile.product_name],
                        evidence_ids=list(finding.evidence_ids),
                    )
                )
    return claims


SCENARIO_LABELS = (
    "编程 Agent",
    "RAG 与知识库问答",
    "长文档分析",
    "实时语音应用",
    "多模态理解",
    "低成本批处理",
    "企业级 API 集成",
)


def build_product_assessments(
    profiles: Sequence[ProductProfile],
) -> list[ProductAssessment]:
    """由统一字段生成逐产品结论，不把未采集字段误判为短板。"""

    assessments: list[ProductAssessment] = []
    for profile in profiles:
        strengths = [
            AnalysisClaim(
                claim=f"{profile.product_name} mentions {feature.name}.",
                claim_type="fact",
                product_names=[profile.product_name],
                evidence_ids=list(feature.evidence_ids),
            )
            for feature in profile.features[:3]
        ]
        shortcomings = [
            AnalysisClaim(
                claim=(
                    f"{profile.product_name} documents the {plan.plan_name} "
                    f"constraint: {limit}."
                ),
                claim_type="fact",
                product_names=[profile.product_name],
                evidence_ids=list(plan.evidence_ids),
            )
            for plan in profile.pricing
            for limit in plan.main_limits[:1]
        ]
        data_gaps = profile_data_gaps(profile)
        evidence_ids = collect_profile_evidence_ids(profile)
        assessments.append(
            ProductAssessment(
                product_name=profile.product_name,
                strengths=strengths,
                shortcomings=shortcomings,
                data_gaps=data_gaps,
                confidence=confidence_from_evidence(evidence_ids),
            )
        )
    return assessments


def profile_data_gaps(profile: ProductProfile) -> list[str]:
    """缺字段只标为待验证资料，不推断产品不支持。"""

    gaps: list[str] = []
    if not profile.models:
        gaps.append("未收集模型能力、模态和上下文窗口资料。")
    if not any(model.rate_limits for model in profile.models) and not profile.rate_limits:
        gaps.append("未收集可验证的限流规则资料。")
    if not profile.enterprise_capabilities:
        gaps.append("未收集企业监控、预算或治理能力资料。")
    if profile.models and not any(
        model.pricing.input_price and model.pricing.output_price
        for model in profile.models
    ):
        gaps.append("未收集可计算的输入与输出 Token 价格组合。")
    return gaps


def confidence_from_evidence(evidence_ids: Sequence[str] | set[str]) -> ConfidenceLevel:
    """以可追溯证据数量给出保守置信度，而非能力强弱评分。"""

    count = len(evidence_ids)
    if count >= 3:
        return "high"
    if count:
        return "medium"
    return "low"


def build_scenario_recommendations(
    profiles: Sequence[ProductProfile],
) -> list[ScenarioRecommendation]:
    """只在统一字段直接支持场景时推荐产品；并列或缺资料时明确待验证。"""

    recommendations: list[ScenarioRecommendation] = []
    for scenario in SCENARIO_LABELS:
        candidates = [
            (profile, scenario_evidence_ids(profile, scenario))
            for profile in profiles
        ]
        candidates = [item for item in candidates if item[1]]
        best_count = max((len(ids) for _, ids in candidates), default=0)
        best = [item for item in candidates if len(item[1]) == best_count]
        if len(best) != 1:
            recommendations.append(
                ScenarioRecommendation(
                    scenario=scenario,
                    recommended_product="暂无可验证推荐",
                    recommendation_reason="当前统一字段不足以形成唯一、可验证的推荐。",
                    cost_capability_tradeoff="未形成成本或能力排名，避免仅按价格判断整体优劣。",
                    confidence="low",
                    unverified_information=[
                        "需补充该场景所需能力、限制与实际工作负载的同范围证据。"
                    ],
                )
            )
            continue
        profile, evidence_ids = best[0]
        recommendations.append(
            ScenarioRecommendation(
                scenario=scenario,
                recommended_product=profile.product_name,
                recommendation_reason=(
                    f"{profile.product_name} 的统一模型字段直接记录了该场景相关能力。"
                ),
                cost_capability_tradeoff=build_cost_capability_tradeoff(profile),
                evidence_ids=evidence_ids,
                primary_limitations=profile_data_gaps(profile),
                confidence="high" if len(evidence_ids) >= 2 else "medium",
                unverified_information=[
                    "尚未验证真实业务负载、延迟和质量表现；不据此作整体产品优劣判断。"
                ],
            )
        )
    return recommendations


def scenario_evidence_ids(
    profile: ProductProfile,
    scenario: str,
) -> list[str]:
    """返回场景所需的明确字段来源；空列表代表资料不足而非不支持。"""

    evidence_ids: list[str] = []
    for model in profile.models:
        capabilities = " ".join(model.model_capabilities or []).casefold()
        modalities = {item.value for item in model.supported_modalities or []}
        supported = {
            "编程 Agent": (
                model.tool_calling == SupportStatus.SUPPORTED
                or model.structured_output == SupportStatus.SUPPORTED
                or "code" in capabilities
            ),
            "RAG 与知识库问答": (
                model.prompt_caching == SupportStatus.SUPPORTED
                or (model.context_window_tokens or 0) >= 32_000
            ),
            "长文档分析": (model.context_window_tokens or 0) >= 128_000,
            "实时语音应用": (
                model.realtime == SupportStatus.SUPPORTED and "audio" in modalities
            ),
            "多模态理解": len(modalities) >= 2,
            "低成本批处理": model.batch_api == SupportStatus.SUPPORTED,
            "企业级 API 集成": bool(profile.enterprise_capabilities),
        }[scenario]
        if supported:
            evidence_ids.extend(item.evidence_id for item in model.source_evidence)
            evidence_ids.extend(model.pricing.evidence_ids())
    if scenario == "企业级 API 集成" and profile.enterprise_capabilities:
        evidence_ids.extend(item.evidence_id for item in profile.source_evidence)
        evidence_ids.extend(item.evidence_id for item in profile.field_evidence)
    return deduplicate_preserving_order(evidence_ids)


def build_cost_capability_tradeoff(profile: ProductProfile) -> str:
    """把成本放在已验证能力的语境中，禁止把价格当成整体排名。"""

    has_token_price = any(
        model.pricing.input_price and model.pricing.output_price
        for model in profile.models
    )
    if has_token_price:
        return "已收集可计算 Token 成本；仍需结合已验证能力、质量和限流评估。"
    return "尚无完整 Token 成本组合；不以单一价格或缺失价格判断产品优劣。"


def build_market_opportunities(
    profiles: Sequence[ProductProfile],
) -> list[MarketOpportunity]:
    """将共同待验证字段转成低置信度产品方向，不声称产品不支持。"""

    product_names = [profile.product_name for profile in profiles]
    model_evidence = deduplicate_preserving_order(
        evidence_id
        for profile in profiles
        for model in profile.models
        for evidence_id in [item.evidence_id for item in model.source_evidence]
    )
    pricing_evidence = deduplicate_preserving_order(
        evidence_id
        for profile in profiles
        for plan in profile.pricing
        for evidence_id in plan.evidence_ids
    )
    all_evidence = deduplicate_preserving_order(
        [*model_evidence, *pricing_evidence]
    )
    opportunity_specs = [
        (
            "跨模型成本可预测性",
            "用户难以在调用前预估跨模型与不同计费方式的总成本。",
            "当前资料记录了部分模型或套餐价格，但工作负载下的成本与能力组合仍需验证。",
            "提供按场景、缓存和 Batch 条件计算并解释成本差异的统一视图。",
            "构建成本预算、路由模拟与价格变更提醒。",
            [*model_evidence, *pricing_evidence],
        ),
        (
            "限流与多模型路由透明度",
            "应用团队难以预判高峰时的吞吐、回退和路由行为。",
            "当前统一字段对限流和路由的覆盖不完整；这表示资料待验证，不表示产品不支持。",
            "将限流、上下文和模型能力归一为可比较的运行约束。",
            "提供运行前容量检查、路由规则说明与回退演练。",
            model_evidence or all_evidence,
        ),
        (
            "企业预算与治理可见性",
            "企业采购与平台团队需要把调用成本、权限和监控放入同一治理流程。",
            "企业能力及预算治理资料在当前画像中仍有待验证部分。",
            "把未验证项显式呈现为采购核验清单，而不是推断能力缺失。",
            "提供预算告警、审计轨迹和供应商能力核验工作流。",
            all_evidence,
        ),
    ]
    return [
        MarketOpportunity(
            title=title,
            user_pain=pain,
            competitor_status=status,
            market_gap=gap,
            product_direction=direction,
            product_names=product_names,
            evidence_ids=deduplicate_preserving_order(evidence_ids),
            inference_confidence=("medium" if evidence_ids else "low"),
        )
        for title, pain, status, gap, direction, evidence_ids in opportunity_specs
    ]


def build_fallback_recommendations(
    analyst_input: AnalystInput,
    opportunity_claims: Sequence[AnalysisClaim] | None = None,
) -> list[ActionableRecommendation]:
    """把有引用的机会点转成目标购买者可执行的保守建议。"""

    opportunities = list(
        opportunity_claims
        if opportunity_claims is not None
        else build_fallback_opportunity_claims(analyst_input.profiles)
    )
    if not opportunities:
        dimension_claims = build_fallback_dimension_claims(analyst_input)
        if dimension_claims:
            opportunities = [
                AnalysisClaim(
                    claim=(
                        "当前同范围资料展示了所选维度的事实差异；"
                        "未被证据覆盖的维度为资料不足。"
                    ),
                    claim_type="interpretation",
                    product_names=deduplicate_preserving_order(
                        product_name
                        for claim in dimension_claims
                        for product_name in claim.product_names
                    ),
                    evidence_ids=deduplicate_preserving_order(
                        evidence_id
                        for claim in dimension_claims
                        for evidence_id in claim.evidence_ids
                    ),
                )
            ]
    market = analyst_input.market_definition
    target_scenario = (
        market.target_buyer
        if market is not None and market.target_buyer is not None
        else market.comparison_level
        if market is not None
        else "本次选型的目标购买者"
    )
    dimension = (
        market.core_dimensions[0]
        if market is not None
        else "已记录差异"
    )
    recommendations: list[ActionableRecommendation] = []

    for opportunity in opportunities:
        if not opportunity.evidence_ids:
            continue
        recommendations.append(
            ActionableRecommendation(
                target_scenario=target_scenario,
                tradeoff_or_gap=opportunity.claim,
                recommended_action=(
                    f"围绕“{dimension}”开展同范围试用或采购核验，"
                    "再根据实际场景做选择。"
                ),
                product_names=list(opportunity.product_names),
                evidence_ids=list(opportunity.evidence_ids),
                limitations=[
                    "资料仅限当前已采集的同范围证据；未覆盖项为资料不足，"
                    "不据此进行产品排名。"
                ],
            )
        )

    return recommendations[:3]


def build_fallback_positioning_claims(
    profiles: Sequence[ProductProfile],
) -> list[AnalysisClaim]:
    """用带 Evidence ID 的画像字段生成一条简短定位分析。"""

    product_names = [profile.product_name for profile in profiles]
    summary_parts: list[str] = []
    evidence_ids: list[str] = []

    for profile in profiles:
        profile_focus = describe_evidence_backed_profile_focus(profile)
        if profile_focus is None:
            continue

        focus_text, focus_evidence_ids = profile_focus
        summary_parts.append(f"{profile.product_name}: {focus_text}")
        evidence_ids.extend(focus_evidence_ids)

    if not summary_parts:
        return []

    claim_text = (
        "In the supplied profiles, "
        + "; ".join(summary_parts)
        + "."
    )
    return [
        AnalysisClaim(
            claim=claim_text,
            claim_type="interpretation",
            product_names=product_names,
            evidence_ids=deduplicate_preserving_order(evidence_ids),
        )
    ]


def describe_evidence_backed_profile_focus(
    profile: ProductProfile,
) -> tuple[str, list[str]] | None:
    """把单个产品的功能或价格证据压缩成定位分析片段。"""

    if profile.features:
        selected_features = profile.features[:2]
        evidence_ids: list[str] = []
        for feature in selected_features:
            evidence_ids.extend(feature.evidence_ids)

        return (
            format_feature_mentions(selected_features),
            deduplicate_preserving_order(evidence_ids),
        )

    if profile.pricing:
        selected_pricing = profile.pricing[:2]
        evidence_ids = []
        for pricing_plan in selected_pricing:
            evidence_ids.extend(pricing_plan.evidence_ids)

        return (
            format_pricing_mentions(selected_pricing),
            deduplicate_preserving_order(evidence_ids),
        )

    return None


def build_fallback_feature_claims(
    profiles: Sequence[ProductProfile],
    revision_feedback: Sequence[str] = (),
) -> list[AnalysisClaim]:
    """把画像中的功能项转换成最短事实 claim。"""

    claims: list[AnalysisClaim] = []
    seen_claim_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    unsupported_feedback_texts = collect_unsupported_feature_feedback_texts(
        revision_feedback
    )

    for profile in profiles:
        for feature in profile.features:
            # Verifier 已点名不受支持的功能不再写回 fallback，避免重试后
            # 报告继续带着同一批无法验证的 claim。
            if is_feature_rejected_by_feedback(
                product_name=profile.product_name,
                feature_name=feature.name,
                unsupported_feedback_texts=unsupported_feedback_texts,
            ):
                continue

            claim = AnalysisClaim(
                claim=f"{profile.product_name} mentions {feature.name}.",
                claim_type="fact",
                product_names=[profile.product_name],
                evidence_ids=list(feature.evidence_ids),
            )
            claim_key = (
                profile.product_name,
                claim.claim,
                tuple(claim.evidence_ids),
            )
            if claim_key in seen_claim_keys:
                continue
            seen_claim_keys.add(claim_key)
            claims.append(claim)

    return claims


def build_fallback_pricing_claims(
    profiles: Sequence[ProductProfile],
) -> list[AnalysisClaim]:
    """把画像中的模型费率或旧版套餐转换成完整语义 claim。"""

    claims: list[AnalysisClaim] = []
    for profile in profiles:
        if profile.models:
            for model in profile.models:
                claims.extend(
                    build_model_pricing_claims(
                        product_name=profile.product_name,
                        model=model,
                    )
                )
            continue

        for pricing_plan in profile.pricing:
            claim_text = format_fallback_pricing_claim(
                product_name=profile.product_name,
                plan_name=pricing_plan.plan_name,
                price=pricing_plan.price,
                billing_cycle=pricing_plan.billing_cycle,
                main_limits=pricing_plan.main_limits,
                unit=pricing_plan.unit,
            )
            claims.append(
                AnalysisClaim(
                    claim=claim_text,
                    claim_type="fact",
                    product_names=[profile.product_name],
                    evidence_ids=list(pricing_plan.evidence_ids),
                )
            )

    return claims


def build_model_pricing_claims(
    product_name: str,
    model: ModelProfile,
) -> list[AnalysisClaim]:
    """从一个模型对象生成带价格方向的事实，绝不把费率拆成套餐。"""

    claims: list[AnalysisClaim] = []
    price_fields = [
        ("input", model.pricing.input_price),
        ("cached input", model.pricing.cached_input_price),
        ("output", model.pricing.output_price),
        (
            "audio input",
            model.pricing.audio_input_price or model.pricing.audio_price,
        ),
        ("audio output", model.pricing.audio_output_price),
        ("Batch input", model.pricing.batch_input_price),
        ("Batch cached input", model.pricing.batch_cached_input_price),
        ("Batch output", model.pricing.batch_output_price),
    ]
    for price_direction, price_value in price_fields:
        rates = price_value if isinstance(price_value, list) else [price_value]
        for rate in rates:
            if rate is None:
                continue
            claims.append(
                AnalysisClaim(
                    claim=format_model_price_claim(
                        model_name=model.model_name,
                        price_direction=price_direction,
                        rate=rate,
                    ),
                    claim_type="fact",
                    product_names=[product_name],
                    evidence_ids=list(rate.evidence_ids),
                )
            )

    # ModelProfile 不保存厂商名；调用方在这里统一回填所属产品，避免复制价格逻辑。
    return claims


def format_model_price_claim(
    model_name: str,
    price_direction: str,
    rate: PriceRate,
) -> str:
    """把原子费率格式化为模型、方向、金额、单位和有效期完整的 claim。"""

    amount_text = format(rate.amount, "f")
    if "." in amount_text:
        amount_text = amount_text.rstrip("0").rstrip(".")
    price_text = (
        f"${amount_text}"
        if rate.currency.value == "USD"
        else f"{amount_text} {rate.currency.value}"
    )
    quantity_text = format_price_quantity(rate.per_quantity)
    unit_text = format_price_unit(rate.unit.value, rate.per_quantity)
    claim = (
        f"{model_name} {price_direction} price is {price_text} "
        f"per {quantity_text}{unit_text}"
    )
    if rate.effective_from is not None:
        claim += f", effective from {rate.effective_from.isoformat()}"
    if rate.effective_to is not None:
        claim += f" through {rate.effective_to.isoformat()}"
    if rate.condition:
        claim += f" ({rate.condition})"
    return claim + "."


def format_price_quantity(per_quantity: int) -> str:
    """用常见自然语言表达价格计量基数。"""

    if per_quantity == 1_000_000:
        return "million "
    if per_quantity == 1_000:
        return "thousand "
    if per_quantity == 1:
        return ""
    return f"{per_quantity:,} "


def format_price_unit(unit: str, per_quantity: int) -> str:
    """把 Schema 单位转换为适合 claim 的可读形式。"""

    labels = {
        "token": "token",
        "request": "request",
        "image": "image",
        "audio_minute": "audio minute",
        "character": "character",
        "second": "second",
        "other": "unit",
    }
    label = labels[unit]
    return label if per_quantity == 1 else f"{label}s"


def format_fallback_pricing_claim(
    product_name: str,
    plan_name: str,
    price: str | None,
    billing_cycle: str | None,
    main_limits: list[str],
    unit: str | None = None,
) -> str:
    """格式化一个不会猜测未知价格的 pricing fallback claim。"""

    price_identity = split_legacy_model_price_identity(plan_name)
    if price is not None and price_identity is not None:
        model_name, price_direction = price_identity
        claim = f"{model_name} {price_direction} price is {price}"
        if (
            unit
            and " per " not in price.casefold()
            and "/" not in price
        ):
            claim += f" per {unit}"
    elif price is None:
        claim = (
            f"{product_name} names {choose_indefinite_article(plan_name)} "
            f"{plan_name} plan without a "
            "public price in the supplied profile"
        )
    else:
        claim = f"{product_name} lists the {plan_name} plan at {price}"

    if should_include_billing_cycle(price, billing_cycle):
        claim += f" with {billing_cycle} billing"

    # 降级分析只保留最稳定的价格事实；限制明细仍在产品概览中展示，
    # 避免 Verifier 因长限制文本的细节差异阻断报告。
    _ = main_limits

    return claim + "."


def split_legacy_model_price_identity(
    plan_name: str,
) -> tuple[str, str] | None:
    """兼容旧 PricingPlan，把末尾价格方向还原为模型费率语义。"""

    normalized_name = plan_name.casefold()
    directions = (
        "batch cached input",
        "batch input",
        "batch output",
        "cached input",
        "audio",
        "input",
        "output",
    )
    for direction in directions:
        suffix = f" {direction}"
        if not normalized_name.endswith(suffix):
            continue
        model_name = plan_name[: -len(suffix)].strip()
        if model_name:
            return model_name, direction
    return None


def format_feature_mentions(features: Sequence[FeatureItem]) -> str:
    """把功能列表写成接近证据事实的 mentions 短句。"""

    feature_names = [feature.name for feature in features]
    return "mentions " + join_human_readable(feature_names)


def format_pricing_mentions(pricing_plans: Sequence[PricingPlan]) -> str:
    """把价格列表写成接近价格事实的 lists/names 短句。"""

    pricing_parts: list[str] = []
    for pricing_plan in pricing_plans:
        if pricing_plan.price is None:
            pricing_parts.append(
                f"names {choose_indefinite_article(pricing_plan.plan_name)} "
                f"{pricing_plan.plan_name} plan without a public price"
            )
        else:
            pricing_parts.append(
                f"lists the {pricing_plan.plan_name} plan at "
                f"{pricing_plan.price}"
            )

    return join_human_readable(pricing_parts)


def build_fallback_opportunity_claims(
    profiles: Sequence[ProductProfile],
    market_definition: MarketDefinition | None = None,
) -> list[AnalysisClaim]:
    """从画像差异中生成个人项目可用的轻量机会点。"""

    claims: list[AnalysisClaim] = []

    pricing_opportunity = (
        build_pricing_clarity_opportunity(profiles)
        if market_dimension_is_selected(market_definition, "pricing")
        else None
    )
    if pricing_opportunity is not None:
        claims.append(pricing_opportunity)

    feature_opportunity = (
        build_feature_contrast_opportunity(profiles)
        if market_dimension_is_selected(market_definition, "features")
        else None
    )
    if feature_opportunity is not None:
        claims.append(feature_opportunity)

    return claims[:3]


def build_pricing_clarity_opportunity(
    profiles: Sequence[ProductProfile],
) -> AnalysisClaim | None:
    """当公开价格不完整时，生成价格透明度机会点。"""

    priced_products: list[str] = []
    missing_price_products: list[str] = []
    evidence_ids: list[str] = []

    for profile in profiles:
        has_public_price = any(
            pricing_plan.price is not None
            for pricing_plan in profile.pricing
        )
        has_missing_price = any(
            pricing_plan.price is None
            for pricing_plan in profile.pricing
        )

        if has_public_price:
            priced_products.append(profile.product_name)
        if has_missing_price:
            missing_price_products.append(profile.product_name)

        for pricing_plan in profile.pricing:
            evidence_ids.extend(pricing_plan.evidence_ids)

    if not priced_products or not missing_price_products:
        return None

    priced_text = join_human_readable(priced_products)
    missing_text = join_human_readable(missing_price_products)
    priced_verb = "includes" if len(priced_products) == 1 else "include"
    missing_verb = "has" if len(missing_price_products) == 1 else "have"
    claim_text = (
        "A practical opportunity is pricing clarity: "
        f"{priced_text} {priced_verb} public prices in the supplied profiles, "
        f"while {missing_text} still {missing_verb} at least one plan without a "
        "public price."
    )

    return AnalysisClaim(
        claim=claim_text,
        claim_type="interpretation",
        product_names=collect_product_names_with_pricing(profiles),
        evidence_ids=deduplicate_preserving_order(evidence_ids),
    )


def build_feature_contrast_opportunity(
    profiles: Sequence[ProductProfile],
) -> AnalysisClaim | None:
    """把各产品首批功能差异转成可读的后续分析机会。"""

    feature_parts: list[str] = []
    evidence_ids: list[str] = []
    product_names: list[str] = []

    for profile in profiles:
        if not profile.features:
            continue

        feature_names = [
            feature.name for feature in profile.features[:2]
        ]
        features_text = join_human_readable(feature_names)
        feature_parts.append(
            f"{profile.product_name} highlights {features_text}"
        )
        product_names.append(profile.product_name)

        for feature in profile.features[:2]:
            evidence_ids.extend(feature.evidence_ids)

    if len(feature_parts) < 2:
        return None

    claim_text = (
        "Another opportunity is to turn feature differences into clearer "
        "buyer scenarios: "
        + "; ".join(feature_parts)
        + "."
    )
    return AnalysisClaim(
        claim=claim_text,
        claim_type="interpretation",
        product_names=product_names,
        evidence_ids=deduplicate_preserving_order(evidence_ids),
    )


def build_fallback_conclusion(
    profiles: Sequence[ProductProfile],
    market_definition: MarketDefinition | None = None,
) -> AnalysisClaim:
    """生成比空泛兜底句更有信息量、但仍不夸大的结论。"""

    product_names = [profile.product_name for profile in profiles]
    product_summaries: list[str] = []

    for profile in profiles:
        summary = summarize_profile_for_conclusion(profile)
        if summary is None:
            continue

        product_summaries.append(summary)

    if product_summaries:
        claim_text = (
            "Based on the supplied profiles, "
            + "; ".join(product_summaries)
            + "."
        )
    else:
        claim_text = (
            "The supplied profiles provide limited comparable details for "
            f"{' and '.join(product_names)}."
        )

    if market_definition is not None:
        buyer = market_definition.target_buyer or "目标购买者"
        claim_text += (
            f" For {buyer} evaluating {market_definition.comparison_level}, "
            "these cited differences are the current decision basis; pending "
            "risks and uncovered dimensions remain 资料不足 and require "
            "buyer-specific validation."
        )

    return AnalysisClaim(
        claim=claim_text,
        claim_type="interpretation",
        product_names=product_names,
        evidence_ids=collect_profile_evidence_ids_in_order(profiles),
    )


def summarize_profile_for_conclusion(
    profile: ProductProfile,
) -> str | None:
    """把单个产品的可见重点压缩进最终结论。"""

    summary_parts: list[str] = []

    if profile.features:
        summary_parts.append(format_feature_mentions(profile.features[:2]))

    priced_pricing_plans = [
        pricing_plan
        for pricing_plan in profile.pricing
        if pricing_plan.price is not None
    ]
    missing_price_plan_names = [
        pricing_plan.plan_name
        for pricing_plan in profile.pricing
        if pricing_plan.price is None
    ]

    if priced_pricing_plans:
        summary_parts.append(
            format_pricing_mentions(priced_pricing_plans[:2])
        )

    if missing_price_plan_names:
        summary_parts.append(
            format_missing_price_summary(missing_price_plan_names[:2])
        )

    if not summary_parts:
        return None

    return f"{profile.product_name} " + join_human_readable(
        summary_parts
    )


def format_missing_price_summary(plan_names: Sequence[str]) -> str:
    """把缺失公开价格的套餐写成更贴近证据的保守表述。"""

    if len(plan_names) == 1:
        plan_name = plan_names[0]
        return (
            f"names {choose_indefinite_article(plan_name)} {plan_name} plan "
            "without a public price in the supplied profile"
        )

    plans = join_human_readable(list(plan_names))
    return (
        f"names {plans} plans without public prices in the supplied profile"
    )


def clean_claim_phrase(text: str) -> str:
    """清理短语结尾标点，避免拼接出来的 claim 有双句号。"""

    cleaned_text = " ".join(text.split())
    return cleaned_text.rstrip(".。;；:：")


def join_human_readable(items: Sequence[str]) -> str:
    """用英文报告常见格式连接短列表，保持 fallback 文本可读。"""

    cleaned_items = [clean_claim_phrase(item) for item in items if item]

    if not cleaned_items:
        return ""
    if len(cleaned_items) == 1:
        return cleaned_items[0]
    if len(cleaned_items) == 2:
        return f"{cleaned_items[0]} and {cleaned_items[1]}"

    leading_items = ", ".join(cleaned_items[:-1])
    return f"{leading_items}, and {cleaned_items[-1]}"


def choose_indefinite_article(text: str) -> str:
    """根据英文计划名首字母选择 a/an，修正 fallback 文案。"""

    cleaned_text = text.strip()
    if not cleaned_text:
        return "a"

    first_letter = cleaned_text[0].lower()
    if first_letter in {"a", "e", "i", "o", "u"}:
        return "an"

    return "a"


def deduplicate_preserving_order(items: Sequence[str]) -> list[str]:
    """按出现顺序去重，避免一个 claim 重复引用同一 Evidence ID。"""

    unique_items: list[str] = []
    seen_items: set[str] = set()

    for item in items:
        if item in seen_items:
            continue

        seen_items.add(item)
        unique_items.append(item)

    return unique_items


def collect_product_names_with_pricing(
    profiles: Sequence[ProductProfile],
) -> list[str]:
    """收集存在价格画像的产品名，供价格机会点引用。"""

    product_names: list[str] = []
    for profile in profiles:
        if not profile.pricing:
            continue

        product_names.append(profile.product_name)

    return product_names


def contains_pricing_language(claim_text: str) -> bool:
    """判断一条 claim 是否包含价格、套餐或计费限制相关措辞。"""

    normalized_text = claim_text.lower()

    # 明确货币符号通常只会出现在价格 claim 中，适合直接拦截。
    for currency_symbol in PRICING_CURRENCY_SYMBOLS:
        if currency_symbol in normalized_text:
            return True

    for pricing_term in CHINESE_PRICING_TERMS:
        if pricing_term in claim_text:
            return True

    for pricing_pattern in PRICING_TEXT_PATTERNS:
        if pricing_pattern.search(normalized_text):
            return True

    has_user_limit = USER_LIMIT_PATTERN.search(normalized_text) is not None
    has_storage_limit = (
        STORAGE_LIMIT_PATTERN.search(normalized_text) is not None
    )
    if has_user_limit and has_storage_limit:
        return True

    return False


def collect_feature_pricing_claim_paths(
    analysis: CompetitiveAnalysis,
) -> list[str]:
    """收集误放在 features 章节中的价格类 claim 路径。"""

    pricing_claim_paths: list[str] = []

    # 只检查 features 章节，因为 pricing 章节本来就允许价格和套餐语言。
    for feature_index, feature_claim in enumerate(analysis.features):
        if contains_pricing_language(feature_claim.claim):
            pricing_claim_paths.append(f"features[{feature_index}]")

    return pricing_claim_paths


def normalize_analysis_output(
    analyst_output: AnalystOutput,
    analyst_input: AnalystInput,
) -> AnalystOutput:
    """把模型可安全收窄的分析文本规范化为更保守的版本。"""

    analysis = analyst_output.analysis
    normalized_analysis = normalize_feature_fact_claims(
        analysis=analysis,
        profiles=analyst_input.profiles,
    )
    normalized_analysis = remove_unsupported_feature_claims_after_feedback(
        analysis=normalized_analysis,
        revision_feedback=analyst_input.revision_feedback,
    )
    normalized_analysis = normalize_feature_claims_after_feedback(
        analysis=normalized_analysis,
        profiles=analyst_input.profiles,
        revision_feedback=analyst_input.revision_feedback,
    )
    normalized_analysis = normalize_pricing_fact_claims(
        analysis=normalized_analysis,
        profiles=analyst_input.profiles,
        market_definition=analyst_input.market_definition,
    )
    normalized_analysis = normalize_opportunities(
        analysis=normalized_analysis,
        profiles=analyst_input.profiles,
        revision_feedback=analyst_input.revision_feedback,
    )
    if not normalized_analysis.dimension_comparisons:
        normalized_analysis = normalized_analysis.model_copy(
            update={
                "dimension_comparisons": build_fallback_dimension_claims(
                    analyst_input
                )
            }
        )
    if not normalized_analysis.recommendations:
        normalized_analysis = normalized_analysis.model_copy(
            update={
                "recommendations": build_fallback_recommendations(
                    analyst_input=analyst_input,
                    opportunity_claims=normalized_analysis.opportunities,
                )
            }
        )
    # 新横向分析完全由统一画像字段派生，避免模型把未检索项写成能力结论。
    normalized_analysis = normalized_analysis.model_copy(
        update={
            "product_assessments": build_product_assessments(
                analyst_input.profiles
            ),
            "scenario_recommendations": build_scenario_recommendations(
                analyst_input.profiles
            ),
            "market_opportunities": build_market_opportunities(
                analyst_input.profiles
            ),
            "conclusion_confidence": confidence_from_evidence(
                collect_profile_evidence_ids_in_order(analyst_input.profiles)
            ),
        }
    )
    normalized_analysis = normalize_conclusion_after_feedback(
        analysis=normalized_analysis,
        profiles=analyst_input.profiles,
        market_definition=analyst_input.market_definition,
        revision_feedback=analyst_input.revision_feedback,
    )
    normalized_analysis = fill_missing_lightweight_analysis_sections(
        analysis=normalized_analysis,
        profiles=analyst_input.profiles,
        market_definition=analyst_input.market_definition,
    )

    if normalized_analysis is analysis:
        return analyst_output

    return AnalystOutput(analysis=normalized_analysis)


def normalize_feature_fact_claims(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
) -> CompetitiveAnalysis:
    """把单产品功能事实收窄成 ProductProfile 中的功能名。"""

    features_by_product = build_features_by_product(profiles)
    normalized_feature_claims: list[AnalysisClaim] = []
    changed = False
    seen_feature_keys: set[tuple[str, str, tuple[str, ...]]] = set()

    for feature_claim in analysis.features:
        normalized_claim = normalize_single_feature_fact_claim(
            claim=feature_claim,
            features_by_product=features_by_product,
        )
        if normalized_claim is None:
            changed = True
            continue

        if normalized_claim is not feature_claim:
            changed = True

        feature_key = (
            normalized_claim.product_names[0]
            if len(normalized_claim.product_names) == 1
            else "",
            normalized_claim.claim,
            tuple(normalized_claim.evidence_ids),
        )
        if feature_key in seen_feature_keys:
            changed = True
            continue

        seen_feature_keys.add(feature_key)
        normalized_feature_claims.append(normalized_claim)

    if not changed:
        return analysis

    return analysis.model_copy(update={"features": normalized_feature_claims})


def remove_unsupported_feature_claims_after_feedback(
    analysis: CompetitiveAnalysis,
    revision_feedback: Sequence[str],
) -> CompetitiveAnalysis:
    """删除 Verifier 已确认 unsupported 的功能 claim。"""

    unsupported_feedback_texts = collect_unsupported_feature_feedback_texts(
        revision_feedback
    )
    if not unsupported_feedback_texts:
        return analysis

    kept_feature_claims: list[AnalysisClaim] = []
    changed = False
    for feature_claim in analysis.features:
        if is_claim_rejected_by_feedback(
            claim=feature_claim,
            unsupported_feedback_texts=unsupported_feedback_texts,
        ):
            changed = True
            continue

        kept_feature_claims.append(feature_claim)

    if not changed:
        return analysis

    return analysis.model_copy(update={"features": kept_feature_claims})


def normalize_feature_claims_after_feedback(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
    revision_feedback: Sequence[str],
) -> CompetitiveAnalysis:
    """重试轮如果 features 被点名，退回画像里的最小功能事实。"""

    if not has_feature_revision_feedback(revision_feedback):
        return analysis

    feature_claims = build_fallback_feature_claims(
        profiles=profiles,
        revision_feedback=revision_feedback,
    )
    return analysis.model_copy(update={"features": feature_claims})


def has_feature_revision_feedback(
    revision_feedback: Sequence[str],
) -> bool:
    """识别 Verifier 反馈是否指向 features 章节。"""

    for feedback_item in revision_feedback:
        normalized_feedback = feedback_item.lower()
        if "features[" not in normalized_feedback:
            continue
        if "unsupported_claim" in normalized_feedback:
            return True
        if "partially_supported" in normalized_feedback:
            return True
        if "conflicting_evidence" in normalized_feedback:
            return True

    return False


def normalize_pricing_fact_claims(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
    market_definition: MarketDefinition | None = None,
) -> CompetitiveAnalysis:
    """把单产品价格事实收窄成 ProductProfile 中的套餐原文。"""

    pricing_plans_by_product = build_pricing_plans_by_product(profiles)
    include_model_pricing = market_dimension_is_selected(
        market_definition,
        "pricing",
    )
    model_pricing_product_names = {
        profile.product_name
        for profile in profiles
        if profile.models and include_model_pricing
    }
    normalized_pricing_claims = [
        claim
        for profile in profiles
        if profile.models and include_model_pricing
        for claim in build_fallback_pricing_claims([profile])
    ]
    changed = False
    seen_pricing_keys: set[tuple[str, str, tuple[str, ...]]] = {
        (
            claim.product_names[0],
            claim.claim,
            tuple(claim.evidence_ids),
        )
        for claim in normalized_pricing_claims
    }

    for pricing_claim in analysis.pricing:
        if (
            len(pricing_claim.product_names) == 1
            and pricing_claim.product_names[0] in model_pricing_product_names
        ):
            changed = True
            continue
        normalized_claim = normalize_single_pricing_fact_claim(
            claim=pricing_claim,
            pricing_plans_by_product=pricing_plans_by_product,
        )
        if normalized_claim is None:
            changed = True
            continue

        if normalized_claim is not pricing_claim:
            changed = True

        pricing_key = (
            normalized_claim.product_names[0]
            if len(normalized_claim.product_names) == 1
            else "",
            normalized_claim.claim,
            tuple(normalized_claim.evidence_ids),
        )
        if pricing_key in seen_pricing_keys:
            changed = True
            continue

        seen_pricing_keys.add(pricing_key)
        normalized_pricing_claims.append(normalized_claim)

    if model_pricing_product_names:
        changed = True

    if not changed:
        return analysis

    return analysis.model_copy(update={"pricing": normalized_pricing_claims})


def normalize_single_pricing_fact_claim(
    claim: AnalysisClaim,
    pricing_plans_by_product: dict[str, list[PricingPlan]],
) -> AnalysisClaim | None:
    """校正单产品价格 fact；无法映射到套餐时删除该 claim。"""

    if claim.claim_type != "fact":
        return claim

    if len(claim.product_names) != 1:
        return claim

    product_name = claim.product_names[0]
    product_pricing_plans = pricing_plans_by_product.get(product_name, [])
    matched_pricing_plan = choose_best_pricing_plan_for_claim(
        claim=claim,
        product_pricing_plans=product_pricing_plans,
    )
    if matched_pricing_plan is None:
        return None

    canonical_claim = format_fallback_pricing_claim(
        product_name=product_name,
        plan_name=matched_pricing_plan.plan_name,
        price=matched_pricing_plan.price,
        billing_cycle=matched_pricing_plan.billing_cycle,
        main_limits=matched_pricing_plan.main_limits,
        unit=matched_pricing_plan.unit,
    )
    return AnalysisClaim(
        claim=canonical_claim,
        claim_type="fact",
        product_names=[product_name],
        evidence_ids=list(matched_pricing_plan.evidence_ids),
    )


def build_pricing_plans_by_product(
    profiles: Sequence[ProductProfile],
) -> dict[str, list[PricingPlan]]:
    """建立产品到价格方案的映射，供价格事实收窄使用。"""

    pricing_plans_by_product: dict[str, list[PricingPlan]] = {}
    for profile in profiles:
        pricing_plans_by_product[profile.product_name] = list(
            profile.pricing
        )
    return pricing_plans_by_product


def choose_best_pricing_plan_for_claim(
    claim: AnalysisClaim,
    product_pricing_plans: Sequence[PricingPlan],
) -> PricingPlan | None:
    """根据 Evidence ID、套餐名和价格文本选择最可能被 claim 描述的套餐。"""

    referenced_evidence_ids = set(claim.evidence_ids)
    claim_tokens = tokenize_pricing_text(claim.claim)
    best_pricing_plan: PricingPlan | None = None
    best_score = 0
    best_plan_name_score = 0

    for pricing_plan in product_pricing_plans:
        pricing_plan_evidence_ids = set(pricing_plan.evidence_ids)
        if not referenced_evidence_ids.intersection(
            pricing_plan_evidence_ids
        ):
            continue

        score, plan_name_score = score_pricing_claim_match(
            claim_text=claim.claim,
            claim_tokens=claim_tokens,
            pricing_plan=pricing_plan,
        )
        if score > best_score:
            best_score = score
            best_plan_name_score = plan_name_score
            best_pricing_plan = pricing_plan

    if best_pricing_plan is None:
        return None

    if best_score <= 0:
        return None

    # 一个价格页常同时列出多个套餐。多套餐时要求 claim 至少命中套餐名，
    # 避免把“Business $0”之类幻觉错映射到同页其他套餐。
    if len(product_pricing_plans) > 1 and best_plan_name_score <= 0:
        return None

    return best_pricing_plan


def score_pricing_claim_match(
    claim_text: str,
    claim_tokens: set[str],
    pricing_plan: PricingPlan,
) -> tuple[int, int]:
    """计算价格 claim 与某个套餐的匹配分数。"""

    normalized_claim_text = claim_text.lower()
    normalized_plan_name = pricing_plan.plan_name.lower()
    plan_name_tokens = tokenize_pricing_text(pricing_plan.plan_name)
    price_tokens = tokenize_pricing_text(pricing_plan.price or "")

    plan_name_score = 0
    if normalized_plan_name in normalized_claim_text:
        plan_name_score += 10

    plan_name_overlap = claim_tokens.intersection(plan_name_tokens)
    plan_name_score += len(plan_name_overlap) * 4

    price_score = 0
    if pricing_plan.price and pricing_plan.price.lower() in normalized_claim_text:
        price_score += 4

    price_overlap = claim_tokens.intersection(price_tokens)
    price_score += len(price_overlap)

    score = plan_name_score + price_score
    return score, plan_name_score


def tokenize_pricing_text(text: str) -> set[str]:
    """把套餐名、价格和 claim 拆成适合宽松匹配的关键词。"""

    tokens: set[str] = set()
    pricing_stop_words = FEATURE_STOP_WORDS.union(
        {"at", "billing", "lists", "names", "plan", "plans", "priced", "tier"}
    )

    for match in FEATURE_TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0)
        if token in pricing_stop_words:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)

    return tokens


def normalize_opportunities(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
    revision_feedback: Sequence[str],
) -> CompetitiveAnalysis:
    """把无证据或已被 Verifier 点名的机会点退回保守 fallback。"""

    if not analysis.opportunities:
        return analysis

    if has_opportunity_revision_feedback(revision_feedback):
        opportunity_claims = build_fallback_opportunity_claims(profiles)
        return analysis.model_copy(update={"opportunities": opportunity_claims})

    if not opportunities_need_fallback(analysis, profiles):
        return analysis

    opportunity_claims = build_fallback_opportunity_claims(profiles)
    return analysis.model_copy(update={"opportunities": opportunity_claims})


def has_opportunity_revision_feedback(
    revision_feedback: Sequence[str],
) -> bool:
    """识别 Verifier 是否已经指出机会点不受支持或证据冲突。"""

    for feedback_item in revision_feedback:
        normalized_feedback = feedback_item.lower()
        if "opportunities[" not in normalized_feedback:
            continue
        if "unsupported_claim" in normalized_feedback:
            return True
        if "partially_supported" in normalized_feedback:
            return True
        if "conflicting_evidence" in normalized_feedback:
            return True

    return False


def opportunities_need_fallback(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
) -> bool:
    """判断模型机会点是否缺少最基本的画像证据引用。"""

    available_evidence_ids = set(collect_profile_evidence_ids_in_order(profiles))
    if not available_evidence_ids:
        return False

    for opportunity_claim in analysis.opportunities:
        if not opportunity_claim.product_names:
            continue
        if not opportunity_claim.evidence_ids:
            return True

    return False


def collect_unsupported_feature_feedback_texts(
    revision_feedback: Sequence[str],
) -> list[str]:
    """只收集 features 章节的 unsupported 反馈，避免误删价格或结论。"""

    feedback_texts: list[str] = []
    for feedback_item in revision_feedback:
        normalized_feedback = feedback_item.lower()
        if not any(
            issue_type in normalized_feedback
            for issue_type in (
                "unsupported_claim",
                "partially_supported",
            )
        ):
            continue
        if "features[" not in normalized_feedback:
            continue

        feedback_texts.append(normalize_feedback_text(feedback_item))

    return feedback_texts


def is_feature_rejected_by_feedback(
    product_name: str,
    feature_name: str,
    unsupported_feedback_texts: Sequence[str],
) -> bool:
    """判断 fallback 功能项是否已被 Verifier 点名删除。"""

    if not unsupported_feedback_texts:
        return False

    canonical_claim = f"{product_name} mentions {feature_name}"
    normalized_claim = normalize_feedback_text(canonical_claim)
    normalized_product = normalize_feedback_text(product_name)
    normalized_feature = normalize_feedback_text(feature_name)

    for feedback_text in unsupported_feedback_texts:
        if normalized_claim and normalized_claim in feedback_text:
            return True
        if (
            normalized_product
            and normalized_feature
            and normalized_product in feedback_text
            and normalized_feature in feedback_text
        ):
            return True

    return False


def is_claim_rejected_by_feedback(
    claim: AnalysisClaim,
    unsupported_feedback_texts: Sequence[str],
) -> bool:
    """判断模型输出的功能 claim 是否已被 Verifier 点名为 unsupported。"""

    normalized_claim = normalize_feedback_text(claim.claim)
    for feedback_text in unsupported_feedback_texts:
        if normalized_claim and normalized_claim in feedback_text:
            return True

        # 单产品功能事实在 normalize 后通常是“产品 mentions 功能名”。
        # 这里再用产品名 + claim 关键词做一次宽松匹配，覆盖模型措辞变化。
        for product_name in claim.product_names:
            normalized_product = normalize_feedback_text(product_name)
            if (
                normalized_product
                and normalized_product in feedback_text
                and feedback_text_contains_claim_tokens(
                    normalized_claim,
                    feedback_text,
                )
            ):
                return True

    return False


def feedback_text_contains_claim_tokens(
    normalized_claim: str,
    feedback_text: str,
) -> bool:
    """用关键词重合判断反馈是否指向同一条功能 claim。"""

    claim_tokens = normalized_claim.split()
    meaningful_tokens = [
        token
        for token in claim_tokens
        if token not in {"mentions", "provides", "supports", "includes"}
    ]
    if not meaningful_tokens:
        return False

    matched_token_count = 0
    for token in meaningful_tokens:
        if token in feedback_text:
            matched_token_count += 1

    return matched_token_count >= max(1, len(meaningful_tokens) - 1)


def normalize_feedback_text(text: str) -> str:
    """把反馈和 claim 统一成适合包含匹配的英文小写文本。"""

    tokens = FEATURE_TOKEN_PATTERN.findall(text.lower())
    return " ".join(tokens)


def normalize_single_feature_fact_claim(
    claim: AnalysisClaim,
    features_by_product: dict[str, list[FeatureItem]],
) -> AnalysisClaim | None:
    """收窄单产品功能 fact；无法匹配画像功能时删除该 claim。"""

    if claim.claim_type != "fact":
        return claim

    if len(claim.product_names) != 1:
        return claim

    product_name = claim.product_names[0]
    product_features = features_by_product.get(product_name, [])
    matched_feature = choose_best_feature_for_claim(
        claim=claim,
        product_features=product_features,
    )
    if matched_feature is None:
        return None

    canonical_claim = f"{product_name} mentions {matched_feature.name}."
    return AnalysisClaim(
        claim=canonical_claim,
        claim_type="fact",
        product_names=[product_name],
        evidence_ids=list(matched_feature.evidence_ids),
    )


def build_features_by_product(
    profiles: Sequence[ProductProfile],
) -> dict[str, list[FeatureItem]]:
    """建立产品到功能项的映射，供功能事实收窄使用。"""

    features_by_product: dict[str, list[FeatureItem]] = {}
    for profile in profiles:
        features_by_product[profile.product_name] = list(profile.features)
    return features_by_product


def choose_best_feature_for_claim(
    claim: AnalysisClaim,
    product_features: Sequence[FeatureItem],
) -> FeatureItem | None:
    """根据 Evidence ID 和词重合度找到 claim 最可能描述的功能。"""

    referenced_evidence_ids = set(claim.evidence_ids)
    claim_tokens = tokenize_feature_text(claim.claim)
    best_feature: FeatureItem | None = None
    best_score = 0

    for feature in product_features:
        feature_evidence_ids = set(feature.evidence_ids)
        if not referenced_evidence_ids.intersection(feature_evidence_ids):
            continue

        feature_score = score_feature_claim_match(
            claim_text=claim.claim,
            claim_tokens=claim_tokens,
            feature=feature,
        )
        if feature_score > best_score:
            best_score = feature_score
            best_feature = feature

    if best_score <= 0:
        return None

    return best_feature


def score_feature_claim_match(
    claim_text: str,
    claim_tokens: set[str],
    feature: FeatureItem,
) -> int:
    """计算功能名与 claim 的重合分数，功能名权重大于描述。"""

    normalized_claim_text = claim_text.lower()
    normalized_feature_name = feature.name.lower()
    feature_name_tokens = tokenize_feature_text(feature.name)
    feature_description_tokens = tokenize_feature_text(feature.description)

    score = 0
    if normalized_feature_name in normalized_claim_text:
        score += 8

    name_overlap = claim_tokens.intersection(feature_name_tokens)
    description_overlap = claim_tokens.intersection(
        feature_description_tokens
    )
    score += len(name_overlap) * 3
    score += len(description_overlap)
    return score


def tokenize_feature_text(text: str) -> set[str]:
    """把英文功能文本拆成适合重合度匹配的关键词集合。"""

    tokens: set[str] = set()
    for match in FEATURE_TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0)
        if token in FEATURE_STOP_WORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def normalize_conclusion_after_feedback(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
    market_definition: MarketDefinition | None,
    revision_feedback: Sequence[str],
) -> CompetitiveAnalysis:
    """重试轮退回可见画像摘要，避免模型把结论越写越宽。"""

    if not revision_feedback:
        return analysis

    return analysis.model_copy(
        update={
            "conclusion": build_fallback_conclusion(
                profiles,
                market_definition=market_definition,
            )
        }
    )


def has_unsupported_conclusion_feedback(
    revision_feedback: Sequence[str],
) -> bool:
    """识别 Verifier 是否已经要求收窄 conclusion。"""

    for feedback_item in revision_feedback:
        normalized_feedback = feedback_item.lower()
        if (
            normalized_feedback.startswith("conclusion ")
            and any(
                issue_type in normalized_feedback
                for issue_type in (
                    "unsupported_claim",
                    "partially_supported",
                )
            )
        ):
            return True
    return False


def fill_missing_lightweight_analysis_sections(
    analysis: CompetitiveAnalysis,
    profiles: Sequence[ProductProfile],
    market_definition: MarketDefinition | None = None,
) -> CompetitiveAnalysis:
    """补齐模型为空或过度保守的定位、机会点和结论。"""

    updates: dict[str, object] = {}

    if not analysis.positioning:
        positioning_claims = build_fallback_positioning_claims(profiles)
        if positioning_claims:
            updates["positioning"] = positioning_claims

    if not analysis.opportunities:
        opportunity_claims = build_fallback_opportunity_claims(
            profiles,
            market_definition=market_definition,
        )
        if opportunity_claims:
            updates["opportunities"] = opportunity_claims

    if is_low_information_conclusion(analysis.conclusion.claim):
        updates["conclusion"] = build_fallback_conclusion(
            profiles,
            market_definition=market_definition,
        )

    if not updates:
        return analysis

    return analysis.model_copy(update=updates)


def is_low_information_conclusion(claim_text: str) -> bool:
    """识别只有范围说明、没有实际分析信息的结论。"""

    normalized_claim = claim_text.lower()
    low_information_markers = (
        "limited to the supplied product profiles",
        "limited to the supplied evidence",
        "limited to the provided evidence",
        "本比较仅限",
        "当前资料不足",
    )

    for marker in low_information_markers:
        if marker in normalized_claim:
            return True

    return False


def collect_profile_evidence_ids_in_order(
    profiles: Sequence[ProductProfile],
) -> list[str]:
    """按画像顺序收集 Evidence ID，并去重保留首次出现。"""

    evidence_ids: list[str] = []
    seen_evidence_ids: set[str] = set()

    for profile in profiles:
        for evidence_id in iter_profile_evidence_ids(profile):
            if evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(evidence_id)
            evidence_ids.append(evidence_id)

    return evidence_ids


def validate_analyst_output(
    raw_output: object,
    analyst_input: AnalystInput,
) -> AnalystOutput:
    """校验输出结构、产品覆盖、claim 类型和 Evidence ID 归属。"""

    normalized_output = remove_unreferenced_fact_claims(raw_output)
    try:
        if isinstance(normalized_output, str):
            analyst_output = AnalystOutput.model_validate_json(
                normalized_output
            )
        else:
            analyst_output = AnalystOutput.model_validate(normalized_output)
    except ValidationError as error:
        raise AnalystValidationError(
            f"Output does not match AnalystOutput: {error}"
        ) from error

    analysis = analyst_output.analysis
    expected_products = [
        profile.product_name for profile in analyst_input.profiles
    ]
    if analysis.products != expected_products:
        raise AnalystValidationError(
            "Analysis products do not match input order: "
            f"expected={expected_products!r}, "
            f"actual={analysis.products!r}"
        )

    evidence_ids_by_product = collect_evidence_ids_by_product(
        analyst_input.profiles
    )
    all_claims = collect_analysis_claims(analysis)
    mentioned_products: set[str] = set()

    for claim in all_claims:
        validate_claim_references(
            claim=claim,
            evidence_ids_by_product=evidence_ids_by_product,
        )
        mentioned_products.update(claim.product_names)

    for recommendation in analysis.recommendations:
        validate_recommendation_references(
            recommendation=recommendation,
            evidence_ids_by_product=evidence_ids_by_product,
        )

    validate_selected_analysis_sections(
        analysis=analysis,
        market_definition=analyst_input.market_definition,
    )

    missing_products = set(expected_products) - mentioned_products
    if missing_products:
        missing_text = ", ".join(sorted(missing_products))
        raise AnalystValidationError(
            f"Products missing from comparison claims: {missing_text}"
        )

    if set(analysis.conclusion.product_names) != set(expected_products):
        raise AnalystValidationError(
            "Conclusion must include every input product."
        )

    available_evidence_ids: set[str] = set()
    for evidence_ids in evidence_ids_by_product.values():
        available_evidence_ids.update(evidence_ids)
    if available_evidence_ids and not analysis.conclusion.evidence_ids:
        raise AnalystValidationError(
            "Conclusion must cite supplied Evidence when Evidence is "
            "available."
        )

    analyst_output = normalize_analysis_output(
        analyst_output=analyst_output,
        analyst_input=analyst_input,
    )
    analysis = analyst_output.analysis
    pricing_claim_paths = collect_feature_pricing_claim_paths(analysis)
    if pricing_claim_paths:
        joined_paths = ", ".join(pricing_claim_paths)
        raise AnalystValidationError(
            f"Feature section contains pricing claims: {joined_paths}"
        )

    return analyst_output


def remove_unreferenced_fact_claims(raw_output: object) -> object:
    """只移除无引用事实，保留同次模型输出中的其余有效分析。"""

    if isinstance(raw_output, str):
        try:
            output_data = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            return raw_output
    elif isinstance(raw_output, dict):
        output_data = deepcopy(raw_output)
    else:
        return raw_output

    if not isinstance(output_data, dict):
        return output_data
    analysis = output_data.get("analysis")
    if not isinstance(analysis, dict):
        return output_data

    claim_sections = (
        "positioning",
        "features",
        "pricing",
        "dimension_comparisons",
        "opportunities",
    )
    for section_name in claim_sections:
        claims = analysis.get(section_name)
        if not isinstance(claims, list):
            continue
        analysis[section_name] = [
            claim
            for claim in claims
            if not (
                isinstance(claim, dict)
                and claim.get("claim_type") == "fact"
                and claim.get("evidence_ids") in (None, [])
            )
        ]

    return output_data


def validate_selected_analysis_sections(
    analysis: CompetitiveAnalysis,
    market_definition: MarketDefinition | None,
) -> None:
    """拒绝模型比较用户未选择的固定维度。"""

    if market_definition is None:
        return
    sections = {
        "features": analysis.features,
        "pricing": analysis.pricing,
    }
    excluded_sections = [
        name
        for name, claims in sections.items()
        if claims and not market_dimension_is_selected(market_definition, name)
    ]
    if excluded_sections:
        excluded_text = ", ".join(excluded_sections)
        raise AnalystValidationError(
            "Analysis contains unselected dimensions: " + excluded_text
        )


def validate_recommendation_references(
    recommendation: ActionableRecommendation,
    evidence_ids_by_product: dict[str, set[str]],
) -> None:
    """确保建议只引用其相关产品名下的已知证据。"""

    allowed_products = set(evidence_ids_by_product)
    unknown_products = set(recommendation.product_names) - allowed_products
    if unknown_products:
        unknown_text = ", ".join(sorted(unknown_products))
        raise AnalystValidationError(
            f"Recommendation references unknown products: {unknown_text}"
        )

    allowed_evidence_ids: set[str] = set()
    for product_name in recommendation.product_names:
        allowed_evidence_ids.update(evidence_ids_by_product[product_name])
    invalid_evidence_ids = (
        set(recommendation.evidence_ids) - allowed_evidence_ids
    )
    if invalid_evidence_ids:
        invalid_text = ", ".join(sorted(invalid_evidence_ids))
        raise AnalystValidationError(
            "Recommendation references evidence outside its products: "
            f"{invalid_text}"
        )


def validate_claim_references(
    claim: AnalysisClaim,
    evidence_ids_by_product: dict[str, set[str]],
) -> None:
    """检查 claim 产品范围、引用存在性和事实的逐产品证据支持。"""

    allowed_products = set(evidence_ids_by_product)
    claim_products = set(claim.product_names)
    unknown_products = claim_products - allowed_products
    if unknown_products:
        unknown_text = ", ".join(sorted(unknown_products))
        raise AnalystValidationError(
            f"Claim references unknown products: {unknown_text}"
        )

    allowed_claim_evidence_ids: set[str] = set()
    for product_name in claim.product_names:
        allowed_claim_evidence_ids.update(
            evidence_ids_by_product[product_name]
        )

    referenced_evidence_ids = set(claim.evidence_ids)
    invalid_evidence_ids = (
        referenced_evidence_ids - allowed_claim_evidence_ids
    )
    if invalid_evidence_ids:
        invalid_text = ", ".join(sorted(invalid_evidence_ids))
        raise AnalystValidationError(
            "Claim references evidence outside its products: "
            f"{invalid_text}"
        )

    if claim.claim_type != "fact":
        return

    # 比较事实涉及多个产品时，每个产品都必须有自己的来源支持。
    unsupported_products: list[str] = []
    for product_name in claim.product_names:
        product_evidence_ids = evidence_ids_by_product[product_name]
        if not referenced_evidence_ids.intersection(product_evidence_ids):
            unsupported_products.append(product_name)

    if unsupported_products:
        unsupported_text = ", ".join(unsupported_products)
        raise AnalystValidationError(
            "Factual claim lacks evidence for products: "
            f"{unsupported_text}"
        )


def collect_profile_evidence_ids(
    profile: ProductProfile,
) -> set[str]:
    """收集一个产品画像中功能和价格项的全部 Evidence ID。"""

    return set(iter_profile_evidence_ids(profile))


def iter_profile_evidence_ids(profile: ProductProfile) -> list[str]:
    """集中暴露画像、模型、价格和限流的可引用来源。"""

    evidence_ids: list[str] = []

    evidence_ids.extend(item.evidence_id for item in profile.source_evidence)
    evidence_ids.extend(item.evidence_id for item in profile.field_evidence)

    for dimension_finding in profile.dimension_findings:
        evidence_ids.extend(dimension_finding.evidence_ids)

    for feature in profile.features:
        evidence_ids.extend(feature.evidence_ids)

    for pricing_plan in profile.pricing:
        evidence_ids.extend(pricing_plan.evidence_ids)

    for rate_limit in profile.rate_limits or []:
        evidence_ids.extend(rate_limit.evidence_ids)

    for model in profile.models:
        evidence_ids.extend(item.evidence_id for item in model.source_evidence)
        evidence_ids.extend(model.pricing.evidence_ids())
        for rate_limit in model.rate_limits or []:
            evidence_ids.extend(rate_limit.evidence_ids)

    return evidence_ids


def collect_evidence_ids_by_product(
    profiles: Sequence[ProductProfile],
) -> dict[str, set[str]]:
    """建立产品到 Evidence ID 集合的映射，供引用归属校验使用。"""

    evidence_ids_by_product: dict[str, set[str]] = {}

    for profile in profiles:
        evidence_ids_by_product[profile.product_name] = (
            collect_profile_evidence_ids(profile)
        )

    return evidence_ids_by_product


def collect_analysis_claims(
    analysis: CompetitiveAnalysis,
) -> list[AnalysisClaim]:
    """按报告章节顺序收集全部 claim，包括最终结论。"""

    claims: list[AnalysisClaim] = []
    claims.extend(analysis.positioning)
    claims.extend(analysis.features)
    claims.extend(analysis.pricing)
    claims.extend(analysis.dimension_comparisons)
    claims.extend(analysis.opportunities)
    for assessment in analysis.product_assessments:
        claims.extend(assessment.strengths)
        claims.extend(assessment.shortcomings)
    for recommendation in analysis.scenario_recommendations:
        product_names = (
            [recommendation.recommended_product]
            if recommendation.recommended_product in analysis.products
            else list(analysis.products)
        )
        claims.append(
            AnalysisClaim(
                claim=(
                    f"{recommendation.scenario}: "
                    f"{recommendation.recommendation_reason}"
                ),
                claim_type="interpretation",
                product_names=product_names,
                evidence_ids=list(recommendation.evidence_ids),
            )
        )
    for opportunity in analysis.market_opportunities:
        claims.append(
            AnalysisClaim(
                claim=(
                    f"{opportunity.title}: {opportunity.competitor_status} "
                    f"{opportunity.market_gap}"
                ),
                claim_type="interpretation",
                product_names=list(opportunity.product_names),
                evidence_ids=list(opportunity.evidence_ids),
            )
        )
    claims.append(analysis.conclusion)
    return claims
