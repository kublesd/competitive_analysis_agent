"""竞品分析流程中共享的数据契约。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


RequiredText = Annotated[str, Field(min_length=1)]
EvidenceId = Annotated[str, Field(min_length=1)]
ConfidenceScore = Annotated[float, Field(ge=0, le=1)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
PositiveDecimal = Annotated[Decimal, Field(gt=0)]
PositiveCount = Annotated[int, Field(gt=0)]


class AvailabilityStatus(str, Enum):
    """标记字段有值、缺失、不适用或仍待确认。"""

    AVAILABLE = "available"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class SupportStatus(str, Enum):
    """统一表达一个模型能力是否得到资料支持。"""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class VerificationStatus(str, Enum):
    """记录抽取结果尚未验证、通过、失败或证据冲突。"""

    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"
    CONFLICTING = "conflicting"


class SourceType(str, Enum):
    """区分厂商官方来源与第三方来源。"""

    OFFICIAL = "official"
    THIRD_PARTY = "third_party"


class Modality(str, Enum):
    """列出模型可直接接收或生成的主要模态。"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class Currency(str, Enum):
    """限制价格使用的币种，未知币种不得伪装成 USD。"""

    USD = "USD"
    CNY = "CNY"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    OTHER = "other"


class PricingUnit(str, Enum):
    """统一 API 费率的计量对象。"""

    TOKEN = "token"
    REQUEST = "request"
    IMAGE = "image"
    AUDIO_MINUTE = "audio_minute"
    CHARACTER = "character"
    SECOND = "second"
    OTHER = "other"


class RateLimitMetric(str, Enum):
    """列出常见限流指标，避免把上下文窗口写进限流字段。"""

    REQUESTS_PER_MINUTE = "requests_per_minute"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    REQUESTS_PER_DAY = "requests_per_day"
    TOKENS_PER_DAY = "tokens_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    OTHER = "other"


class ContractModel(BaseModel):
    """为外部输入提供统一、严格的 Pydantic 校验规则。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class SourceReference(ContractModel):
    """保存统一比较字段引用的来源摘要和采集时间。"""

    evidence_id: EvidenceId
    title: RequiredText
    url: HttpUrl
    source_type: SourceType
    collected_at: datetime


class FieldEvidence(ContractModel):
    """说明一个抽取字段由哪段原文支持以及为何匹配该字段。"""

    field_path: RequiredText
    evidence_id: EvidenceId
    quote: RequiredText
    rationale: RequiredText
    confidence: ConfidenceScore


class PriceRate(ContractModel):
    """表示一项带币种、计量基数和来源的原子费率。"""

    amount: NonNegativeDecimal
    currency: Currency
    per_quantity: PositiveCount
    unit: PricingUnit
    condition: RequiredText | None = None
    max_context_tokens: PositiveCount | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    evidence_ids: list[EvidenceId] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "PriceRate":
        """拒绝重复来源，避免同一价格重复计算引用覆盖率。"""

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Price evidence IDs must be unique.")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("Price effective_to cannot precede effective_from.")
        return self


class ModelPricing(ContractModel):
    """把同一模型的所有价格类型收拢在一个固定结构中。"""

    input_price: PriceRate | list[PriceRate] | None = None
    cached_input_price: PriceRate | list[PriceRate] | None = None
    output_price: PriceRate | list[PriceRate] | None = None
    audio_input_price: PriceRate | list[PriceRate] | None = None
    audio_output_price: PriceRate | list[PriceRate] | None = None
    # 兼容既有已抽取数据；新数据必须使用 audio_input_price。
    audio_price: PriceRate | list[PriceRate] | None = None
    batch_input_price: PriceRate | list[PriceRate] | None = None
    batch_cached_input_price: PriceRate | list[PriceRate] | None = None
    batch_output_price: PriceRate | list[PriceRate] | None = None
    batch_discount_percent: Annotated[Decimal, Field(ge=0, le=100)] | None = (
        None
    )
    batch_evidence_ids: list[EvidenceId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_batch_discount_evidence(self) -> "ModelPricing":
        """校验 Batch 引用和同类型多有效期价格。"""

        has_discount = self.batch_discount_percent is not None
        has_evidence = bool(self.batch_evidence_ids)
        if has_discount != has_evidence:
            raise ValueError(
                "Batch discount and batch evidence IDs must be provided "
                "together."
            )
        if len(self.batch_evidence_ids) != len(set(self.batch_evidence_ids)):
            raise ValueError("Batch evidence IDs must be unique.")

        for price_value in self._price_values():
            for price in (
                price_value if isinstance(price_value, list) else [price_value]
            ):
                if price is None:
                    continue
                if (
                    price.currency != Currency.USD
                    or price.per_quantity != 1_000_000
                    or price.unit != PricingUnit.TOKEN
                ):
                    raise ValueError(
                        "Model token prices must be USD per 1M tokens."
                    )
            if not isinstance(price_value, list):
                continue
            if not price_value:
                raise ValueError("Price history cannot be empty.")
            periods = {
                (price.effective_from, price.effective_to)
                for price in price_value
            }
            if all(start is None and end is None for start, end in periods):
                conditions = [price.condition for price in price_value]
                if any(condition is None for condition in conditions) or len(
                    conditions
                ) != len(set(conditions)):
                    raise ValueError(
                        "Multiple undated prices require unique conditions."
                    )
            elif len(periods) != len(price_value) or any(
                start is None and end is None for start, end in periods
            ):
                raise ValueError("Price effective periods must be unique.")
        return self

    def _price_values(self) -> list[PriceRate | list[PriceRate] | None]:
        """返回全部价格字段，供校验和引用汇总复用。"""

        return [
            self.input_price,
            self.cached_input_price,
            self.output_price,
            self.audio_input_price,
            self.audio_output_price,
            self.audio_price,
            self.batch_input_price,
            self.batch_cached_input_price,
            self.batch_output_price,
        ]

    def evidence_ids(self) -> set[str]:
        """汇总所有费率和 Batch 折扣引用的 Evidence ID。"""

        evidence_ids = set(self.batch_evidence_ids)
        for price_value in self._price_values():
            price_rates = (
                price_value if isinstance(price_value, list) else [price_value]
            )
            for price_rate in price_rates:
                if price_rate is None:
                    continue
                evidence_ids.update(price_rate.evidence_ids)
        return evidence_ids

    @property
    def has_batch_pricing(self) -> bool:
        """返回当前结构是否包含任一 Batch 价格或折扣。"""

        return any(
            value is not None
            for value in [
                self.batch_input_price,
                self.batch_cached_input_price,
                self.batch_output_price,
                self.batch_discount_percent,
            ]
        )


class RateLimit(ContractModel):
    """保存一个明确的限流指标、适用层级和来源。"""

    metric: RateLimitMetric
    limit: PositiveDecimal
    tier: RequiredText | None = None
    region: RequiredText | None = None
    notes: RequiredText | None = None
    evidence_ids: list[EvidenceId] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> "RateLimit":
        """同一个限流项不能重复引用同一 Evidence。"""

        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Rate-limit evidence IDs must be unique.")
        return self


class ModelProfile(ContractModel):
    """用固定字段保存一个模型的能力、模态、限制和完整价格矩阵。"""

    model_name: RequiredText
    model_capabilities: list[RequiredText] | None = None
    supported_modalities: list[Modality] | None = None
    context_window_tokens: PositiveCount | None = None
    max_output_tokens: PositiveCount | None = None
    tool_calling: SupportStatus = SupportStatus.MISSING
    structured_output: SupportStatus = SupportStatus.MISSING
    realtime: SupportStatus = SupportStatus.MISSING
    prompt_caching: SupportStatus = SupportStatus.MISSING
    batch_api: SupportStatus = SupportStatus.MISSING
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    rate_limits: list[RateLimit] | None = None
    source_evidence: list[SourceReference] = Field(min_length=1)
    extraction_confidence: ConfidenceScore
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    @model_validator(mode="after")
    def validate_model_contract(self) -> "ModelProfile":
        """拒绝拆价模型名、重复值和未绑定当前模型来源的事实。"""

        normalized_name = self.model_name.casefold()
        price_direction_suffixes = (
            " input",
            " output",
            " cached input",
            " cache input",
            " batch input",
            " batch output",
        )
        if normalized_name.endswith(price_direction_suffixes):
            raise ValueError(
                "Price direction belongs in ModelPricing, not model_name."
            )

        if self.model_capabilities is not None and len(
            self.model_capabilities
        ) != len(set(self.model_capabilities)):
            raise ValueError("Model capabilities must be unique.")
        if self.supported_modalities is not None and len(
            self.supported_modalities
        ) != len(set(self.supported_modalities)):
            raise ValueError("Supported modalities must be unique.")

        source_evidence_ids = {
            item.evidence_id for item in self.source_evidence
        }
        if len(source_evidence_ids) != len(self.source_evidence):
            raise ValueError("Model source Evidence IDs must be unique.")

        referenced_evidence_ids = self.pricing.evidence_ids()
        for rate_limit in self.rate_limits or []:
            referenced_evidence_ids.update(rate_limit.evidence_ids)
        unknown_evidence_ids = referenced_evidence_ids - source_evidence_ids
        if unknown_evidence_ids:
            unknown_text = ", ".join(sorted(unknown_evidence_ids))
            raise ValueError(
                "Model facts reference Evidence outside source_evidence: "
                f"{unknown_text}"
            )

        if self.pricing.has_batch_pricing and self.batch_api != (
            SupportStatus.SUPPORTED
        ):
            raise ValueError(
                "Batch pricing requires batch_api='supported'."
            )
        return self


class MarketDefinition(ContractModel):
    """定义一次竞品分析必须遵守的市场范围。"""

    market_name: RequiredText
    product_category: RequiredText
    target_buyer: RequiredText | None = None
    comparison_level: RequiredText
    pricing_scope: Literal["api", "subscription"] = "subscription"
    monthly_call_count: PositiveCount = 1_000
    core_dimensions: list[RequiredText] = Field(min_length=1)
    exclusions: list[RequiredText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_lists(self) -> "MarketDefinition":
        """拒绝重复维度和排除项，避免同一范围规则重复执行。"""

        if len(self.core_dimensions) != len(set(self.core_dimensions)):
            raise ValueError("Core dimensions must be unique.")
        if len(self.exclusions) != len(set(self.exclusions)):
            raise ValueError("Exclusions must be unique.")
        return self


class ResearchTask(ContractModel):
    """描述 Planner 交给 Researcher 的一项独立调研任务。"""

    product_name: RequiredText
    topic: RequiredText
    query: RequiredText


class Evidence(ContractModel):
    """保存一条可追溯的搜索证据及其采集上下文。"""

    evidence_id: EvidenceId
    product_name: RequiredText
    topic: RequiredText
    title: RequiredText
    url: HttpUrl
    snippet: RequiredText
    raw_content: str | None = None
    source_type: SourceType
    scope_status: Literal["in_scope", "out_of_scope", "uncertain"] = (
        "in_scope"
    )
    scope_reason: RequiredText = "Provided as in-scope evidence."
    collected_at: datetime


class FeatureItem(ContractModel):
    """描述一个由证据支持的产品功能。"""

    name: RequiredText
    description: RequiredText
    evidence_ids: list[EvidenceId] = Field(min_length=1)


class DimensionFinding(ContractModel):
    """保存一个用户核心维度下的证据化事实；资料不足时列表为空。"""

    dimension: RequiredText
    facts: list[RequiredText] = Field(default_factory=list)
    evidence_ids: list[EvidenceId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fact_evidence(self) -> "DimensionFinding":
        """有事实时必须有证据，无事实时不得挂空洞引用。"""

        if self.facts and not self.evidence_ids:
            raise ValueError("Dimension facts require evidence IDs.")
        if not self.facts and self.evidence_ids:
            raise ValueError("Empty dimension findings cannot cite evidence.")
        return self


class PricingPlan(ContractModel):
    """描述一个价格方案；未知价格和计费周期使用 None。"""

    plan_name: RequiredText
    price: RequiredText | None = None
    unit: RequiredText | None = None
    billing_cycle: RequiredText | None = None
    service_level: RequiredText | None = None
    threshold: RequiredText | None = None
    main_limits: list[str] = Field(default_factory=list)
    evidence_ids: list[EvidenceId] = Field(min_length=1)


class ProductProfile(ContractModel):
    """汇总产品级事实，并兼容旧版自由结构字段。"""

    product_name: RequiredText
    positioning: RequiredText | None = None
    positioning_status: AvailabilityStatus = AvailabilityStatus.MISSING
    target_users: list[str] = Field(default_factory=list)
    target_users_status: AvailabilityStatus = AvailabilityStatus.MISSING
    models: list[ModelProfile] = Field(default_factory=list)
    rate_limits: list[RateLimit] | None = None
    enterprise_capabilities: list[RequiredText] | None = None
    enterprise_capabilities_status: AvailabilityStatus = (
        AvailabilityStatus.MISSING
    )
    source_evidence: list[SourceReference] = Field(default_factory=list)
    field_evidence: list[FieldEvidence] = Field(default_factory=list)
    extraction_confidence: ConfidenceScore | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

    # 以下字段是旧版抽取和报告链路仍在使用的兼容入口。
    features: list[FeatureItem] = Field(default_factory=list)
    dimension_findings: list[DimensionFinding] = Field(default_factory=list)
    pricing: list[PricingPlan] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unified_profile(self) -> "ProductProfile":
        """保持字段状态、模型身份和来源引用一致。"""

        if "positioning_status" not in self.model_fields_set:
            object.__setattr__(
                self,
                "positioning_status",
                self._inferred_status(self.positioning),
            )
        if "target_users_status" not in self.model_fields_set:
            object.__setattr__(
                self,
                "target_users_status",
                self._inferred_status(self.target_users or None),
            )
        if "enterprise_capabilities_status" not in self.model_fields_set:
            object.__setattr__(
                self,
                "enterprise_capabilities_status",
                self._inferred_status(self.enterprise_capabilities),
            )

        self._validate_availability(
            "positioning",
            self.positioning,
            self.positioning_status,
        )
        self._validate_availability(
            "target_users",
            self.target_users or None,
            self.target_users_status,
        )
        self._validate_availability(
            "enterprise_capabilities",
            self.enterprise_capabilities,
            self.enterprise_capabilities_status,
        )

        normalized_model_names = [
            model.model_name.casefold() for model in self.models
        ]
        if len(normalized_model_names) != len(set(normalized_model_names)):
            raise ValueError("Model names must be unique within a product.")

        source_evidence_ids = {
            item.evidence_id for item in self.source_evidence
        }
        if len(source_evidence_ids) != len(self.source_evidence):
            raise ValueError("Product source Evidence IDs must be unique.")
        return self

    @staticmethod
    def _inferred_status(value: Any) -> AvailabilityStatus:
        """旧字段有内容时标记 available，否则明确标记 missing。"""

        if value is None:
            return AvailabilityStatus.MISSING
        return AvailabilityStatus.AVAILABLE

    @staticmethod
    def _validate_availability(
        field_name: str,
        value: Any,
        status: AvailabilityStatus,
    ) -> None:
        """阻止 missing 状态携带内容或 available 状态缺少内容。"""

        if status == AvailabilityStatus.AVAILABLE and value is None:
            raise ValueError(f"{field_name} is required when available.")
        if status in {
            AvailabilityStatus.MISSING,
            AvailabilityStatus.NOT_APPLICABLE,
        } and value is not None:
            raise ValueError(
                f"{field_name} must be null or empty when status is "
                f"'{status.value}'."
            )


class WorkflowState(ContractModel):
    """保存一次竞品分析从输入到报告的共享状态。"""

    target_product: RequiredText
    competitors: list[RequiredText] = Field(min_length=1)
    market_definition: MarketDefinition
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    product_profiles: list[ProductProfile] = Field(default_factory=list)

    # 分析、验证和报告的具体契约会在对应阶段定义。
    analysis_result: dict[str, Any] | None = None
    verification_result: dict[str, Any] | None = None
    final_report: str | None = None
    retry_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)

    @property
    def dimensions(self) -> list[str]:
        """兼容现有节点名称，维度以市场定义为唯一来源。"""

        return self.market_definition.core_dimensions
