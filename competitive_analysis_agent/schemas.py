"""竞品分析流程中共享的数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


RequiredText = Annotated[str, Field(min_length=1)]
EvidenceId = Annotated[str, Field(min_length=1)]


class ContractModel(BaseModel):
    """为外部输入提供统一、严格的 Pydantic 校验规则。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


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
    source_type: Literal["official", "third_party"]
    collected_at: datetime


class FeatureItem(ContractModel):
    """描述一个由证据支持的产品功能。"""

    name: RequiredText
    description: RequiredText
    evidence_ids: list[EvidenceId] = Field(min_length=1)


class PricingPlan(ContractModel):
    """描述一个价格方案；未知价格和计费周期使用 None。"""

    plan_name: RequiredText
    price: RequiredText | None = None
    billing_cycle: RequiredText | None = None
    main_limits: list[str] = Field(default_factory=list)
    evidence_ids: list[EvidenceId] = Field(min_length=1)


class ProductProfile(ContractModel):
    """汇总单个产品的定位、功能、价格和公开限制。"""

    product_name: RequiredText
    positioning: RequiredText | None = None
    target_users: list[str] = Field(default_factory=list)
    features: list[FeatureItem] = Field(default_factory=list)
    pricing: list[PricingPlan] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WorkflowState(ContractModel):
    """保存一次竞品分析从输入到报告的共享状态。"""

    target_product: RequiredText
    competitors: list[RequiredText] = Field(min_length=1)
    dimensions: list[RequiredText] = Field(min_length=1)
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    product_profiles: list[ProductProfile] = Field(default_factory=list)

    # 分析、验证和报告的具体契约会在对应阶段定义。
    analysis_result: dict[str, Any] | None = None
    verification_result: dict[str, Any] | None = None
    final_report: str | None = None
    retry_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
