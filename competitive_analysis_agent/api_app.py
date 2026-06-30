"""FastAPI 服务层：把 HTTP 请求转换成现有竞品分析工作流调用。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import Field
import uvicorn

from competitive_analysis_agent import ui_service
from competitive_analysis_agent.health import build_health_report
from competitive_analysis_agent.logging_config import configure_application_logging
from competitive_analysis_agent.researcher import ResearchError
from competitive_analysis_agent.schemas import ContractModel, RequiredText
from competitive_analysis_agent.verifier import VerificationIssue


AnalysisRunner = Callable[[ui_service.AnalysisRequest], ui_service.AnalysisRunResult]

CONFIGURATION_ERROR_NAMES = {
    "LiveModelConfigurationError",
    "LivePlannerConfigurationError",
    "LiveExtractorConfigurationError",
    "LiveAnalystConfigurationError",
    "LiveVerifierConfigurationError",
    "ApplicationSearchConfigurationError",
}


class ApiAnalysisRequest(ContractModel):
    """描述 API 调用方提交的一次竞品分析请求。"""

    target_product: RequiredText
    competitors: list[RequiredText] = Field(min_length=1)
    dimensions: list[RequiredText] = Field(min_length=1)
    official_domains_by_product: dict[str, list[RequiredText]] = Field(
        default_factory=dict
    )

    def to_service_request(self) -> ui_service.AnalysisRequest:
        """转换为 Streamlit 与 API 共用的应用服务请求。"""

        return ui_service.AnalysisRequest(
            target_product=self.target_product,
            competitors=self.competitors,
            dimensions=self.dimensions,
            official_domains_by_product=self.official_domains_by_product,
        )


class ApiEvidenceResponse(ContractModel):
    """返回给 API 调用方的证据摘要，不包含网页原文。"""

    evidence_id: RequiredText
    product_name: RequiredText
    topic: RequiredText
    title: RequiredText
    url: RequiredText
    snippet_preview: RequiredText
    source_type: Literal["official", "third_party"]
    collected_at: datetime


class ApiAnalysisResponse(ContractModel):
    """保存一次 API 分析完成后的报告、轨迹和可复核来源。"""

    final_report: RequiredText
    stage_history: list[RequiredText] = Field(min_length=1)
    verification_passed: bool
    verification_issues: list[VerificationIssue] = Field(default_factory=list)
    evidence: list[ApiEvidenceResponse] = Field(default_factory=list)
    research_errors: list[ResearchError] = Field(default_factory=list)

    @classmethod
    def from_run_result(
        cls,
        result: ui_service.AnalysisRunResult,
    ) -> "ApiAnalysisResponse":
        """把内部运行结果转换成适合 HTTP JSON 返回的形状。"""

        evidence_responses: list[ApiEvidenceResponse] = []
        for evidence in result.evidence:
            evidence_responses.append(
                ApiEvidenceResponse(
                    evidence_id=evidence.evidence_id,
                    product_name=evidence.product_name,
                    topic=evidence.topic,
                    title=evidence.title,
                    url=str(evidence.url),
                    snippet_preview=ui_service.truncate_text(
                        evidence.snippet
                    ),
                    source_type=evidence.source_type,
                    collected_at=evidence.collected_at,
                )
            )

        return cls(
            final_report=result.final_report,
            stage_history=result.stage_history,
            verification_passed=result.verification_result.passed,
            verification_issues=result.verification_result.issues,
            evidence=evidence_responses,
            research_errors=result.research_errors,
        )


class ApiErrorResponse(ContractModel):
    """统一的 API 错误响应，避免把内部异常原文直接暴露给调用方。"""

    message: RequiredText
    error_type: RequiredText


class ApiErrorEnvelope(ContractModel):
    """匹配 FastAPI HTTPException 的 detail 外层结构。"""

    detail: ApiErrorResponse


def create_app(
    analysis_runner: AnalysisRunner | None = None,
    *,
    configure_logging: bool = True,
) -> FastAPI:
    """创建 FastAPI 应用；测试可以注入假的分析函数。"""

    if analysis_runner is None:
        current_analysis_runner = run_api_analysis
    else:
        current_analysis_runner = analysis_runner
    lifespan = create_lifespan() if configure_logging else None
    app = FastAPI(
        title="Competitive Analysis Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def read_health() -> dict[str, object]:
        """返回服务健康状态和脱敏后的配置状态。"""

        return build_health_report()

    @app.post(
        "/analyses",
        response_model=ApiAnalysisResponse,
        responses={
            422: {"model": ApiErrorEnvelope},
            500: {"model": ApiErrorEnvelope},
            503: {"model": ApiErrorEnvelope},
        },
    )
    def create_analysis(
        request: ApiAnalysisRequest,
    ) -> ApiAnalysisResponse:
        """同步运行一次竞品分析，并返回 Markdown 报告和来源摘要。"""

        try:
            service_request = request.to_service_request()
            result = current_analysis_runner(service_request)
        except Exception as error:
            raise build_http_error(error) from error

        return ApiAnalysisResponse.from_run_result(result)

    return app


def run_api_analysis(
    request: ui_service.AnalysisRequest,
) -> ui_service.AnalysisRunResult:
    """使用 API 入口标记运行共享分析服务。"""

    return ui_service.run_analysis(
        request,
        entrypoint="api",
    )


def create_lifespan() -> Callable[[FastAPI], AsyncIterator[None]]:
    """创建 FastAPI 生命周期钩子，在服务启动时再配置日志。"""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        configure_application_logging()
        yield

    return lifespan


def build_http_error(error: Exception) -> HTTPException:
    """把内部异常转换成 HTTP 状态码和脱敏错误体。"""

    error_type = type(error).__name__
    if error_type in CONFIGURATION_ERROR_NAMES:
        status_code = 503
    elif isinstance(error, ValueError):
        status_code = 422
    else:
        status_code = 500

    error_response = ApiErrorResponse(
        message=ui_service.describe_user_error(error),
        error_type=error_type,
    )
    return HTTPException(
        status_code=status_code,
        detail=error_response.model_dump(),
    )


def main() -> None:
    """使用 uvicorn 启动本地 API 服务。"""

    uvicorn.run(
        "competitive_analysis_agent.api_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


app = create_app()


if __name__ == "__main__":
    main()
