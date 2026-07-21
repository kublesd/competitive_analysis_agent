"""搜索供应商边界与确定性结果规范化。"""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import Field, HttpUrl, ValidationError

from competitive_analysis_agent.schemas import ContractModel, RequiredText


SourceType = Literal["official", "third_party"]
SearchStatus = Literal["success", "error"]
SearchErrorCode = Literal["timeout", "provider_error"]
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
TAVILY_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"


class SearchRequest(ContractModel):
    """描述一次搜索调用所需的查询、官方域名和结果上限。"""

    query: RequiredText
    official_domains: list[RequiredText] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=10)
    include_raw_content: bool = False
    search_depth: Literal["basic", "advanced"] = "basic"
    chunks_per_source: int = Field(default=3, ge=1, le=3)
    extract_query: str | None = None
    extract_top_results: int = Field(default=0, ge=0, le=5)


class ProviderSearchResult(ContractModel):
    """表示搜索供应商返回、尚未规范化的一条记录。"""

    title: RequiredText
    url: HttpUrl
    snippet: RequiredText
    raw_content: str | None = None
    extracted_content: bool = False
    extraction_error: str | None = None


class SearchResult(ContractModel):
    """表示项目内部统一使用的一条搜索结果。"""

    title: RequiredText
    url: HttpUrl
    snippet: RequiredText
    raw_content: str | None = None
    extracted_content: bool = False
    extraction_error: str | None = None
    source_type: SourceType


class SearchError(ContractModel):
    """表示调用方可以处理的搜索失败。"""

    code: SearchErrorCode
    message: RequiredText


class SearchResponse(ContractModel):
    """统一封装成功结果或受控错误。"""

    status: SearchStatus
    results: list[SearchResult] = Field(default_factory=list)
    error: SearchError | None = None


class SearchProvider(Protocol):
    """约定所有搜索供应商都接收同一种请求。"""

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        """执行搜索并返回供应商记录。"""


class FakeSearchProvider:
    """使用固定数据模拟搜索供应商，供测试和本地开发使用。"""

    def __init__(
        self,
        results_by_query: Mapping[
            str,
            Sequence[ProviderSearchResult | Mapping[str, object]],
        ],
        failures_by_query: Mapping[str, Exception] | None = None,
    ) -> None:
        self._results_by_query = results_by_query
        self._failures_by_query = failures_by_query or {}

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        """按查询返回固定结果，或抛出测试指定的异常。"""

        failure = self._failures_by_query.get(request.query)
        if failure is not None:
            raise failure

        raw_results = self._results_by_query.get(request.query, [])
        validated_results: list[ProviderSearchResult] = []

        # 模拟真实供应商边界：每条外部记录进入项目时都要先校验。
        for raw_result in raw_results:
            validated_result = ProviderSearchResult.model_validate(raw_result)
            validated_results.append(validated_result)

        return validated_results


class TavilySearchProviderError(RuntimeError):
    """表示 Tavily 网络、鉴权或响应契约错误。"""


class TavilySearchProvider:
    """调用 Tavily Search API，并映射成项目统一的供应商结果。"""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = TAVILY_SEARCH_ENDPOINT,
        extract_endpoint: str = TAVILY_EXTRACT_ENDPOINT,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
    ) -> None:
        normalized_api_key = api_key.strip()
        if not normalized_api_key:
            raise ValueError("Tavily API key must not be empty.")
        if max_retries < 0:
            raise ValueError("Tavily max_retries must not be negative.")

        self._api_key = normalized_api_key
        self._endpoint = endpoint
        self._extract_endpoint = extract_endpoint
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def search(self, request: SearchRequest) -> list[ProviderSearchResult]:
        """执行搜索，并按请求对少量候选 URL 追加 Tavily Extract。"""

        payload: dict[str, object] = {
            "query": request.query,
            "search_depth": request.search_depth,
            "max_results": request.max_results,
            "include_answer": False,
            # 追加 Extract 时不重复请求整页正文，Search 摘要保留为降级数据。
            "include_raw_content": (
                request.include_raw_content
                and request.extract_top_results == 0
            ),
            "include_images": False,
        }
        if request.official_domains:
            # 用户显式提供域名时优先搜索官方资料，避免模型猜测来源身份。
            payload["include_domains"] = request.official_domains
        if request.search_depth == "advanced":
            payload["chunks_per_source"] = request.chunks_per_source

        response_payload = self._send_request_with_retries(
            payload,
            endpoint=self._endpoint,
            operation="search",
        )
        raw_results = response_payload.get("results")
        if not isinstance(raw_results, list):
            raise TavilySearchProviderError(
                "Tavily response does not contain a results list."
            )

        provider_results: list[ProviderSearchResult] = []
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue

            mapped_result = {
                "title": raw_result.get("title"),
                "url": raw_result.get("url"),
                "snippet": raw_result.get("content"),
                "raw_content": normalize_optional_text(
                    raw_result.get("raw_content")
                ),
            }
            try:
                provider_result = ProviderSearchResult.model_validate(
                    mapped_result
                )
            except ValidationError:
                # 单条外部记录损坏时跳过，不让其破坏同一查询的其他有效来源。
                continue
            provider_results.append(provider_result)

        if request.extract_top_results == 0 or not provider_results:
            return provider_results

        try:
            return self._add_extracted_content(
                provider_results=provider_results,
                request=request,
            )
        except (TimeoutError, TavilySearchProviderError) as error:
            # Extract 只是增强步骤；保留 Search 结果，并把降级原因交给 Reporter。
            provider_results[0] = provider_results[0].model_copy(
                update={"extraction_error": str(error)}
            )
            return provider_results

    def _add_extracted_content(
        self,
        provider_results: list[ProviderSearchResult],
        request: SearchRequest,
    ) -> list[ProviderSearchResult]:
        """对排名靠前的候选页执行查询聚焦的高级正文提取。"""

        selected_results = provider_results[: request.extract_top_results]
        payload: dict[str, object] = {
            "urls": [str(result.url) for result in selected_results],
            "extract_depth": "advanced",
            "format": "markdown",
            "query": request.extract_query or request.query,
            "chunks_per_source": 3,
        }
        response_payload = self._send_request_with_retries(
            payload,
            endpoint=self._extract_endpoint,
            operation="extract",
        )
        raw_results = response_payload.get("results")
        if not isinstance(raw_results, list):
            raise TavilySearchProviderError(
                "Tavily extract response does not contain a results list."
            )

        extracted_content_by_url: dict[str, str] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            raw_url = raw_result.get("url")
            raw_content = normalize_optional_text(
                raw_result.get("raw_content")
            )
            if raw_url is None or raw_content is None:
                continue
            try:
                extracted_content_by_url[normalize_url(str(raw_url))] = (
                    raw_content
                )
            except ValueError:
                continue

        enriched_results: list[ProviderSearchResult] = []
        extracted_count = 0
        for provider_result in provider_results:
            normalized_result_url = normalize_url(provider_result.url)
            extracted_content = extracted_content_by_url.get(
                normalized_result_url
            )
            if extracted_content is None:
                enriched_results.append(provider_result)
                continue
            enriched_results.append(
                provider_result.model_copy(
                    update={
                        "raw_content": extracted_content,
                        "extracted_content": True,
                    }
                )
            )
            extracted_count += 1

        if extracted_count == 0:
            raise TavilySearchProviderError(
                "Tavily extract returned no usable content."
            )
        return enriched_results

    def _send_request_with_retries(
        self,
        payload: dict[str, object],
        *,
        endpoint: str,
        operation: str,
    ) -> dict[str, object]:
        """对瞬时网络失败最多重试一次，避免单个任务轻易丢失证据。"""

        last_error: Exception | None = None
        total_attempts = self._max_retries + 1
        for attempt_number in range(total_attempts):
            try:
                return self._send_request(
                    payload,
                    endpoint=endpoint,
                    operation=operation,
                )
            except TimeoutError as error:
                last_error = error
            except TavilySearchProviderError as error:
                if not is_retryable_tavily_error(error):
                    raise
                last_error = error

            is_last_attempt = attempt_number == total_attempts - 1
            if is_last_attempt:
                break

        if last_error is not None:
            raise last_error

        raise TavilySearchProviderError(
            f"Tavily {operation} retry loop failed."
        )

    def _send_request(
        self,
        payload: dict[str, object],
        *,
        endpoint: str,
        operation: str,
    ) -> dict[str, object]:
        """发送 HTTP 请求，并把底层异常转换成不包含密钥的错误。"""

        request_body = json.dumps(payload).encode("utf-8")
        http_request = Request(
            endpoint,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "competitive-analysis-agent/0.1",
            },
            method="POST",
        )

        try:
            with urlopen(
                http_request,
                timeout=self._timeout_seconds,
            ) as response:
                response_text = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as error:
            raise TimeoutError(f"Tavily {operation} timed out.") from error
        except HTTPError as error:
            raise TavilySearchProviderError(
                f"Tavily {operation} returned HTTP status {error.code}."
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(
                    f"Tavily {operation} timed out."
                ) from error
            raise TavilySearchProviderError(
                f"Tavily {operation} network request failed."
            ) from error

        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise TavilySearchProviderError(
                f"Tavily {operation} returned invalid JSON."
            ) from error

        if not isinstance(parsed_response, dict):
            raise TavilySearchProviderError(
                f"Tavily {operation} returned an invalid response object."
            )
        return parsed_response


class SearchAdapter:
    """调用任意搜索供应商，并返回稳定的项目内部响应。"""

    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    def search(self, request: SearchRequest) -> SearchResponse:
        """执行搜索，将异常转换成受控错误，并规范化成功结果。"""

        try:
            provider_results = self._provider.search(request)
        except TimeoutError:
            # 超时是可恢复失败，调用方可以记录错误后继续其他任务。
            return SearchResponse(
                status="error",
                error=SearchError(
                    code="timeout",
                    message="Search provider timed out.",
                ),
            )
        except Exception as error:
            # 供应商格式错误或其他异常都不能让工作流直接崩溃。
            return SearchResponse(
                status="error",
                error=SearchError(
                    code="provider_error",
                    message=f"Search provider failed: {error}",
                ),
            )

        normalized_results = normalize_search_results(
            provider_results=provider_results,
            official_domains=request.official_domains,
            max_results=request.max_results,
        )
        return SearchResponse(status="success", results=normalized_results)


def normalize_url(url: str | HttpUrl) -> str:
    """规范化网页 URL，使等价地址能够被稳定去重。"""

    parsed_url = urlsplit(str(url))
    scheme = parsed_url.scheme.lower()
    hostname = parsed_url.hostname
    if scheme not in {"http", "https"} or hostname is None:
        raise ValueError("Search result URL must use HTTP or HTTPS.")

    normalized_host = hostname.lower().rstrip(".")
    port = parsed_url.port
    uses_default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    if port is not None and not uses_default_port:
        normalized_host = f"{normalized_host}:{port}"

    normalized_path = parsed_url.path or "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")

    # 跟踪参数和 fragment 不影响页面主体，删除后可识别同一来源。
    query_pairs = parse_qsl(parsed_url.query, keep_blank_values=True)
    meaningful_query_pairs: list[tuple[str, str]] = []
    for key, value in query_pairs:
        if key.lower().startswith("utm_"):
            continue
        meaningful_query_pairs.append((key, value))

    meaningful_query_pairs.sort()
    normalized_query = urlencode(meaningful_query_pairs)

    return urlunsplit(
        (
            scheme,
            normalized_host,
            normalized_path,
            normalized_query,
            "",
        )
    )


def classify_source(
    url: str | HttpUrl,
    official_domains: Sequence[str],
) -> SourceType:
    """根据域名判断来源；厂商社区内容仍按第三方资料处理。"""

    result_hostname = urlsplit(str(url)).hostname
    if result_hostname is None:
        return "third_party"

    normalized_result_hostname = result_hostname.lower().rstrip(".")
    if {"community", "discuss", "forum"}.intersection(
        normalized_result_hostname.split(".")
    ):
        return "third_party"
    for official_domain in official_domains:
        normalized_official_domain = _normalize_domain(official_domain)
        is_exact_domain = (
            normalized_result_hostname == normalized_official_domain
        )
        is_subdomain = normalized_result_hostname.endswith(
            f".{normalized_official_domain}"
        )
        if is_exact_domain or is_subdomain:
            return "official"

    return "third_party"


def normalize_search_results(
    provider_results: Sequence[ProviderSearchResult],
    official_domains: Sequence[str],
    max_results: int,
) -> list[SearchResult]:
    """按供应商顺序规范化、去重并限制搜索结果数量。"""

    normalized_results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for provider_result in provider_results:
        normalized_url = normalize_url(provider_result.url)
        if normalized_url in seen_urls:
            continue

        source_type = classify_source(
            normalized_url,
            official_domains,
        )
        normalized_result = SearchResult(
            title=provider_result.title,
            url=normalized_url,
            snippet=provider_result.snippet,
            raw_content=normalize_optional_text(
                provider_result.raw_content
            ),
            extracted_content=provider_result.extracted_content,
            extraction_error=normalize_optional_text(
                provider_result.extraction_error
            ),
            source_type=source_type,
        )
        normalized_results.append(normalized_result)
        seen_urls.add(normalized_url)

        if len(normalized_results) >= max_results:
            break

    return normalized_results


def _normalize_domain(domain: str) -> str:
    """把用户配置的域名转换成可比较的主机名。"""

    stripped_domain = domain.strip().lower()
    if "://" in stripped_domain:
        hostname = urlsplit(stripped_domain).hostname
        if hostname is None:
            return stripped_domain.rstrip(".")
        return hostname.rstrip(".")

    domain_without_path = stripped_domain.split("/", maxsplit=1)[0]
    domain_without_port = domain_without_path.split(":", maxsplit=1)[0]
    return domain_without_port.rstrip(".")


def is_retryable_tavily_error(error: TavilySearchProviderError) -> bool:
    """判断 Tavily 错误是否像瞬时网络问题，只有这类错误才重试。"""

    return str(error).endswith("network request failed.")


def normalize_optional_text(value: object) -> str | None:
    """把外部可选文本转换成 None 或去空白后的字符串。"""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    return text
