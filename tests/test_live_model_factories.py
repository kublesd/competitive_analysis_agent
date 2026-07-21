"""验证真实模型工厂只传递 OpenAI 兼容接口可接受的基础参数。"""

from __future__ import annotations

from collections.abc import Callable

from competitive_analysis_agent.config import Settings
from competitive_analysis_agent.live_analyst import create_live_analyst
from competitive_analysis_agent.live_extractor import create_live_extractor
from competitive_analysis_agent.live_verifier import (
    build_live_verifier_input,
    create_live_verifier,
)
from competitive_analysis_agent.verifier import find_deterministic_issues


class FakeStructuredModel:
    """占位结构化模型；工厂测试只关心构造参数，不会真正调用模型。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """防止测试误触发真实调用。"""

        raise AssertionError("Factory test should not invoke the model.")


class FakeChatOpenAI:
    """记录 ChatOpenAI 构造参数，并模拟 LangChain 的结构化输出入口。"""

    created_kwargs: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.created_kwargs.append(kwargs)

    def with_structured_output(
        self,
        schema: type[object],
        *,
        method: str,
        include_raw: bool,
    ) -> FakeStructuredModel:
        """返回占位结构化模型，让各组件工厂可以完成包装。"""

        assert method == "json_mode"
        assert include_raw is True
        return FakeStructuredModel()


def test_live_verifier_fixture_reaches_semantic_model() -> None:
    """真实 Verifier fixture 不应在模型调用前被确定性门禁截断。"""

    assert find_deterministic_issues(build_live_verifier_input()) == []


def test_live_model_factories_do_not_send_unsupported_thinking_flag(
    monkeypatch,
) -> None:
    """Gemini OpenAI 兼容接口不接受 enable_thinking，默认请求不能携带它。"""

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    FakeChatOpenAI.created_kwargs = []
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1/",
        llm_model="test-model",
        tavily_api_key=None,
    )
    factories: list[Callable[[Settings], object]] = [
        create_live_extractor,
        create_live_analyst,
        create_live_verifier,
    ]

    # 三个真实 LLM 节点共用同类 OpenAI 兼容参数，避免某个节点再次触发 400。
    for factory in factories:
        factory(settings)

    assert len(FakeChatOpenAI.created_kwargs) == len(factories)
    for kwargs in FakeChatOpenAI.created_kwargs:
        assert "extra_body" not in kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "https://example.com/v1/"
        assert kwargs["model"] == "test-model"


def test_gemini_25_model_factories_send_reasoning_effort_none(
    monkeypatch,
) -> None:
    """Gemini 2.5 Flash 工厂应发送官方支持的 thinking 控制参数。"""

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    FakeChatOpenAI.created_kwargs = []
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        llm_model="gemini-2.5-flash",
        tavily_api_key=None,
    )

    create_live_extractor(settings)

    assert FakeChatOpenAI.created_kwargs[0]["reasoning_effort"] == "none"
    assert "extra_body" not in FakeChatOpenAI.created_kwargs[0]


def test_extractor_factory_uses_timeout_above_observed_tail_latency(
    monkeypatch,
) -> None:
    """Extractor 单次超时应覆盖已观测到的慢结构化响应。"""

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    FakeChatOpenAI.created_kwargs = []
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1/",
        llm_model="test-model",
        tavily_api_key=None,
    )

    create_live_extractor(settings)

    assert FakeChatOpenAI.created_kwargs[0]["timeout"] == 90
    assert FakeChatOpenAI.created_kwargs[0]["max_retries"] == 1


def test_siliconflow_qwen3_factories_disable_thinking(monkeypatch) -> None:
    """Qwen3 的三个模型节点应共用非思考模式。"""

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    FakeChatOpenAI.created_kwargs = []
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="https://api.siliconflow.cn/v1",
        llm_model="Qwen/Qwen3-8B",
        tavily_api_key=None,
    )

    factories: list[Callable[[Settings], object]] = [
        create_live_extractor,
        create_live_analyst,
        create_live_verifier,
    ]
    for factory in factories:
        factory(settings)

    for kwargs in FakeChatOpenAI.created_kwargs:
        assert kwargs["extra_body"] == {"enable_thinking": False}
