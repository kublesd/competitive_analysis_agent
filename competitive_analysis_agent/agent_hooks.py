"""Agent 运行生命周期 Hooks，用于把观测逻辑从业务节点中分离出来。"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from time import perf_counter
from typing import Protocol


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    """保存一次 Agent 运行中所有 Hook 共享的脱敏上下文。"""

    analysis_id: str
    entrypoint: str
    started_at: float
    configuration_summary: dict[str, object] = field(default_factory=dict)

    def elapsed_ms(self) -> int:
        """返回从运行开始到当前时刻的毫秒数。"""

        return int((perf_counter() - self.started_at) * 1000)


@dataclass(frozen=True, slots=True)
class AgentStageContext:
    """保存单个工作流阶段的 Hook 上下文。"""

    stage_name: str
    attempt_index: int
    retry_count: int
    started_at: float

    def elapsed_ms(self) -> int:
        """返回当前阶段已经运行的毫秒数。"""

        return int((perf_counter() - self.started_at) * 1000)


class AgentHook(Protocol):
    """定义 Agent 运行生命周期的同步 Hook 接口。"""

    def on_run_started(self, context: AgentRunContext) -> None:
        """在一次 Agent 运行开始时触发。"""

    def on_stage_started(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        input_summary: dict[str, object],
    ) -> None:
        """在某个工作流阶段开始时触发。"""

    def on_stage_completed(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        output_summary: dict[str, object],
    ) -> None:
        """在某个工作流阶段成功完成时触发。"""

    def on_stage_failed(
        self,
        run_context: AgentRunContext,
        stage_context: AgentStageContext,
        error_summary: dict[str, object],
    ) -> None:
        """在某个工作流阶段抛出异常时触发。"""

    def on_run_completed(
        self,
        context: AgentRunContext,
        result_summary: dict[str, object],
    ) -> None:
        """在一次 Agent 运行成功完成时触发。"""

    def on_run_failed(
        self,
        context: AgentRunContext,
        error_summary: dict[str, object],
    ) -> None:
        """在一次 Agent 运行失败时触发。"""


class HookManager:
    """按顺序调度多个 Hook，并隔离 Hook 自身失败。"""

    def __init__(
        self,
        run_context: AgentRunContext,
        hooks: list[AgentHook] | None = None,
    ) -> None:
        self.run_context = run_context
        self._hooks = hooks or []
        self._stage_attempts: dict[str, int] = {}

    def create_stage_context(
        self,
        stage_name: str,
        retry_count: int,
    ) -> AgentStageContext:
        """为一次阶段调用创建上下文，并计算同名阶段的第几次尝试。"""

        attempt_index = self._stage_attempts.get(stage_name, 0) + 1
        self._stage_attempts[stage_name] = attempt_index
        return AgentStageContext(
            stage_name=stage_name,
            attempt_index=attempt_index,
            retry_count=retry_count,
            started_at=perf_counter(),
        )

    def on_run_started(self) -> None:
        """通知所有 Hook：运行已经开始。"""

        self._call_hooks("on_run_started", self.run_context)

    def on_stage_started(
        self,
        stage_context: AgentStageContext,
        input_summary: dict[str, object],
    ) -> None:
        """通知所有 Hook：阶段已经开始。"""

        self._call_hooks(
            "on_stage_started",
            self.run_context,
            stage_context,
            input_summary,
        )

    def on_stage_completed(
        self,
        stage_context: AgentStageContext,
        output_summary: dict[str, object],
    ) -> None:
        """通知所有 Hook：阶段已经成功完成。"""

        self._call_hooks(
            "on_stage_completed",
            self.run_context,
            stage_context,
            output_summary,
        )

    def on_stage_failed(
        self,
        stage_context: AgentStageContext,
        error_summary: dict[str, object],
    ) -> None:
        """通知所有 Hook：阶段执行失败。"""

        self._call_hooks(
            "on_stage_failed",
            self.run_context,
            stage_context,
            error_summary,
        )

    def on_run_completed(
        self,
        result_summary: dict[str, object],
    ) -> None:
        """通知所有 Hook：运行已经成功完成。"""

        self._call_hooks(
            "on_run_completed",
            self.run_context,
            result_summary,
        )

    def on_run_failed(
        self,
        error_summary: dict[str, object],
    ) -> None:
        """通知所有 Hook：运行已经失败。"""

        self._call_hooks("on_run_failed", self.run_context, error_summary)

    def _call_hooks(self, method_name: str, *arguments: object) -> None:
        """逐个调用 Hook；Hook 异常只写日志，不影响 Agent 主流程。"""

        for hook in self._hooks:
            try:
                method = getattr(hook, method_name)
                method(*arguments)
            except Exception as error:
                LOGGER.warning(
                    "hook_failed hook=%s method=%s error_type=%s",
                    type(hook).__name__,
                    method_name,
                    type(error).__name__,
                )
