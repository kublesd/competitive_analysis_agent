"""记录真实模型边界的完整输入、输出和错误，便于后台排查。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from competitive_analysis_agent.logging_config import MODEL_IO_LOGGER_NAME


LOGGER = logging.getLogger(MODEL_IO_LOGGER_NAME)
MODEL_IO_SCHEMA_VERSION = 1
_MODEL_IO_CONTEXT: ContextVar[dict[str, object]] = ContextVar(
    "model_io_context",
    default={},
)


@contextmanager
def model_io_context(
    *,
    analysis_id: str,
    entrypoint: str,
    stage: str,
    attempt_index: int,
    retry_count: int,
) -> Iterator[None]:
    """为当前节点内的模型调用补充 analysis_id 和阶段上下文。"""

    context = {
        "analysis_id": analysis_id,
        "entrypoint": entrypoint,
        "stage": stage,
        "attempt_index": attempt_index,
        "retry_count": retry_count,
    }
    token = _MODEL_IO_CONTEXT.set(context)
    try:
        yield
    finally:
        _MODEL_IO_CONTEXT.reset(token)


def log_model_request(
    component: str,
    messages: Sequence[Mapping[str, str]],
) -> str:
    """写入一次模型请求，并返回可关联响应和错误的调用 ID。"""

    call_id = uuid4().hex[:12]
    event = build_base_event(
        event_type="model_request",
        component=component,
        call_id=call_id,
    )
    serialized_messages = serialize_messages(messages)
    event["message_count"] = len(serialized_messages)
    event["input_chars"] = count_message_chars(serialized_messages)
    event["messages"] = serialized_messages
    write_model_io_event(event)
    return call_id


def log_model_response(
    component: str,
    call_id: str,
    response: object,
) -> None:
    """写入一次模型响应，保留结构化输出和 raw content。"""

    event = build_base_event(
        event_type="model_response",
        component=component,
        call_id=call_id,
    )
    event["response"] = serialize_model_value(response)
    write_model_io_event(event)


def log_model_error(
    component: str,
    call_id: str,
    error: Exception,
) -> None:
    """写入一次模型调用错误；只记录异常类型，不保存供应商错误原文。"""

    event = build_base_event(
        event_type="model_error",
        component=component,
        call_id=call_id,
    )
    event["error_type"] = type(error).__name__
    write_model_io_event(event)


def build_base_event(
    *,
    event_type: str,
    component: str,
    call_id: str,
) -> dict[str, object]:
    """构造模型 I/O 事件的公共字段。"""

    context = _MODEL_IO_CONTEXT.get()
    event = {
        "schema_version": MODEL_IO_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "component": component,
        "call_id": call_id,
        "analysis_id": context.get("analysis_id"),
        "entrypoint": context.get("entrypoint"),
        "stage": context.get("stage"),
        "attempt_index": context.get("attempt_index"),
        "retry_count": context.get("retry_count"),
    }
    return event


def serialize_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """复制模型 messages，避免后续调用方修改影响日志内容。"""

    serialized_messages: list[dict[str, str]] = []
    for message in messages:
        serialized_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    return serialized_messages


def count_message_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """统计消息正文字符数，帮助定位超时或上下文过大的调用。"""

    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("content", "")))
    return total_chars


def serialize_model_value(value: object) -> object:
    """把 LangChain/Pydantic 输出转换成可写入 JSONL 的结构。"""

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): serialize_model_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [serialize_model_value(item) for item in value]

    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": type(value).__name__,
            "content": serialize_model_value(content),
        }

    return {
        "type": type(value).__name__,
        "text": str(value),
    }


def write_model_io_event(event: dict[str, Any]) -> None:
    """把模型 I/O 事件写入 JSONL；失败时不影响主流程。"""

    try:
        event_json = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        LOGGER.info(event_json)
    except Exception as error:
        logging.getLogger(__name__).warning(
            "model_io_log_failed event_type=%s error_type=%s",
            event.get("event_type"),
            type(error).__name__,
        )
