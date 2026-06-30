"""配置竞品分析应用的控制台日志与本地轮转文件。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


APPLICATION_LOGGER_NAME = "competitive_analysis_agent"
AGENT_EVENT_LOGGER_NAME = "competitive_analysis_agent.agent_events"
DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_LOG_FILE_NAME = "application.log"
DEFAULT_AGENT_EVENT_LOG_FILE_NAME = "agent-events.jsonl"
DEFAULT_MAX_LOG_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3
LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)
JSONL_LOG_FORMAT = "%(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_application_logging(
    log_directory: Path | None = None,
    *,
    log_level: int = logging.INFO,
    include_console: bool = True,
) -> Path:
    """配置应用日志并返回当前日志文件路径。

    重复调用不会为同一文件或控制台增加重复 Handler。日志目录无法创建时会明确
    抛出异常，让启动入口显示真实配置问题，而不是静默丢失后台日志。
    """

    current_log_directory = log_directory or DEFAULT_LOG_DIRECTORY
    current_log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = (
        current_log_directory / DEFAULT_LOG_FILE_NAME
    ).resolve()
    agent_event_log_file_path = get_agent_event_log_path(
        current_log_directory
    )

    application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    application_logger.setLevel(log_level)
    application_logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )
    file_handler_added = _add_file_handler_if_missing(
        application_logger,
        log_file_path,
        formatter,
        log_level,
    )
    if include_console:
        _add_console_handler_if_missing(
            application_logger,
            formatter,
            log_level,
        )

    _configure_agent_event_logger(
        agent_event_log_file_path,
        log_level,
    )

    if file_handler_added:
        application_logger.info(
            "application_logging_configured log_file=%s "
            "agent_event_log_file=%s level=%s",
            log_file_path,
            agent_event_log_file_path,
            logging.getLevelName(log_level),
        )

    return log_file_path


def get_agent_event_log_path(log_directory: Path | None = None) -> Path:
    """返回结构化 Agent 事件 JSONL 文件路径。"""

    current_log_directory = log_directory or DEFAULT_LOG_DIRECTORY
    return (
        current_log_directory / DEFAULT_AGENT_EVENT_LOG_FILE_NAME
    ).resolve()


def _add_file_handler_if_missing(
    logger: logging.Logger,
    log_file_path: Path,
    formatter: logging.Formatter,
    log_level: int,
) -> bool:
    """按绝对路径识别已有文件 Handler，避免 Streamlit 重跑后重复写日志。"""

    normalized_path = str(log_file_path)
    for handler in logger.handlers:
        if getattr(handler, "_application_log_path", None) == normalized_path:
            handler.setLevel(log_level)
            return False

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=DEFAULT_MAX_LOG_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler._application_log_path = normalized_path
    logger.addHandler(file_handler)
    return True


def _configure_agent_event_logger(
    agent_event_log_file_path: Path,
    log_level: int,
) -> None:
    """配置专门写 JSONL 的 Agent 事件 Logger。"""

    agent_event_logger = logging.getLogger(AGENT_EVENT_LOGGER_NAME)
    agent_event_logger.setLevel(log_level)
    agent_event_logger.propagate = False

    formatter = logging.Formatter(JSONL_LOG_FORMAT)
    _add_agent_event_handler_if_missing(
        agent_event_logger,
        agent_event_log_file_path,
        formatter,
        log_level,
    )


def _add_agent_event_handler_if_missing(
    logger: logging.Logger,
    log_file_path: Path,
    formatter: logging.Formatter,
    log_level: int,
) -> None:
    """按绝对路径识别已有 JSONL Handler，避免重复写事件。"""

    normalized_path = str(log_file_path)
    for handler in logger.handlers:
        if getattr(handler, "_agent_event_log_path", None) == normalized_path:
            handler.setLevel(log_level)
            return

    event_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=DEFAULT_MAX_LOG_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    event_handler.setLevel(log_level)
    event_handler.setFormatter(formatter)
    event_handler._agent_event_log_path = normalized_path
    logger.addHandler(event_handler)


def _add_console_handler_if_missing(
    logger: logging.Logger,
    formatter: logging.Formatter,
    log_level: int,
) -> None:
    """增加一个控制台 Handler，让双击启动窗口也能看到后台事件。"""

    for handler in logger.handlers:
        if getattr(handler, "_application_console_handler", False):
            handler.setLevel(log_level)
            return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler._application_console_handler = True
    logger.addHandler(console_handler)
