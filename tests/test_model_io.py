import json
import logging
import tempfile
import unittest
from pathlib import Path

from competitive_analysis_agent.logging_config import (
    AGENT_EVENT_LOGGER_NAME,
    APPLICATION_LOGGER_NAME,
    MODEL_IO_LOGGER_NAME,
    configure_application_logging,
    get_model_io_log_path,
)
from competitive_analysis_agent.model_io import (
    log_model_error,
    log_model_request,
    log_model_response,
    model_io_context,
)
from competitive_analysis_agent.planner import PlannerOutput


class ModelIoLoggingTest(unittest.TestCase):
    def tearDown(self) -> None:
        """关闭测试添加的 Handler，避免 Windows 文件锁影响清理。"""

        for logger_name in (
            APPLICATION_LOGGER_NAME,
            AGENT_EVENT_LOGGER_NAME,
            MODEL_IO_LOGGER_NAME,
        ):
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def test_model_io_log_records_context_messages_response_and_error(
        self,
    ) -> None:
        # 模型 I/O 日志需要保留完整 messages 和结构化响应，方便后台排查。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        log_directory = Path(temporary_directory.name)
        configure_application_logging(
            log_directory,
            include_console=False,
        )

        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]
        response = {
            "parsed": PlannerOutput.model_validate(
                {
                    "tasks": [
                        {
                            "product_name": "Atlas Notes",
                            "topic": "features",
                            "query": "Atlas Notes official features",
                        }
                    ]
                }
            )
        }
        with model_io_context(
            analysis_id="analysis123",
            entrypoint="streamlit",
            stage="planner",
            attempt_index=1,
            retry_count=0,
        ):
            call_id = log_model_request("Planner", messages)
            log_model_response("Planner", call_id, response)
            log_model_error("Planner", call_id, TimeoutError("secret text"))

        log_lines = get_model_io_log_path(log_directory).read_text(
            encoding="utf-8"
        ).splitlines()
        events = [json.loads(line) for line in log_lines]

        self.assertEqual(
            [event["event_type"] for event in events],
            ["model_request", "model_response", "model_error"],
        )
        self.assertEqual(events[0]["analysis_id"], "analysis123")
        self.assertEqual(events[0]["stage"], "planner")
        self.assertEqual(events[0]["component"], "Planner")
        self.assertEqual(events[0]["messages"], messages)
        self.assertEqual(events[0]["input_chars"], 24)
        self.assertEqual(events[1]["call_id"], call_id)
        self.assertEqual(
            events[1]["response"]["parsed"]["tasks"][0]["product_name"],
            "Atlas Notes",
        )
        self.assertEqual(events[2]["error_type"], "TimeoutError")
        self.assertNotIn("secret text", json.dumps(events[2]))


if __name__ == "__main__":
    unittest.main()
