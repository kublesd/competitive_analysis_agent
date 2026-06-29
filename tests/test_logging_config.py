import logging
import tempfile
import unittest
from pathlib import Path

from competitive_analysis_agent.logging_config import (
    APPLICATION_LOGGER_NAME,
    configure_application_logging,
)


class LoggingConfigTest(unittest.TestCase):
    def tearDown(self) -> None:
        """关闭测试添加的 Handler，避免 Windows 文件锁影响临时目录清理。"""

        logger = logging.getLogger(APPLICATION_LOGGER_NAME)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_configure_writes_log_file_without_duplicate_handlers(self) -> None:
        # Streamlit 会重复执行脚本，同一路径只能绑定一个文件 Handler。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        log_directory = Path(temporary_directory.name)

        log_path = configure_application_logging(
            log_directory,
            include_console=False,
        )
        configure_application_logging(
            log_directory,
            include_console=False,
        )
        logger = logging.getLogger(
            "competitive_analysis_agent.test"
        )
        logger.info("analysis_completed analysis_id=test123")

        file_handlers = [
            handler
            for handler in logging.getLogger(
                APPLICATION_LOGGER_NAME
            ).handlers
            if getattr(handler, "_application_log_path", None)
        ]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual(file_handlers[0].maxBytes, 5 * 1024 * 1024)
        self.assertEqual(file_handlers[0].backupCount, 3)
        self.assertTrue(log_path.is_file())
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("analysis_id=test123", log_text)

    def test_log_configuration_does_not_record_secret_values(self) -> None:
        # 配置日志本身只记录文件位置和级别，不读取或输出 API Key。
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        log_path = configure_application_logging(
            Path(temporary_directory.name),
            include_console=False,
        )

        log_text = log_path.read_text(encoding="utf-8")
        self.assertNotIn("api_key", log_text.lower())
        self.assertNotIn("authorization", log_text.lower())


if __name__ == "__main__":
    unittest.main()
