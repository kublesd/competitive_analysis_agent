import json
import unittest
from datetime import datetime, timezone
from time import perf_counter

from competitive_analysis_agent.agent_hooks import AgentRunContext
from competitive_analysis_agent.logging_config import AGENT_EVENT_LOGGER_NAME
from competitive_analysis_agent.observability import (
    JsonlLoggingHook,
    StagePayloadSummarizer,
)
from competitive_analysis_agent.schemas import Evidence


class ObservabilityTest(unittest.TestCase):
    def test_jsonl_hook_writes_fixed_event_fields(self) -> None:
        # JSONL Hook 输出固定字段，后续可以稳定按 analysis_id 查询。
        hook = JsonlLoggingHook()
        context = AgentRunContext(
            analysis_id="run123",
            entrypoint="test",
            started_at=perf_counter(),
            configuration_summary={"dimension_count": 1},
        )

        with self.assertLogs(
            AGENT_EVENT_LOGGER_NAME,
            level="INFO",
        ) as captured_logs:
            hook.on_run_started(context)

        event_json = captured_logs.output[0].split(":", maxsplit=2)[2]
        event = json.loads(event_json)
        self.assertEqual(
            set(event),
            {
                "schema_version",
                "timestamp",
                "event_type",
                "analysis_id",
                "entrypoint",
                "stage",
                "status",
                "duration_ms",
                "summary",
            },
        )
        self.assertEqual(event["event_type"], "run_started")
        self.assertEqual(event["analysis_id"], "run123")
        self.assertEqual(event["entrypoint"], "test")

    def test_stage_summaries_exclude_sensitive_content(self) -> None:
        # 摘要只记录数量和 ID，不保存 Evidence 正文、报告正文或官方域名原文。
        collected_at = datetime(2026, 6, 30, 8, 0, tzinfo=timezone.utc)
        evidence = [
            Evidence(
                evidence_id="E1",
                product_name="Atlas Notes",
                topic="features",
                title="Atlas feature page",
                url="https://secret.example.com/features",
                snippet="secret-snippet-must-not-be-logged",
                raw_content="secret-raw-content-must-not-be-logged",
                source_type="official",
                collected_at=collected_at,
            )
        ]
        state = {
            "target_product": "Atlas Notes",
            "competitors": ["Beacon Docs"],
            "dimensions": ["features"],
            "official_domains_by_product": {
                "Atlas Notes": ["secret.example.com"],
            },
            "max_results_per_task": 3,
            "research_tasks": [],
            "evidence": evidence,
            "research_errors": [],
            "product_profiles": [],
            "analysis_result": None,
            "verification_result": None,
            "final_report": (
                "# Report\n\nsecret-final-report-body-must-not-be-logged"
            ),
            "retry_count": 0,
            "retry_pending": False,
            "stage_history": [],
        }
        summarizer = StagePayloadSummarizer()

        researcher_input = summarizer.build_stage_input_summary(
            "researcher",
            state,
        )
        extractor_input = summarizer.build_stage_input_summary(
            "extractor",
            state,
        )
        reporter_output = summarizer.build_stage_output_summary(
            "reporter",
            state,
        )
        summary_text = json.dumps(
            [researcher_input, extractor_input, reporter_output],
            ensure_ascii=False,
        )

        self.assertNotIn("secret.example.com", summary_text)
        self.assertNotIn("secret-snippet-must-not-be-logged", summary_text)
        self.assertNotIn("secret-raw-content-must-not-be-logged", summary_text)
        self.assertNotIn("secret-final-report-body-must-not-be-logged", summary_text)
        self.assertNotIn("raw_content", summary_text)
        self.assertIn("source_content_chars", summary_text)
        self.assertIn("final_report_chars", summary_text)


if __name__ == "__main__":
    unittest.main()
