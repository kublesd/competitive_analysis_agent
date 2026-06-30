import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.planner import (
    FakePlannerModel,
    LangChainPlannerModel,
    Planner,
    PlannerError,
    PlannerInput,
    PlannerOutput,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "planner_outputs.json"


def _load_planner_outputs() -> dict:
    """读取固定模型输出，确保 Planner 测试不调用真实模型。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_planner_input() -> PlannerInput:
    """创建测试共用的两产品、两维度 Planner 输入。"""

    return PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        dimensions=["features", "pricing"],
    )


class FakeChatModel:
    """模拟 LangChain ChatModel 的 with_structured_output 接口。"""

    def __init__(self, structured_model: FakePlannerModel) -> None:
        self.structured_model = structured_model
        self.received_schema: type[PlannerOutput] | None = None
        self.received_method: str | None = None
        self.received_include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type[PlannerOutput],
        *,
        method: str,
        include_raw: bool,
    ) -> FakePlannerModel:
        self.received_schema = schema
        self.received_method = method
        self.received_include_raw = include_raw
        return self.structured_model


class FakeRawMessage:
    """模拟 LangChain 在解析失败时返回的原始 AIMessage。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FailingPlannerModel:
    """模拟真实模型客户端抛出包含敏感文本的异常。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """始终抛错，验证 Planner 的页面错误详情会脱敏。"""

        raise RuntimeError("secret-token-must-not-be-shown")


class PlannerTest(unittest.TestCase):
    def test_valid_output_covers_every_product_and_dimension(self) -> None:
        # 两个产品乘以两个维度，应得到四条独立调研任务。
        fixture = _load_planner_outputs()
        model = FakePlannerModel([fixture["valid"]])
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        task_pairs = {
            (task.product_name, task.topic) for task in tasks
        }
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            task_pairs,
            {
                ("Atlas Notes", "features"),
                ("Atlas Notes", "pricing"),
                ("Beacon Docs", "features"),
                ("Beacon Docs", "pricing"),
            },
        )
        self.assertEqual(model.invocation_count, 1)

    def test_missing_coverage_is_repaired_once(self) -> None:
        # 首次漏掉一个维度时，Planner 应反馈错误并接受一次修复。
        fixture = _load_planner_outputs()
        model = FakePlannerModel(
            [fixture["missing_coverage"], fixture["valid"]]
        )
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(model.invocation_count, 2)
        repair_message = model.received_messages[1][-1]["content"]
        self.assertIn("missing=Beacon Docs/pricing", repair_message)

    def test_invalid_output_stops_after_one_failed_repair(self) -> None:
        # 连续两次格式错误后立即停止，防止无限模型调用。
        fixture = _load_planner_outputs()
        model = FakePlannerModel(
            [fixture["invalid_shape"], fixture["invalid_shape"]]
        )
        planner = Planner(model)

        with self.assertRaises(PlannerError):
            planner.plan(_build_planner_input())

        self.assertEqual(model.invocation_count, 2)

    def test_langchain_wrapper_binds_planner_output_schema(self) -> None:
        # LangChain 模型必须先绑定 PlannerOutput，再执行结构化调用。
        fixture = _load_planner_outputs()
        structured_model = FakePlannerModel([fixture["valid"]])
        chat_model = FakeChatModel(structured_model)
        planner_model = LangChainPlannerModel(chat_model)
        planner = Planner(planner_model)

        tasks = planner.plan(_build_planner_input())

        self.assertIs(chat_model.received_schema, PlannerOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(structured_model.invocation_count, 1)

    def test_langchain_raw_parse_failure_enters_repair_flow(self) -> None:
        # LangChain 解析失败时，原始文本仍应交给 Planner 的一次修复逻辑。
        fixture = _load_planner_outputs()
        invalid_json = json.dumps(
            fixture["invalid_shape"],
            ensure_ascii=False,
        )
        model_responses = [
            {
                "raw": FakeRawMessage(invalid_json),
                "parsed": None,
                "parsing_error": ValueError("fixture parse failure"),
            },
            {
                "raw": FakeRawMessage(""),
                "parsed": PlannerOutput.model_validate(fixture["valid"]),
                "parsing_error": None,
            },
        ]
        structured_model = FakePlannerModel(model_responses)
        planner_model = LangChainPlannerModel(FakeChatModel(structured_model))
        planner = Planner(planner_model)

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(structured_model.invocation_count, 2)

    def test_model_call_failure_exposes_safe_public_detail(self) -> None:
        # 供应商异常可能包含敏感文本，页面详情只能展示脱敏后的定位信息。
        planner = Planner(FailingPlannerModel())

        with self.assertRaises(PlannerError) as captured_error:
            planner.plan(_build_planner_input())

        public_detail = captured_error.exception.public_detail
        self.assertIn("Planner 调用模型服务失败", public_detail)
        self.assertIn("底层异常类型：RuntimeError", public_detail)
        self.assertIn("2 个产品、2 个维度", public_detail)
        self.assertNotIn("secret-token-must-not-be-shown", public_detail)

    def test_duplicate_dimensions_are_rejected_before_model_call(self) -> None:
        # 重复维度会制造重复任务，应在进入模型前直接拒绝。
        with self.assertRaises(ValidationError):
            PlannerInput(
                target_product="Atlas Notes",
                competitors=["Beacon Docs"],
                dimensions=["pricing", "pricing"],
            )


if __name__ == "__main__":
    unittest.main()
