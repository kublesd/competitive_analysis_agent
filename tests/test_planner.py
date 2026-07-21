import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from competitive_analysis_agent.planner import (
    FakePlannerModel,
    LangChainPlannerModel,
    Planner,
    PlannerInput,
    PlannerOutput,
)
from competitive_analysis_agent.schemas import MarketDefinition


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "planner_outputs.json"


def _load_planner_outputs() -> dict:
    """读取固定模型输出，确保 Planner 测试不调用真实模型。"""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_planner_input() -> PlannerInput:
    """创建测试共用的两产品、两维度 Planner 输入。"""

    return PlannerInput(
        target_product="Atlas Notes",
        competitors=["Beacon Docs"],
        market_definition=MarketDefinition(
            market_name="团队知识管理工具",
            product_category="SaaS 协作软件",
            target_buyer="中型企业 IT 与业务负责人",
            comparison_level="企业订阅产品",
            core_dimensions=["features", "pricing"],
            exclusions=["消费端套餐", "API 用量价格"],
        ),
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
    def test_deterministic_planner_does_not_call_model(self) -> None:
        # 任务矩阵完全由已校验输入决定，不应再受模型格式或可用性影响。
        tasks = Planner(FailingPlannerModel()).plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            tasks[0].query,
            (
                "Atlas Notes SaaS 协作软件 企业订阅产品 features official "
                "exclude 消费端套餐 exclude API 用量价格"
            ),
        )

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
        self.assertEqual(model.invocation_count, 0)

    def test_model_missing_coverage_cannot_change_deterministic_plan(self) -> None:
        # 旧模型即使漏维度，也不会再影响确定性任务矩阵。
        fixture = _load_planner_outputs()
        model = FakePlannerModel(
            [fixture["missing_coverage"], fixture["valid"]]
        )
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(model.invocation_count, 0)
        self.assertIn(
            ("Beacon Docs", "pricing"),
            {(task.product_name, task.topic) for task in tasks},
        )

    def test_deterministic_queries_always_include_market_scope(self) -> None:
        # 查询范围直接由结构化输入拼接，不依赖模型逐字复制。
        fixture = _load_planner_outputs()
        model = FakePlannerModel(
            [fixture["missing_query_scope"], fixture["valid"]]
        )
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(model.invocation_count, 0)
        self.assertTrue(
            all("企业订阅产品" in task.query for task in tasks)
        )

    def test_queries_contain_market_scope_and_exclusions(self) -> None:
        # 市场定义必须逐条进入查询，供 Researcher 后续聚焦和审计。
        fixture = _load_planner_outputs()
        model = FakePlannerModel([fixture["valid"]])
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        for task in tasks:
            self.assertIn("SaaS 协作软件", task.query)
            self.assertIn("企业订阅产品", task.query)
            self.assertIn("exclude 消费端套餐", task.query)
            self.assertIn("exclude API 用量价格", task.query)

    def test_invalid_model_output_is_not_consulted(self) -> None:
        # 确定性 Planner 不再因模型 JSON 格式错误阻塞工作流。
        fixture = _load_planner_outputs()
        model = FakePlannerModel(
            [fixture["invalid_shape"], fixture["invalid_shape"]]
        )
        planner = Planner(model)

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)
        self.assertEqual(model.invocation_count, 0)

    def test_langchain_wrapper_binds_planner_output_schema(self) -> None:
        # LangChain 模型必须先绑定 PlannerOutput，再执行结构化调用。
        fixture = _load_planner_outputs()
        structured_model = FakePlannerModel([fixture["valid"]])
        chat_model = FakeChatModel(structured_model)
        planner_model = LangChainPlannerModel(chat_model)
        raw_output = planner_model.invoke([])
        tasks = PlannerOutput.model_validate(raw_output).tasks

        self.assertIs(chat_model.received_schema, PlannerOutput)
        self.assertEqual(chat_model.received_method, "json_mode")
        self.assertTrue(chat_model.received_include_raw)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(structured_model.invocation_count, 1)

    def test_legacy_langchain_wrapper_preserves_raw_parse_failure(self) -> None:
        # 旧包装器仍保留原始文本，兼容已有独立调用方。
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
        raw_output = planner_model.invoke([])

        self.assertEqual(raw_output, invalid_json)
        self.assertEqual(structured_model.invocation_count, 1)

    def test_model_failure_cannot_block_planning(self) -> None:
        # Planner 不调用供应商，因此供应商不可用也能继续进入 Researcher。
        planner = Planner(FailingPlannerModel())

        tasks = planner.plan(_build_planner_input())

        self.assertEqual(len(tasks), 4)

    def test_duplicate_dimensions_are_rejected_before_model_call(self) -> None:
        # 重复维度会制造重复任务，应在进入模型前直接拒绝。
        with self.assertRaises(ValidationError):
            PlannerInput(
                target_product="Atlas Notes",
                competitors=["Beacon Docs"],
                market_definition=MarketDefinition(
                    market_name="团队知识管理工具",
                    product_category="SaaS 协作软件",
                    comparison_level="企业订阅产品",
                    core_dimensions=["pricing", "pricing"],
                ),
            )


if __name__ == "__main__":
    unittest.main()
