"""Planner 节点：把分析目标拆分成结构化调研任务。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from competitive_analysis_agent.schemas import (
    ContractModel,
    ResearchTask,
    RequiredText,
)


PLANNER_SYSTEM_PROMPT = """
你是竞品分析流程中的 Planner。

你的唯一职责是把产品和分析维度拆成搜索任务，不要搜索网页，也不要回答产品事实。

要求：
1. 目标产品和每个竞品都必须覆盖全部分析维度。
2. 每条任务只包含一个产品和一个 topic。
3. topic 必须直接使用输入中的分析维度。
4. query 应包含产品名称、topic 和 official，便于后续搜索官方资料。
5. 不得添加输入中不存在的产品或分析维度。
6. product_name 和 topic 必须逐字复制输入中的值，不得翻译或改写。
7. 任务数量必须等于产品数量乘以分析维度数量。
8. 只输出 JSON 对象，不要添加 Markdown 或解释。
9. JSON 格式必须是：
   {"tasks": [{"product_name": "...", "topic": "...", "query": "..."}]}
""".strip()


class PlannerInput(ContractModel):
    """保存 Planner 的用户输入，并拒绝重复产品或维度。"""

    target_product: RequiredText
    competitors: list[RequiredText] = Field(min_length=1)
    dimensions: list[RequiredText] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_values(self) -> "PlannerInput":
        """确保任务矩阵不会因重复输入产生含义不清的任务。"""

        products = [self.target_product, *self.competitors]
        if len(products) != len(set(products)):
            raise ValueError("Target product and competitors must be unique.")

        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("Dimensions must be unique.")

        return self

    @property
    def products(self) -> list[str]:
        """按目标产品优先的顺序返回全部待调研产品。"""

        return [self.target_product, *self.competitors]


class PlannerOutput(ContractModel):
    """约束结构化模型必须返回非空调研任务列表。"""

    tasks: list[ResearchTask] = Field(min_length=1)


class PlannerModel(Protocol):
    """约定 Planner 所需的最小结构化模型调用接口。"""

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """根据消息返回可被 PlannerOutput 校验的对象。"""


class StructuredChatModel(Protocol):
    """描述 LangChain ChatModel 的结构化输出能力。"""

    def with_structured_output(
        self,
        schema: type[PlannerOutput],
        *,
        method: Literal["json_mode"],
        include_raw: Literal[True],
    ) -> PlannerModel:
        """绑定 Pydantic 输出 Schema，并返回可调用对象。"""


class LangChainPlannerModel:
    """把 LangChain ChatModel 包装成 Planner 所需的模型接口。"""

    def __init__(self, chat_model: StructuredChatModel) -> None:
        # 硅基流动官方支持 json_object，因此显式使用 LangChain JSON mode。
        self._structured_model = chat_model.with_structured_output(
            PlannerOutput,
            method="json_mode",
            include_raw=True,
        )

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """执行调用，并在解析失败时保留模型原始文本供 Planner 修复。"""

        structured_response = self._structured_model.invoke(messages)
        if not isinstance(structured_response, dict):
            return structured_response

        response_wrapper_keys = {"raw", "parsed", "parsing_error"}
        if not response_wrapper_keys.intersection(structured_response):
            return structured_response

        parsed_output = structured_response.get("parsed")
        if parsed_output is not None:
            return parsed_output

        raw_message = structured_response.get("raw")
        raw_content = getattr(raw_message, "content", raw_message)
        return raw_content


class FakePlannerModel:
    """按顺序返回固定响应，用于无 API Key 的确定性测试。"""

    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.invocation_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]]) -> object:
        """返回下一条固定响应，并记录调用次数和消息。"""

        copied_messages = [message.copy() for message in messages]
        self.received_messages.append(copied_messages)

        if self.invocation_count >= len(self._responses):
            raise RuntimeError("Fake planner has no response left.")

        response = self._responses[self.invocation_count]
        self.invocation_count += 1
        return response


class PlannerError(RuntimeError):
    """表示 Planner 调用失败或有限修复后仍无法生成有效任务。"""

    def __init__(
        self,
        message: str,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        # public_detail 会显示到页面，因此只保存脱敏后的定位信息。
        self.public_detail = public_detail or message


class PlannerValidationError(ValueError):
    """表示模型输出格式错误或任务覆盖不完整。"""


class Planner:
    """生成、校验并在必要时修复一次调研任务计划。"""

    def __init__(self, model: PlannerModel) -> None:
        self._model = model

    def plan(self, planner_input: PlannerInput) -> list[ResearchTask]:
        """生成任务列表；首次校验失败时最多请求一次修复。"""

        initial_messages = build_planner_messages(planner_input)
        raw_output = self._invoke_model(
            messages=initial_messages,
            planner_input=planner_input,
        )

        try:
            validated_output = validate_planner_output(
                raw_output,
                planner_input,
            )
            return validated_output.tasks
        except PlannerValidationError as first_error:
            # 只修复一次，避免错误输出导致无限调用和不可控成本。
            repair_messages = build_repair_messages(
                initial_messages=initial_messages,
                raw_output=raw_output,
                validation_error=str(first_error),
            )
            repaired_output = self._invoke_model(
                messages=repair_messages,
                planner_input=planner_input,
            )

        try:
            validated_repair = validate_planner_output(
                repaired_output,
                planner_input,
            )
            return validated_repair.tasks
        except PlannerValidationError as second_error:
            raise PlannerError(
                "Planner output remained invalid after one repair: "
                f"{second_error}"
            ) from second_error

    def _invoke_model(
        self,
        messages: list[dict[str, str]],
        planner_input: PlannerInput,
    ) -> object:
        """调用模型，并把供应商异常转换成 PlannerError。"""

        try:
            return self._model.invoke(messages)
        except Exception as error:
            public_detail = build_model_call_failure_detail(
                planner_input=planner_input,
                error=error,
            )
            raise PlannerError(
                f"Planner model call failed: {error}",
                public_detail=public_detail,
            ) from error


def build_model_call_failure_detail(
    planner_input: PlannerInput,
    error: Exception,
) -> str:
    """生成可展示到 UI 的模型失败摘要，不包含供应商原始错误文本。"""

    product_count = len(planner_input.products)
    dimension_count = len(planner_input.dimensions)
    expected_task_count = product_count * dimension_count
    error_type = type(error).__name__
    error_text = str(error).lower()

    if "401" in error_text or "invalid token" in error_text:
        reason = (
            "模型服务认证失败，通常是 LLM_API_KEY 无效、过期，"
            "或应用仍在使用旧的环境变量值。"
        )
        next_step = "请更新 .env.example 中的 LLM_API_KEY 后重新启动应用。"
    else:
        reason = (
            "模型服务调用失败，可能是网络、超时、额度、模型名称或接口兼容性问题。"
        )
        next_step = (
            "请检查 LLM_BASE_URL、LLM_MODEL、额度和网络连通性后重试。"
        )

    return (
        "Planner 调用模型服务失败。"
        f"底层异常类型：{error_type}。"
        f"{reason}"
        f"本次输入包含 {product_count} 个产品、{dimension_count} 个维度，"
        f"预计生成 {expected_task_count} 条调研任务。"
        f"{next_step}"
    )


def build_planner_messages(
    planner_input: PlannerInput,
) -> list[dict[str, str]]:
    """把结构化输入转换成模型可读取的 system 和 user 消息。"""

    input_json = json.dumps(
        planner_input.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    expected_task_count = (
        len(planner_input.products) * len(planner_input.dimensions)
    )
    user_message = (
        "请根据以下输入生成调研任务。\n"
        f"必须生成恰好 {expected_task_count} 条任务。\n\n"
        f"{input_json}"
    )

    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_repair_messages(
    initial_messages: list[dict[str, str]],
    raw_output: object,
    validation_error: str,
) -> list[dict[str, str]]:
    """把校验错误反馈给模型，要求只修复结构化任务。"""

    repair_messages = [message.copy() for message in initial_messages]
    repair_instruction = (
        "上一次输出没有通过校验，请只修复输出，不要执行搜索。\n"
        f"校验错误：{validation_error}\n"
        f"上一次输出：{raw_output!r}"
    )
    repair_messages.append(
        {"role": "user", "content": repair_instruction}
    )
    return repair_messages


def validate_planner_output(
    raw_output: object,
    planner_input: PlannerInput,
) -> PlannerOutput:
    """校验模型输出结构及产品与维度覆盖范围。"""

    try:
        if isinstance(raw_output, str):
            planner_output = PlannerOutput.model_validate_json(raw_output)
        else:
            planner_output = PlannerOutput.model_validate(raw_output)
    except ValidationError as error:
        raise PlannerValidationError(
            f"Output does not match PlannerOutput: {error}"
        ) from error

    validate_task_coverage(planner_output.tasks, planner_input)
    return planner_output


def validate_task_coverage(
    tasks: Sequence[ResearchTask],
    planner_input: PlannerInput,
) -> None:
    """确保每个产品与维度恰好对应一条任务。"""

    expected_pairs: set[tuple[str, str]] = set()
    for product_name in planner_input.products:
        for dimension in planner_input.dimensions:
            expected_pairs.add((product_name, dimension))

    actual_pairs: list[tuple[str, str]] = []
    for task in tasks:
        actual_pairs.append((task.product_name, task.topic))

    unique_actual_pairs = set(actual_pairs)
    duplicate_pairs = _find_duplicate_pairs(actual_pairs)
    missing_pairs = expected_pairs - unique_actual_pairs
    unexpected_pairs = unique_actual_pairs - expected_pairs

    error_parts: list[str] = []
    if missing_pairs:
        error_parts.append(
            f"missing={_format_pairs(missing_pairs)}"
        )
    if unexpected_pairs:
        error_parts.append(
            f"unexpected={_format_pairs(unexpected_pairs)}"
        )
    if duplicate_pairs:
        error_parts.append(
            f"duplicates={_format_pairs(duplicate_pairs)}"
        )

    if error_parts:
        error_summary = "; ".join(error_parts)
        raise PlannerValidationError(
            f"Task coverage is invalid: {error_summary}"
        )


def _find_duplicate_pairs(
    pairs: Sequence[tuple[str, str]],
) -> set[tuple[str, str]]:
    """找出重复出现的产品与维度组合。"""

    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs: set[tuple[str, str]] = set()

    for pair in pairs:
        if pair in seen_pairs:
            duplicate_pairs.add(pair)
        seen_pairs.add(pair)

    return duplicate_pairs


def _format_pairs(pairs: set[tuple[str, str]]) -> str:
    """以稳定顺序格式化组合，便于测试和修复提示阅读。"""

    sorted_pairs = sorted(pairs)
    formatted_pairs: list[str] = []
    for product_name, dimension in sorted_pairs:
        formatted_pairs.append(f"{product_name}/{dimension}")

    return ", ".join(formatted_pairs)
