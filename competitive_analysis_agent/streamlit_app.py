"""Streamlit 页面：收集竞品输入并展示 LangGraph 分析报告。"""

from __future__ import annotations

import logging

import streamlit as st

from competitive_analysis_agent import ui_service
from competitive_analysis_agent.logging_config import (
    configure_application_logging,
)


SESSION_DEFAULTS = {
    "final_report": None,
    "stage_history": [],
    "evidence": [],
    "verification_passed": None,
    "research_error_count": 0,
    "error_message": None,
}
LOGGER = logging.getLogger(__name__)


def initialize_session_state() -> None:
    """初始化当前浏览器会话中的报告、状态和错误字段。"""

    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_previous_result() -> None:
    """提交新任务前清除旧结果，避免用户误读上一次报告。"""

    for key, default_value in SESSION_DEFAULTS.items():
        st.session_state[key] = default_value


def render_sources() -> None:
    """在报告下方单独展示可点击来源，方便快速复核。"""

    evidence = st.session_state["evidence"]
    if not evidence:
        return

    with st.expander("查看资料来源", expanded=False):
        for item in evidence:
            st.markdown(
                f"- **{item.evidence_id}** "
                f"[{item.title}]({str(item.url)}) "
                f"({item.product_name} / {item.topic})"
            )


def render_saved_result() -> None:
    """渲染 Session State 中保存的错误、轨迹、报告和下载按钮。"""

    error_message = st.session_state["error_message"]
    if error_message:
        st.error(error_message)

    stage_history = st.session_state["stage_history"]
    if stage_history:
        st.subheader("执行状态")
        st.caption(ui_service.build_stage_summary(stage_history))

    final_report = st.session_state["final_report"]
    if final_report is None:
        return

    if st.session_state["research_error_count"]:
        st.warning(
            "部分研究任务未完成，报告中的“数据限制”章节已保留详情。"
        )

    if st.session_state["verification_passed"] is False:
        st.warning("最终验证未通过，请重点查看报告顶部的验证警告。")

    st.subheader("分析报告")
    st.markdown(final_report)
    render_sources()
    st.download_button(
        label="下载 Markdown 报告",
        data=final_report,
        file_name="competitive-analysis-report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def run_submitted_analysis(
    target_product: str,
    competitors_text: str,
    dimensions: list[str],
    official_domains_text: str,
) -> None:
    """处理一次表单提交，并把成功或失败结果写入 Session State。"""

    clear_previous_result()
    status_container = st.status("准备运行分析...", expanded=True)

    def update_progress(stage_name: str) -> None:
        """把 LangGraph 节点事件显示为用户可理解的阶段状态。"""

        stage_label = ui_service.STAGE_LABELS.get(
            stage_name,
            stage_name,
        )
        status_container.write(stage_label)

    try:
        analysis_request = ui_service.create_analysis_request(
            target_product=target_product,
            competitors_text=competitors_text,
            dimensions=dimensions,
            official_domains_text=official_domains_text,
        )
        result = ui_service.run_analysis(
            analysis_request,
            progress_callback=update_progress,
            entrypoint="streamlit",
        )
    except Exception as error:
        LOGGER.warning(
            "analysis_submission_failed error_type=%s",
            type(error).__name__,
        )
        st.session_state["error_message"] = (
            ui_service.describe_user_error(error)
        )
        status_container.update(
            label="分析未完成",
            state="error",
            expanded=True,
        )
        return

    st.session_state["final_report"] = result.final_report
    st.session_state["stage_history"] = result.stage_history
    st.session_state["evidence"] = result.evidence
    st.session_state["verification_passed"] = (
        result.verification_result.passed
    )
    st.session_state["research_error_count"] = len(
        result.research_errors
    )
    status_container.update(
        label="分析完成",
        state="complete",
        expanded=False,
    )


def main() -> None:
    """渲染最小可用的竞品分析页面。"""

    configure_application_logging()
    st.set_page_config(
        page_title="AI 竞品分析 Agent",
        layout="wide",
    )
    initialize_session_state()

    st.title("AI 竞品分析 Agent")
    st.write(
        "输入分析目标后，页面会运行 Planner、Researcher、Extractor、"
        "Analyst、Verifier 和 Reporter。"
    )
    st.info(
        "当前使用真实模型和 Tavily 搜索。为提高来源可靠性，"
        "建议为每个产品填写官方域名。"
    )

    with st.form("analysis_form"):
        target_product = st.text_input(
            "目标产品",
            value=ui_service.DEFAULT_TARGET_PRODUCT,
        )
        competitors_text = st.text_area(
            "竞品（每行一个，也可使用逗号分隔）",
            value="\n".join(ui_service.DEFAULT_COMPETITORS),
        )
        selected_dimensions = st.multiselect(
            "常用分析维度",
            options=ui_service.AVAILABLE_DIMENSIONS,
            default=ui_service.DEFAULT_DIMENSIONS,
        )
        custom_dimensions_text = st.text_area(
            "自定义分析维度（每行一个，也可使用逗号分隔）",
            value="",
            help=(
                "例如：coding、research、enterprise_security、ecosystem。"
            ),
        )
        dimensions = ui_service.build_analysis_dimensions(
            selected_dimensions=selected_dimensions,
            custom_dimensions_text=custom_dimensions_text,
        )
        if dimensions:
            st.caption("本次分析维度：" + "、".join(dimensions))
        official_domains_text = st.text_area(
            "官方域名（每行使用 产品=域名）",
            value=ui_service.DEFAULT_OFFICIAL_DOMAINS_TEXT,
            help=(
                "填写后，搜索会限定到对应官方域名。多个域名可用逗号分隔。"
            ),
        )
        submitted = st.form_submit_button(
            "开始分析",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        run_submitted_analysis(
            target_product=target_product,
            competitors_text=competitors_text,
            dimensions=dimensions,
            official_domains_text=official_domains_text,
        )

    render_saved_result()


if __name__ == "__main__":
    main()
