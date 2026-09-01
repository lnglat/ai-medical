"""LangGraph 的纯条件路由函数。

路由函数只读取共享状态并返回下一条边的标签，绝不在这里调用模型、数据库或
图谱工具。这样测试可以单独覆盖每个分支，也避免路由决定和业务执行相互耦合。
"""

from typing import Literal

from src.graph.state import MedicalState


def route_after_triage(state: MedicalState) -> Literal["emergency_end", "preconsult"]:
    """根据分诊结果选择紧急结束或继续预问诊。"""
    triage_result = state.get("triage_result") or {}
    # 缺少分诊结果时，默认进入 emergency_end
    if not triage_result.get("should_continue_preconsult", False):
        return "emergency_end"
    return "preconsult"


def route_after_preconsult(
    state: MedicalState,
) -> Literal["retrieve", "ask_user", "summarize", "failed"]:
    """根据预问诊状态选择检索、暂停、摘要或失败结束分支。"""
    # 存在“先失败、再检索、再等待、最后总结”的控制顺序
    if state.get("conversation_status") == "failed":
        return "failed"
    if state.get("need_graph_retrieval"):
        return "retrieve"
    if state.get("conversation_status") == "waiting_user":
        return "ask_user"
    return "summarize"
