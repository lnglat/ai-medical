"""LangGraph 各节点之间唯一允许共享的医疗问诊状态。

LangGraph 会把节点返回的“小字典”合并进 ``MedicalState``，所以节点不应复制
整份状态。带 ``Annotated[..., reducer]`` 的字段不会被新值直接覆盖，而是由
reducer 按规则合并，适合消息、证据、错误和审计记录等累计数据。
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.models.schemas import ConversationStatus, RetrievalIntent


class MedicalState(TypedDict, total=False):
    """一次会话从患者主诉到完成摘要的运行时状态。

    ``total=False`` 表示某些字段可以在流程后续节点才出现。例如摘要节点之前
    没有 ``summary``；每个节点只返回本次真正改变的字段即可。
    """

    # 用于 checkpointer 和 API 把同一患者的多轮消息关联为一个会话。
    session_id: str
    # add_messages 能识别消息 ID 并追加/更新历史，支持多轮对话恢复。
    messages: Annotated[list[BaseMessage], add_messages]
    chief_complaint: str  # 患者最初主诉，后续节点不应擅自改写。
    patient_profile: dict  # PatientProfile.model_dump() 的结果。
    consultation_slots: dict  # ConsultationSlots.model_dump() 的结果，随追问逐步补全。
    triage_result: dict | None  # TriageResult 的字典，供路由和摘要读取。
    # 多次检索的证据需要保留，因此用 operator.add 拼接而非覆盖。
    retrieved_evidence: Annotated[list[dict], operator.add]
    differential_evidence: Annotated[list[dict], operator.add]
    differential_directions: list[dict]
    check_evidence: Annotated[list[dict], operator.add]
    possible_evaluations: list[dict]
    current_question: str | None  # 状态为 waiting_user 时，前端展示的唯一问题。
    question_count: int  # 已提出的问题数，用于限制无限追问。
    max_question_count: int  # 配置给出的单次问诊最大追问数。
    differential_question_count: int  # 已展示的图谱鉴别追问数，独立限制为至多两轮。
    need_graph_retrieval: bool  # 预问诊节点只提出请求；检索节点才真正执行工具。
    retrieval_intent: RetrievalIntent | None  # 限制工具只能查询允许的知识关系。
    retrieval_entities: list[str]  # 本次检索使用的标准化前实体名称。
    conversation_status: ConversationStatus  # 条件边据此选择等待、总结或结束分支。
    summary: dict | None  # 仅摘要 Agent 写入的 PreconsultSummary 字典。
    medical_record_draft: dict | None  # 仅摘要 Agent 写入的 MedicalRecordDraft 字典。
    # 每个节点可追加自己的错误和审计信息，便于定位一次会话发生过什么。
    errors: Annotated[list[dict], operator.add]
    audit_log: Annotated[list[dict], operator.add]
