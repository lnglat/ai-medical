"""FastAPI 的 HTTP 输入与输出模型。

本模块只描述接口收发的数据，不处理分诊或问诊业务。它复用 ``models.schemas``
中的领域模型，避免 HTTP 层和 Agent 层分别定义同一种医疗数据。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.schemas import (
    GraphEvidence,
    MedicalError,
    MedicalRecordDraft,
    PatientProfile,
    PreconsultSummary,
    RetrievalIntent,
    TriageResult,
)


class RootResponse(BaseModel):
    """根路径返回的项目导航信息，方便浏览器直接确认服务已经启动。"""

    model_config = ConfigDict(extra="forbid")

    service: str
    message: str
    docs_url: str
    health_url: str
    dependencies_health_url: str


class MessageRequest(BaseModel):
    """患者发送一条消息时的请求体。

    ``session_id`` 位于 URL 路径中，正文只接收当前消息和首次可选的患者档案。
    该模型会由 FastAPI 在调用路由函数之前自动校验。
    """
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    patient_profile: PatientProfile | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message", "request_id")
    @classmethod
    def strip_non_empty_text(cls, value: str | None) -> str | None:
        """去掉传输层多余空白，并拒绝只包含空格的消息或幂等键。"""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能只包含空白字符")
        return normalized


class WaitingUserResponse(BaseModel):
    """信息不足时返回；前端据此展示唯一的下一轮核心问题。"""
    model_config = ConfigDict(extra="forbid")

    conversation_status: Literal["waiting_user"] = "waiting_user"
    session_id: str
    current_question: str
    question_count: int = Field(ge=0)
    triage_result: TriageResult | None = None


class EmergencyResponse(BaseModel):
    """命中红旗风险后返回；流程不会继续普通预问诊。"""
    model_config = ConfigDict(extra="forbid")

    conversation_status: Literal["emergency_ended"] = "emergency_ended"
    session_id: str
    triage_result: TriageResult
    message: str


class CompletedResponse(BaseModel):
    """问诊完成响应；摘要含安全筛选后的排查方向和图谱检查，证据不足时为空。"""
    model_config = ConfigDict(extra="forbid")

    conversation_status: Literal["completed"] = "completed"
    session_id: str
    summary: PreconsultSummary
    medical_record_draft: MedicalRecordDraft


class FailedResponse(BaseModel):
    """发生不可恢复错误时返回；错误使用统一模型，避免暴露底层异常。"""
    model_config = ConfigDict(extra="forbid")

    conversation_status: Literal["failed"] = "failed"
    session_id: str
    errors: list[MedicalError] = Field(default_factory=list)
    message: str


class HealthResponse(BaseModel):
    """应用存活检查；不触发数据库连接或大型模型加载。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    checkpoint_backend: str


class DependencyHealthResponse(BaseModel):
    """外部依赖准备状态，与应用本身是否存活分开返回。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_initialized", "unavailable"]
    session_service: str
    detail: str | None = None


class NormalizationTrace(BaseModel):
    """从真实 GraphEvidence 还原的最小实体标准化信息。"""

    model_config = ConfigDict(extra="forbid")

    input_entity: str
    normalized_entity: str
    method: str
    score: float | None = None


class TraceAuditEvent(BaseModel):
    """允许进入演示接口的审计字段白名单，不包含患者原文或底层异常。"""

    model_config = ConfigDict(extra="forbid")

    node: str
    route: str | None = None
    evidence_count: int | None = None
    error_category: str | None = None


class DemoTraceResponse(BaseModel):
    """只读演示轨迹；所有证据均来自当前会话的 LangGraph 检查点。"""

    model_config = ConfigDict(extra="forbid")

    available: bool
    retrieval_intent: RetrievalIntent | None = None
    retrieval_entities: list[str] = Field(default_factory=list)
    normalization_results: list[NormalizationTrace] = Field(default_factory=list)
    retrieved_evidence: list[GraphEvidence] = Field(default_factory=list)
    audit_events: list[TraceAuditEvent] = Field(default_factory=list)


ConsultationResponse = WaitingUserResponse | EmergencyResponse | CompletedResponse | FailedResponse
