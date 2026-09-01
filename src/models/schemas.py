"""医疗预问诊的共享领域模型。

本模块相当于各业务模块共同使用的“数据表单”。分诊、预问诊、摘要和 API
都通过这些 Pydantic 模型交换数据；模型会在创建时检查输入，随后可用
``model_dump()`` 转为适合写入 LangGraph 状态或 JSON 响应的普通字典。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Literal 将可选值限制为固定集合，避免不同 Agent 自行发明状态字符串。
UrgencyLevel = Literal["emergency", "urgent", "routine"]
ConversationStatus = Literal["triaging", "preconsulting", "waiting_user", "summarizing", "completed", "emergency_ended", "failed"]
RetrievalIntent = Literal["symptom_to_department", "symptom_to_disease", "disease_to_check"]
ErrorCategory = Literal["validation", "state", "llm", "database", "retrieval", "internal"]
KnowledgeEntityType = Literal[
    "disease", "symptom", "department", "check", "drug", "food", "cause", "people"
]


class PatientProfile(BaseModel):
    """患者基础信息。

    由 API 入口接收，并由分诊规则读取年龄、性别和妊娠状态等风险相关信息。
    未提供的字段保留为 ``None``，不能把未知信息当成否定信息。
    """

    #禁止在模型中定义未声明的额外字段
    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=0, le=130)
    # 性别未知时使用 "unknown"，而不是 None，避免后续逻辑误判。
    sex: Literal["male", "female", "unknown"] = "unknown"
    pregnancy: bool | None = None


class ConsultationSlots(BaseModel):
    """预问诊逐轮收集的结构化信息槽位。

    预问诊 Agent 每轮从患者回答中补充这些字段，再将 ``model_dump()`` 的结果
    写回共享状态。列表使用 ``default_factory``，保证每位患者拥有独立列表。
    """
    model_config = ConfigDict(extra="forbid")

    symptom: list[str] = Field(default_factory=list)
    onset: str | None = None
    duration: str | None = None
    severity: str | None = None
    location: str | None = None
    characteristics: str | None = None
    aggravating_or_relieving_factors: str | None = None
    accompanying_symptoms: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    medication_history: list[str] = Field(default_factory=list)
    allergy_history: list[str] = Field(default_factory=list)
    special_population: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    """分诊 Agent 的结构化输出。

    工作流路由读取 ``should_continue_preconsult``：为 ``False`` 时直接进入紧急
    结束分支，红旗症状不能被后续普通问诊覆盖。
    """
    # 先检查每个字段 —— 值都合法，单个过
    model_config = ConfigDict(extra="forbid")

    urgency: UrgencyLevel
    recommended_departments: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    rationale: str
    should_continue_preconsult: bool  # 要不要继续普通问诊

    # 给模型加一道"整表复核"关卡,查验逻辑是否合理
    @model_validator(mode="after")
    def emergency_must_stop_preconsult(self) -> "TriageResult":
        """拒绝“已判定紧急却继续普通问诊”的矛盾状态。

        路由层会直接信任 ``should_continue_preconsult``，因此这个不变量必须在
        共享模型中兜底，避免 Agent 或外部输入绕过紧急结束分支。
        """

        if (self.urgency == "emergency" or self.red_flags) and self.should_continue_preconsult:
            raise ValueError("紧急等级或红旗症状存在时不得继续普通预问诊")
        return self


class GraphEvidence(BaseModel):
    """知识图谱工具返回的一条只读证据，而非患者诊断结论。"""
    model_config = ConfigDict(extra="forbid")

    source_entity: str
    relation: str
    target_entity: str
    # 可选稳定 ID 和类型让真实 RAG 结果能追溯到 离线医学数据清洗、实体对齐与索引构建 图节点；保留默认值可兼容旧 fake。
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    source_entity_type: KnowledgeEntityType | None = None
    target_entity_type: KnowledgeEntityType | None = None
    source_records: list[str] = Field(default_factory=list)
    score: float | None = None
    evidence_source: str | None = None


class PreconsultDecision(BaseModel):
    """预问诊 Agent 的结构化决策，不是诊断结果。

    Agent 将本轮提取到的槽位和下一问打包在这里；工作流据此决定是否调用图谱工具、
    等待患者回答，或把状态交给摘要 Agent。
    """

    model_config = ConfigDict(extra="forbid")

    slot_updates: dict[str, object] = Field(default_factory=dict)
    next_question: str | None = None
    need_graph_retrieval: bool = False
    retrieval_intent: RetrievalIntent | None = None
    retrieval_entities: list[str] = Field(default_factory=list)
    should_end_preconsult: bool = False


class PreconsultSummary(BaseModel):
    """摘要 Agent 输出的预问诊摘要，供患者展示或后续人工查看。"""
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str
    present_illness_summary: str
    key_positive_findings: list[str] = Field(default_factory=list)
    key_negative_findings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    triage_recommendation: str
    safety_notice: str


class MedicalRecordDraft(BaseModel):
    """根据已采集事实整理的病历草稿，不包含诊断或处方。"""
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str
    history_of_present_illness: str
    past_history: str
    medication_and_allergy_history: str
    preliminary_department: str
    note: str


class MedicalError(BaseModel):
    """可安全序列化的错误记录。

    节点发生可预期错误时返回此模型的字典形式，而不是直接抛出给患者；工作流可
    将它累计到 ``errors`` 和审计日志中，用于后续降级、排障或 API 失败响应。
    """

    model_config = ConfigDict(extra="forbid")

    category: ErrorCategory
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)
    retryable: bool = False
    node: str | None = None
