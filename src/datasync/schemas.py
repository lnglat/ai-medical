"""离线数据流水线使用的 Pydantic 契约。

这些模型和在线问诊的 ``src.models.schemas`` 完全分开。这样数据工程字段的变化
不会意外改变 LangGraph 的共享状态。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EntityType = Literal[
    "disease", "symptom", "department", "check", "drug", "food", "cause", "people"
]

class StrictModel(BaseModel):
    """禁止悄悄接收拼错字段的基础模型。"""

    model_config = ConfigDict(extra="forbid")


class RawMedicalRecord(BaseModel):
    """原始知识图谱的一行；原始数据阶段暂时允许任何类型。"""

    model_config = ConfigDict(extra="allow")

    name: Any = None
    desc: Any = ""
    symptom: Any = Field(default_factory=list)
    department: Any = Field(default_factory=list)
    check: Any = Field(default_factory=list)
    drug: Any = Field(default_factory=list)
    eat: Any = Field(default_factory=list)
    not_eat: Any = Field(default_factory=list)
    cause: Any = ""
    people: Any = ""


class CleanMedicalRecord(StrictModel):
    """清洗成功的疾病记录，所有可多值字段统一为字符串列表。"""

    record_id: str
    # 追溯原始文件和行号
    source_file: str
    source_line: int = Field(ge=1)
    name: str = Field(min_length=1)
    desc: str = ""
    symptoms: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    foods_recommended: list[str] = Field(default_factory=list)
    foods_avoided: list[str] = Field(default_factory=list)
    causes: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)


class RejectedRecord(StrictModel):
    """无法修复的输入及原因，保证坏数据也可以追溯。"""

    source_file: str
    source_line: int = Field(ge=1)
    reason: str
    raw_record: Any


class EntityMappingRecord(StrictModel):
    """原始词到标准词的一条可解释映射。"""

    entity_id: str
    entity_type: EntityType
    original_text: str
    standard_text: str
    source: str
    method: Literal["reviewed_mysql", "embedding_cluster", "low_confidence_candidate", "exact"]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False
    rationale: str


class GraphNodeRecord(StrictModel):
    """可通过稳定 ID 幂等写入 Neo4j 的节点。"""

    node_id: str
    label: str
    entity_type: EntityType
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_records: list[str] = Field(min_length=1)


class GraphRelationRecord(StrictModel):
    """图关系始终从疾病节点指向关联实体节点。

    一条关系可能由多条原始医学记录共同支持，因此来源使用
    ``source_records`` 列表。后续图谱 RAG 的解析模型和 fixture 也按列表读取。
    """

    relation_id: str
    source_id: str
    relation_type: str
    target_id: str
    # 一条去重后的图关系可能由多条原始医学记录共同支持，因此必须保留全部来源。
    source_records: list[str] = Field(min_length=1)


class VectorDocumentRecord(StrictModel):
    """供 Chroma 消费的文本与 metadata；embedding 可在 apply 时计算。"""

    document_id: str
    text: str
    # metadata 只允许 Chroma 可接受的标量类型；不能直接放 Python 列表或复杂对象。
    metadata: dict[str, str | int | float | bool]


class InputProfile(StrictModel):
    """只读扫描得到的输入数据概况。"""

    path: str
    sha256: str
    encoding: str = "utf-8"
    total_lines: int
    blank_lines: int
    valid_json_records: int
    invalid_json_records: int
    field_counts: dict[str, int]
    empty_field_counts: dict[str, int]
    empty_field_ratios: dict[str, float]


class PipelineStats(StrictModel):
    """各阶段数量统计，manifest 和命令行共用。"""

    input_count: int = 0
    cleaned_count: int = 0
    rejected_count: int = 0
    entity_count: int = 0
    mapping_count: int = 0
    low_confidence_count: int = 0
    node_count: int = 0
    relation_count: int = 0
    vector_document_count: int = 0
