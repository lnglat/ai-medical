"""在线医学实体标准化：MySQL 精确映射优先，Chroma 语义召回兜底。

本模块只读取 离线医学数据清洗、实体对齐与索引构建 已导入的数据，不执行离线聚类，也不向数据库写入内容。真实连接、
Chroma collection 和嵌入器都由组合知识工具注入，因此导入模块时不会访问外部服务。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from src.services.database import get_mysql_connection

# frozenset 是不可修改集合，防止运行时误加实体类型
ENTITY_TYPES = frozenset(
    {"disease", "symptom", "department", "check", "drug", "food", "cause", "people"}
)
# 检查 Chroma 返回的 metadata 是否与 Neo4j 契约一致
ENTITY_LABELS = {
    "disease": "Disease", "symptom": "Symptom", "department": "Department",
    "check": "Check", "drug": "Drug", "food": "Food", "cause": "Cause",
    "people": "People",
}


class TextEmbedder(Protocol):
    """在线查询向量生成器的最小接口。"""

    def encode(self, texts: Sequence[str]) -> Any:
        """把一批文本转换为与 离线医学数据清洗、实体对齐与索引构建 Chroma collection 相同空间中的向量。"""


class EntityNormalizationError(RuntimeError):
    """MySQL、Chroma 或嵌入模型不可用时抛出的明确检索错误。"""

    # super() 代表父类 RuntimeError；
    # super().__init__(...) 调用父类的初始化方法，设置异常信息。
    def __init__(self, dependency: str, message: str) -> None:
        super().__init__(f"{dependency} 实体标准化失败：{message}")
        self.dependency = dependency


@dataclass(frozen=True)
class NormalizedEntity:
    """一次标准化结果，供后续用稳定 ID 查询 Neo4j。"""

    original_text: str # 患者原始表达
    standard_text: str
    entity_id: str # 稳定 ID
    entity_type: str
    method: Literal["mysql_exact", "chroma_semantic"] # MySQL 精确或 Chroma 语义
    score: float # 标准化相似分数


def _first(value: Any) -> Any:
    """读取 Chroma ``query`` 返回的第一条结果，兼容空结果。"""

    if not isinstance(value, list) or not value:
        return None
    first_batch = value[0]
    return first_batch[0] if isinstance(first_batch, list) and first_batch else None


class MedicalEntityNormalizer:
    """执行 MySQL 精确查询，并在未命中时使用 Chroma 语义召回。

    ``mysql_connection_factory`` 每次调用返回一个只读短连接；``collection`` 是 离线医学数据清洗、实体对齐与索引构建
    约定的 ``ai_medical`` collection；``embedder`` 必须加载与 离线医学数据清洗、实体对齐与索引构建 相同的 BGE 模型。
    外部依赖一旦抛错会被标记为不可用，并向上抛出错误，不能伪装成“没有命中”。
    """

    _MYSQL_SQL = (
        "SELECT id, std_name, entity_schema FROM entity_mapping "
        "WHERE synonym = %s AND entity_schema = %s "
        "ORDER BY is_reviewed DESC LIMIT 1"
    )

    def __init__(
        self,
        *,
        mysql_connection_factory: Callable[[], Any] = get_mysql_connection,
        collection: Any,
        embedder: TextEmbedder,
        semantic_max_distance: float = 0.35,
    ) -> None:
        # 构造函数接收三个依赖和一个阈值
        if semantic_max_distance < 0:
            raise ValueError("semantic_max_distance 不能小于 0")
        self._mysql_connection_factory = mysql_connection_factory
        self._collection = collection
        self._embedder = embedder
        self.semantic_max_distance = semantic_max_distance
        self.dependency_status = {"mysql": True, "chroma": True, "embedding": True}

    def normalize(self, text: str, entity_type: str) -> NormalizedEntity | None:
        """标准化一个实体；合法但确实无匹配时返回 ``None``。

        精确查询会读取所有 离线医学数据清洗、实体对齐与索引构建 映射，而非只读取 ``is_reviewed=1`` 的历史人工词表；
        这是因为自动映射也是 离线医学数据清洗、实体对齐与索引构建 正式产物的一部分。只有未精确命中时才计算查询向量。
        """

        cleaned = text.strip()
        if not cleaned:
            return None
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"不支持的实体类型：{entity_type}")
        exact = self._normalize_from_mysql(cleaned, entity_type)
        # MySQL 精确命中：直接返回    MySQL 正常查询但没找到：使用 Chroma；
        # MySQL 自身故障：抛异常，不允许假装成“没找到”并继续
        return exact if exact is not None else self._normalize_from_chroma(cleaned, entity_type)

    def _normalize_from_mysql(self, text: str, entity_type: str) -> NormalizedEntity | None:
        """使用参数化 SQL 查询完整同义词，防止把患者文本拼入 SQL。"""

        try:
            with self._mysql_connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(self._MYSQL_SQL, (text, entity_type))
                    row = cursor.fetchone()
        # 捕获连接、超时、SQL 等异常
        except Exception as exc:
            self.dependency_status["mysql"] = False
            raise EntityNormalizationError("mysql", str(exc)) from exc
        if not row:
            return None
        if isinstance(row, Mapping):
            entity_id, standard, actual_type = row["id"], row["std_name"], row["entity_schema"]
        else:
            entity_id, standard, actual_type = row[0], row[1], row[2]
        if str(actual_type) != entity_type or not str(entity_id).startswith(f"{entity_type}_"):
            self.dependency_status["mysql"] = False
            raise EntityNormalizationError("mysql", "返回记录不符合 离线医学数据清洗、实体对齐与索引构建 实体类型或稳定 ID 契约")
        return NormalizedEntity(
            original_text=text,
            standard_text=str(standard),
            entity_id=str(entity_id),
            entity_type=str(actual_type),
            method="mysql_exact",
            score=1.0,
        )

    def _normalize_from_chroma(self, text: str, entity_type: str) -> NormalizedEntity | None:
        """按 离线医学数据清洗、实体对齐与索引构建 metadata 的 ``entity_type`` 过滤语义候选并校验来源。"""

        try:
            vectors = self._embedder.encode([text])
            vector = vectors[0]
            # Sentence Transformers 通常返回 NumPy 数组，Chroma 更适合普通 Python 列表，因此通过 tolist() 转换。
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
        except Exception as exc:
            self.dependency_status["embedding"] = False
            raise EntityNormalizationError("embedding", str(exc)) from exc
        # 查询 Chroma
        try:
            result = self._collection.query(
                query_embeddings=[vector],# 查询向量
                n_results=1,# 只取最接近的一个标准实体
                where={"entity_type": entity_type},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self.dependency_status["chroma"] = False
            raise EntityNormalizationError("chroma", str(exc)) from exc

        # 校验 Chroma metadata
        entity_id = _first(result.get("ids"))
        standard = _first(result.get("documents"))
        metadata = _first(result.get("metadatas"))
        distance = _first(result.get("distances"))
        if entity_id is None or standard is None or metadata is None or distance is None:
            return None
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("entity_type") != entity_type
            or metadata.get("label") != ENTITY_LABELS[entity_type]
            or metadata.get("source") != "ai-medical:t2"
            or not str(entity_id).startswith(f"{entity_type}_")
        ):
            self.dependency_status["chroma"] = False
            raise EntityNormalizationError("chroma", "召回 metadata 不符合 离线医学数据清洗、实体对齐与索引构建 数据契约")
        # 距离太远，不能强行把不相关表达映射成某个医学实体
        numeric_distance = float(distance)
        if numeric_distance > self.semantic_max_distance:
            return None
        return NormalizedEntity(
            original_text=text,
            standard_text=str(standard),
            entity_id=str(entity_id),
            entity_type=entity_type,
            method="chroma_semantic",
            score=max(0.0, min(1.0, 1.0 - numeric_distance)),
        )


def normalize_medical_entity(text: str, entity_type: str) -> str | None:
    """旧的 MySQL 精确标准化兼容入口；新代码应注入 ``MedicalEntityNormalizer``。

    该函数不执行 Chroma 兜底，保留它只是为了避免已有调用方突然失效。
    """

    sql = MedicalEntityNormalizer._MYSQL_SQL
    try:
        with get_mysql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (text.strip(), entity_type))
                row = cursor.fetchone()
    except Exception as exc:
        raise EntityNormalizationError("mysql", str(exc)) from exc
    if not row:
        return None
    return str(row["std_name"] if isinstance(row, Mapping) else row[1])
