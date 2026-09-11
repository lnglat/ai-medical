"""可插拔图谱 RAG 检索 组合知识工具：校验 离线医学数据清洗、实体对齐与索引构建 产物后串联标准化、向量召回与图查询。

工作流只注入本模块的 ``Neo4jMedicalKnowledgeTool``，无需知道 MySQL、Chroma、
Neo4j 的调用顺序。工厂不会创建 collection 或导入数据；这些写操作属于 真实三库初始化与正式数据导入。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.models.schemas import GraphEvidence, RetrievalIntent
from src.services.database import get_mysql_config, get_neo4j_driver
from src.tools.entity_normalizer import MedicalEntityNormalizer, NormalizedEntity, TextEmbedder
from src.tools.graph_retriever import Neo4jGraphRetriever

# 规定 离线医学数据清洗、实体对齐与索引构建 实体类型与 Neo4j 标签的对应关系
LABELS = {
    "disease": "Disease", "symptom": "Symptom", "department": "Department",
    "check": "Check", "drug": "Drug", "food": "Food", "cause": "Cause",
    "people": "People",
}
# 规定每种关系的目标类型
RELATION_TARGET_TYPES = {
    "HAS_SYMPTOM": "symptom",
    "BELONGS_TO_DEPARTMENT": "department",
    "RECOMMENDS_CHECK": "check",
    "COMMON_DRUG": "drug",
    "RECOMMENDS_FOOD": "food",
    "AVOIDS_FOOD": "food",
    "HAS_CAUSE": "cause",
    "AFFECTS_PEOPLE": "people",
}
# 每种在线检索意图应该标准化成什么类型
INTENT_ENTITY_TYPES: dict[RetrievalIntent, str] = {
    "symptom_to_department": "symptom",
    "symptom_to_disease": "symptom",
    "disease_to_check": "disease",
    "symptom_to_differential": "symptom",
}
ARTIFACT_FILES = {
    "cleaned_medical_records.jsonl", "entity_mapping.jsonl", "graph_nodes.jsonl",
    "graph_relations.jsonl", "vector_documents.jsonl", "rejected_records.jsonl",
}
CLEANED_FIELDS = {
    "record_id", "source_file", "source_line", "name", "desc", "symptoms",
    "departments", "checks", "drugs", "foods_recommended", "foods_avoided",
    "causes", "people",
}
REJECTED_FIELDS = {"source_file", "source_line", "reason", "raw_record"}
MAPPING_FIELDS = {
    "entity_id", "entity_type", "original_text", "standard_text", "source", "method",
    "confidence", "needs_review", "rationale",
}
# 每类 JSONL 的严格字段集合
NODE_FIELDS = {"node_id", "label", "entity_type", "name", "aliases", "source_records"}
RELATION_FIELDS = {"relation_id", "source_id", "relation_type", "target_id", "source_records"}
VECTOR_FIELDS = {"document_id", "text", "metadata"}
VECTOR_METADATA_FIELDS = {"entity_type", "label", "source", "related_entity_ids"}


class ArtifactContractError(RuntimeError):
    """离线医学数据清洗、实体对齐与索引构建 产物缺失、摘要错误或字段不符合 DATA_CONTRACT 时的启动错误。"""

# @dataclass 自动生成初始化方法。frozen=True 表示创建后不能随意修改
@dataclass(frozen=True)
class ArtifactContractReport:
    """成功校验后的只读统计，可供健康检查和演示记录使用。"""

    processed_dir: Path
    counts: dict[str, int]
    artifact_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    """分块计算大文件摘要，避免一次把正式产物全部读入内存太大，降低内存峰值。"""

    digest = hashlib.sha256()
    # "rb" 以二进制读取文件，避免编码转换影响摘要
    with path.open("rb") as handle:
        # 每次读取 1 MB; b""表示读到空字节时结束
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            # 返回十六进制摘要的str
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """逐行解析 JSONL，并把行号和文件名加入可读并解析错误。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            # 跳过空行
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ArtifactContractError(f"{path.name}:{line_number} 不是有效 UTF-8 JSON") from exc
            if not isinstance(value, dict):
                raise ArtifactContractError(f"{path.name}:{line_number} 必须是 JSON 对象")
            yield line_number, value


def _require_exact_fields(filename: str, line: int, row: Mapping[str, Any], fields: set[str]) -> None:
    """按 离线医学数据清洗、实体对齐与索引构建 严格模型拒绝缺失或偷偷新增的字段。"""

    missing = fields.difference(row)
    extra = set(row).difference(fields)
    if missing or extra:
        raise ArtifactContractError(
            f"{filename}:{line} 字段不符合契约；缺少={sorted(missing)}，多余={sorted(extra)}"
        )


def validate_t2_artifacts(
    processed_dir: Path,
    *,
    require_full: bool = True,
) -> ArtifactContractReport:
    """在连接 MySQL、Chroma、Neo4j 之前，先验证离线数据是否可信、完整、相互一致。

    可插拔图谱 RAG 检索 不调用 离线医学数据清洗、实体对齐与索引构建 内部流水线函数，也不会尝试修复或重建产物。任何不一致都会阻止
    真实知识工具构建。校验包括关系方向、``source_records`` 列表、Chroma metadata、
    跨文件稳定 ID 以及 manifest 数量。
    """

    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactContractError(f"缺少 离线医学数据清洗、实体对齐与索引构建 manifest：{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactContractError("manifest.json 不是有效 UTF-8 JSON") from exc
    # manifest 中的摘要文件集合必须与 ARTIFACT_FILES 完全相同
    expected_digests = manifest.get("artifact_sha256")
    if not isinstance(expected_digests, dict) or set(expected_digests) != ARTIFACT_FILES:
        raise ArtifactContractError("manifest artifact_sha256 文件集合不符合 离线医学数据清洗、实体对齐与索引构建 契约")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ArtifactContractError("manifest 缺少 parameters")
    if require_full and parameters.get("limit") is not None:
        raise ArtifactContractError("真实 RAG 拒绝使用带 limit 的抽样产物")
    if parameters.get("embedding_enabled") is not True:
        raise ArtifactContractError("真实 RAG 要求 离线医学数据清洗、实体对齐与索引构建 使用 BGE 构建向量产物")

    # 校验六个文件摘要
    actual_digests: dict[str, str] = {}
    # 逐个检查文件是否存在
    for filename in sorted(ARTIFACT_FILES):
        path = processed_dir / filename
        if not path.is_file():
            raise ArtifactContractError(f"缺少 离线医学数据清洗、实体对齐与索引构建 产物：{path}")
        actual_digests[filename] = _sha256(path)
        if actual_digests[filename] != expected_digests[filename]:
            raise ArtifactContractError(f"离线医学数据清洗、实体对齐与索引构建 产物摘要不匹配：{filename}")
        
    # 校验清洗记录与拒绝记录
    known_records: set[str] = set()
    cleaned_count = 0
    for line, row in _jsonl(processed_dir / "cleaned_medical_records.jsonl"):
        _require_exact_fields("cleaned_medical_records.jsonl", line, row, CLEANED_FIELDS)
        known_records.add(str(row["record_id"]))
        cleaned_count += 1
    rejected_count = 0
    for line, row in _jsonl(processed_dir / "rejected_records.jsonl"):
        _require_exact_fields("rejected_records.jsonl", line, row, REJECTED_FIELDS)
        rejected_count += 1

    # 校验图节点
    node_types: dict[str, str] = {}
    node_names: dict[str, str] = {}
    node_count = 0
    for line, row in _jsonl(processed_dir / "graph_nodes.jsonl"):
        _require_exact_fields("graph_nodes.jsonl", line, row, NODE_FIELDS)
        node_id, entity_type = str(row["node_id"]), str(row["entity_type"])
        if entity_type not in LABELS or row["label"] != LABELS[entity_type]:
            raise ArtifactContractError(f"graph_nodes.jsonl:{line} 标签与实体类型不匹配")
        if node_id in node_types:
            raise ArtifactContractError(f"graph_nodes.jsonl:{line} 出现重复 node_id")
        # source_records必须是列表,不能为空,每个元素必须是字符串 
        sources = row["source_records"]
        if (not isinstance(sources, list) or not sources
                or not all(isinstance(item, str) for item in sources)
                # 当前集合中的所有元素，是否都包含在 known_records 集合中。
                or not set(sources).issubset(known_records)):
            raise ArtifactContractError(f"graph_nodes.jsonl:{line} source_records 无效")
        # aliases 必须是字符串列表
        if not isinstance(row["aliases"], list) or not all(isinstance(item, str) for item in row["aliases"]):
            raise ArtifactContractError(f"graph_nodes.jsonl:{line} aliases 必须是字符串列表")
        node_types[node_id] = entity_type
        node_names[node_id] = str(row["name"])
        node_count += 1

    # 校验实体映射
    mapping_ids: set[str] = set() # 所有标准实体 ID
    mapping_keys: set[tuple[str, str]] = set() # 所有 (实体类型, 原始词)
    mapping_count = 0 # 映射数量
    for line, row in _jsonl(processed_dir / "entity_mapping.jsonl"):
        _require_exact_fields("entity_mapping.jsonl", line, row, MAPPING_FIELDS)
        entity_id, entity_type = str(row["entity_id"]), str(row["entity_type"])
        key = (entity_type, str(row["original_text"]))
        if key in mapping_keys or node_types.get(entity_id) != entity_type:
            raise ArtifactContractError(f"entity_mapping.jsonl:{line} 映射键或稳定 ID 无效")
        mapping_keys.add(key)
        mapping_ids.add(entity_id)
        mapping_count += 1
    if mapping_ids != set(node_types):
        raise ArtifactContractError("entity_mapping 的标准实体 ID 集合与图节点不一致")

    # 校验图关系
    relation_count = 0
    relation_ids: set[str] = set()
    for line, row in _jsonl(processed_dir / "graph_relations.jsonl"):
        _require_exact_fields("graph_relations.jsonl", line, row, RELATION_FIELDS)
        source_id, target_id, relation = str(row["source_id"]), str(row["target_id"]), str(row["relation_type"])
        relation_id = str(row["relation_id"])
        if relation_id in relation_ids:
            raise ArtifactContractError(f"graph_relations.jsonl:{line} 出现重复 relation_id")
        if node_types.get(source_id) != "disease" or node_types.get(target_id) != RELATION_TARGET_TYPES.get(relation):
            raise ArtifactContractError(f"graph_relations.jsonl:{line} 关系方向或类型不符合契约")
        sources = row["source_records"]
        if (not isinstance(sources, list) or not sources
                or not all(isinstance(item, str) for item in sources)
                or not set(sources).issubset(known_records)):
            raise ArtifactContractError(f"graph_relations.jsonl:{line} source_records 必须是有效列表")
        relation_ids.add(relation_id)
        relation_count += 1

    # 校验 Chroma 文档
    document_ids: set[str] = set() # 拒绝 重复文档 ID和非字典 metadata。
    vector_count = 0
    for line, row in _jsonl(processed_dir / "vector_documents.jsonl"):
        _require_exact_fields("vector_documents.jsonl", line, row, VECTOR_FIELDS)
        document_id, metadata = str(row["document_id"]), row["metadata"]
        if document_id in document_ids or not isinstance(metadata, dict):
            raise ArtifactContractError(f"vector_documents.jsonl:{line} ID 重复或 metadata 无效")
        entity_type = node_types.get(document_id)
        related_ids = str(metadata.get("related_entity_ids", "")).split(",") if isinstance(metadata, dict) else []
        related_ids = [item for item in related_ids if item]
        if (
            set(metadata) != VECTOR_METADATA_FIELDS
            or metadata.get("entity_type") != entity_type
            or metadata.get("label") != LABELS.get(entity_type)
            or metadata.get("source") != "ai-medical:t2"
            or row["text"] != node_names.get(document_id)
            or related_ids != sorted(set(related_ids))
            or not set(related_ids).issubset(node_types)
            or any(isinstance(value, (list, dict)) or value is None for value in metadata.values())
        ):
            raise ArtifactContractError(f"vector_documents.jsonl:{line} metadata 不符合 Chroma 契约")
        document_ids.add(document_id)
        vector_count += 1
    if document_ids != set(node_types):
        raise ArtifactContractError("vector document ID 集合与图节点不一致")

    # 对比 manifest 数量并返回报告
    counts = {
        "cleaned_count": cleaned_count, "rejected_count": rejected_count,
        "mapping_count": mapping_count, "node_count": node_count,
        "relation_count": relation_count, "vector_document_count": vector_count,
    }
    stats = manifest.get("stats")
    if not isinstance(stats, dict) or any(stats.get(key) != value for key, value in counts.items()):
        raise ArtifactContractError("manifest stats 与实际 离线医学数据清洗、实体对齐与索引构建 产物数量不一致")
    return ArtifactContractReport(processed_dir.resolve(), counts, actual_digests)


class LocalBGEQueryEmbedder:
    """使用本地 BGE 模型，把用户查询转换成向量，用于 Chroma 语义检索。"""

    def __init__(self, model_path: Path, *, batch_size: int = 32) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> Any:
        """使用与 离线医学数据清洗、实体对齐与索引构建 一致的归一化 BGE 向量参数编码文本。"""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                str(self.model_path), device="cpu", local_files_only=True
            )
        return self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )


class Neo4jMedicalKnowledgeTool:
    """实现工作流 ``MedicalKnowledgeTool`` Protocol 的真实组合检索工具。

    MedicalEntityNormalizer	将用户文本转为标准实体与稳定 ID
    Neo4jGraphRetriever	基于标准实体从 Neo4j 查询图谱关系
    Neo4jMedicalKnowledgeTool	编排前两者、去重、统一输出"""

    def __init__(self, *, normalizer: MedicalEntityNormalizer, retriever: Neo4jGraphRetriever) -> None:
        self.normalizer = normalizer
        self.retriever = retriever

    def search(
        self,
        *,
        entities: list[str],
        intent: RetrievalIntent,
        limit: int = 5,
    ) -> list[GraphEvidence]:
        """依次执行实体标准化和图查询，跨实体去重并限制总返回数量。"""

        # 检查参数
        if intent not in INTENT_ENTITY_TYPES:
            raise ValueError(f"不支持的知识检索意图：{intent}")
        if not 1 <= limit <= 50:
            raise ValueError("limit 必须在 1 到 50 之间")
        expected_type = INTENT_ENTITY_TYPES[intent]
        merged: dict[tuple[Any, ...], GraphEvidence] = {} # 来源实体 + 关系 + 目标实体\

        # 去除重复、空白实体
        # dict.fromkeys 在保留患者输入顺序的同时去掉重复实体，减少数据库调用。
        for raw_entity in dict.fromkeys(item.strip() for item in entities if item.strip()):
            # MySQL精确标准化或Chroma语义标准化
            normalized = self.normalizer.normalize(raw_entity, expected_type)
            if normalized is None:
                continue
            # 在 Neo4j 查询图谱
            # 最终方向检查每个疾病最多取 3 条，确保至多 3 个方向都获得查询机会。
            query_limit = 3 if intent == "disease_to_check" else limit
            evidence = self.retriever.retrieve(
                entity=normalized.standard_text,
                entity_id=normalized.entity_id,
                intent=intent,
                limit=query_limit,
            )
            for item in evidence:
                # model_copy复制一个模型对象，并且可以在复制时修改部分字段；原对象保持不变
                enriched = item.model_copy(update={
                    "score": normalized.score,
                    "evidence_source": f"{normalized.method}+neo4j",
                })
                key = (
                    enriched.source_entity_id or enriched.source_entity,
                    enriched.relation,
                    enriched.target_entity_id or enriched.target_entity,
                )
                previous = merged.get(key)
                if previous is None:
                    merged[key] = enriched
                else:
                    merged[key] = previous.model_copy(update={
                        "source_records": list(dict.fromkeys(previous.source_records + enriched.source_records)),
                        "score": max(previous.score or 0.0, enriched.score or 0.0),
                    })
                if intent not in {"symptom_to_differential", "disease_to_check"} and len(merged) >= limit:
                    return list(merged.values())
        if intent == "symptom_to_differential":
            # 所有患者阳性症状都先获得一次查询机会，再优先保留“输入症状→疾病”
            # 支持边，避免首个高连接度症状耗尽预算导致后续症状无法参与共同支持。
            def evidence_priority(item: GraphEvidence) -> tuple[int, str, str]:
                if item.source_entity_type == "symptom" and item.target_entity_type == "disease":
                    rank = 0
                elif item.target_entity_type == "symptom":
                    rank = 1
                else:
                    rank = 2
                return rank, item.source_entity, item.target_entity

            return sorted(merged.values(), key=evidence_priority)[:limit]
        if intent == "disease_to_check":
            return list(merged.values())[:limit]
        return list(merged.values())


def _mysql_read_connection(timeout_seconds: int = 5) -> Any:
    """创建带连接/读取超时的 MySQL 短连接；查询仍由参数化 SELECT 完成。"""

    import pymysql

    config = get_mysql_config() | {
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": timeout_seconds,
        "read_timeout": timeout_seconds,
        "write_timeout": timeout_seconds,
    }
    return pymysql.connect(**config)


def build_medical_knowledge_tool(
    *,
    processed_dir: Path | None = None, # 本地数据目录
    mysql_connection_factory: Callable[[], Any] | None = None,
    neo4j_driver_factory: Callable[[], Any] = get_neo4j_driver,
    chroma_client: Any = None,
    embedder: TextEmbedder | None = None,
) -> Neo4jMedicalKnowledgeTool:
    """构建生产知识工具；先验产物失败时绝不连接任何数据库。

    Chroma 使用 ``get_collection`` 而非 ``get_or_create_collection``，确保在线启动不会
    越权执行 真实三库初始化与正式数据导入 的初始化职责。调用方可注入 fake 依赖完成离线测试。
    """

    output = processed_dir or settings.data_dir / "processed"
    validate_t2_artifacts(output, require_full=True)

    if chroma_client is None:
        import chromadb

        chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        collection = chroma_client.get_collection(settings.chroma_collection)
    except Exception as exc:
        # 主动抛出一个运行时异常，让程序停止继续构建知识检索工具
        raise RuntimeError(
            # !r 表示使用变量的 repr() 形式，显示引号
            f"Chroma collection {settings.chroma_collection!r} 不可用；请先由 真实三库初始化与正式数据导入 初始化并导入 离线医学数据清洗、实体对齐与索引构建 产物"
        ) from exc
    active_embedder = embedder or LocalBGEQueryEmbedder(settings.embedding_model_path)
    normalizer = MedicalEntityNormalizer(
        mysql_connection_factory=mysql_connection_factory or _mysql_read_connection,
        collection=collection,
        embedder=active_embedder,
    )
    retriever = Neo4jGraphRetriever(
        driver_factory=neo4j_driver_factory,
        database=settings.neo4j_database,
    )
    return Neo4jMedicalKnowledgeTool(normalizer=normalizer, retriever=retriever)
