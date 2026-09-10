"""确定性本地产物输出与显式、幂等、非破坏性的数据库适配器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from src.datasync.entity_alignment import TextEmbedder
from src.datasync.cleaning import stable_id
from src.datasync.graph_builder import LABELS, RELATION_TYPES, validate_graph
from src.datasync.schemas import (
    CleanMedicalRecord, EntityMappingRecord, GraphNodeRecord, GraphRelationRecord,
    RejectedRecord, VectorDocumentRecord,
)


@dataclass(frozen=True)
class ValidatedArtifacts:
    """通过整包摘要、当前模型和交叉引用校验后才可交给数据库适配器的数据。"""

    cleaned: list[CleanMedicalRecord]
    rejected: list[RejectedRecord]
    mappings: list[EntityMappingRecord]
    nodes: list[GraphNodeRecord]
    relations: list[GraphRelationRecord]
    documents: list[VectorDocumentRecord]
    manifest: dict[str, Any]

# 输入Pydantic模型 → 输出 JSON 兼容dict;输入普通dict → 原样浅拷贝成 dict。
def _plain(item: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    return item.model_dump(mode="json") if isinstance(item, BaseModel) else dict(item)


def write_jsonl(path: Path, items: Iterable[BaseModel | Mapping[str, Any]]) -> str:
    """以先写临时文件、校验安全后再原子替换的方式,把一批数据写成 JSONL 格式文件,并返回整个文件内容的 SHA-256 摘要"""

    # 创建父目录,已经存在不报错静默跳过
    path.parent.mkdir(parents=True, exist_ok=True)
    # path.suffix — 扩展名
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    # "w" — 写模式(会覆盖文件)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            line = json.dumps(_plain(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    # 把临时文件原地重命名成目标文件
    temporary.replace(path)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    """以稳定格式写普通 JSON，返回内容摘要。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_artifact_bundle(
    output_dir: Path,
    *,
    require_full: bool = False,
    require_embeddings: bool = False,
) -> ValidatedArtifacts:
    """在建立数据库连接前验证 manifest、文件摘要、模型和跨文件引用。

    该函数一次性读取并解析已校验的字节，避免先验摘要、后重新读取产生的
    检查与使用时间差。任何一项不一致都会中止 apply。

    六个产物文件 SHA-256 是否与 manifest 一致。
    每行 JSON 是否符合 Pydantic 模型。
    节点 ID 是否稳定。
    关系端点是否存在。
    关系 ID 是否符合稳定 ID 算法。
    映射 ID、节点 ID、向量文档 ID 是否一致。
    manifest 数量是否与文件实际数量相同。
    """

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("manifest 缺少 parameters")
    if require_full and parameters.get("limit") is not None:
        raise ValueError("正式 apply 拒绝导入带 limit 的抽样产物")
    if require_embeddings and parameters.get("embedding_enabled") is not True:
        raise ValueError("正式 apply 要求使用 BGE 构建的产物")

    # 代码里期待的文件名集合
    specifications: dict[str, type[BaseModel]] = {
        "cleaned_medical_records.jsonl": CleanMedicalRecord,
        "rejected_records.jsonl": RejectedRecord,
        "entity_mapping.jsonl": EntityMappingRecord,
        "graph_nodes.jsonl": GraphNodeRecord,
        "graph_relations.jsonl": GraphRelationRecord,
        "vector_documents.jsonl": VectorDocumentRecord,
    }
    parsed: dict[str, list[BaseModel]] = {}
    expected_digests = manifest.get("artifact_sha256")
    if not isinstance(expected_digests, dict) or set(expected_digests) != set(specifications):
        raise ValueError("manifest 的 artifact_sha256 文件集合不完整或包含未知产物")
    for filename, model in specifications.items():
        # 一次性按二进制读入整个文件,得到的是 bytes,不做任何解码
        raw = (output_dir / filename).read_bytes()
        actual_digest = hashlib.sha256(raw).hexdigest()
        if actual_digest != expected_digests[filename]:
            raise ValueError(f"产物摘要不匹配：{filename}")
        text = raw.decode("utf-8")
        # line.strip()过滤掉空行
        # text.splitlines()按换行拆成行的列表
        # 把单个JSON字符串直接解析成对应模型并做类型校验
        parsed[filename] = [model.model_validate_json(line) for line in text.splitlines() if line.strip()]

    # jsonl后的数据在逻辑上说得通
    cleaned = parsed["cleaned_medical_records.jsonl"]
    rejected = parsed["rejected_records.jsonl"]
    mappings = parsed["entity_mapping.jsonl"]
    nodes = parsed["graph_nodes.jsonl"]
    relations = parsed["graph_relations.jsonl"]
    documents = parsed["vector_documents.jsonl"]
    cleaned_ids = [item.record_id for item in cleaned]
    if len(cleaned_ids) != len(set(cleaned_ids)):
        raise ValueError("cleaned_medical_records 包含重复 record_id")
    # 把 ID 列表转成集合
    known_records = set(cleaned_ids)
    # 检验图节点和关系是否存在重复节点 ID 和悬空关系
    validate_graph(nodes, relations)
    # 把 node_id 映射到节点对象。
    node_by_id = {item.node_id: item for item in nodes}
    # 节点 ID 必须能由类型+名称重新推导
    if any(item.node_id != stable_id(item.entity_type, item.name) for item in nodes):
        raise ValueError("图节点稳定 ID 与实体类型/标准名称不一致")
    # entity_type 对应的 label 必须是映射表里写死的那个值。
    if any(item.label != LABELS[item.entity_type] for item in nodes):
        raise ValueError("图节点 label 与 entity_type 不一致")
    # 节点的引用必须指向存在的清洗记录
    if any(not set(item.source_records) <= known_records for item in nodes):
        raise ValueError("图节点引用了未知清洗记录")
    for relation in relations:
        if relation.relation_type not in RELATION_TYPES:
            raise ValueError(f"未知关系类型：{relation.relation_type}")
        if not set(relation.source_records) <= known_records:
            raise ValueError(f"关系引用未知清洗记录：{relation.relation_id}")
        expected_relation_id = stable_id(
            "relation", f"{relation.source_id}|{relation.relation_type}|{relation.target_id}"
        )
        if relation.relation_id != expected_relation_id:
            raise ValueError(f"关系稳定 ID 不匹配：{relation.relation_id}")
    for item in mappings:
        if item.entity_id != stable_id(item.entity_type, item.standard_text):
            raise ValueError(f"实体映射稳定 ID 不匹配：{item.original_text}")
    mapping_keys = [(item.entity_type, item.original_text) for item in mappings]
    if len(mapping_keys) != len(set(mapping_keys)):
        raise ValueError("entity_mapping 包含重复的实体类型/原词")
    node_ids = {item.node_id for item in nodes}
    if {item.entity_id for item in mappings} != node_ids:
        raise ValueError("实体映射的标准实体 ID 集合必须与图节点 ID 集合一致")
    document_ids = [item.document_id for item in documents]
    if len(document_ids) != len(set(document_ids)) or set(document_ids) != node_ids:
        raise ValueError("向量文档 ID 必须唯一并与图节点 ID 集合完全一致")
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for relation in relations:
        adjacency[relation.source_id].add(relation.target_id)
        adjacency[relation.target_id].add(relation.source_id)
    for item in documents:
        node = node_by_id[item.document_id]
        expected_related = ",".join(sorted(adjacency[item.document_id]))
        if (
            item.text != node.name
            or item.metadata.get("entity_type") != node.entity_type
            or item.metadata.get("label") != node.label
            or item.metadata.get("source") != "ai-medical:t2"
            or item.metadata.get("related_entity_ids") != expected_related
        ):
            raise ValueError(f"向量文档与图节点/关系契约不一致：{item.document_id}")

    stats = manifest.get("stats", {})
    expected_counts = {
        "cleaned_count": len(cleaned), "rejected_count": len(rejected),
        "mapping_count": len(mappings), "node_count": len(nodes),
        "relation_count": len(relations), "vector_document_count": len(documents),
    }
    if any(stats.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("manifest 统计与实际产物数量不一致")
    return ValidatedArtifacts(
        cleaned=cleaned, rejected=rejected, mappings=mappings, nodes=nodes,
        relations=relations, documents=documents, manifest=manifest,
    )


def load_reviewed_mysql_mapping(connection: Any) -> dict[tuple[str, str], str]:
    """只读取人工审核映射；返回 同义词、标准名、实体类型并组装成查询字典"""

    query = (
        "SELECT synonym, std_name, entity_schema FROM entity_mapping "
        "WHERE review_status = %s"
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (1,)) # review_status=1 的执行
        rows = cursor.fetchall()  # 一次性取回全部结果行
    # cursor 的配置决定,同一批查询的所有行类型必然一致,查一行就够
    if rows and not isinstance(rows[0], Mapping):
        raise TypeError("MySQL cursor 必须返回字典行")
    return {(row["entity_schema"], row["synonym"]): row["std_name"] for row in rows}


def apply_mysql_mappings(connection: Any, mappings: Sequence[EntityMappingRecord]) -> int:
    """用实体类型和完整同义词摘要联合键 upsert；不删除旧数据。

    离线医学数据清洗、实体对齐与索引构建 的 cause 等实体可能超过 1,000 个中文字符，无法把完整 utf8mb4 文本直接放进
    InnoDB 联合索引。数据库仍保存完整 synonym，同时用它的 SHA-256 二进制摘要作为
    等价唯一键；在线查询最终仍比较完整文本，不会返回只匹配前缀的记录。
    """

    sql = (
        "INSERT INTO entity_mapping "
        "(id, synonym, std_name, entity_schema, synonym_hash, review_status) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON DUPLICATE KEY UPDATE "
        "std_name=IF(review_status=1, std_name, VALUES(std_name)), "
        "id=IF(review_status=1, id, VALUES(id)), "
        "review_status=IF(review_status=1, review_status, VALUES(review_status))"
    )
    values = [
        (item.entity_id, item.original_text, item.standard_text, item.entity_type,
         hashlib.sha256(item.original_text.encode("utf-8")).digest(),
         1 if item.method == "reviewed_mysql" else (0 if item.needs_review else 2))
        for item in mappings
    ]
    with connection.cursor() as cursor:
        cursor.executemany(sql, values)
        affected = cursor.rowcount # 累加的所有受影响行数
    connection.commit() # 写入数据,必须 commit 才生效
    return affected


def apply_neo4j_graph(
    driver: Any,
    nodes: Sequence[GraphNodeRecord],
    relations: Sequence[GraphRelationRecord],
    *,
    batch_size: int = 500,
    database: str | None = None,
) -> dict[str, int]:
    """仅用 MERGE 按稳定 ID 增量写图；标签和关系来自代码白名单。"""

    allowed_labels = set(LABELS.values())
    allowed_relations = RELATION_TYPES
    node_labels = {node.node_id: node.label for node in nodes}
    # 记录发送给数据库的节点/关系数量
    node_count = relation_count = 0
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        for label in sorted(allowed_labels):
            # 把node.label == label 的节点转成 JSON 兼容 dict,凑成一个列表。
            rows = [node.model_dump(mode="json") for node in nodes if node.label == label]
            query = (
                # UNWIND $rows AS row — 把参数 $rows(一个数组)逐行展开
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.node_id}}) "
                # SET 更新节点
                "SET n.name=row.name, n.entity_type=row.entity_type, n.aliases=row.aliases, "
                "n.source_records=row.source_records"
            )
            for start in range(0, len(rows), batch_size):
                # consume()显式消费结果流,强制查询真正跑完
                session.run(query, rows=rows[start:start + batch_size]).consume()
                # 累加本批发送的节点数,返回值的语义是"处理了 N 个节点",不是"新增了 N 个节点"
                node_count += len(rows[start:start + batch_size])
        for relation_type in sorted(allowed_relations):
            rows = [item.model_dump(mode="json") for item in relations if item.relation_type == relation_type]
            target_labels = {node_labels[item.target_id] for item in relations if item.relation_type == relation_type}
            if len(target_labels) > 1:
                raise ValueError(f"关系 {relation_type} 指向了多个节点标签：{sorted(target_labels)}")
            # DATA_CONTRACT 规定所有关系均从 Disease 指向固定类型。显式标签让 Neo4j
            # 使用各标签上的 id 唯一索引，避免无标签 MATCH 随图规模增长反复全图扫描。
            target_label = next(iter(target_labels), None)
            if target_label is None:
                continue
            query = (
                # MATCH 找不到端点时这一行会被跳过；标签只来自代码白名单。
                f"UNWIND $rows AS row MATCH (a:Disease {{id: row.source_id}}), "
                f"(b:{target_label} {{id: row.target_id}}) "
                f"MERGE (a)-[r:{relation_type} {{id: row.relation_id}}]->(b) "
                "SET r.source_records=row.source_records"
            )
            for start in range(0, len(rows), batch_size):
                session.run(query, rows=rows[start:start + batch_size]).consume()
                relation_count += len(rows[start:start + batch_size])
    return {"neo4j_nodes": node_count, "neo4j_relations": relation_count}


def apply_chroma_documents(
    collection: Any,
    documents: Sequence[VectorDocumentRecord],
    embedder: TextEmbedder,
    *,
    batch_size: int = 5000,
) -> int:
    """按稳定 ID 分批 upsert Chroma；不清空集合，也不删除既有文档。"""

    if not documents:
        return 0
    if batch_size <= 0:
        raise ValueError("Chroma batch_size 必须大于 0")
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        embeddings = embedder.encode([item.text for item in batch])
        # upsert 的语义：ID 不存在：插入文档、metadata 和向量；ID 已存在：更新该文档、metadata 和向量；
        # 不会清空 Collection；不会删除其他已有文档。
        collection.upsert(
            ids=[item.document_id for item in batch], # 必须唯一，且通常是字符串
            documents=[item.text for item in batch],
            metadatas=[item.metadata for item in batch],
            embeddings=list(embeddings), # 每条向量的维度必须相同
        )
    return len(documents)
