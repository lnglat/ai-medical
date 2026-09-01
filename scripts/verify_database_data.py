"""按 离线医学数据清洗、实体对齐与索引构建 稳定 ID 集合核验 MySQL、Neo4j 和 Chroma 的真实导入结果。

该命令只执行查询，不写数据库。共享服务中可能包含其他项目的数据，因此不能比较
全库总数；这里逐批查询本次 manifest 对应的键和 ID，并报告缺失、重复或字段不一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasync.entity_alignment import LocalBgeEmbedder
from src.datasync.exporters import ValidatedArtifacts, validate_artifact_bundle


def _batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """按数据库可接受的小批次切分序列，避免一次查询参数过多。"""

    if size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(items), size):
        # 使用生成器逐批产生切片，不一次复制所有批次
        yield items[start:start + size]


def _summary(expected: int, found: int, missing: list[str], **extra: Any) -> dict[str, Any]:
    """生成三个数据库共用的核验摘要，并只保留少量示例供排错。"""

    result = {
        "expected": expected,
        "found": found,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "ok": expected == found and not missing,
    }
    result.update(extra)
    return result


def verify_mysql(connection: Any, artifacts: ValidatedArtifacts, *, batch_size: int = 500) -> dict[str, Any]:
    """按 ``(entity_schema, synonym)`` 联合键检查映射及其标准实体字段。"""

    expected = {
        (item.entity_type, item.original_text): (item.entity_id, item.standard_text)
        for item in artifacts.mappings
    }
    found: dict[tuple[str, str], tuple[str, str]] = {}
    keys = list(expected)
    with connection.cursor() as cursor:
        for batch in _batches(keys, batch_size):
            predicates = " OR ".join("(entity_schema=%s AND synonym=%s)" for _ in batch)
            # 把 [(实体类型, 同义词), ...] 展平，生成与 SQL 中多个 %s 占位符顺序一致的参数列表
            parameters = [value for key in batch for value in key]
            cursor.execute(
                "SELECT id, synonym, std_name, entity_schema FROM entity_mapping WHERE " + predicates,
                parameters,
            )
            for row in cursor.fetchall():
                found[(row["entity_schema"], row["synonym"])] = (row["id"], row["std_name"])
    # 校验数据库是否映射确实
    missing_keys = [f"{schema}:{synonym}" for schema, synonym in keys if (schema, synonym) not in found]
    # 校验字段是否映射错位
    mismatched = [
        f"{schema}:{synonym}"
        for (schema, synonym), value in expected.items()
        if (schema, synonym) in found and found[(schema, synonym)] != value
    ]
    result = _summary(len(expected), len(found), missing_keys)
    result.update({
        "mismatched_count": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "ok": result["ok"] and not mismatched,
    })
    return result


def verify_neo4j(
    driver: Any,
    artifacts: ValidatedArtifacts,
    *,
    database: str,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """在显式目标库中按节点 ID、关系 ID 查询，并检测同一 ID 的重复记录。"""

    node_ids = [item.node_id for item in artifacts.nodes]
    relation_ids = [item.relation_id for item in artifacts.relations]
    found_nodes: list[str] = []
    found_relations: list[str] = []
    with driver.session(database=database) as session:
        # 按标签分组
        for label in sorted({item.label for item in artifacts.nodes}):
            # 取出当前标签的所有期望节点 ID
            label_ids = [item.node_id for item in artifacts.nodes if item.label == label]
            # 按标签和 ID 批量查询
            for batch in _batches(label_ids, batch_size):
                # extend() 会将一个可迭代对象中的元素逐个添加到列表末尾
                found_nodes.extend(
                    row["id"]
                    for row in session.run(
                        f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id",
                        ids=list(batch),
                    )
                )
        # 查询关系
        for batch in _batches(relation_ids, batch_size):
            found_relations.extend(
                row["id"]
                for row in session.run(
                    "MATCH ()-[r]->() WHERE r.id IN $ids RETURN r.id AS id",
                    ids=list(batch),
                )
            )
    # 每个 ID 出现次数
    node_counter = Counter(found_nodes)
    relation_counter = Counter(found_relations)
    # 检查缺失
    missing_nodes = [item for item in node_ids if item not in node_counter]
    missing_relations = [item for item in relation_ids if item not in relation_counter]
    # 检查重复
    duplicate_nodes = [item for item, count in node_counter.items() if count != 1]
    duplicate_relations = [item for item, count in relation_counter.items() if count != 1]
    # 形成摘要
    nodes = _summary(len(node_ids), len(node_counter), missing_nodes)
    relations = _summary(len(relation_ids), len(relation_counter), missing_relations)
    nodes.update({"duplicate_count": len(duplicate_nodes), "ok": nodes["ok"] and not duplicate_nodes})
    relations.update({
        "duplicate_count": len(duplicate_relations),
        "ok": relations["ok"] and not duplicate_relations,
    })
    return {"database": database, "nodes": nodes, "relations": relations,
            "ok": nodes["ok"] and relations["ok"]}


def verify_chroma(collection: Any, artifacts: ValidatedArtifacts, *, batch_size: int = 1000) -> dict[str, Any]:
    """按文档稳定 ID 获取记录，同时检查文本和关键 metadata 未发生错位。"""

    expected = {item.document_id: item for item in artifacts.documents}
    found: dict[str, tuple[str | None, dict[str, Any]]] = {}
    ids = list(expected)
    for batch in _batches(ids, batch_size):
        result = collection.get(ids=list(batch), include=["documents", "metadatas"])
        result_ids = result.get("ids", [])
        documents = result.get("documents") or [None] * len(result_ids)
        metadatas = result.get("metadatas") or [{}] * len(result_ids)
        # strict=True要求三个列表长度完全相同
        for document_id, text, metadata in zip(result_ids, documents, metadatas, strict=True):
            found[document_id] = (text, metadata or {})
    # 缺失文档
    missing = [document_id for document_id in ids if document_id not in found]
    mismatched = []
    # 错位列表
    for document_id, item in expected.items():
        if document_id not in found:
            continue
        text, metadata = found[document_id]
        if text != item.text or any(metadata.get(key) != value for key, value in item.metadata.items()):
            mismatched.append(document_id)
    summary = _summary(len(expected), len(found), missing)
    summary.update({
        "mismatched_count": len(mismatched),
        "mismatched_examples": mismatched[:10],
        "ok": summary["ok"] and not mismatched,
    })
    return summary


def semantic_probe(collection: Any, query: str, embedder: Any) -> dict[str, Any]:
    """执行一次真实向量召回，证明 collection 不仅存在而且可被语义查询。"""

    vector = list(embedder.encode([query]))[0]
    result = collection.query(
        query_embeddings=[vector], n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    return {
        "query": query,
        "result_ids": (result.get("ids") or [[]])[0],
        "documents": (result.get("documents") or [[]])[0],
        "distances": (result.get("distances") or [[]])[0],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读核验 真实三库初始化与正式数据导入 三类数据库导入结果")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    # 控制每批向数据库发送多少 ID
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--semantic-query", help="可选：用本地 BGE 执行一次真实 Chroma 语义召回")
    return parser


def main(argv: list[str] | None = None) -> int:
    """连接三类真实数据库，输出分系统状态；任一不一致时返回非零退出码。"""

    args = _parser().parse_args(argv)
    from src.config.settings import settings
    from src.services.database import close_database_connections, get_mysql_connection, get_neo4j_driver
    import chromadb

    artifacts = validate_artifact_bundle(
        args.processed_dir, require_full=True, require_embeddings=True
    )
    report: dict[str, Any] = {}
    try:
        with get_mysql_connection() as connection:
            report["mysql"] = verify_mysql(connection, artifacts, batch_size=args.batch_size)
        report["neo4j"] = verify_neo4j(
            get_neo4j_driver(), artifacts,
            database=settings.neo4j_database, batch_size=args.batch_size,
        )
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_collection(settings.chroma_collection)
        report["chroma"] = verify_chroma(collection, artifacts, batch_size=args.batch_size)
        if args.semantic_query:
            embedder = LocalBgeEmbedder(
                settings.embedding_model_path, batch_size=settings.datasync_batch_size
            )
            report["semantic_probe"] = semantic_probe(collection, args.semantic_query, embedder)
        report["ok"] = all(report[name]["ok"] for name in ("mysql", "neo4j", "chroma"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    except Exception as exc:
        report.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    finally:
        close_database_connections()


if __name__ == "__main__":
    raise SystemExit(main())
