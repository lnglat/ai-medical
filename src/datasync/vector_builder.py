"""创建双向邻接表。即使图关系在 Neo4j 中是“疾病 → 症状”，向量文档的 metadata 里，疾病和症状都知道彼此关联"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.datasync.schemas import GraphNodeRecord, GraphRelationRecord, VectorDocumentRecord


def build_vector_documents(
    nodes: Iterable[GraphNodeRecord], relations: Iterable[GraphRelationRecord]
) -> list[VectorDocumentRecord]:
    """每个标准实体生成一条文档，metadata 只使用 Chroma 支持的标量。"""

    nodes = list(nodes)
    related: dict[str, set[str]] = defaultdict(set)
    # 向量构建阶段额外维护一个内存中的双向映射
    for relation in relations:
        related[relation.source_id].add(relation.target_id)
        related[relation.target_id].add(relation.source_id)
    documents = [
        VectorDocumentRecord(
            document_id=node.node_id,
            text=node.name,
            metadata={
                "entity_type": node.entity_type,
                "label": node.label,
                "source": "ai-medical:t2",
                "related_entity_ids": ",".join(sorted(related[node.node_id])),
            },
        )
        # nodes列表中的元素，按照每个元素的 node_id 属性进行升序排序
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    ids = [item.document_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("vector_documents 包含重复 document_id")
    return documents
