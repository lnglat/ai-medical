"""把标准化实体转换成节点和有明确方向的图关系。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.datasync.cleaning import stable_id
from src.datasync.entity_alignment import FIELD_ENTITY_TYPES
from src.datasync.schemas import (
    CleanMedicalRecord, EntityMappingRecord, EntityType, GraphNodeRecord, GraphRelationRecord,
)


LABELS: dict[EntityType, str] = {
    "disease": "Disease", "symptom": "Symptom", "department": "Department",
    "check": "Check", "drug": "Drug", "food": "Food", "cause": "Cause", "people": "People",
}

# 所有关系统一从 Disease 指向目标，可插拔图谱 RAG 检索 反向检索症状时使用 ``(s)<-[:HAS_SYMPTOM]-(d)``。
FIELD_RELATIONS = {
    "symptoms": "HAS_SYMPTOM", "departments": "BELONGS_TO_DEPARTMENT",
    "checks": "RECOMMENDS_CHECK", "drugs": "COMMON_DRUG",
    "foods_recommended": "RECOMMENDS_FOOD", "foods_avoided": "AVOIDS_FOOD",
    "causes": "HAS_CAUSE", "people": "AFFECTS_PEOPLE",
}
RELATION_TYPES = frozenset(FIELD_RELATIONS.values())


def build_graph(
    records: Iterable[CleanMedicalRecord], mappings: Iterable[EntityMappingRecord]
) -> tuple[list[GraphNodeRecord], list[GraphRelationRecord]]:
    """生成去重节点与关系，并保留原词别名和来源记录。"""

    records = list(records)
    mappings = list(mappings)
    # lookup（原词→标准词）。
    lookup = {(item.entity_type, item.original_text): item.standard_text for item in mappings}
    # aliases（标准词→原词集合） defaultdict:缺键自动建 set自动去重 list重复项会累积
    aliases: dict[tuple[EntityType, str], set[str]] = defaultdict(set)
    for item in mappings:
        aliases[(item.entity_type, item.standard_text)].add(item.original_text)

    # 记录节点来源
    sources: dict[tuple[EntityType, str], set[str]] = defaultdict(set)
    # 记录关系来源
    relation_sources: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for record in records:
        # 索引疾病标准词
        disease = lookup[("disease", record.name)]
        # 生成（类型, 标准名），供 sources 用
        disease_key = ("disease", disease)
        # 记录节点来源
        sources[disease_key].add(record.record_id)
        # 生成确定性 ID
        disease_id = stable_id("disease", disease)
        # 遍历该疾病的除疾病外的每一field与实体类型对应
        for field, relation_type in FIELD_RELATIONS.items():
            entity_type = FIELD_ENTITY_TYPES[field]
            # 处理每一field
            for original in getattr(record, field):
                standard = lookup[(entity_type, original)]
                target_id = stable_id(entity_type, standard)
                sources[(entity_type, standard)].add(record.record_id)
                # 构建关系的确定id 只要起点、关系类型、终点都相同，就始终是同一关系
                relation_id = stable_id("relation", f"{disease_id}|{relation_type}|{target_id}")
                relation_sources[(relation_id, disease_id, relation_type, target_id)].add(
                    record.record_id
                )
    # 构建节点
    nodes = [
        GraphNodeRecord(
            node_id=stable_id(entity_type, standard), label=LABELS[entity_type],
            entity_type=entity_type, name=standard,
            # 构建去掉标准词的同义词库
            aliases=sorted(aliases[(entity_type, standard)] - {standard}),
            source_records=sorted(source_records),
        )
        for (entity_type, standard), source_records in sorted(sources.items())
    ]
    # 构建关系
    relations = sorted(
        (
            GraphRelationRecord(
                relation_id=relation_id,
                source_id=source_id,
                relation_type=relation_type,
                target_id=target_id,
                source_records=sorted(source_records),
            )
            for (relation_id, source_id, relation_type, target_id), source_records
            in relation_sources.items()
        ),
        key=lambda item: item.relation_id,
    )
    validate_graph(nodes, relations)
    return nodes, relations


def validate_graph(nodes: Iterable[GraphNodeRecord], relations: Iterable[GraphRelationRecord]) -> None:
    """拒绝重复节点和悬空关系，避免把坏产物交给 Neo4j。"""

    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("graph_nodes 包含重复 node_id")
    known = set(node_ids)
    relation_ids: set[str] = set()
    for relation in relations:
        if relation.relation_id in relation_ids:
            raise ValueError(f"重复 relation_id: {relation.relation_id}")
        relation_ids.add(relation.relation_id)
        if relation.source_id not in known or relation.target_id not in known:
            raise ValueError(f"悬空关系: {relation.relation_id}")
