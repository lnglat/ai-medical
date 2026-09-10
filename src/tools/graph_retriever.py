"""基于 离线医学数据清洗、实体对齐与索引构建 图结构的 Neo4j 只读白名单查询。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from neo4j import Query

from src.config.settings import settings
from src.models.schemas import GraphEvidence, RetrievalIntent
from src.services.database import get_neo4j_driver


# 模板、标签、方向和关系名全部遵守 离线医学数据清洗、实体对齐与索引构建 图谱数据约定。患者文本与稳定 ID 只能作为
# 参数传入；本模块没有接收或执行大模型生成 Cypher 的入口。
QUERY_TEMPLATES: dict[RetrievalIntent, str] = {
    "symptom_to_department": """
        MATCH (s:Symptom)<-[r1:HAS_SYMPTOM]-(d:Disease)
              -[r2:BELONGS_TO_DEPARTMENT]->(dept:Department)
        WHERE s.id = $entity_id OR ($entity_id IS NULL AND s.name = $entity)
        RETURN s.name AS source, s.id AS source_id, s.entity_type AS source_type,
               'HAS_SYMPTOM/BELONGS_TO_DEPARTMENT' AS relation,
               dept.name AS target, dept.id AS target_id, dept.entity_type AS target_type,
               coalesce(r1.source_records, []) + coalesce(r2.source_records, []) AS source_records
        LIMIT $limit
    """,
    "symptom_to_disease": """
        MATCH (s:Symptom)<-[r:HAS_SYMPTOM]-(d:Disease)
        WHERE s.id = $entity_id OR ($entity_id IS NULL AND s.name = $entity)
        RETURN s.name AS source, s.id AS source_id, s.entity_type AS source_type,
               'HAS_SYMPTOM' AS relation,
               d.name AS target, d.id AS target_id, d.entity_type AS target_type,
               coalesce(r.source_records, []) AS source_records
        LIMIT $limit
    """,
    "disease_to_check": """
        MATCH (d:Disease)-[r:RECOMMENDS_CHECK]->(c:Check)
        WHERE d.id = $entity_id OR ($entity_id IS NULL AND d.name = $entity)
        RETURN d.name AS source, d.id AS source_id, d.entity_type AS source_type,
               'RECOMMENDS_CHECK' AS relation,
               c.name AS target, c.id AS target_id, c.entity_type AS target_type,
               coalesce(r.source_records, []) AS source_records
        LIMIT $limit
    """,
    "symptom_to_differential": """
        MATCH (input:Symptom)<-[input_rel:HAS_SYMPTOM]-(d:Disease)
        WHERE input.id = $entity_id OR ($entity_id IS NULL AND input.name = $entity)
        WITH input, d, input_rel
        ORDER BY size(coalesce(input_rel.source_records, [])) DESC, d.name
        LIMIT 3
        CALL {
            WITH input, d, input_rel
            RETURN input.name AS source, input.id AS source_id,
                   input.entity_type AS source_type, 'HAS_SYMPTOM' AS relation,
                   d.name AS target, d.id AS target_id, d.entity_type AS target_type,
                   coalesce(input_rel.source_records, []) AS source_records, 0 AS result_rank
            UNION ALL
            WITH input, d, input_rel
            MATCH (d)-[symptom_rel:HAS_SYMPTOM]->(related:Symptom)
            WITH d, symptom_rel, related ORDER BY related.name LIMIT 4
            RETURN d.name AS source, d.id AS source_id, d.entity_type AS source_type,
                   'HAS_SYMPTOM' AS relation,
                   related.name AS target, related.id AS target_id,
                   related.entity_type AS target_type,
                   coalesce(symptom_rel.source_records, []) AS source_records, 1 AS result_rank
            UNION ALL
            WITH input, d, input_rel
            MATCH (d)-[check_rel:RECOMMENDS_CHECK]->(check:Check)
            WITH d, check_rel, check ORDER BY check.name LIMIT 3
            RETURN d.name AS source, d.id AS source_id, d.entity_type AS source_type,
                   'RECOMMENDS_CHECK' AS relation,
                   check.name AS target, check.id AS target_id,
                   check.entity_type AS target_type,
                   coalesce(check_rel.source_records, []) AS source_records, 2 AS result_rank
        }
        RETURN source, source_id, source_type, relation, target, target_id,
               target_type, source_records
        ORDER BY result_rank, source, target
        LIMIT $limit
    """,
}


class GraphRetrievalError(RuntimeError):
    """Neo4j 查询失败或返回坏数据时使用的明确错误。"""


class Neo4jGraphRetriever:
    """执行参数化 Neo4j 查询并转换为共享 ``GraphEvidence`` 模型。"""

    def __init__(
        self,
        *,
        driver_factory: Callable[[], Any] = get_neo4j_driver,
        database: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        self._driver_factory = driver_factory
        self.database = database or settings.neo4j_database
        self.timeout_seconds = timeout_seconds
        self.available = True

    def retrieve(
        self,
        *,
        entity: str,
        entity_id: str | None,
        intent: RetrievalIntent,
        limit: int = 5,
    ) -> list[GraphEvidence]:
        """按白名单意图读取图谱，返回至多 ``limit`` 条可追溯证据。

        组合意图固定最多展开 3 个候选方向，并为每个方向限制关联症状和检查，
        防止高连接度节点无界放大结果。
        """

        if intent not in QUERY_TEMPLATES:
            raise ValueError(f"不支持的图谱检索意图：{intent}")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        try:
            records, _, _ = self._driver_factory().execute_query(
                Query(QUERY_TEMPLATES[intent], timeout=self.timeout_seconds),
                # parameters_：患者相关内容全部参数化；
                parameters_={"entity": entity, "entity_id": entity_id, "limit": limit},
                database_=self.database,
            )
            return [self._to_evidence(row) for row in records]
        except (ValueError, GraphRetrievalError):
            raise
        except Exception as exc:
            self.available = False
            raise GraphRetrievalError(f"Neo4j 图谱查询失败：{exc}") from exc

    # 虽然写在类里面，但它不需要使用当前对象 self，把它当成普通函数使用
    @staticmethod
    def _to_evidence(row: Mapping[str, Any]) -> GraphEvidence:
        """ 把Neo4j 返回的一条记录转换为项目共享模型 GraphEvidence。"""

        try:
            return GraphEvidence(
                source_entity=row["source"],
                relation=row["relation"],
                target_entity=row["target"],
                source_entity_id=row.get("source_id"),
                target_entity_id=row.get("target_id"),
                source_entity_type=row.get("source_type"),
                target_entity_type=row.get("target_type"),
                source_records=list(dict.fromkeys(row.get("source_records") or [])),
                evidence_source="neo4j",
            )
        except Exception as exc:
            raise GraphRetrievalError(f"Neo4j 返回字段不符合 GraphEvidence 契约：{exc}") from exc


def retrieve_medical_graph(
    entity: str,
    intent: RetrievalIntent,
    limit: int = 5,
    *,
    entity_id: str | None = None,
) -> list[GraphEvidence]:
    """向后兼容的函数入口，内部使用新的可注入检索器。"""

    return Neo4jGraphRetriever().retrieve(
        entity=entity,
        entity_id=entity_id,
        intent=intent,
        limit=limit,
    )
