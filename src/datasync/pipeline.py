"""串联 inspect 检查输入 → clean 清洗和校验 → align 实体对齐 → build_graph 构建图节点和关系 
→ build_vectors 生成向量文档 → validate 检查产物是否合法 → export 写入 data/processed。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.datasync.cleaning import clean_jsonl, inspect_jsonl
from src.datasync.entity_alignment import LocalBgeEmbedder, TextEmbedder, align_entities, collect_entities
from src.datasync.exporters import write_json, write_jsonl
from src.datasync.graph_builder import build_graph, validate_graph
from src.datasync.schemas import EntityType, PipelineStats
from src.datasync.vector_builder import build_vector_documents


@dataclass(frozen=True)
class PipelineConfig:
    """流水线的可复现配置；数据库 apply 不属于本地构建过程。"""

    input_path: Path
    output_dir: Path
    embedding_model_path: Path | None = None
    use_embeddings: bool = True
    batch_size: int = 128
    cluster_threshold: float = 0.88
    review_threshold: float = 0.80
    limit: int | None = None


@dataclass(frozen=True)
class PipelineResult:
    """返回给 CLI 和测试的产物路径、数量及内存结果。"""

    stats: PipelineStats
    output_dir: Path
    manifest: dict


def run_pipeline(
    config: PipelineConfig,
    *,
    reviewed_mapping: Mapping[tuple[EntityType, str], str] | None = None,
    embedder: TextEmbedder | None = None,
) -> PipelineResult:
    """执行完整离线构建并写本地产物，不连接或写入任何数据库。

    ``reviewed_mapping`` 和 ``embedder`` 都通过依赖注入提供；因此外部服务失败时，
    调用方可以传空映射或禁用嵌入，流水线仍会以 exact 模式安全完成。
    """
    # 阻止错误配置，例如审核阈值比自动合并阈值更高，或批次大小为负数。
    if not 0.0 <= config.review_threshold <= config.cluster_threshold <= 1.0:
        raise ValueError("阈值必须满足 0 <= review_threshold <= cluster_threshold <= 1")
    if config.batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit 必须大于 0")
    # 输入
    profile = inspect_jsonl(config.input_path)
    # 清洗
    cleaned, rejected = clean_jsonl(config.input_path, limit=config.limit)
    # 统计
    counters = collect_entities(cleaned)
    # 测试可注入fake embedder；生产环境可使用本地 BGE。两者都通过同一个 TextEmbedder 接口调用。
    if embedder is None and config.use_embeddings:
        if config.embedding_model_path is None:
            raise ValueError("启用嵌入时必须提供 embedding_model_path")
        embedder = LocalBgeEmbedder(config.embedding_model_path, batch_size=config.batch_size)
    # 实体对齐
    mappings = align_entities(
        counters, reviewed_mapping=reviewed_mapping, embedder=embedder,
        cluster_threshold=config.cluster_threshold, review_threshold=config.review_threshold,
    )
    # 创建图节点和关系
    nodes, relations = build_graph(cleaned, mappings)
    # 创建双向邻接表
    documents = build_vector_documents(nodes, relations)
    validate_graph(nodes, relations)

    # 统计
    stats = PipelineStats(
        input_count=len(cleaned) + len(rejected), cleaned_count=len(cleaned),
        rejected_count=len(rejected), entity_count=len(nodes), mapping_count=len(mappings),
        low_confidence_count=sum(item.needs_review for item in mappings),
        node_count=len(nodes), relation_count=len(relations),
        vector_document_count=len(documents),
    )
    output = config.output_dir
    # 每次 write_jsonl() 都返回该文件的 SHA256，存进 manifest。
    artifact_digests = {
        "cleaned_medical_records.jsonl": write_jsonl(output / "cleaned_medical_records.jsonl", cleaned),
        "entity_mapping.jsonl": write_jsonl(output / "entity_mapping.jsonl", mappings),
        "graph_nodes.jsonl": write_jsonl(output / "graph_nodes.jsonl", nodes),
        "graph_relations.jsonl": write_jsonl(output / "graph_relations.jsonl", relations),
        "vector_documents.jsonl": write_jsonl(output / "vector_documents.jsonl", documents),
        "rejected_records.jsonl": write_jsonl(output / "rejected_records.jsonl", rejected),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "local_build_database_dry_run",
        "input": profile.model_dump(mode="json"),
        "parameters": {
            "batch_size": config.batch_size,
            "cluster_threshold": config.cluster_threshold,
            "review_threshold": config.review_threshold,
            "limit": config.limit,
            "embedding_enabled": embedder is not None,
            "embedding_model": str(config.embedding_model_path) if embedder is not None else None,
            "embedding_device": "cpu" if embedder is not None else None,
            "embedding_dimension": getattr(embedder, "dimension", None) if embedder is not None else None,
            "clustering_strategy": "representative_cosine_exact_or_deterministic_lsh",
            "standard_selection": "frequency_desc,information_completeness_desc,unicode_asc",
        },
        "stages": {
            "inspect": {"valid_json_records": profile.valid_json_records, "invalid_json_records": profile.invalid_json_records},
            "clean": {"cleaned_count": len(cleaned), "rejected_count": len(rejected)},
            "align": {"mapping_count": len(mappings), "low_confidence_count": stats.low_confidence_count},
            "build_graph": {"node_count": len(nodes), "relation_count": len(relations)},
            "build_vectors": {"vector_document_count": len(documents)},
            "validate": {"dangling_relations": 0, "duplicate_vector_ids": 0},
            "export": {"artifact_count": len(artifact_digests)},
        },
        "stats": stats.model_dump(mode="json"),
        "artifact_sha256": artifact_digests,
    }
    write_json(output / "manifest.json", manifest)
    return PipelineResult(stats=stats, output_dir=output, manifest=manifest)
