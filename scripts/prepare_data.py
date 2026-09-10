"""离线医学数据清洗、实体对齐与索引构建 数据准备命令；默认只构建本地产物，数据库始终是 dry-run。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 直接执行 ``python scripts/prepare_data.py`` 时，Python 默认只把 scripts 放进导入路径。
# 加入项目根目录后才能导入 src；这不会修改系统或 Conda 环境。
# resolve()相对路径转换为绝对路径
# parents[0]是直接父目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasync.entity_alignment import LocalBgeEmbedder
from src.datasync.exporters import (
    apply_chroma_documents, apply_mysql_mappings, apply_neo4j_graph,
    load_reviewed_mysql_mapping, validate_artifact_bundle,
)
from src.datasync.pipeline import PipelineConfig, run_pipeline


def _parser() -> argparse.ArgumentParser:
    """定义命令行参数；危险写入需要两个显式开关和配置开关。"""
    
    parser = argparse.ArgumentParser(description="构建 ai-medical 的 离线医学数据清洗、实体对齐与索引构建 检索数据产物")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/knowledge_graph/medical_kg.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--limit", type=int, help="仅处理前 N 行，适合快速验证")
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--use-embeddings", dest="use_embeddings", action="store_true", default=True,
        help="使用本地 BGE 做离线实体聚类（默认）",
    )
    # action="store_true" :默认关、显式开
    # action="store_false" :默认开、显式关
    embedding_group.add_argument(
        "--no-embeddings", dest="use_embeddings", action="store_false",
        help="显式降级为 exact 对齐，不加载 BGE",
    )
    parser.add_argument(
        "--embedding-model", type=Path,
        default=PROJECT_ROOT / "pretrained/bge-base-zh-v1.5",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("DATASYNC_BATCH_SIZE", "128")))
    parser.add_argument(
        "--cluster-threshold", type=float,
        default=float(os.getenv("ENTITY_CLUSTER_THRESHOLD", "0.88")),
    )
    parser.add_argument(
        "--review-threshold", type=float,
        default=float(os.getenv("ENTITY_REVIEW_THRESHOLD", "0.80")),
    )
    parser.add_argument(
        "--use-reviewed-mysql", action="store_true",
        help="只读加载 MySQL 中 review_status=1 的历史人工审核映射；失败时降级为空映射",
    )
    parser.add_argument("--apply", action="store_true", help="把产物幂等增量写入既有数据库")
    parser.add_argument(
        "--confirm-apply", action="store_true",
        help="确认已在命令执行前向用户报告影响范围并取得明确授权",
    )
    return parser


def _apply(output: Path) -> dict[str, int]:
    """执行三个非破坏性适配器；调用前已完成全部显式授权检查。"""

    import chromadb

    from src.config.settings import settings
    from src.services.database import close_database_connections, get_mysql_connection, get_neo4j_driver

    # 完整性校验必须先于任何数据库连接；失败时三个外部系统都不会被触碰。

    # 先校验 output 目录里的产物是否完整、格式是否正确
    '''文件 SHA-256 是否与 manifest.json 一致；
    JSONL 每行是否符合数据模型；
    节点、关系、映射和向量文档之间的 ID 是否一致；
    关系是否引用不存在的节点；
    当前不是 --limit 产生的抽样数据；
    本地产物是否确实由 BGE embedding 构建'''
    artifacts = validate_artifact_bundle(output, require_full=True, require_embeddings=True)
    result: dict[str, int] = {}
    '''依次写 MySQL、Neo4j、Chroma 三个库 → 无论成败都关连接 → 返回每个系统写入的条数统计'''
    try:
        with get_mysql_connection() as connection:
            ''' 新词条：插入 entity_mapping 表；
                同一个实体类型和同义词已存在：更新标准词和 ID；
                如果已有记录被标记为 review_status=1（人工审核），则保留人工审核结果，不会被本地数据覆盖；
                最后执行 connection.commit()，使 MySQL 写入真正生效。
                返回的受影响行数 '''
            result["mysql_mappings"] = apply_mysql_mappings(connection, artifacts.mappings)
        # 返回[ "neo4j_nodes": 节点处理数量,"neo4j_relations": 关系处理数量]
        # result.update(...) 会把这两个键合并到结果中
        result.update(apply_neo4j_graph(
            get_neo4j_driver(), artifacts.nodes, artifacts.relations,
            database=settings.neo4j_database,
        ))
        # 创建 Chroma 本地持久化客户端
        chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        # 取得已有的 Collection
        collection = chroma_client.get_collection(
            settings.chroma_collection
        )
        # 加载本地 BGE 向量模型
        embedder = LocalBgeEmbedder(settings.embedding_model_path, batch_size=settings.datasync_batch_size)
        # 确定 Chroma 最大批量大小
        max_batch_size = (
            chroma_client.get_max_batch_size()
            if hasattr(chroma_client, "get_max_batch_size") else 5000
        )
        # 写入/处理文档数
        result["chroma_documents"] = apply_chroma_documents(
            collection, artifacts.documents, embedder, batch_size=max_batch_size
        )
    except Exception as exc:
        print(json.dumps({
            "database_writes_performed": "partial" if result else False,
            "completed_writes": result,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    finally:
        close_database_connections()
    return result


def _reviewed_reference(enabled: bool, *, required: bool = False) -> dict:
    """可选读取审核词表；任何连接错误都明确告警并安全降级。"""
    # enabled ← 命令行 --use-reviewed-mysql（想不想用审核词表）
    # required ← 命令行 --apply（是不是要真写数据库）

    if not enabled:
        return {}
    from src.services.database import get_mysql_connection

    try:
        with get_mysql_connection() as connection:
            mapping = load_reviewed_mysql_mapping(connection)
        print(f"已只读加载 {len(mapping)} 条审核映射", file=sys.stderr)
        return mapping
    except Exception as exc:  # 外部依赖失败不应阻塞本地数据构建
        if required:
            raise SystemExit(
                f"拒绝 apply：无法读取人工审核映射：{type(exc).__name__}: {exc}"
            ) from exc
        print(f"警告：MySQL 审核映射不可用，降级为本地对齐：{type(exc).__name__}: {exc}", file=sys.stderr)
        return {}


def main(argv: list[str] | None = None) -> int:
    """构建并汇报影响范围；默认绝不连接数据库。"""
# parse_args() 是"把命令行输入变成程序里可用的数据"的那一步——识别参数、转换类型、填默认值、校验错误
    args = _parser().parse_args(argv)
    if not args.use_embeddings:
        print("警告：已显式禁用 BGE，未审核实体将保持 exact 独立映射", file=sys.stderr)
    config = PipelineConfig(
        input_path=args.input, output_dir=args.output, embedding_model_path=args.embedding_model,
        use_embeddings=args.use_embeddings, batch_size=args.batch_size,
        cluster_threshold=args.cluster_threshold, review_threshold=args.review_threshold,
        limit=args.limit,
    )
    if args.apply and not args.use_reviewed_mysql:
        raise SystemExit("拒绝 apply：必须同时传入 --use-reviewed-mysql，避免图谱与人工审核词表不一致")
    if args.apply and args.limit is not None:
        raise SystemExit("拒绝 apply：正式数据库导入不能使用 --limit 抽样产物")
    if args.apply and not args.use_embeddings:
        raise SystemExit("拒绝 apply：正式数据库导入必须启用本地 BGE")
    reviewed_mapping = _reviewed_reference(
        args.use_reviewed_mysql, required=args.apply
    )
    result = run_pipeline(config, reviewed_mapping=reviewed_mapping)
    impact = {
        "mode": "apply_requested" if args.apply else "dry-run",
        "output_dir": str(result.output_dir),
        "database_apply_requested": bool(args.apply),
        "database_writes_performed": False,
        "stats": result.stats.model_dump(mode="json"),
    }
    print(json.dumps(impact, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0
    if not args.confirm_apply:
        raise SystemExit("拒绝 apply：必须先取得用户明确授权，再同时传入 --apply --confirm-apply")
    from src.config.settings import settings

    if not settings.datasync_apply_enabled:
        raise SystemExit("拒绝 apply：DATASYNC_APPLY_ENABLED=false")
    writes = _apply(args.output)
    print(json.dumps({
        "database_writes_performed": True,
        "writes": writes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
