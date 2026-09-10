"""真实三库初始化与正式数据导入 数据库对象初始化命令。

默认模式只连接外部服务做只读检查并打印计划；只有同时传入 ``--apply`` 和
``--confirm-init`` 才执行幂等 DDL。命令不会导入业务数据，数据导入仍由
``scripts/prepare_data.py`` 的 离线医学数据清洗、实体对齐与索引构建 适配器完成。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasync.exporters import validate_artifact_bundle
from src.datasync.graph_builder import LABELS

# 固定使用的 MySQL 表
_MYSQL_TABLE = "entity_mapping"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_identifier(value: str, *, field: str) -> str:
    """校验只能来自配置、但必须写入 DDL 的数据库标识符。

    数据值可以使用 SQL 参数，数据库名却不能；因此这里使用严格白名单，避免把
    ``MYSQL_DATABASE`` 或 ``NEO4J_DATABASE`` 中的任意文本拼进管理语句。
    """

    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} 只能包含英文字母、数字和下划线")
    return value


def _constraint_name(label: str) -> str:
    '''小写初始化'''
    return f"{label.lower()}_id_unique"


def expected_neo4j_constraints() -> dict[str, str]:
    """返回“约束名 → 节点标签”，与 DATA_CONTRACT 的八类实体一一对应。"""

    return {_constraint_name(label): label for label in sorted(set(LABELS.values()))}


def _mysql_server_connection(settings: Any):
    """连接 MySQL 服务本身，而非目标库，使尚未建库的环境也能被初始化。"""

    import pymysql

    if not settings.mysql_user or settings.mysql_password is None:
        raise RuntimeError("缺少 MYSQL_USER 或 MYSQL_PASSWORD")
    password = settings.mysql_password.get_secret_value()
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=password,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def inspect_mysql(connection: Any, database: str) -> dict[str, bool]:
    """只读检查项目库、映射表、联合唯一键和 ID 普通索引是否存在。"""
    # information_schema保存数据库本身的结构信息
    with connection.cursor() as cursor:
        # SCHEMATA：数据库列表
        cursor.execute(
            "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s",
            (database,),
        )
        database_exists = cursor.fetchone() is not None # 返回bool
        # TABLES：表和视图列表
        cursor.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
            (database, _MYSQL_TABLE),
        )
        table_exists = cursor.fetchone() is not None
        # 读取索引信息
        # INDEX_NAME：索引名称。NON_UNIQUE：0 表示唯一，1 表示非唯一。
        # SEQ_IN_INDEX：字段在联合索引里的顺序。COLUMN_NAME：字段名
        cursor.execute(
            "SELECT INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME "
            "FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY INDEX_NAME, SEQ_IN_INDEX",
            (database, _MYSQL_TABLE),
        )
        rows = cursor.fetchall()

    # 把多行索引信息组合起来
    indexes: dict[str, dict[str, Any]] = {}
    for row in rows:
        # setdefault(): 索引第一次出现：创建结构。同一个联合索引后续字段：复用原结构
        item = indexes.setdefault(
            row["INDEX_NAME"], {"unique": not bool(row["NON_UNIQUE"]), "columns": []}
        )
        item["columns"].append(row["COLUMN_NAME"])
    # 识别两个KEY是否为唯一键    
    composite_unique = any(
        item["unique"] and item["columns"] == ["entity_schema", "synonym"]
        for item in indexes.values()
    )
    hash_unique = any(
        item["unique"] and item["columns"] == ["entity_schema", "synonym_hash"]
        for item in indexes.values()
    )
    # 确认是否建立索引
    id_indexes = [item for item in indexes.values() if item["columns"] and item["columns"][0] == "id"]
    return {
        "database_exists": database_exists,
        "table_exists": table_exists,
        "composite_unique_exists": composite_unique,
        "hash_unique_exists": hash_unique,
        # id_index_exists=True：可以通过稳定 ID 加快查询
        "id_index_exists": bool(id_indexes),
        # id_has_unique_index=False：同一标准实体 ID 允许对应多个 synonym
        "id_has_unique_index": any(item["unique"] for item in id_indexes),
    }


def initialize_mysql(connection: Any, database: str) -> None:
    """创建缺失的 MySQL 对象；所有语句均可安全重复执行。"""

    name = _safe_identifier(database, field="MYSQL_DATABASE")
    statements = [
        f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
        f"""CREATE TABLE IF NOT EXISTS `{name}`.`{_MYSQL_TABLE}` (
            `id` VARCHAR(64) NOT NULL,
            `synonym` TEXT NOT NULL,
            `std_name` TEXT NOT NULL,
            `entity_schema` VARCHAR(32) NOT NULL,
            `synonym_hash` BINARY(32) NOT NULL,
            `review_status` TINYINT UNSIGNED NOT NULL DEFAULT 0,
            PRIMARY KEY (`entity_schema`, `synonym_hash`),
            KEY `idx_entity_mapping_id` (`id`),
            KEY `idx_entity_mapping_lookup` (`entity_schema`, `synonym`(191))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    ]
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    connection.commit()
    state = inspect_mysql(connection, name)
    # 如果 id 是唯一键，会阻止“同一标准实体对应多个 synonym”
    if state["id_has_unique_index"]:
        raise RuntimeError("entity_mapping.id 存在唯一索引，需人工迁移为普通索引后再初始化")
    schema_changed = False
    with connection.cursor() as cursor:
        if not state["composite_unique_exists"] and not state["hash_unique_exists"]:
            raise RuntimeError(
                "entity_mapping 缺少同义词联合唯一键；为避免破坏旧表，拒绝自动迁移"
            )
        if not state["id_index_exists"]:
            cursor.execute(
                f"ALTER TABLE `{name}`.`{_MYSQL_TABLE}` ADD KEY `idx_entity_mapping_id` (`id`)"
            )
            schema_changed = True
    if schema_changed:
        connection.commit()


def inspect_neo4j(driver: Any, database: str) -> dict[str, Any]:
    """从 system 库检查目标库，再从目标库读取已有约束；不会执行写操作。"""
    # system 是 Neo4j 内置的管理数据库，专门用于查询和管理数据库列表
    with driver.session(database="system") as session:
        databases = {row["name"] for row in session.run("SHOW DATABASES YIELD name RETURN name")}
    if database not in databases:
        return {"database_exists": False, "existing_constraints": []}
    # 读取约束名
    with driver.session(database=database) as session:
        names = [
            row["name"]
            for row in session.run("SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name")
        ]
    return {"database_exists": True, "existing_constraints": names}


def initialize_neo4j(driver: Any, database: str) -> None:
    """显式创建目标 Neo4j 库及八个 ID 唯一约束，不删除任何已有对象。"""

    name = _safe_identifier(database, field="NEO4J_DATABASE")
    # 检查数据库是否存在
    state = inspect_neo4j(driver, name)
    if not state["database_exists"]:
        # 目标库不存在，尝试从 system 库创建
        with driver.session(database="system") as session:
            # .consume() 强制驱动消费结果，确保语句真正执行完成
            session.run(f"CREATE DATABASE `{name}` IF NOT EXISTS").consume()
    with driver.session(database=name) as session:
        # 八种标签逐一创建 ID 唯一约束
        for constraint, label in expected_neo4j_constraints().items():
            session.run(
                f"CREATE CONSTRAINT `{constraint}` IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE n.id IS UNIQUE"
            ).consume()


def inspect_chroma(path: Path, collection_name: str) -> dict[str, Any]:
    """检查 Chroma collection；目录不存在时不创建客户端，保持计划模式无写入。"""

    if not path.exists():
        return {"directory_exists": False, "collection_exists": False}
    import chromadb
    # 连接本地持久化目录
    client = chromadb.PersistentClient(path=str(path))
    names = {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }
    return {"directory_exists": True, "collection_exists": collection_name in names}


def initialize_chroma(path: Path, collection_name: str) -> None:
    """创建或复用契约规定的 collection；绝不清空已有文档。"""

    import chromadb
    # parents=True：父目录不存在时一起创建。
    # exist_ok=True：目录已经存在时不报错。
    path.mkdir(parents=True, exist_ok=True)
    chromadb.PersistentClient(path=str(path)).get_or_create_collection(name=collection_name)


def build_impact_plan(artifacts: Any, settings: Any) -> dict[str, Any]:
    """把本次产物规模和计划对象整理成不含密码的可审计输出。"""

    return {
        "mode": "read_only_plan",
        # 统计计划导入的对象数量,即本地数据的数量
        "artifact_counts": {
            "mysql_mappings": len(artifacts.mappings),
            "neo4j_nodes": len(artifacts.nodes),
            "neo4j_relations": len(artifacts.relations),
            "chroma_documents": len(artifacts.documents),
        },
        # 明确数据库的配置
        "targets": {
            "mysql": {"database": settings.mysql_database, "table": _MYSQL_TABLE},
            "neo4j": {
                "database": settings.neo4j_database,
                "constraints": list(expected_neo4j_constraints()),
            },
            "chroma": {
                "path": str(settings.chroma_dir),
                "collection": settings.chroma_collection,
            },
        },
        "prohibited_operations": ["DROP", "DELETE", "DETACH DELETE", "collection reset"],
    }


def _parser() -> argparse.ArgumentParser:
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="检查或幂等初始化 真实三库初始化与正式数据导入 数据库对象")
    # type=Path 会把字符串转换为 Path 对象
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--apply", action="store_true", help="执行幂等初始化")
    parser.add_argument(
        "--confirm-init", action="store_true",
        help="确认已向用户报告影响范围并取得本次真实初始化授权",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """先验证产物和检查现状；计划输出完成后才可能进入显式写入分支。"""

    args = _parser().parse_args(argv)
    if args.confirm_init and not args.apply:
        raise SystemExit("--confirm-init 只能与 --apply 同时使用")
    from src.config.settings import settings
    from src.services.database import close_database_connections, get_neo4j_driver

    artifacts = validate_artifact_bundle(
        args.processed_dir, require_full=True, require_embeddings=True
    )
    plan = build_impact_plan(artifacts, settings)
    mysql_connection = None
    try:
        # 连接mysql但尚未创建数据库
        mysql_connection = _mysql_server_connection(settings)
        driver = get_neo4j_driver()
        plan["current_state"] = {
            "mysql": inspect_mysql(mysql_connection, settings.mysql_database),
            "neo4j": inspect_neo4j(driver, settings.neo4j_database),
            "chroma": inspect_chroma(settings.chroma_dir, settings.chroma_collection),
        }
        # indent=2：两空格缩进，便于人工审阅。
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        # 打印计划后立即成功退出
        if not args.apply:
            return 0
        if not args.confirm_init:
            raise SystemExit("拒绝初始化：请先报告以上影响范围并取得授权，再传入 --apply --confirm-init")

        initialize_mysql(mysql_connection, settings.mysql_database)
        initialize_neo4j(driver, settings.neo4j_database)
        initialize_chroma(settings.chroma_dir, settings.chroma_collection)
        print(json.dumps({"database_initialization": "completed"}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {
            "database_initialization": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            # 即使凭据缺失，也返回已经离线验证过的对象和导入规模；其中不含密码。
            "impact_plan": plan,
        }
        # sys.stderr：错误输出通道
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    finally:
        if mysql_connection is not None:
            mysql_connection.close()
        close_database_connections()


if __name__ == "__main__":
    # 会把返回值作为进程退出码交给终端
    raise SystemExit(main())
