from functools import lru_cache

from neo4j import GraphDatabase
from src.config.settings import settings


@lru_cache
def get_neo4j_driver():
    """惰性创建 Neo4j 驱动；只有真实访问时才要求密码。"""

    if settings.neo4j_password is None or not settings.neo4j_password.get_secret_value():
        raise RuntimeError("缺少 NEO4J_PASSWORD，无法连接 Neo4j")
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )


def get_mysql_config() -> dict:
    """返回 MySQL 连接参数；连接本身仍由调用方按需创建。"""

    # 本地 MySQL 开发实例可能明确配置为空密码。``None`` 表示配置项缺失，空字符串
    # 则表示用户确实选择了无密码账号；两者不能混为一谈。
    if not settings.mysql_user or settings.mysql_password is None:
        raise RuntimeError("缺少 MYSQL_USER 或 MYSQL_PASSWORD 配置，无法连接 MySQL")

    return {
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "user": settings.mysql_user,
        "password": settings.mysql_password.get_secret_value(),
        "database": settings.mysql_database,
        "charset": "utf8mb4",
    }


def get_mysql_connection():
    """惰性创建字典游标连接，模块导入时不会访问数据库。"""

    import pymysql

    return pymysql.connect(**get_mysql_config(), cursorclass=pymysql.cursors.DictCursor)


def close_database_connections() -> None:
    """关闭缓存的 Neo4j 驱动；MySQL 短连接由其上下文管理器关闭。"""

    if get_neo4j_driver.cache_info().currsize:
        get_neo4j_driver().close()
        get_neo4j_driver.cache_clear()
