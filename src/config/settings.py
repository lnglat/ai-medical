from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """读取项目运行配置。

    密码和 API Key 使用 ``SecretStr`` 保存，避免它们因日志、异常或调试时打印
    Settings 对象而被意外显示。账号、数据库地址和模型名称仍由 ``.env`` 提供；
    非敏感的本地运行参数直接在代码中给出默认值。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # 外部服务配置提供不连接网络的安全默认值。真正创建连接时，连接工厂会再
    # 校验账号和密码；因此仅导入工作流或运行离线测试不再强制要求 .env。
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr | None = None
    # 当前本地 Neo4j Community 仅提供默认 neo4j 数据库，不支持 CREATE DATABASE。
    neo4j_database: str = "neo4j"
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: SecretStr | None = None
    mysql_database: str = "ai_medical"

    # 当前项目只使用 DeepSeek
    llm_provider: str = "deepseek"
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-chat"
    embedding_device: str = "cpu"
    max_question_count: int = 6
    # 产品目标要求真实 RAG；单元测试如需隔离外部服务，应显式传入 rag_enabled=False。
    rag_enabled: bool = True
    llm_enabled: bool = True
    checkpoint_backend: str = "memory"
    # 面试 trace 只读取 checkpointer 已有状态；默认关闭，避免普通部署暴露内部轨迹。
    demo_trace_enabled: bool = False

    # 相对路径由下方校验器转换为以项目根目录为基准的绝对路径。
    data_dir: Path = PROJECT_ROOT / "data"
    embedding_model_path: Path = PROJECT_ROOT / "pretrained/bge-base-zh-v1.5"
    chroma_dir: Path = PROJECT_ROOT / "data/vectorstore"

    # 离线医学数据清洗、实体对齐与索引构建 离线流水线参数都有安全默认值；只有显式启用 apply 开关才允许数据库写入。
    datasync_batch_size: int = 128
    entity_cluster_threshold: float = 0.88
    entity_review_threshold: float = 0.80
    datasync_apply_enabled: bool = False
    chroma_collection: str = "ai_medical"

    # 只要 Pydantic 创建配置对象并处理这三个字段中的任一个，就会调用下面的 resolve_project_relative_path() 
    # mode="before"校验器在 Pydantic 对字段执行默认类型转换和校验之前运行。
    @field_validator("data_dir", "embedding_model_path", "chroma_dir", mode="before")
    @classmethod # 下面的方法定义为类方法
    def resolve_project_relative_path(cls, value: str | Path) -> Path:
        """将相对数据路径转换成相对于项目根目录的绝对路径。"""
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
