"""预问诊 Agent 与知识检索工具之间的依赖注入契约。

Agent 只知道这里声明的能力，不需要知道 Neo4j、MySQL 或 Chroma 的连接细节。
工作流构建时可以注入真实工具或测试 fake；生产配置不得使用空工具掩盖数据库故障。
"""

from typing import Protocol

from src.models.schemas import GraphEvidence, RetrievalIntent


class MedicalKnowledgeTool(Protocol):
    """可提供只读医学图谱证据的工具接口。

    预问诊 Agent 调用 ``search`` 时传入待检索实体和白名单意图，得到证据列表；
    工具只能提供辅助信息，不能决定 LangGraph 下一跳或作出诊断。
    """
    def search(
        self,
        *,
        entities: list[str],
        intent: RetrievalIntent,
        limit: int = 5,
    ) -> list[GraphEvidence]:
        """按白名单检索意图返回至多 ``limit`` 条可解释的只读图谱证据。"""
