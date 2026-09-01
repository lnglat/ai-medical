"""医疗预问诊 LangGraph 最小主流程。

本模块只负责把节点和条件边组装为状态机。真实 Agent、测试假 Agent 和知识
工具都从 ``build_medical_graph`` 注入，因此导入本模块或运行单元测试时不会
连接数据库，也不需要配置大模型密钥。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.agents.preconsult_agent import preconsult_node
from src.agents.summary_agent import summary_node
from src.agents.triage_agent import triage_node
from src.config.settings import settings
from src.graph.routers import route_after_preconsult, route_after_triage
from src.graph.state import MedicalState
from src.models.schemas import GraphEvidence
from src.tools.protocols import MedicalKnowledgeTool


NodeCallable = Callable[[MedicalState], dict[str, Any]]


# @dataclass 省掉写构造函数、打印对象、判断对象是否相等的重复代码
# frozen=True 保证里面装的函数引用在运行期间恒定不变。
@dataclass(frozen=True)
class WorkflowNodes:
    """工作流可替换的 Agent 节点集合，主要供测试和后续真实 Agent 接入。"""

    triage: NodeCallable = triage_node
    preconsult: NodeCallable = preconsult_node
    summary: NodeCallable = summary_node


def intake_node(state: MedicalState) -> dict:
    """登记本轮流程的执行日志到audit_log，并将控制权交给分诊节点。"""
    return {"conversation_status": "triaging", "audit_log": [{"node": "intake"}]}


def make_graph_retrieval_node(
    tool: MedicalKnowledgeTool | None,
    *,
    rag_enabled: bool,
) -> NodeCallable:
    """创建绑定工具依赖的检索节点。

    节点从预问诊 Agent 写入状态的实体和意图读取请求；RAG 关闭时不调用工具。
    无论证据是否为空，都会清除本次检索请求，保证回到预问诊节点时不会死循环。
    """

    def graph_retrieval_node(state: MedicalState) -> dict:
        '''
        retrieval_entities：待查询实体，例如 ["头痛"]
        retrieval_intent：检索意图，例如 "symptom_to_department"
        need_graph_retrieval：预问诊节点是否请求检索
        '''
        entities = state.get("retrieval_entities", [])
        intent = state.get("retrieval_intent")
        # list[GraphEvidence] 是类型标注，说明它应该是“由 GraphEvidence 对象组成的列表”,实际赋值为[]
        evidence: list[GraphEvidence] = []

        if rag_enabled and entities and intent:
            '''
            单元测试显式设置 rag_enabled=False 时绝不访问图谱工具。
            没有实体或意图时不发起无意义查询。
            真实工具、空工具和测试工具都可以被替换使用。
            '''
            # build_medical_graph 已保证启用 RAG 时必须注入真实工具；这里的断言
            # 同时帮助类型检查器理解 tool 不会是 None。
            assert tool is not None
            evidence = tool.search(entities=entities, intent=intent)
        return {
            "retrieved_evidence": [item.model_dump() for item in evidence],
            "need_graph_retrieval": False,
            # 保留 intent 作为“本轮已尝试检索”的标记；预问诊 Agent 据此不重复请求。
            "retrieval_entities": [],
            "audit_log": [
                {
                    "node": "graph_retrieval",
                    "rag_enabled": rag_enabled,
                    # 检索后状态会清空 retrieval_entities；审计中保留最小请求摘要，
                    # 供 安全、错误处理与可观测性 只读 trace 展示，绝不保存完整患者消息。
                    "retrieval_intent": intent,
                    "retrieval_entities": list(entities),
                    "evidence_count": len(evidence),
                }
            ],
        }

    return graph_retrieval_node


def emergency_end_node(state: MedicalState) -> dict:
    """会话状态标记为紧急结束状态，清空当前追问，写入审计日志。"""
    return {
        "conversation_status": "emergency_ended",
        "current_question": None,
        "audit_log": [{"node": "emergency_end"}],
    }


def build_medical_graph(
    *,
    # 保存和恢复多轮会话状态
    checkpointer: BaseCheckpointSaver | None = None,
    # 注入真实、空实现或测试用检索工具
    knowledge_tool: MedicalKnowledgeTool | None = None,
    # 临时覆盖配置中的 RAG 开关
    rag_enabled: bool | None = None,
    # 注入真实或测试用 Agent 节点
    nodes: WorkflowNodes | None = None,
):
    """编译可运行的医疗问诊状态图。
    
    ``checkpointer`` 由调用方传入，例如测试的内存 saver 或将来的会话服务；
    ``nodes`` 支持注入假 Agent。单元测试可显式关闭 RAG 来隔离外部服务；产品
    配置启用 RAG 时必须注入真实知识工具，否则构建阶段立即报错。
    """
    active_nodes = nodes or WorkflowNodes()
    active_rag_enabled = settings.rag_enabled if rag_enabled is None else rag_enabled
    if active_rag_enabled and knowledge_tool is None:
        raise ValueError("启用 RAG 时必须注入真实 MedicalKnowledgeTool")

    graph = StateGraph(MedicalState)
    graph.add_node("intake", intake_node)
    graph.add_node("triage", active_nodes.triage)
    graph.add_node("preconsult", active_nodes.preconsult)
    graph.add_node(
        "graph_retrieval",
        make_graph_retrieval_node(knowledge_tool, rag_enabled=active_rag_enabled),
    )
    graph.add_node("summary", active_nodes.summary)
    graph.add_node("emergency_end", emergency_end_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "triage")
    graph.add_conditional_edges("triage", route_after_triage)
    graph.add_conditional_edges(
        "preconsult",
        route_after_preconsult,
        {
            "retrieve": "graph_retrieval",
            "ask_user": END,
            "summarize": "summary",
            "failed": END,
        },
    )
    graph.add_edge("graph_retrieval", "preconsult")
    graph.add_edge("summary", END)
    graph.add_edge("emergency_end", END)
    return graph.compile(checkpointer=checkpointer)
