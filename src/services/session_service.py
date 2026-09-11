"""基于 LangGraph checkpointer 的多轮会话服务。

FastAPI 只负责 HTTP 收发；本模块负责判断一次请求是首轮还是续轮，并始终把
``session_id`` 放进 ``configurable.thread_id``。这样 LangGraph 的 checkpointer
会按会话隔离状态，续轮只需追加一条 ``HumanMessage``，不必由 API 自己复制整份
医疗状态。
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from src.config.settings import settings
from src.models.schemas import ConsultationSlots, PatientProfile


class SessionServiceError(RuntimeError):
    """会话服务可预期错误的基类，HTTP 层会将其转换为安全提示。"""


class SessionNotFoundError(SessionServiceError):
    """读取了尚未创建的会话。"""


class SessionClosedError(SessionServiceError):
    """阻止已完成或紧急终止的会话继续添加消息。"""


class SessionConflictError(SessionServiceError):
    """阻止续轮修改首轮患者档案。"""


class SessionService:
    """封装已编译 LangGraph 的创建、恢复和并发隔离逻辑。

    ``graph`` 需要提供 ``invoke`` 和 ``get_state``，通常是
    :func:`src.graph.workflow.build_medical_graph` 返回的已编译图。每个会话使用
    一把独立锁，避免同一患者的两个并发请求读取同一个旧检查点后相互覆盖；不同
    会话仍可并行执行。
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        # session-001 的两个请求不能同时修改状态； session-001 和 session-002 可以同时运行
        self._locks_guard = Lock()
        self._session_locks: dict[str, RLock] = {}

    # @staticmethod 表示该函数不需要读取 self
    @staticmethod
    def _config(session_id: str) -> dict[str, dict[str, str]]:
        """生成 LangGraph checkpointer 识别会话所需的固定配置。"""

        return {"configurable": {"thread_id": session_id}}

    def _lock_for(self, session_id: str) -> RLock:
        """惰性创建会话锁；守卫锁只保护锁字典本身。"""

        with self._locks_guard:
            return self._session_locks.setdefault(session_id, RLock())

    def _state_or_none(self, session_id: str) -> dict[str, Any] | None:
        """读取最近检查点；空 ``values`` 表示该 thread_id 从未运行。"""

        # get_state读取某个会话当前最新的工作流状态
        snapshot = self._graph.get_state(self._config(session_id))
        # 读取最新状态快照中的 State 数据
        values = dict(snapshot.values or {})
        return values if values.get("session_id") else None

    @staticmethod
    def _has_request_id(state: Mapping[str, Any], request_id: str) -> bool:
        """在持久化消息 ID 中查找幂等键，避免重复请求再次增加追问轮数。"""

        return any(
            getattr(message, "id", None) == request_id
            for message in state.get("messages", []) or []
        )

    # 创建状态快照中的 State 数据结构
    @staticmethod
    def _initial_state(
        *,
        session_id: str,
        message: str,
        patient_profile: PatientProfile,
        message_id: str,
    ) -> dict[str, Any]:
        """构造首轮完整状态；后续轮次不会再次调用本函数。"""

        return {
            "session_id": session_id,
            "messages": [HumanMessage(content=message, id=message_id)],
            "chief_complaint": message,
            "patient_profile": patient_profile.model_dump(),
            "consultation_slots": ConsultationSlots(symptom=[message]).model_dump(),
            "triage_result": None,
            "retrieved_evidence": [],
            "differential_evidence": [],
            "differential_directions": [],
            "check_evidence": [],
            "possible_evaluations": [],
            "current_question": None,
            "question_count": 0,
            "max_question_count": settings.max_question_count,
            "differential_question_count": 0,
            "need_graph_retrieval": False,
            "retrieval_intent": None,
            "retrieval_entities": [],
            "conversation_status": "triaging",
            "summary": None,
            "medical_record_draft": None,
            "errors": [],
            "audit_log": [],
        }

    def get_state(self, session_id: str) -> dict[str, Any]:
        """返回会话最近状态，供 API 查询；未知会话会明确报错。"""

        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("session_id 不能为空")
        with self._lock_for(normalized_id):
            state = self._state_or_none(normalized_id)
            if state is None:
                raise SessionNotFoundError("会话不存在，请先发送首条消息")
            return state

    def send_message(
        self,
        *,
        session_id: str,
        message: str,
        patient_profile: PatientProfile | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """创建或恢复会话并执行一轮工作流。

        首轮输入包含完整初始化字段；续轮输入只有新的 ``HumanMessage``。调用方可
        提供 ``request_id`` 作为幂等键：相同键再次到达时直接返回最近检查点，不会
        重跑 Agent。未提供键时每次请求都被视为患者的新回答。
        """

        normalized_id = session_id.strip()
        normalized_message = message.strip()
        normalized_request_id = request_id.strip() if request_id else None
        if not normalized_id:
            raise ValueError("session_id 不能为空")
        if not normalized_message:
            raise ValueError("message 不能为空或只包含空白字符")
        
        # 第二个请求必须等第一个请求完成 graph.invoke() 后才能继续
        with self._lock_for(normalized_id):
            previous = self._state_or_none(normalized_id)
            # 会话已存在；客户端传了 request_id；历史消息里已经出现这个 ID
            if previous and normalized_request_id and self._has_request_id(previous, normalized_request_id):
                return previous

            if previous:
                status = previous.get("conversation_status")
                if status in {"completed", "emergency_ended"}:
                    raise SessionClosedError(f"会话已经处于 {status} 状态，不能继续追加消息")

                # 患者档案属于首轮事实。续轮可重复提交完全相同的内容，但不能改写。
                if patient_profile is not None:
                    previous_profile = PatientProfile.model_validate(previous.get("patient_profile", {}))
                    if patient_profile != previous_profile:
                        raise SessionConflictError("患者基础资料只能在会话首轮设置")
                # 续轮追加只有 messages
                # 随机 ID 保证不同患者消息不会被 add_messages reducer 当成同一消息覆盖
                graph_input = {
                    "messages": [
                        HumanMessage(
                            content=normalized_message,
                            id=normalized_request_id or uuid4().hex,
                        )
                    ]
                }
            # 者没有提供档案，创建默认档案
            else:
                graph_input = self._initial_state(
                    session_id=normalized_id,
                    message=normalized_message,
                    patient_profile=patient_profile or PatientProfile(),
                    message_id=normalized_request_id or uuid4().hex,
    )
# graph_input首轮时是完整状态,续轮时只有messages
# MemorySaver 根据 thread_id： 找到旧检查点；将新输入合并进去；执行 LangGraph；保存新的检查点；返回执行后的完整状态。
            result = dict(self._graph.invoke(graph_input, config=self._config(normalized_id)))
            if result.get("session_id") != normalized_id:
                raise SessionServiceError("工作流返回了不一致的会话标识")
            return result
