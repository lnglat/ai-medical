"""FastAPI 应用入口和 LangGraph 会话响应映射。

模块导入只创建轻量的 FastAPI 路由，不连接 MySQL、Neo4j、Chroma，也不加载
BGE。注册 HTTP 路由；把 LangGraph 状态转换成 API 响应。测试可向 :func:`create_app` 注入
使用 fake Agent 的 ``SessionService``，因此默认 API 测试完全离线。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path as FilePath
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Path, status
from fastapi.staticfiles import StaticFiles

# 模型负责校验 HTTP 输入输出
from src.api.schemas import (
    CompletedResponse, # 问诊完成
    ConsultationResponse, # 前面几种问诊响应的联合类型
    DemoTraceResponse, # 面试演示所需的只读安全轨迹
    DependencyHealthResponse, # 业务依赖状态
    EmergencyResponse, # 紧急风险终止
    FailedResponse, # 工作流失败
    HealthResponse, # 应用存活状态
    MessageRequest, # 患者发送的请求。
    NormalizationTrace,
    RootResponse, # 首页导航信息
    TraceAuditEvent,
    WaitingUserResponse, # 系统还要继续追问
)
from src.config.settings import settings
from src.graph.workflow import build_medical_graph
from src.models.schemas import (
    GraphEvidence,
    MedicalError,
    MedicalRecordDraft,
    PreconsultSummary,
    TriageResult,
)
from src.services.session_service import (
    SessionClosedError,
    SessionConflictError,
    SessionNotFoundError,
    SessionService,
)


FRONTEND_DIST = FilePath(__file__).resolve().parents[2] / "frontend" / "dist"
logger = logging.getLogger("ai_medical.api")


def build_default_session_service() -> SessionService:
    """按配置构建生产会话服务，但只在首个患者请求到达时调用。

    当前 多轮会话恢复、Checkpointer 与 FastAPI 只实现内存 checkpointer；其他后端必须以后单独配置，不能隐式在旧
    MySQL 中建会话表。RAG 开启时使用 可插拔图谱 RAG 检索 的真实工具工厂，不提供空工具降级。
    """

    if settings.checkpoint_backend != "memory":
        raise RuntimeError(
            f"暂不支持 CHECKPOINT_BACKEND={settings.checkpoint_backend!r}；只支持内存检查点"
        )

    from langgraph.checkpoint.memory import MemorySaver

    knowledge_tool = None
    if settings.rag_enabled:
        # 延迟导入使 /health 和 OpenAPI 文档不依赖数据库或嵌入模型。
        from src.tools.knowledge_tool import build_medical_knowledge_tool

        knowledge_tool = build_medical_knowledge_tool()
    graph = build_medical_graph(
        # 让 LangGraph 保存每个 thread_id 的状态
        checkpointer=MemorySaver(),
        # 注入图谱 RAG 工具
        knowledge_tool=knowledge_tool,
        # 是否启用检索分支
        rag_enabled=settings.rag_enabled,
    )
    return SessionService(graph)


def _safe_failed_response(
    session_id: str,
    *,
    code: str,
    message: str,
    category: str = "internal",
) -> FailedResponse:
    """构造不会泄露内部异常和密钥的统一失败响应。"""

    error = MedicalError(
        category=category,
        code=code,
        message=message,
        retryable=True,
        node="api",
    )
    return FailedResponse(session_id=session_id, errors=[error], message=message)


def _safe_error_category(exc: Exception) -> tuple[str, str]:
    """只根据异常类型和依赖标签分类，不把异常正文返回或写入日志。"""

    current: BaseException | None = exc
    for _ in range(8):
        if current is None:
            break
        dependency = str(getattr(current, "dependency", "")).lower()
        name = type(current).__name__.lower()
        if dependency in {"mysql", "chroma", "neo4j"} or "graphretrieval" in name:
            return "database", "rag_dependency_failed"
        if dependency == "embedding" or "retrieval" in name:
            return "retrieval", "rag_retrieval_failed"
        current = current.__cause__ or current.__context__
    return "internal", "session_execution_failed"


def _request_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """提取不含患者消息的请求摘要，供结构化日志和排障使用。"""

    audit = state.get("audit_log", []) or []
    errors = state.get("errors", []) or []
    return {
        "nodes": [str(item.get("node")) for item in audit if isinstance(item, Mapping) and item.get("node")],
        "status": str(state.get("conversation_status", "unknown")),
        "evidence_count": len(state.get("retrieved_evidence", []) or []),
        "error_categories": [
            str(item.get("category"))
            for item in errors
            if isinstance(item, Mapping) and item.get("category")
        ],
    }


def _state_to_trace(state: Mapping[str, Any]) -> DemoTraceResponse:
    """把检查点状态压缩成白名单 trace，不重新查询任何外部依赖。"""

    audit = state.get("audit_log", []) or []
    retrieval_event: Mapping[str, Any] = {}
    for item in reversed(audit):
        if isinstance(item, Mapping) and item.get("node") == "graph_retrieval":
            retrieval_event = item
            break

    intent = state.get("retrieval_intent") or retrieval_event.get("retrieval_intent")
    raw_entities = state.get("retrieval_entities") or retrieval_event.get("retrieval_entities") or []
    retrieval_entities = [str(item) for item in raw_entities]
    evidence = [
        GraphEvidence.model_validate(item)
        for item in (state.get("retrieved_evidence", []) or [])
    ]

    normalizations: list[NormalizationTrace] = []
    seen: set[tuple[str, str, str]] = set()
    for item in evidence:
        source = item.evidence_source or ""
        method = source.removesuffix("+neo4j")
        if method not in {"mysql_exact", "chroma_semantic"}:
            continue
        input_entity = retrieval_entities[0] if len(retrieval_entities) == 1 else item.source_entity
        key = (input_entity, item.source_entity, method)
        if key in seen:
            continue
        seen.add(key)
        normalizations.append(
            NormalizationTrace(
                input_entity=input_entity,
                normalized_entity=item.source_entity,
                method=method,
                score=item.score,
            )
        )

    audit_events = [
        TraceAuditEvent(
            node=str(item["node"]),
            route=str(item["route"]) if item.get("route") is not None else None,
            evidence_count=int(item["evidence_count"])
            if item.get("evidence_count") is not None else None,
            error_category=str(item["error_category"])
            if item.get("error_category") is not None else None,
        )
        for item in audit
        if isinstance(item, Mapping) and item.get("node")
    ]
    return DemoTraceResponse(
        available=bool(intent or retrieval_entities or evidence),
        retrieval_intent=intent,
        retrieval_entities=retrieval_entities,
        normalization_results=normalizations,
        retrieved_evidence=evidence,
        audit_events=audit_events,
    )


def state_to_response(state: Mapping[str, Any]) -> ConsultationResponse:
    """把工作流最终状态校验为四种稳定 HTTP 响应之一。"""

    session_id = str(state.get("session_id", ""))
    conversation_status = state.get("conversation_status")
    if conversation_status == "waiting_user":
        triage_data = state.get("triage_result")
        return WaitingUserResponse(
            session_id=session_id,
            current_question=str(state.get("current_question") or ""),
            question_count=int(state.get("question_count", 0)),
            triage_result=TriageResult.model_validate(triage_data) if triage_data else None,
        )
    if conversation_status == "emergency_ended":
        return EmergencyResponse(
            session_id=session_id,
            triage_result=TriageResult.model_validate(state.get("triage_result")),
            message="检测到需要立即线下评估的风险信号，请尽快联系急救或前往急诊。",
        )
    if conversation_status == "completed":
        return CompletedResponse(
            session_id=session_id,
            summary=PreconsultSummary.model_validate(state.get("summary")),
            medical_record_draft=MedicalRecordDraft.model_validate(
                state.get("medical_record_draft")
            ),
        )
    if conversation_status == "failed":
        return FailedResponse(
            session_id=session_id,
            errors=[MedicalError.model_validate(item) for item in state.get("errors", [])],
            message="本次预问诊暂时无法继续，请稍后重试或联系线下医务人员。",
        )
    raise ValueError(f"工作流停在不可返回的状态：{conversation_status!r}")


def create_app(
    *,
    # # 允许直接传入一个已经创建好的服务，主要用于测试
    session_service: SessionService | None = None, 
    # 第一次患者请求到达时调用这个工厂
    session_service_factory: Callable[[], SessionService] = build_default_session_service,
    frontend_dist: FilePath | None = FRONTEND_DIST,
) -> FastAPI:
    """创建可注入依赖的 FastAPI 应用。

    参数 ``session_service`` 供离线测试直接注入；未传入时，通过带锁的延迟工厂
    创建生产服务，避免并发首请求重复初始化大型依赖。
    """

    application = FastAPI(title="ai-medical", description="多 Agent 预问诊演示系统")
    application.state.session_service = session_service
    application.state.session_service_factory = session_service_factory
    # 创建一把全局初始化锁
    application.state.session_service_lock = Lock()
    application.state.session_service_error = None

    def get_service() -> SessionService:
        """取得应用级单例会话服务，初始化失败时保留安全状态供健康检查读取。"""

        if application.state.session_service is not None:
            return application.state.session_service
        # 同一时间只允许一个线程初始化服务
        with application.state.session_service_lock:
            if application.state.session_service is None:
                try:
                    application.state.session_service = application.state.session_service_factory()
                    application.state.session_service_error = None
                except Exception as exc:
                    application.state.session_service_error = type(exc).__name__
                    raise
        return application.state.session_service

    @application.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        """返回轻量首页和常用入口，不初始化会话、数据库或嵌入模型。"""

        return RootResponse(
            service="ai-medical",
            message="AI 医疗预问诊服务已启动。本系统仅用于风险分诊和信息整理，不构成诊断。",
            docs_url="/docs",
            health_url="/health",
            dependencies_health_url="/health/dependencies",
        )

    # 装饰器把下面的函数注册成 GET 路由
    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """只确认 Web 应用存活，绝不因健康探针加载数据库或模型。"""

        return HealthResponse(checkpoint_backend=settings.checkpoint_backend)

    @application.get(
        "/api/health/dependencies",
        response_model=DependencyHealthResponse,
        include_in_schema=False,
    )
    @application.get("/health/dependencies", response_model=DependencyHealthResponse)
    def dependency_health() -> DependencyHealthResponse:
        """报告会话/外部依赖是否已初始化，不主动触发昂贵连接。"""

        if application.state.session_service is not None:
            return DependencyHealthResponse(status="ready", session_service="ready")
        if application.state.session_service_error:
            return DependencyHealthResponse(
                status="unavailable",
                session_service="unavailable",
                detail="会话依赖初始化失败，请检查项目数据和外部服务配置。",
            )
        return DependencyHealthResponse(
            status="not_initialized",
            session_service="not_initialized",
            detail="尚无患者请求，外部依赖未被加载或探测。",
        )

    def checked_service(session_id: str) -> SessionService:
        """将延迟初始化失败转换成不泄露内部堆栈的 HTTP 503。"""

        try:
            return get_service()
        except Exception as exc:
            response = _safe_failed_response(
                session_id,
                code="dependencies_unavailable",
                message="问诊依赖暂时不可用，请稍后重试。",
            )
            # 用指定的 HTTP 状态码和错误内容返回给客户端
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.model_dump(mode="json"),
            ) from exc

    @application.get(
        "/api/sessions/{session_id}",
        response_model=ConsultationResponse,
        include_in_schema=False,
    )
    @application.get(
        "/sessions/{session_id}",
        response_model=ConsultationResponse,
    )
    def get_session(
        session_id: str = Path(min_length=1, max_length=128),
    ) -> ConsultationResponse:
        """读取最近一次患者可见结果；未知会话明确返回 404。"""

        service = checked_service(session_id)
        try:
            return state_to_response(service.get_state(session_id))
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    # 当客户端向某个地址发送 POST 请求 时，执行它下面紧跟着的 Python 函数
    @application.post(
        "/api/sessions/{session_id}/messages",
        response_model=ConsultationResponse,
        include_in_schema=False,
    )
    @application.post(
        "/sessions/{session_id}/messages",
        response_model=ConsultationResponse,
    )
    def send_message(
        payload: MessageRequest,
        session_id: str = Path(min_length=1, max_length=128),
    ) -> ConsultationResponse:
        """发送首轮主诉或向同一 checkpointer 会话追加一轮患者回答。"""

        service = checked_service(session_id)
        try:
            state = service.send_message(
                session_id=session_id,
                message=payload.message,
                patient_profile=payload.patient_profile,
                request_id=payload.request_id,
            )
            summary = _request_summary(state)
            logger.info(
                "consultation_request session_id=%s request_id=%s status=%s nodes=%s "
                "evidence_count=%s error_categories=%s",
                session_id,
                payload.request_id or "generated",
                summary["status"],
                summary["nodes"],
                summary["evidence_count"],
                summary["error_categories"],
            )
            return state_to_response(state)
        except (SessionClosedError, SessionConflictError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except Exception as exc:
            category, code = _safe_error_category(exc)
            logger.warning(
                "consultation_failed session_id=%s request_id=%s error_category=%s",
                session_id,
                payload.request_id or "generated",
                category,
            )
            response = _safe_failed_response(
                session_id,
                code=code,
                message="本轮问诊处理失败，请稍后重试。",
                category=category,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.model_dump(mode="json"),
            ) from exc

    @application.get(
        "/api/sessions/{session_id}/trace",
        response_model=DemoTraceResponse,
        include_in_schema=False,
    )
    @application.get(
        "/sessions/{session_id}/trace",
        response_model=DemoTraceResponse,
    )
    def get_session_trace(
        session_id: str = Path(min_length=1, max_length=128),
    ) -> DemoTraceResponse:
        """读取现有检查点的安全轨迹；关闭时表现为接口不可用。"""

        if not settings.demo_trace_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="演示轨迹未启用",
            )
        service = checked_service(session_id)
        try:
            return _state_to_trace(service.get_state(session_id))
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            response = _safe_failed_response(
                session_id,
                code="trace_unavailable",
                message="知识检索详情暂时不可用。",
                category="internal",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.model_dump(mode="json"),
            ) from exc

    # 构建产物是可选的只读资源。先注册所有 API 路由，再挂载前端，避免 /app
    # 影响既有接口；开发阶段尚未 build 时，FastAPI 仍可独立启动和测试。
    if frontend_dist is not None and (frontend_dist / "index.html").is_file():
        application.mount(
            "/app",
            StaticFiles(directory=frontend_dist, html=True, check_dir=True),
            name="frontend",
        )

    return application


# Uvicorn 使用 ``src.api.app:app`` 启动。这里不会创建图或访问任何外部依赖。
app = create_app()
