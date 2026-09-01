import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getDependencyHealth,
  getSession,
  getTrace,
  hasTraceData,
  sendMessage,
  type ConsultationResponse,
  type DemoTrace,
  type DependencyHealth,
} from "./api";
import {
  EmergencyPanel,
  ErrorPanel,
  EvidencePanel,
  MessageList,
  ProgressBar,
  SummaryPanel,
  type ChatMessage,
} from "./components";

const SESSION_KEY = "ai-medical-session-id";
const EXAMPLES = [
  "发热、咳嗽两天，晚上更明显",
  "上腹部疼痛，饭后加重",
  "突然胸痛并且呼吸困难",
];

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

interface BrowserSession {
  id: string;
  shouldRestore: boolean;
}

function initialSession(): BrowserSession {
  const existing = sessionStorage.getItem(SESSION_KEY);
  if (existing) return { id: existing, shouldRestore: true };
  const created = newId();
  sessionStorage.setItem(SESSION_KEY, created);
  return { id: created, shouldRestore: false };
}

export default function App() {
  const [browserSession, setBrowserSession] = useState(initialSession);
  const sessionId = browserSession.id;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [response, setResponse] = useState<ConsultationResponse | null>(null);
  const [trace, setTrace] = useState<DemoTrace | null>(null);
  const [dependency, setDependency] = useState<DependencyHealth | null>(null);
  const [sending, setSending] = useState(false);
  const [pageError, setPageError] = useState<ApiError | null>(null);
  const [inputError, setInputError] = useState("");
  const composing = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    getDependencyHealth()
      .then((value) => { if (active) setDependency(value); })
      .catch(() => { if (active) setDependency({ status: "unavailable", session_service: "unavailable" }); });
    if (!browserSession.shouldRestore) {
      return () => { active = false; };
    }
    getSession(sessionId)
      .then((value) => {
        if (!active) return;
        setResponse(value);
        if (value.conversation_status === "waiting_user") {
          setMessages([{ id: `restored-${sessionId}`, role: "assistant", content: value.current_question }]);
        }
        if (value.conversation_status !== "emergency_ended" && value.conversation_status !== "failed") {
          getTrace(sessionId)
            .then((result) => { if (active) setTrace(result); })
            .catch((error) => {
              if (active) setPageError(error instanceof ApiError
                ? error
                : new ApiError("server", null, "知识检索详情读取失败，请稍后重试。"));
            });
        }
      })
      .catch((error) => {
        if (!active || (error instanceof ApiError && error.status === 404)) return;
        setPageError(error instanceof ApiError
          ? error
          : new ApiError("server", null, "会话恢复失败，请开始新的预问诊。"));
      });
    return () => { active = false; };
  }, [browserSession, sessionId]);

  useEffect(() => {
    if (typeof bottomRef.current?.scrollIntoView === "function") {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, response]);

  const terminal = response?.conversation_status === "completed" ||
    response?.conversation_status === "emergency_ended" ||
    response?.conversation_status === "failed" || pageError?.kind === "conflict";
  const traceAvailable = hasTraceData(trace);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const value = input.trim();
    if (!value || sending || terminal) {
      if (!value) setInputError("请先描述当前最主要的不适。");
      return;
    }

    setInputError("");
    setPageError(null);
    setSending(true);
    setInput("");
    const userMessage: ChatMessage = { id: newId(), role: "user", content: value };
    setMessages((current) => [...current, userMessage]);
    try {
      const result = await sendMessage(sessionId, value, newId());
      setResponse(result);
      if (result.conversation_status === "waiting_user") {
        setMessages((current) => [
          ...current,
          { id: newId(), role: "assistant", content: result.current_question },
        ]);
      }
      if (result.conversation_status !== "emergency_ended" && result.conversation_status !== "failed") {
        setTrace(await getTrace(sessionId));
      } else {
        setTrace(null);
      }
      setDependency((current) => current ? { ...current, status: "ready", session_service: "ready" } : current);
    } catch (error) {
      const safeError = error instanceof ApiError
        ? error
        : new ApiError("server", null, "本轮问诊处理失败，请稍后重试。");
      if (safeError.kind === "validation") setInputError(safeError.message);
      else setPageError(safeError);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !composing.current) {
      event.preventDefault();
      void submit();
    }
  }

  function resetConsultation() {
    const id = newId();
    sessionStorage.setItem(SESSION_KEY, id);
    setBrowserSession({ id, shouldRestore: false });
    setMessages([]);
    setInput("");
    setResponse(null);
    setTrace(null);
    setPageError(null);
    setInputError("");
  }

  const dependencyLabel = dependency?.status === "ready" ? "就绪" :
    dependency?.status === "unavailable" ? "不可用" : "未初始化";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__line">
          <a className="brand" href="/app/" aria-label="ai-medical 首页">
            <span className="brand__pulse" aria-hidden="true" />
            <span>ai-medical</span>
            <small>预问诊台</small>
          </a>
          <div className="topbar__actions">
            <span className={`dependency dependency--${dependency?.status ?? "not_initialized"}`}>
              <i aria-hidden="true" />依赖：{dependencyLabel}
            </span>
            <button className="secondary-button" type="button" onClick={resetConsultation}>新预问诊</button>
          </div>
        </div>
        <ProgressBar response={response} hasError={Boolean(pageError)} traceAvailable={traceAvailable} />
      </header>

      <main className="consultation">
        {!messages.length && !response && !pageError && (
          <section className="welcome" aria-labelledby="welcome-title">
            <div className="welcome__index">01 / 风险初筛</div>
            <h1 id="welcome-title">开始一次预问诊</h1>
            <p>我会先排查紧急风险，再逐步整理症状信息。请不要输入姓名、证件号、电话或住址。</p>
            <div className="examples" aria-label="症状描述示例">
              {EXAMPLES.map((example) => (
                <button type="button" key={example} onClick={() => setInput(example)}>{example}</button>
              ))}
            </div>
          </section>
        )}

        <MessageList messages={messages} />
        {response?.conversation_status === "completed" && (
          <SummaryPanel summary={response.summary} draft={response.medical_record_draft} />
        )}
        {response?.conversation_status === "emergency_ended" && <EmergencyPanel response={response} />}
        {response?.conversation_status === "failed" && <ErrorPanel response={response} />}
        {pageError && <ErrorPanel message={pageError.message} category={pageError.kind} />}
        {!pageError && (response?.conversation_status === "waiting_user" || response?.conversation_status === "completed") && (
          <EvidencePanel trace={trace} />
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="composer-area">
        <form className="composer" onSubmit={submit}>
          <label className="sr-only" htmlFor="symptom-input">描述症状</label>
          <textarea
            id="symptom-input"
            value={input}
            onChange={(event) => { setInput(event.target.value); setInputError(""); }}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => { composing.current = true; }}
            onCompositionEnd={() => { composing.current = false; }}
            placeholder={terminal ? "本次问诊已结束" : "描述当前最主要的不适…"}
            rows={2}
            maxLength={4000}
            disabled={sending || terminal}
          />
          <button type="submit" disabled={sending || terminal || !input.trim()}>
            {sending ? "正在整理…" : "发送"}
          </button>
        </form>
        {inputError && <p className="input-error" role="alert">{inputError}</p>}
        <p className="disclaimer">本系统仅用于风险分诊和预问诊信息整理，不构成诊断或治疗建议。</p>
      </footer>
    </div>
  );
}
