import type {
  ConsultationResponse,
  DemoTrace,
  EmergencyResponse,
  FailedResponse,
  GraphEvidence,
  MedicalRecordDraft,
  PreconsultSummary,
} from "./api";
import { hasTraceData } from "./api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

type StepState = "pending" | "current" | "done" | "failed" | "emergency";

export function ProgressBar({
  response,
  hasError,
}: {
  response: ConsultationResponse | null;
  hasError: boolean;
}) {
  const states: StepState[] = ["current", "pending", "pending"];
  if (response?.conversation_status === "emergency_ended") {
    states[0] = "emergency";
  } else if (response?.conversation_status === "waiting_user") {
    states[0] = "done";
    states[1] = "current";
  } else if (response?.conversation_status === "completed") {
    states.fill("done");
  } else if (response?.conversation_status === "failed" || hasError) {
    states[0] = "done";
    states[1] = "failed";
  }

  const labels = ["风险初筛", "预问诊采集", "信息整理"];
  const stateText: Record<StepState, string> = {
    pending: "未开始",
    current: "当前",
    done: "完成",
    failed: "失败",
    emergency: "紧急停止",
  };
  return (
    <ol className="progress" aria-label="问诊进度">
      {labels.map((label, index) => (
        <li className={`progress__step progress__step--${states[index]}`} key={label}>
          <span className="progress__mark" aria-hidden="true">
            {states[index] === "done" ? "✓" : index + 1}
          </span>
          <span>{label}</span>
          <small>{stateText[states[index]]}</small>
        </li>
      ))}
    </ol>
  );
}

const urgencyLabels = {
  routine: "常规",
  urgent: "建议尽快就医",
  emergency: "紧急",
} as const;

export function PreconsultStatus({
  response,
}: {
  response: ConsultationResponse | null;
}) {
  if (response?.conversation_status !== "waiting_user" || !response.triage_result) return null;
  const triage = response.triage_result;
  return (
    <section className={`preconsult-status preconsult-status--${triage.urgency}`} aria-label="预问诊状态">
      <div>
        <span>风险初筛</span>
        <strong>{urgencyLabels[triage.urgency]}</strong>
      </div>
      <div>
        <span>建议科室</span>
        <strong>{triage.recommended_departments.join("、") || "待进一步确认"}</strong>
      </div>
      <div>
        <span>信息采集</span>
        <strong>已追问 {response.question_count} 轮</strong>
      </div>
    </section>
  );
}

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  if (!messages.length) return null;
  return (
    <section className="messages" aria-label="本次问诊对话" aria-live="polite">
      {messages.map((message) => (
        <div className={`message message--${message.role}`} key={message.id}>
          <span className="message__role">{message.role === "user" ? "你" : "预问诊助手"}</span>
          <p>{message.content}</p>
        </div>
      ))}
    </section>
  );
}

function TextList({ items, empty = "未提供" }: { items: string[]; empty?: string }) {
  return items.length ? (
    <ul className="fact-list">
      {items.map((item) => <li key={item}>{item}</li>)}
    </ul>
  ) : <p className="empty-text">{empty}</p>;
}

function RecordRows({ draft }: { draft: MedicalRecordDraft }) {
  const rows = [
    ["主诉", draft.chief_complaint],
    ["现病史", draft.history_of_present_illness],
    ["既往史", draft.past_history],
    ["用药及过敏史", draft.medication_and_allergy_history],
    ["建议科室", draft.preliminary_department],
    ["备注", draft.note],
  ];
  return (
    <dl className="record-grid">
      {rows.map(([label, value]) => (
        <div key={label}><dt>{label}</dt><dd>{value || "未提供"}</dd></div>
      ))}
    </dl>
  );
}

export function SummaryPanel({
  summary,
  draft,
}: {
  summary: PreconsultSummary;
  draft: MedicalRecordDraft;
}) {
  return (
    <section className="result-card result-card--summary" aria-labelledby="summary-title">
      <div className="result-card__eyebrow">信息整理完成</div>
      <h2 id="summary-title">预问诊摘要</h2>
      <dl className="summary-lead">
        <div><dt>主诉</dt><dd>{summary.chief_complaint}</dd></div>
        <div><dt>现病史摘要</dt><dd>{summary.present_illness_summary}</dd></div>
        <div><dt>分诊建议</dt><dd>{summary.triage_recommendation}</dd></div>
      </dl>
      <div className="summary-columns">
        <div><h3>已采集信息</h3><TextList items={summary.key_positive_findings} /></div>
        <div><h3>明确否认</h3><TextList items={summary.key_negative_findings} /></div>
        <div><h3>仍需补充</h3><TextList items={summary.missing_information} empty="暂无" /></div>
      </div>
      <div className="safety-note">{summary.safety_notice}</div>
      <details className="record-details">
        <summary>查看病历草稿</summary>
        <RecordRows draft={draft} />
      </details>
    </section>
  );
}

export function EmergencyPanel({ response }: { response: EmergencyResponse }) {
  return (
    <section className="result-card result-card--emergency" role="alert" aria-labelledby="emergency-title">
      <div className="result-card__eyebrow">紧急停止</div>
      <h2 id="emergency-title">发现需要立即线下评估的风险信号</h2>
      <p className="emergency-action">请立即联系当地急救或前往急诊。</p>
      <TextList items={response.triage_result.red_flags} empty="后端未返回具体红旗信号" />
      <p className="panel-detail">{response.message}</p>
    </section>
  );
}

export function ErrorPanel({
  response,
  message,
  category,
}: {
  response?: FailedResponse;
  message?: string;
  category?: string;
}) {
  const safeMessage = response?.message || message || "本次预问诊暂时无法继续。";
  const safeCategory = response?.errors[0]?.category || category || "服务错误";
  return (
    <section className="result-card result-card--error" role="alert" aria-labelledby="error-title">
      <div className="result-card__eyebrow">处理失败 · {safeCategory}</div>
      <h2 id="error-title">本轮问诊未能完成</h2>
      <p>{safeMessage}</p>
      <p className="panel-detail">可以稍后重试；若当前会话已结束，请开始新的预问诊。</p>
    </section>
  );
}

function EvidenceRow({ evidence }: { evidence: GraphEvidence }) {
  return (
    <li className="evidence-item">
      <div className="evidence-relation">
        <span>{evidence.source_entity}</span>
        <b>{evidence.relation}</b>
        <span>{evidence.target_entity}</span>
      </div>
      <dl className="evidence-meta">
        {evidence.source_entity_id && <div><dt>源 ID</dt><dd>{evidence.source_entity_id}</dd></div>}
        {evidence.target_entity_id && <div><dt>目标 ID</dt><dd>{evidence.target_entity_id}</dd></div>}
        {evidence.evidence_source && <div><dt>来源</dt><dd>{evidence.evidence_source}</dd></div>}
        {typeof evidence.score === "number" && <div><dt>相似度</dt><dd>{evidence.score.toFixed(3)}</dd></div>}
      </dl>
      {!!evidence.source_records?.length && (
        <p className="source-records">来源记录：{evidence.source_records.join("、")}</p>
      )}
    </li>
  );
}

export function EvidencePanel({ trace }: { trace: DemoTrace | null }) {
  const evidence = trace?.retrieved_evidence ?? trace?.evidence ?? [];
  const hasTrace = hasTraceData(trace);
  return (
    <details className="evidence-panel">
      <summary>
        <span>真实知识参考</span>
        <small>{hasTrace ? `${evidence.length} 条图谱关系` : "详情不可用"}</small>
      </summary>
      {!hasTrace ? (
        <p className="trace-empty">本轮未提供知识检索详情</p>
      ) : (
        <div className="trace-content">
          <dl className="trace-overview">
            <div><dt>检索意图</dt><dd>{trace?.retrieval_intent || "未记录"}</dd></div>
            <div><dt>检索实体</dt><dd>{trace?.retrieval_entities?.join("、") || "未记录"}</dd></div>
          </dl>
          {!!trace?.normalization_results?.length && (
            <section><h3>实体标准化</h3>
              <ul className="normalization-list">
                {trace.normalization_results.map((item, index) => (
                  <li key={`${item.input_entity}-${index}`}>
                    <span>{item.input_entity || "未知实体"} → {item.normalized_entity || "未命中"}</span>
                    <small>{item.method || "方式未记录"}{typeof item.score === "number" ? ` · ${item.score.toFixed(3)}` : ""}</small>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {evidence.length ? <ul className="evidence-list">{evidence.map((item, index) => (
            <EvidenceRow evidence={item} key={`${item.source_entity_id || item.source_entity}-${item.relation}-${item.target_entity_id || item.target_entity}-${index}`} />
          ))}</ul> : <p className="trace-empty">本轮 trace 未包含图谱关系</p>}
        </div>
      )}
    </details>
  );
}
