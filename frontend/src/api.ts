export interface TriageResult {
  urgency: "emergency" | "urgent" | "routine";
  recommended_departments: string[];
  red_flags: string[];
  rationale: string;
  should_continue_preconsult: boolean;
}

export interface PatientProfile {
  age: number | null;
  sex: "male" | "female" | "unknown";
  pregnancy: boolean | null;
}

export interface WaitingUserResponse {
  conversation_status: "waiting_user";
  session_id: string;
  current_question: string;
  question_count: number;
  triage_result: TriageResult | null;
}

export interface PreconsultSummary {
  chief_complaint: string;
  present_illness_summary: string;
  key_positive_findings: string[];
  key_negative_findings: string[];
  missing_information: string[];
  triage_recommendation: string;
  safety_notice: string;
}

export interface MedicalRecordDraft {
  chief_complaint: string;
  history_of_present_illness: string;
  past_history: string;
  medication_and_allergy_history: string;
  preliminary_department: string;
  note: string;
}

export interface CompletedResponse {
  conversation_status: "completed";
  session_id: string;
  summary: PreconsultSummary;
  medical_record_draft: MedicalRecordDraft;
}

export interface EmergencyResponse {
  conversation_status: "emergency_ended";
  session_id: string;
  triage_result: TriageResult;
  message: string;
}

export interface MedicalError {
  category: string;
  code: string;
  message: string;
  retryable: boolean;
  node?: string | null;
}

export interface FailedResponse {
  conversation_status: "failed";
  session_id: string;
  errors: MedicalError[];
  message: string;
}

export type ConsultationResponse =
  | WaitingUserResponse
  | CompletedResponse
  | EmergencyResponse
  | FailedResponse;

export interface DependencyHealth {
  status: "ready" | "not_initialized" | "unavailable";
  session_service: string;
  detail?: string | null;
}

export interface GraphEvidence {
  source_entity: string;
  relation: string;
  target_entity: string;
  source_entity_id?: string | null;
  target_entity_id?: string | null;
  source_entity_type?: string | null;
  target_entity_type?: string | null;
  source_records?: string[];
  score?: number | null;
  evidence_source?: string | null;
}

export interface NormalizationTrace {
  input_entity?: string;
  normalized_entity?: string;
  method?: string;
  score?: number | null;
}

export interface DemoTrace {
  available?: boolean;
  retrieval_intent?: string | null;
  retrieval_entities?: string[];
  normalization_results?: NormalizationTrace[];
  retrieved_evidence?: GraphEvidence[];
  evidence?: GraphEvidence[];
}

type ErrorKind = "validation" | "conflict" | "dependency" | "server" | "network";

export class ApiError extends Error {
  constructor(
    public readonly kind: ErrorKind,
    public readonly status: number | null,
    message: string,
  ) {
    super(message);
  }
}

function backendMessage(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const message = (detail as { message?: unknown }).message;
    return typeof message === "string" ? message : null;
  }
  return null;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      signal: controller.signal,
    });
    const body: unknown = await response.json().catch(() => null);
    if (response.ok) return body as T;

    // 只采纳后端已经为患者准备的安全消息，绝不把原始对象或堆栈渲染到页面。
    const safe = backendMessage(body);
    if (response.status === 422) {
      throw new ApiError("validation", 422, safe ?? "请输入有效的症状描述后再发送。");
    }
    if (response.status === 409) {
      throw new ApiError("conflict", 409, safe ?? "当前问诊已结束，请开始新的预问诊。");
    }
    if (response.status === 503) {
      throw new ApiError("dependency", 503, safe ?? "问诊依赖暂时不可用，请稍后重试。");
    }
    throw new ApiError("server", response.status, "本轮问诊处理失败，请稍后重试。");
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("network", null, "无法连接问诊服务，请检查服务是否已启动。");
  } finally {
    window.clearTimeout(timer);
  }
}

export function getDependencyHealth(): Promise<DependencyHealth> {
  return requestJson("/health/dependencies");
}

export function getSession(sessionId: string): Promise<ConsultationResponse> {
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function sendMessage(
  sessionId: string,
  message: string,
  requestId: string,
  patientProfile?: PatientProfile,
): Promise<ConsultationResponse> {
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    body: JSON.stringify({
      message,
      request_id: requestId,
      ...(patientProfile ? { patient_profile: patientProfile } : {}),
    }),
  });
}

export function hasTraceData(trace: DemoTrace | null): boolean {
  return Boolean(
    trace && trace.available !== false &&
    (trace.retrieval_intent || trace.retrieval_entities?.length ||
      trace.normalization_results?.length || trace.retrieved_evidence?.length ||
      trace.evidence?.length),
  );
}

export async function getTrace(sessionId: string): Promise<DemoTrace | null> {
  try {
    return await requestJson(`/sessions/${encodeURIComponent(sessionId)}/trace`);
  } catch (error) {
    // 只有接口不存在或功能被关闭才属于正常空态；500/503 必须继续抛出，
    // 否则真实 RAG 故障会被误显示为“本轮没有检索详情”。
    if (error instanceof ApiError && (error.status === 403 || error.status === 404)) return null;
    throw error;
  }
}
