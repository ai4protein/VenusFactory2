export interface ResearchDelivery {
  candidate_id: string; family_id?: string; kind?: "model" | "method"; name?: string; version?: string;
  model_package: string; package?: string; test_report: string; model_card: string; readme?: string;
}

import { shareRead } from "./readRequests";
import { validateUploads } from "./uploadLimits";
import { quotaError, QUOTA_CHANGED } from "./onlineQuota";
import { getChatSessionAuthHeaders } from "./api";

export type GoalBudget = { max_cycles: number; model_calls: number; tool_calls: number; wall_seconds: number };
export type ExecutionSelection = { harness: string; model_id: string };
export type ExecutionCatalog = {
  mode: string; default: ExecutionSelection; readiness: string;
  models: Array<{ id: string; label: string; provider: string; model: string; protocol: string; free?: boolean }>;
  harnesses: Array<{ id: string; modes: string[]; protocols: string[]; status: string }>;
  combinations: Array<{ harness: string; model_id: string; ready: boolean; reasons: string[] }>;
};
export type ResearchAttachment = { id: string; name: string; size: number; sha256: string };
export type ResearchDevelopment = {
  count: number;
  generations: Array<{
    id: string; name: string; strategy: string; variant: string; parent: string | null; hypothesis: string;
    validation: { metric: string; score: number; direction: "minimize" | "maximize" };
    comparison: { parent_score: number; delta_in_preferred_direction: number; improved_on_validation: boolean } | null;
  }>;
};
export type GoalSpec = { attachments?: string[]; execution?: ExecutionSelection; driver?: string; parameters?: Record<string, string>; objective: string; budget: GoalBudget };
export type PaperProgress = {
  case: string; paper_id: string; title: string; doi: string; reproduction_level: string;
  covered_steps: string[]; excluded_steps: string[]; verified: boolean; full_paper_reproduced: false;
  criteria: Array<{ name: string; passed: boolean }>;
};
export type GeneratedStepEntry = {
  entry_id: string; paper: { title: string; doi: string };
  interface: { id: string; description: string };
  scope: { covered_steps: string[]; excluded_steps: string[] };
  origin_record_kind: string;
};
export type CodeStepProgress = {
  evidence_available?: boolean; paper?: { title: string; doi: string };
  interface?: { id: string; description: string };
  scope?: { covered_steps: string[]; excluded_steps: string[] };
  imported?: boolean; registered?: boolean; reused?: boolean; published?: boolean;
  completed_inputs?: number; required_inputs?: number; capability_id?: string; source_sha256?: string; entry_id?: string;
};
export type ScienceGoal = {
  queue?: { id:string; resource:string; status:string; position:number; cancel_requested:boolean };
  id: string; status: string; objective: string; driver: string; worker_alive: boolean;
  created_at: number; model?: string; budget?: GoalBudget;
  execution?: { harness: string; provider: string; model: string; protocol: string; profile_sha256: string };
  usage?: { model_calls: number; tool_calls: number; wall_seconds: number };
  criteria?: Array<{ name: string; passed: boolean; evidence_count: number }>;
  cycles?: Array<{ index: number; phase: string; status: string; summary: string; generation?: number }>;
  inflight_index?: number | null;
  project?: { file_count: number; available: boolean; status: string; research_completed: boolean; files: string[] };
  parameters?: Record<string, string>;
  research?: {
    development?: ResearchDevelopment;
    deliveries?: ResearchDelivery[];
    selected_deliveries?: ResearchDelivery[];
    pending_interaction?: import("./interactions").UserInteraction | null;
    research_completed?: boolean;
    publication_reviews?: { id: string; name: string; status: string; summary: string; issues: string[] }[]; pending_updates?: number; inputs?: ResearchAttachment[]; live?: { phase?: { role: string; episode: number; updated_at: number }; checkpoint?: { summary: string; completed_steps: string[]; artifacts?: Array<{ path: string; sha256: string }> }; stages?: Array<{ role: string; episode: number; status: string; started_at: number; ended_at?: number }>; timeline?: Array<{ role: string; episode: number; tool: string; status: string; timestamp: number }>; tool?: { tool: string; status: string; progress?: { stdout_tail?: string; progress?: { step: number; total: number; metrics: Record<string, unknown> } } }; execution?: { queue?: { id:string; resource:string; status:string; position:number }; script: string; elapsed_seconds: number; stdout_tail: string; stderr_tail: string; progress?: { step: number; total: number; metrics: Record<string, unknown> } }; files?: Array<{ path: string; bytes: number }> }; plan?: { summary: string; steps: Array<{ id: string; title: string; acceptance: string; status: string }> }; questions?: string[]; artifacts?: Array<{ path: string; sha256: string }>; capabilities?: Array<{ id: string; name: string }>; last_review?: { summary: string } } | null;
  activity?: Array<{ id: string; episode: number; stage?: string; timestamp?: number; role: string; content: string }>;
  papers?: PaperProgress[];
  code_step?: CodeStepProgress | null;
  reproduction_recipes?: Array<{ case?: string; version_id: string; available: boolean; reused: boolean; imported?: boolean; plugin?: string; scope?: string }>;
  capabilities?: Array<{ generation: number; model_id: string; source_sha256?: string; experience_id?: string; reused: boolean }>;
};
export type GoalCatalog = {
  default_driver?: string; available: boolean; model: string | null; execution?: ExecutionCatalog | null;
  drivers: Array<{ id: string; title: string; cases: string[]; criteria: string[]; maximum_budget?: GoalBudget; case_details?: Record<string, { scope: string; suggested_budget: GoalBudget; minimum_cycles?: number; paper_count?: number; label?: { en: string; zh: string }; excluded_steps?: string[] }> }>;
  default_budget?: GoalBudget; maximum_budget?: GoalBudget;
};

async function request<T>(path: string, sessionId = "", body?: unknown): Promise<T> {
  const response = await fetch(`/api/chat${path}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store",
    headers: { ...getChatSessionAuthHeaders(sessionId), "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) quotaError(result);
  else if (body !== undefined) window.dispatchEvent(new Event(QUOTA_CHANGED));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : result.detail?.message || `Request failed (${response.status})`);
  return result as T;
}
const root = (sid: string) => `/sessions/${encodeURIComponent(sid)}/science-goals`;
export const goalCatalog = () => shareRead("goal-catalog", () => request<GoalCatalog>("/science-goals/catalog?view=composer"));
export const listGoals = (sid: string) => shareRead("goals:" + sid + JSON.stringify(getChatSessionAuthHeaders(sid)), () => request<{ goals: ScienceGoal[] }>(root(sid), sid));
export const createGoal = (sid: string, spec: GoalSpec, request_id: string) => request<ScienceGoal>(root(sid), sid, { ...spec, request_id });
export const controlGoal = (sid: string, id: string, action: "pause" | "resume" | "cancel") => request<ScienceGoal>(`${root(sid)}/${id}/${action}`, sid, {});

export const generatedSteps = (sid: string, caseId: string) => request<{ entries: GeneratedStepEntry[] }>(`${root(sid)}/generated-steps?case=${encodeURIComponent(caseId)}`, sid);

export const provideResearchInput = (sid: string, id: string, text: string, plan?: { summary: string; steps: Array<{ id: string; title: string; acceptance: string }> } | null, attachments: string[] = []) => request<ScienceGoal>(`${root(sid)}/${id}/input`, sid, { text, plan, attachments });
export const answerResearchInteraction = (sid: string, id: string, interaction: import("./interactions").UserResponse) => request<ScienceGoal>(`${root(sid)}/${id}/input`, sid, { interaction });
export async function uploadResearchInputs(sid: string, files: File[]): Promise<{ files: ResearchAttachment[] }> {
  await validateUploads(files);
  const body = new FormData(); files.forEach(file => body.append("files", file));
  const response = await fetch(`/api/chat${root(sid)}/uploads`, { method: "POST", headers: getChatSessionAuthHeaders(sid), body });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `Upload failed (${response.status})`);
  return result;
}

export async function downloadResearchProject(sid: string, id: string, format: "zip" | "report") {
  const response = await fetch(`/api/chat${root(sid)}/${encodeURIComponent(id)}/project?format=${format}`, {
    headers: getChatSessionAuthHeaders(sid), cache: "no-store",
  });
  if (!response.ok) { const value = await response.json(); throw new Error(value.detail || "Project download failed"); }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a"); link.href = url; link.download = `research-${id}.${format === "zip" ? "zip" : "md"}`;
  document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
