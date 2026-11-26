import type { ChatSnapshot } from "../../lib/api";
import type { ChatMode } from "../../lib/useModelRegistry";
import type { ChatMessages } from "./chatMessages";

export function checkpointForSession(snapshot: ChatSnapshot | null, chatMode: ChatMode) {
  const waitingFor = snapshot?.waiting_for || "";
  const snapStatus = snapshot?.status || "";
  // Older snapshots identify their mode through the engine or the waiting status.
  const agentSnapshot = snapshot?.chat_mode === "science_agent" || snapshot?.engine === "kimi-code"
    || waitingFor === "kimi_question" || waitingFor === "kimi_approval"
    || snapStatus === "waiting_for_kimi_question" || snapStatus === "waiting_for_kimi_approval";
  const expertSnapshot = snapshot?.chat_mode === "science_expert" || snapshot?.engine === "graph"
    || ["waiting_for_clarification", "waiting_for_plan_confirmation", "waiting_for_sub_report_review",
        "waiting_for_step_review", "waiting_for_iteration"].includes(snapStatus);
  if (agentSnapshot || (!expertSnapshot && chatMode === "science_agent")) {
    if (waitingFor === "kimi_question" || snapStatus === "waiting_for_kimi_question" || snapshot?.kimi_pending_question) return "ask";
    if (waitingFor === "kimi_approval" || snapStatus === "waiting_for_kimi_approval" || snapshot?.kimi_pending_approval) return "approve";
    return null;
  }
  if (snapStatus === "waiting_for_clarification" && (!waitingFor || waitingFor === "clarification")) return "clarify";
  switch (snapStatus) {
    case "waiting_for_plan_confirmation": return "plan";
    case "waiting_for_sub_report_review": return "subReport";
    case "waiting_for_step_review": return "step";
    case "waiting_for_iteration": return "iteration";
    default: return null;
  }
}

export type ConversationCheckpoint = ReturnType<typeof checkpointForSession>;

export function checkpointPlaceholder(checkpoint: ConversationCheckpoint, hasInteraction: boolean, t: ChatMessages) {
  if (hasInteraction) return t.composerWaitingAskUser;
  switch (checkpoint) {
    case "ask": return t.composerWaitingAskUser;
    case "approve": return t.composerWaitingApproval;
    case "clarify": return t.composerWaitingClarification;
    case "plan": return t.composerWaitingPlan;
    case "subReport": return t.composerWaitingSubReport;
    case "step": return t.composerWaitingStep;
    case "iteration": return t.composerWaitingIteration;
    default: return t.composerWaiting;
  }
}
