import type { ComponentProps } from "react";
import type { ChatSnapshot } from "../../lib/api";
import type { ChatMessages } from "./chatMessages";
import type { ConversationCheckpoint } from "./conversationCheckpoint";
import { AskUserCard } from "../../components/AskUserCard";
import { ApprovalCard } from "../../components/ApprovalCard";
import { ClarificationForm } from "../../components/ClarificationForm";
import { PlanEditor } from "../../components/PlanEditor";
import { SubReportCheckpoint } from "../../components/SubReportCheckpoint";
import { StepCheckpoint } from "../../components/StepCheckpoint";
import { IterationDecision } from "../../components/IterationDecision";
import { InteractionCard } from "../../components/InteractionCard";

type Props = {
  snapshot: ChatSnapshot;
  sessionId: string;
  agentView: boolean;
  agentWorking: boolean;
  streaming: boolean;
  running: boolean;
  checkpoint: ConversationCheckpoint;
  t: ChatMessages;
  submitAskUser: ComponentProps<typeof AskUserCard>["onSubmit"];
  submitApproval: ComponentProps<typeof ApprovalCard>["onDecide"];
  submitClarification: ComponentProps<typeof ClarificationForm>["onSubmit"];
  confirmPlan: ComponentProps<typeof PlanEditor>["onConfirm"];
  handleSubReportDecision: ComponentProps<typeof SubReportCheckpoint>["onDecide"];
  handleStepDecision: ComponentProps<typeof StepCheckpoint>["onDecide"];
  handleIterationDecision: ComponentProps<typeof IterationDecision>["onDecide"];
  submitInteraction: ComponentProps<typeof InteractionCard>["onSubmit"];
};

export function ConversationCheckpoints({
  checkpoint, snapshot, sessionId, agentView, agentWorking, streaming, running, t,
  submitInteraction, submitAskUser, submitApproval, submitClarification, confirmPlan,
  handleSubReportDecision, handleStepDecision, handleIterationDecision,
}: Props) {
  return <>
    {agentView && snapshot.pending_interaction &&
      <InteractionCard key={snapshot.pending_interaction.id} interaction={snapshot.pending_interaction}
        sessionId={sessionId} disabled={agentWorking} onSubmit={submitInteraction}/>}
    {/* Science Agent (kimi-code) gates — never mix with Expert forms */}
    {checkpoint === "ask" && !streaming &&
      (snapshot.clarification_questions?.length ?? 0) > 0 && (
        <div className="agent-gate-wrap">
          <AskUserCard
            key={`ask-${snapshot.waiting_for || ""}-${snapshot.kimi_pending_question?.question_id || ""}-${(snapshot.clarification_questions || []).map((q) => q.question).join("|").slice(0, 80)}`}
            questions={snapshot.clarification_questions}
            onSubmit={submitAskUser}
            disabled={running}
          />
        </div>
      )}
    {checkpoint === "approve" && !streaming && (
        <div className="agent-gate-wrap">
          <ApprovalCard
            key={`appr-${snapshot.waiting_for || ""}-${snapshot.kimi_pending_approval?.approval_id || ""}`}
            toolName={snapshot.kimi_pending_approval?.tool_name || ""}
            prompt={
              snapshot.kimi_pending_approval?.approval_prompt ||
              snapshot.approval_prompt ||
              ""
            }
            planMarkdown={
              snapshot.kimi_pending_approval?.plan_markdown ||
              snapshot.plan_markdown ||
              ""
            }
            optionLabels={
              snapshot.kimi_pending_approval?.option_labels ||
              snapshot.clarification_questions?.[0]?.options ||
              []
            }
            onDecide={submitApproval}
            disabled={running}
          />
        </div>
      )}
    {/* Science Expert (LangGraph PI→CB→MLS→SC) checkpoints only.
        Waiting gates must mount even if streamingIdx briefly lags —
        otherwise ClarificationForm stays hidden behind a Thinking pulse. */}
    {checkpoint === "clarify" &&
      (snapshot.clarification_questions?.length ?? 0) > 0 && (
        <div className="expert-checkpoint-card">
          <div className="expert-checkpoint-title">{t.checkpointClarify}</div>
          <ClarificationForm
            key={`${snapshot.waiting_for}-${(snapshot.clarification_questions || []).map((q) => q.question).join("|").slice(0, 120)}`}
            questions={snapshot.clarification_questions}
            onSubmit={submitClarification}
            disabled={running}
          />
        </div>
      )}
    {checkpoint === "plan" &&
      (snapshot.plan?.length ?? 0) > 0 && (
        <div className="expert-checkpoint-card">
          <div className="expert-checkpoint-title">{t.checkpointPlan}</div>
          <PlanEditor
            plan={snapshot.plan}
            onConfirm={confirmPlan}
            disabled={running}
          />
        </div>
      )}
    {checkpoint === "subReport" && (
      <div className="expert-checkpoint-card">
        <div className="expert-checkpoint-title">{t.checkpointSubReport}</div>
        <SubReportCheckpoint
          onDecide={handleSubReportDecision}
          disabled={running}
        />
      </div>
    )}
    {checkpoint === "step" && (
      <div className="expert-checkpoint-card">
        <div className="expert-checkpoint-title">{t.checkpointStep}</div>
        <StepCheckpoint
          onDecide={handleStepDecision}
          disabled={running}
        />
      </div>
    )}
    {checkpoint === "iteration" && (
      <div className="expert-checkpoint-card">
        <div className="expert-checkpoint-title">{t.checkpointIteration}</div>
        <IterationDecision
          onDecide={handleIterationDecision}
          disabled={running}
        />
      </div>
    )}
  </>;
}
