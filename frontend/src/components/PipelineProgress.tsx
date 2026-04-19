import type { PlanStep } from "../lib/api";

const STAGES = [
  { key: "analyze", label: "Analyze", icon: "🤔" },
  { key: "research", label: "Research", icon: "🔍" },
  { key: "plan", label: "Plan", icon: "📋" },
  { key: "execute", label: "Execute", icon: "⏳" },
  { key: "summarize", label: "Summarize", icon: "📝" },
] as const;

type StageKey = (typeof STAGES)[number]["key"];

const STATUS_TO_STAGE: Record<string, StageKey | "done"> = {
  started: "analyze",
  analyzing: "analyze",
  new_request: "analyze",
  waiting_for_clarification: "analyze",
  research_planning_done: "research",
  resume_research: "research",
  research_search_done: "research",
  research_step_done: "research",
  research_steps_done: "research",
  researched: "plan",
  waiting_for_plan_confirmation: "plan",
  resume_execution: "execute",
  executing: "execute",
  execution_failed: "execute",
  waiting_for_step_review: "execute",
  waiting_for_sub_report_review: "execute",
  waiting_for_iteration: "summarize",
  completed: "done",
};

function stageState(
  idx: number,
  activeIdx: number,
  failed: boolean
): "done" | "active" | "pending" | "failed" {
  if (idx < activeIdx) return "done";
  if (idx === activeIdx) return failed ? "failed" : "active";
  return "pending";
}

interface PipelineProgressProps {
  status: string;
  plan: PlanStep[];
  toolExecutions: Array<Record<string, unknown>>;
}

export function PipelineProgress({
  status,
  plan,
  toolExecutions,
}: PipelineProgressProps) {
  if (!status) return null;

  if (status === "completed") {
    return (
      <div className="pipeline-bar pipeline-done">
        {STAGES.map((stage) => (
          <div key={stage.key} className="pipeline-step step-done">
            <span className="pipeline-step-icon">✓</span>
            <span className="pipeline-step-label">{stage.label}</span>
          </div>
        ))}
      </div>
    );
  }

  if (status === "stopped" || status === "stopping") {
    return (
      <div className="pipeline-bar pipeline-simple pipeline-stopped">
        <span className="pipeline-simple-text">{status === "stopping" ? "Stopping…" : "Stopped"}</span>
      </div>
    );
  }

  if (status === "chat_mode") {
    return (
      <div className="pipeline-bar pipeline-simple">
        <span className="pipeline-pulse-dot" />
        <span className="pipeline-simple-text">Responding…</span>
      </div>
    );
  }

  const activeStageKey = STATUS_TO_STAGE[status];
  if (!activeStageKey) return null;

  const isFullPipeline =
    plan.length > 0 ||
    ["researched", "waiting_for_plan_confirmation", "resume_execution", "executing", "execution_failed"].includes(status);

  if (!isFullPipeline && !["research_planning_done", "resume_research", "research_search_done",
    "research_step_done", "research_steps_done"].includes(status)) {
    return (
      <div className="pipeline-bar pipeline-simple">
        <span className="pipeline-pulse-dot" />
        <span className="pipeline-simple-text">Analyzing…</span>
      </div>
    );
  }

  const activeIdx =
    activeStageKey === "done"
      ? STAGES.length
      : STAGES.findIndex((s) => s.key === activeStageKey);
  const isFailed = status === "execution_failed";

  const stepProgress =
    activeStageKey === "execute" && plan.length > 0
      ? `${Math.min(toolExecutions.length, plan.length)}/${plan.length}`
      : null;

  return (
    <div className="pipeline-bar">
      {STAGES.map((stage, i) => {
        const state = stageState(i, activeIdx, isFailed && stage.key === "execute");
        return (
          <div key={stage.key} className={`pipeline-step step-${state}`}>
            <span className="pipeline-step-icon">
              {state === "done" ? "✓" : stage.icon}
            </span>
            <span className="pipeline-step-label">
              {stage.label}
              {stage.key === "execute" && stepProgress && (
                <span className="pipeline-step-count">{stepProgress}</span>
              )}
            </span>
            {state === "active" && <span className="pipeline-pulse-dot" />}
          </div>
        );
      })}
    </div>
  );
}
