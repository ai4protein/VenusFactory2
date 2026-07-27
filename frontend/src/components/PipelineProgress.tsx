import type { PlanStep } from "../lib/api";
import { useLang } from "../lib/i18n";

const STAGE_KEYS = ["analyze", "research", "plan", "execute", "summarize"] as const;
type StageKey = (typeof STAGE_KEYS)[number];

const STRINGS = {
  en: {
    analyze: "Analyze",
    research: "Research",
    plan: "Plan",
    execute: "Execute",
    summarize: "Summarize",
    stopping: "Stopping…",
    stopped: "Stopped",
    responding: "Responding…",
    analyzing: "Analyzing…",
  },
  zh: {
    analyze: "分析",
    research: "检索",
    plan: "规划",
    execute: "执行",
    summarize: "汇总",
    stopping: "停止中…",
    stopped: "已停止",
    responding: "回复中…",
    analyzing: "分析中…",
  },
};

const STAGE_ICONS: Record<StageKey, string> = {
  analyze: "🤔",
  research: "🔍",
  plan: "📋",
  execute: "⏳",
  summarize: "📝",
};

const STATUS_TO_STAGE: Record<string, StageKey | "done"> = {
  started: "analyze",
  analyzing: "analyze",
  new_request: "analyze",
  waiting_for_clarification: "analyze",
  // Agent uses separate statuses; PipelineProgress is Expert-only in ChatPage.
  replan_research: "research",
  research_planning_done: "research",
  resume_research: "research",
  research_search_done: "research",
  research_step_done: "research",
  research_steps_done: "research",
  waiting_for_sub_report_review: "research",
  researched: "plan",
  waiting_for_plan_confirmation: "plan",
  resume_execution: "execute",
  executing: "execute",
  execution_failed: "execute",
  waiting_for_step_review: "execute",
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

function isExecutionTool(t: Record<string, unknown>): boolean {
  const name = String(t.tool_name || t.name || "").toLowerCase();
  const kind = String(t.kind || "").toLowerCase();
  if (kind === "research_search" || name === "research_search" || name.startsWith("query_")) {
    return false;
  }
  return true;
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
  const t = useLang().t(STRINGS);
  if (!status) return null;

  const stages = STAGE_KEYS.map((key) => ({
    key,
    label: t[key],
    icon: STAGE_ICONS[key],
  }));

  if (status === "completed") {
    return (
      <div className="pipeline-bar pipeline-done">
        {stages.map((stage) => (
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
        <span className="pipeline-simple-text">{status === "stopping" ? t.stopping : t.stopped}</span>
      </div>
    );
  }

  if (status === "chat_mode") {
    return (
      <div className="pipeline-bar pipeline-simple">
        <span className="pipeline-pulse-dot" />
        <span className="pipeline-simple-text">{t.responding}</span>
      </div>
    );
  }

  const activeStageKey = STATUS_TO_STAGE[status];
  if (!activeStageKey) return null;

  const isFullPipeline =
    plan.length > 0 ||
    ["researched", "waiting_for_plan_confirmation", "resume_execution", "executing", "execution_failed"].includes(status);

  const researchStatuses = [
    "research_planning_done",
    "replan_research",
    "resume_research",
    "research_search_done",
    "research_step_done",
    "research_steps_done",
    "waiting_for_sub_report_review",
  ];

  if (!isFullPipeline && !researchStatuses.includes(status)) {
    return (
      <div className="pipeline-bar pipeline-simple">
        <span className="pipeline-pulse-dot" />
        <span className="pipeline-simple-text">{t.analyzing}</span>
      </div>
    );
  }

  const activeIdx =
    activeStageKey === "done"
      ? stages.length
      : stages.findIndex((s) => s.key === activeStageKey);
  const isFailed = status === "execution_failed";

  const execCount = toolExecutions.filter(isExecutionTool).length;
  const stepProgress =
    activeStageKey === "execute" && plan.length > 0
      ? `${Math.min(execCount, plan.length)}/${plan.length}`
      : null;

  return (
    <div className="pipeline-bar">
      {stages.map((stage, i) => {
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
