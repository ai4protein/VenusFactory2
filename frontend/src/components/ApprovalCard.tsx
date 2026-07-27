import { useEffect, useMemo, useState } from "react";
import { renderMarkdown } from "../lib/markdown";
import { useLang } from "../lib/i18n";

export type ApprovalDecision = {
  decision: "approved" | "rejected";
  selected_label?: string;
  feedback?: string;
};

type Props = {
  toolName?: string;
  prompt?: string;
  planMarkdown?: string;
  optionLabels?: string[];
  onDecide: (decision: ApprovalDecision) => void;
  disabled?: boolean;
};

const STRINGS = {
  en: {
    title: "Approval required",
    subtitle: "Review the plan or tool call, then approve or reject.",
    tool: "Tool",
    plan: "Plan",
    showPlan: "Show plan",
    hidePlan: "Hide plan",
    approaches: "Choose an approach",
    approve: "Approve",
    reject: "Reject",
    rejectReason: "Reason (optional)",
    rejectPlaceholder: "Why are you rejecting?",
  },
  zh: {
    title: "需要你的批准",
    subtitle: "请审阅计划或工具调用，然后批准或拒绝。",
    tool: "工具",
    plan: "计划",
    showPlan: "展开计划",
    hidePlan: "收起计划",
    approaches: "选择方案",
    approve: "批准",
    reject: "拒绝",
    rejectReason: "拒绝理由（可选）",
    rejectPlaceholder: "简要说明拒绝原因…",
  },
};

const REJECT_SET = new Set(["reject", "拒绝", "reject and exit", "dismiss", "取消", "deny", "no"]);
const APPROVE_SET = new Set(["approve", "批准", "确认", "yes", "ok", "allow", "允许"]);

function shortToolName(name: string): string {
  if (!name) return "";
  const parts = name.split("__");
  return parts[parts.length - 1] || name;
}

export function ApprovalCard({
  toolName = "",
  prompt = "",
  planMarkdown = "",
  optionLabels = [],
  onDecide,
  disabled,
}: Props) {
  const t = useLang().t(STRINGS);
  const approaches = useMemo(
    () =>
      (optionLabels || []).filter((label) => {
        const key = label.trim().toLowerCase();
        return key && !REJECT_SET.has(key) && !APPROVE_SET.has(key);
      }),
    [optionLabels]
  );
  const [selectedApproach, setSelectedApproach] = useState(approaches[0] || "");
  const [feedback, setFeedback] = useState("");
  const [planOpen, setPlanOpen] = useState(true);

  useEffect(() => {
    setSelectedApproach(approaches[0] || "");
    setFeedback("");
    setPlanOpen(true);
  }, [approaches, toolName, planMarkdown]);

  const planHtml = planMarkdown ? renderMarkdown(planMarkdown) : "";
  const displayTool = shortToolName(toolName);

  return (
    <div className="agent-gate-card approval-card">
      <div className="agent-gate-card-header">
        <span className="agent-gate-card-icon approval-card-icon" aria-hidden="true">!</span>
        <div className="agent-gate-card-titles">
          <div className="agent-gate-card-title">{t.title}</div>
          <div className="agent-gate-card-subtitle">{prompt || t.subtitle}</div>
        </div>
      </div>

      {displayTool ? (
        <div className="approval-tool-row">
          <span className="approval-tool-label">{t.tool}</span>
          <code className="approval-tool-name">{displayTool}</code>
        </div>
      ) : null}

      {planMarkdown ? (
        <div className="approval-plan">
          <button
            type="button"
            className="approval-plan-toggle"
            onClick={() => setPlanOpen((v) => !v)}
            disabled={disabled}
          >
            <span>{t.plan}</span>
            <span className="approval-plan-chev">{planOpen ? "▾" : "▸"}</span>
            <span className="approval-plan-toggle-hint">{planOpen ? t.hidePlan : t.showPlan}</span>
          </button>
          {planOpen ? (
            <div
              className="approval-plan-body chat-msg-body"
              dangerouslySetInnerHTML={{ __html: planHtml }}
            />
          ) : null}
        </div>
      ) : null}

      {approaches.length > 0 ? (
        <div className="approval-approaches">
          <div className="approval-approaches-label">{t.approaches}</div>
          <div className="ask-user-options">
            {approaches.map((label) => (
              <button
                key={label}
                type="button"
                className={`ask-user-option${selectedApproach === label ? " selected" : ""}`}
                onClick={() => setSelectedApproach(label)}
                disabled={disabled}
                aria-pressed={selectedApproach === label}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="approval-reject-reason">
        <label className="approval-reject-label" htmlFor="approval-reject-feedback">
          {t.rejectReason}
        </label>
        <input
          id="approval-reject-feedback"
          type="text"
          className="ask-user-custom-input"
          placeholder={t.rejectPlaceholder}
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          disabled={disabled}
        />
      </div>

      <div className="agent-gate-card-actions">
        <button
          type="button"
          className="agent-gate-btn agent-gate-btn-primary"
          onClick={() =>
            onDecide({
              decision: "approved",
              selected_label: selectedApproach || undefined,
            })
          }
          disabled={disabled || (approaches.length > 0 && !selectedApproach)}
        >
          {t.approve}
        </button>
        <button
          type="button"
          className="agent-gate-btn agent-gate-btn-danger"
          onClick={() =>
            onDecide({
              decision: "rejected",
              feedback: feedback.trim() || undefined,
            })
          }
          disabled={disabled}
        >
          {t.reject}
        </button>
      </div>
    </div>
  );
}
