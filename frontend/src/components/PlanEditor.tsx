import { useState } from "react";
import type { PlanStep } from "../lib/api";

type Props = {
  plan: PlanStep[];
  onConfirm: (plan: PlanStep[], autoExecute: boolean) => void;
  disabled?: boolean;
};

export function PlanEditor({ plan, onConfirm, disabled }: Props) {
  const [steps, setSteps] = useState<PlanStep[]>(plan.map(s => ({ ...s })));

  function updateDescription(idx: number, desc: string) {
    setSteps(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], task_description: desc };
      return next;
    });
  }

  function removeStep(idx: number) {
    setSteps(prev => {
      const next = prev.filter((_, i) => i !== idx);
      return next.map((s, i) => ({ ...s, step: i + 1 }));
    });
  }

  function moveStep(idx: number, direction: "up" | "down") {
    setSteps(prev => {
      const next = [...prev];
      const targetIdx = direction === "up" ? idx - 1 : idx + 1;
      if (targetIdx < 0 || targetIdx >= next.length) return prev;
      [next[idx], next[targetIdx]] = [next[targetIdx], next[idx]];
      return next.map((s, i) => ({ ...s, step: i + 1 }));
    });
  }

  return (
    <div className="plan-editor">
      {steps.map((step, idx) => (
        <div key={idx} className="plan-editor-step">
          <div className="plan-editor-step-header">
            <span className="plan-editor-step-num">Step {idx + 1}</span>
            <span className="plan-editor-tool-badge">{step.tool_name}</span>
            <div className="plan-editor-step-actions">
              <button
                className="plan-editor-btn"
                onClick={() => moveStep(idx, "up")}
                disabled={disabled || idx === 0}
                title="Move up"
              >
                ↑
              </button>
              <button
                className="plan-editor-btn"
                onClick={() => moveStep(idx, "down")}
                disabled={disabled || idx === steps.length - 1}
                title="Move down"
              >
                ↓
              </button>
              <button
                className="plan-editor-btn plan-editor-btn-delete"
                onClick={() => removeStep(idx)}
                disabled={disabled || steps.length <= 1}
                title="Remove step"
              >
                ×
              </button>
            </div>
          </div>
          <textarea
            className="plan-editor-desc"
            value={step.task_description}
            onChange={e => updateDescription(idx, e.target.value)}
            disabled={disabled}
            rows={2}
          />
        </div>
      ))}
      <div className="plan-editor-actions">
        <button
          className="plan-editor-confirm"
          onClick={() => onConfirm(steps, false)}
          disabled={disabled || steps.length === 0}
        >
          Confirm Plan
        </button>
        <button
          className="plan-editor-auto-execute"
          onClick={() => onConfirm(steps, true)}
          disabled={disabled || steps.length === 0}
          title="Confirm and execute all steps automatically without pausing"
        >
          Confirm & Auto-execute
        </button>
      </div>
    </div>
  );
}
