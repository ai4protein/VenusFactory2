type Props = {
  onDecide: (action: "satisfied" | "modify_plan" | "continue") => void;
  disabled?: boolean;
};

export function IterationDecision({ onDecide, disabled }: Props) {
  return (
    <div className="iteration-decision">
      <div className="iteration-decision-options">
        <button
          className="iteration-btn iteration-btn-satisfied"
          onClick={() => onDecide("satisfied")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#10003;</span>
          <span className="iteration-btn-label">Satisfied</span>
          <span className="iteration-btn-hint">Mark task as complete</span>
        </button>
        <button
          className="iteration-btn iteration-btn-modify"
          onClick={() => onDecide("modify_plan")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#8635;</span>
          <span className="iteration-btn-label">Modify & Re-execute</span>
          <span className="iteration-btn-hint">Edit the plan and run again</span>
        </button>
        <button
          className="iteration-btn iteration-btn-continue"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#43;</span>
          <span className="iteration-btn-label">Continue Analysis</span>
          <span className="iteration-btn-hint">Provide new instructions</span>
        </button>
      </div>
    </div>
  );
}
