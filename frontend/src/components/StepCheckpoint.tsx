type Props = {
  onDecide: (action: "continue" | "abort") => void;
  disabled?: boolean;
};

export function StepCheckpoint({ onDecide, disabled }: Props) {
  return (
    <div className="step-checkpoint">
      <div className="step-checkpoint-options">
        <button
          className="step-checkpoint-btn step-checkpoint-btn-continue"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9654;</span>
          <span>Continue</span>
        </button>
        <button
          className="step-checkpoint-btn step-checkpoint-btn-abort"
          onClick={() => onDecide("abort")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9724;</span>
          <span>Stop & Summarize</span>
        </button>
      </div>
    </div>
  );
}
