import { useState } from "react";

type Props = {
  onDecide: (action: "continue" | "skip" | "rewrite", comment?: string) => void;
  disabled?: boolean;
};

export function SubReportCheckpoint({ onDecide, disabled }: Props) {
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");

  return (
    <div className="step-checkpoint">
      <div className="step-checkpoint-options">
        <button
          className="step-checkpoint-btn step-checkpoint-btn-continue"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9654;</span>
          <span>Continue Research</span>
        </button>
        <button
          className="step-checkpoint-btn step-checkpoint-btn-rewrite"
          onClick={() => setShowComment(!showComment)}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9998;</span>
          <span>Comment & Rewrite</span>
        </button>
        <button
          className="step-checkpoint-btn step-checkpoint-btn-abort"
          onClick={() => onDecide("skip")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9193;</span>
          <span>Skip to Report</span>
        </button>
      </div>
      {showComment && (
        <div className="sub-report-comment-area">
          <textarea
            className="sub-report-comment-input"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Describe what changes you'd like to the sub-report..."
            rows={3}
            disabled={disabled}
          />
          <button
            className="step-checkpoint-btn step-checkpoint-btn-continue sub-report-submit-btn"
            onClick={() => {
              onDecide("rewrite", comment);
              setComment("");
              setShowComment(false);
            }}
            disabled={disabled || !comment.trim()}
          >
            <span className="step-checkpoint-btn-icon">&#10148;</span>
            <span>Submit & Rewrite</span>
          </button>
        </div>
      )}
    </div>
  );
}
