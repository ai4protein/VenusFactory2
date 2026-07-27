import { useState } from "react";
import { useLang } from "../lib/i18n";

type Props = {
  onDecide: (action: "continue" | "skip" | "rewrite", comment?: string) => void;
  disabled?: boolean;
};

const STRINGS = {
  en: {
    continue: "Continue",
    comment: "Revise",
    skip: "Skip to report",
    placeholder: "What should change in this sub-report?",
    submit: "Submit rewrite"
  },
  zh: {
    continue: "继续",
    comment: "修改",
    skip: "跳过至报告",
    placeholder: "希望小报告如何修改？",
    submit: "提交并重写"
  }
};

export function SubReportCheckpoint({ onDecide, disabled }: Props) {
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");
  const t = useLang().t(STRINGS);

  return (
    <div className="step-checkpoint expert-checkpoint-actions">
      <div className="step-checkpoint-options">
        <button
          type="button"
          className="step-checkpoint-btn step-checkpoint-btn-continue step-checkpoint-btn-primary"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9654;</span>
          <span>{t.continue}</span>
        </button>
        <button
          type="button"
          className="step-checkpoint-btn step-checkpoint-btn-rewrite"
          onClick={() => setShowComment(!showComment)}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9998;</span>
          <span>{t.comment}</span>
        </button>
        <button
          type="button"
          className="step-checkpoint-btn step-checkpoint-btn-abort"
          onClick={() => onDecide("skip")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9193;</span>
          <span>{t.skip}</span>
        </button>
      </div>
      {showComment && (
        <div className="sub-report-comment-area">
          <textarea
            className="sub-report-comment-input"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder={t.placeholder}
            rows={3}
            disabled={disabled}
          />
          <button
            type="button"
            className="step-checkpoint-btn step-checkpoint-btn-continue step-checkpoint-btn-primary sub-report-submit-btn"
            onClick={() => {
              onDecide("rewrite", comment);
              setComment("");
              setShowComment(false);
            }}
            disabled={disabled || !comment.trim()}
          >
            <span className="step-checkpoint-btn-icon">&#10148;</span>
            <span>{t.submit}</span>
          </button>
        </div>
      )}
    </div>
  );
}
