import { useLang } from "../lib/i18n";

type Props = {
  onDecide: (action: "continue" | "abort") => void;
  disabled?: boolean;
};

const STRINGS = {
  en: {
    continue: "Continue",
    stop: "Stop & Summarize"
  },
  zh: {
    continue: "继续",
    stop: "停止并总结"
  }
};

export function StepCheckpoint({ onDecide, disabled }: Props) {
  const t = useLang().t(STRINGS);
  return (
    <div className="step-checkpoint">
      <div className="step-checkpoint-options">
        <button
          className="step-checkpoint-btn step-checkpoint-btn-continue"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9654;</span>
          <span>{t.continue}</span>
        </button>
        <button
          className="step-checkpoint-btn step-checkpoint-btn-abort"
          onClick={() => onDecide("abort")}
          disabled={disabled}
        >
          <span className="step-checkpoint-btn-icon">&#9724;</span>
          <span>{t.stop}</span>
        </button>
      </div>
    </div>
  );
}
