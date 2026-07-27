import { useEffect, useState } from "react";
import type { ClarificationAnswer, ClarificationQuestion } from "../lib/api";
import { useLang } from "../lib/i18n";

type Props = {
  questions: ClarificationQuestion[];
  onSubmit: (answers: ClarificationAnswer[]) => void;
  disabled?: boolean;
};

const STRINGS = {
  en: {
    title: "Agent needs your input",
    subtitle: "Answer the questions below so the agent can continue.",
    multipleHint: "multiple choice",
    specifyPlaceholder: "Type your answer…",
    submit: "Submit",
  },
  zh: {
    title: "Agent 正在等待你的回答",
    subtitle: "请回答下列问题，以便 Agent 继续执行。",
    multipleHint: "可多选",
    specifyPlaceholder: "输入自定义回答…",
    submit: "提交",
  },
};

function isOtherOption(q: ClarificationQuestion, idx: number): boolean {
  const opt = q.options[idx] || "";
  if (q.allow_other && q.other_label && opt === q.other_label) return true;
  const lower = opt.toLowerCase();
  return lower === "other" || lower === "其他";
}

export function AskUserCard({ questions, onSubmit, disabled }: Props) {
  const t = useLang().t(STRINGS);
  const [answers, setAnswers] = useState<ClarificationAnswer[]>(
    questions.map((_, i) => ({ question_index: i, selected_options: [], custom_text: "" }))
  );

  useEffect(() => {
    setAnswers(questions.map((_, i) => ({ question_index: i, selected_options: [], custom_text: "" })));
  }, [questions]);

  function toggleOption(qIdx: number, optIdx: number) {
    setAnswers((prev) => {
      const next = [...prev];
      const ans = { ...next[qIdx] };
      const q = questions[qIdx];
      if (q.allow_multiple) {
        if (ans.selected_options.includes(optIdx)) {
          ans.selected_options = ans.selected_options.filter((i) => i !== optIdx);
        } else {
          ans.selected_options = [...ans.selected_options, optIdx];
        }
      } else {
        ans.selected_options = [optIdx];
      }
      next[qIdx] = ans;
      return next;
    });
  }

  function setCustomText(qIdx: number, text: string) {
    setAnswers((prev) => {
      const next = [...prev];
      next[qIdx] = { ...next[qIdx], custom_text: text };
      return next;
    });
  }

  const hasOtherSelected = (qIdx: number) =>
    answers[qIdx].selected_options.some((i) => isOtherOption(questions[qIdx], i));

  const allAnswered = answers.every((a, qIdx) => {
    if (a.selected_options.length === 0) return false;
    if (hasOtherSelected(qIdx) && !a.custom_text.trim()) return false;
    return true;
  });

  return (
    <div className="agent-gate-card ask-user-card">
      <div className="agent-gate-card-header">
        <span className="agent-gate-card-icon" aria-hidden="true">?</span>
        <div className="agent-gate-card-titles">
          <div className="agent-gate-card-title">{t.title}</div>
          <div className="agent-gate-card-subtitle">{t.subtitle}</div>
        </div>
      </div>
      <div className="ask-user-questions">
        {questions.map((q, qIdx) => (
          <div key={qIdx} className="ask-user-question">
            {q.header ? <div className="ask-user-question-header">{q.header}</div> : null}
            <div className="ask-user-question-text">
              {questions.length > 1 ? `${qIdx + 1}. ` : ""}
              {q.question}
              {q.allow_multiple ? (
                <span className="ask-user-multi-hint">{t.multipleHint}</span>
              ) : null}
            </div>
            <div className="ask-user-options">
              {q.options.map((opt, optIdx) => {
                const selected = answers[qIdx].selected_options.includes(optIdx);
                return (
                  <button
                    key={optIdx}
                    type="button"
                    className={`ask-user-option${selected ? " selected" : ""}`}
                    onClick={() => toggleOption(qIdx, optIdx)}
                    disabled={disabled}
                    aria-pressed={selected}
                  >
                    {opt}
                  </button>
                );
              })}
            </div>
            {hasOtherSelected(qIdx) && (
              <input
                type="text"
                className="ask-user-custom-input"
                placeholder={t.specifyPlaceholder}
                value={answers[qIdx].custom_text}
                onChange={(e) => setCustomText(qIdx, e.target.value)}
                disabled={disabled}
              />
            )}
          </div>
        ))}
      </div>
      <div className="agent-gate-card-actions">
        <button
          type="button"
          className="agent-gate-btn agent-gate-btn-primary"
          onClick={() => onSubmit(answers)}
          disabled={disabled || !allAnswered}
        >
          {t.submit}
        </button>
      </div>
    </div>
  );
}
