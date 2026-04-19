import { useState } from "react";
import type { ClarificationQuestion, ClarificationAnswer } from "../lib/api";

type Props = {
  questions: ClarificationQuestion[];
  onSubmit: (answers: ClarificationAnswer[]) => void;
  disabled?: boolean;
};

export function ClarificationForm({ questions, onSubmit, disabled }: Props) {
  const [answers, setAnswers] = useState<ClarificationAnswer[]>(
    questions.map((_, i) => ({ question_index: i, selected_options: [], custom_text: "" }))
  );

  function toggleOption(qIdx: number, optIdx: number) {
    setAnswers(prev => {
      const next = [...prev];
      const ans = { ...next[qIdx] };
      const q = questions[qIdx];
      if (q.allow_multiple) {
        if (ans.selected_options.includes(optIdx)) {
          ans.selected_options = ans.selected_options.filter(i => i !== optIdx);
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
    setAnswers(prev => {
      const next = [...prev];
      next[qIdx] = { ...next[qIdx], custom_text: text };
      return next;
    });
  }

  const isOtherOption = (q: ClarificationQuestion, idx: number) => {
    const opt = q.options[idx]?.toLowerCase();
    return opt === "other" || opt === "其他";
  };

  const hasOtherSelected = (qIdx: number) => {
    const q = questions[qIdx];
    return answers[qIdx].selected_options.some(i => isOtherOption(q, i));
  };

  const allAnswered = answers.every(a => a.selected_options.length > 0);

  return (
    <div className="clarification-form">
      {questions.map((q, qIdx) => (
        <div key={qIdx} className="clarification-question">
          <div className="clarification-question-text">
            {qIdx + 1}. {q.question}
            {q.allow_multiple && <span className="clarification-multi-hint">(可多选 / multiple)</span>}
          </div>
          <div className="clarification-options">
            {q.options.map((opt, optIdx) => {
              const selected = answers[qIdx].selected_options.includes(optIdx);
              return (
                <label
                  key={optIdx}
                  className={`clarification-option${selected ? " selected" : ""}`}
                >
                  <input
                    type={q.allow_multiple ? "checkbox" : "radio"}
                    name={`q-${qIdx}`}
                    checked={selected}
                    onChange={() => toggleOption(qIdx, optIdx)}
                    disabled={disabled}
                  />
                  <span>{opt}</span>
                </label>
              );
            })}
          </div>
          {hasOtherSelected(qIdx) && (
            <input
              type="text"
              className="clarification-custom-input"
              placeholder="Please specify..."
              value={answers[qIdx].custom_text}
              onChange={e => setCustomText(qIdx, e.target.value)}
              disabled={disabled}
            />
          )}
        </div>
      ))}
      <button
        className="clarification-submit"
        onClick={() => onSubmit(answers)}
        disabled={disabled || !allAnswered}
      >
        Submit
      </button>
    </div>
  );
}
