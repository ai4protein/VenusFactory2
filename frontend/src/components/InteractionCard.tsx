import { useId } from "react";
import { useLang } from "../lib/i18n";
import { useResearchDraft } from "../lib/useResearchDraft";
import { canAnswer, makeResponse, type InteractionDraft, type UserInteraction, type UserResponse } from "../lib/interactions";
import "./InteractionCard.css";

export function InteractionCard({ interaction, sessionId, disabled, onSubmit }: {
  interaction: UserInteraction; sessionId: string; disabled?: boolean; onSubmit: (response: UserResponse) => Promise<void>;
}) {
  const { lang } = useLang(); const zh = lang === "zh"; const formId = useId();
  const [draft, setDraft] = useResearchDraft<InteractionDraft>(`interaction:${sessionId}:${interaction.id}`, { answers: {}, feedback: "" });
  const say = (en: string, cn: string) => zh ? cn : en;
  const submit = (decision: UserResponse["decision"]) => { if (!disabled) void onSubmit(makeResponse(interaction, draft, decision)); };
  return <section className="interaction-card" aria-label={say("Your input", "请补充信息")}>
    <h3>{interaction.title || say("A quick question", "请补充一下")}</h3>
    {interaction.message && <div className="interaction-message">{interaction.message}</div>}
    <form onSubmit={e => { e.preventDefault(); if (interaction.kind === "question" && canAnswer(interaction, draft)) submit("answer"); }}>
      <fieldset disabled={disabled} className="interaction-fields">
        {interaction.questions.map(q => {
          const answer = draft.answers[q.id] || { selected: [], text: "" };
          const update = (value: typeof answer) => setDraft(d => ({ ...d, answers: { ...d.answers, [q.id]: value } }));
          return <fieldset key={q.id} className="interaction-question"><legend>{q.question}</legend>
            {q.options.length > 0 && <div className="interaction-options">{q.options.map((option, index) =>
              <label key={index} className={answer.selected.includes(index) ? "is-selected" : ""}>
                <input type={q.multiple ? "checkbox" : "radio"} name={`${formId}-${q.id}`} checked={answer.selected.includes(index)}
                  onChange={() => update({ ...answer, selected: q.multiple ? answer.selected.includes(index) ? answer.selected.filter(n => n !== index) : [...answer.selected, index] : [index] })}/>
                <span>{option}</span>
              </label>)}</div>}
            {q.allow_text && <label className="interaction-text"><span>{q.options.length ? say("Your own answer or details", "自行填写或补充") : say("Your answer", "你的回答")}</span>
              <textarea rows={2} maxLength={4000} value={answer.text} onChange={e => update({ ...answer, text: e.target.value })}/></label>}
          </fieldset>;
        })}
        {interaction.kind !== "question" && <label className="interaction-text"><span>{say("Comments or changes", "意见或修改内容")}</span>
          <textarea rows={3} maxLength={4000} value={draft.feedback} onChange={e => setDraft(d => ({ ...d, feedback: e.target.value }))}/></label>}
        <div className="interaction-actions">
          {interaction.kind === "question" ? <>
            <button type="button" onClick={() => submit("cancel")}>{say("Skip", "暂不回答")}</button>
            <button type="submit" className="interaction-primary" disabled={disabled || !canAnswer(interaction, draft)}>{say("Continue", "继续")}</button>
          </> : <>
            <button type="button" onClick={() => submit("reject")}>{say("Decline", "不同意")}</button>
            <button type="button" disabled={disabled || !draft.feedback.trim()} onClick={() => submit("revise")}>{say("Request changes", "按意见修改")}</button>
            <button type="button" className="interaction-primary" onClick={() => submit("approve")}>{say("Approve", "同意")}</button>
          </>}
        </div>
      </fieldset>
    </form>
  </section>;
}
