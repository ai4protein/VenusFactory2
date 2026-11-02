import { useResearchDraft } from "../lib/useResearchDraft";
import { useState } from "react";
import { useLang } from "../lib/i18n";
import { provideResearchInput, type ScienceGoal, type ResearchAttachment } from "../lib/scienceGoals";
import { ResearchInputs } from "./ResearchInputs";

export function ResearchRevision({ goal, sessionId, onSaved }: { goal: ScienceGoal; sessionId: string; onSaved: () => Promise<void> }) {
  const { lang } = useLang(); const zh = lang === "zh";
  type Plan = { summary: string; steps: Array<{ id: string; title: string; acceptance: string }> };
  const [draft, setDraft] = useResearchDraft("revision:" + goal.id, { open: false, text: "", attachments: [] as ResearchAttachment[], plan: null as Plan | null });
  const { open, text, attachments, plan } = draft;
  const setOpen = (open: boolean) => setDraft(d => ({ ...d, open }));
  const setText = (text: string) => setDraft(d => ({ ...d, text }));
  const setAttachments = (attachments: ResearchAttachment[]) => setDraft(d => ({ ...d, attachments }));
  const setPlan = (plan: Plan | null) => setDraft(d => ({ ...d, plan }));
  const [busy, setBusy] = useState(false); const [uploading, setUploading] = useState(false); const [error, setError] = useState("");
  function edit() {
    const p = goal.research?.plan;
    if (!plan) setPlan(p ? { summary: p.summary, steps: p.steps.map(({ id, title, acceptance }) => ({ id, title, acceptance })) } : null);
    setOpen(true);
  }
  const original = goal.research?.plan;
  const planChanged = Boolean(plan) && JSON.stringify(plan) !== JSON.stringify(original ? { summary: original.summary, steps: original.steps.map(({ id, title, acceptance }) => ({ id, title, acceptance })) } : null);
  async function save() {
    setBusy(true); setError("");
    try {
      const original = goal.research?.plan;
      const unchanged = original && JSON.stringify(plan) === JSON.stringify({ summary: original.summary, steps: original.steps.map(({ id, title, acceptance }) => ({ id, title, acceptance })) });
      await provideResearchInput(sessionId, goal.id, text, unchanged ? null : plan, attachments.map(f => f.id)); setDraft({ open: false, text: "", attachments: [], plan: null }); await onSaved(); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  return <div className="research-revision">
    {!open && <button type="button" onClick={edit}>{zh ? "调整计划 / 补充材料" : "Adjust plan / Add materials"}</button>}
    {!!goal.research?.pending_updates && <small role="status">{zh ? "将在下一轮应用" : "Applies next iteration"}</small>}
    {open && <form onSubmit={e => { e.preventDefault(); void save(); }}>
      <div className="research-revision-head"><strong>{zh ? "调整研究" : "Update research"}</strong><button type="button" aria-label={zh ? "关闭" : "Close"} onClick={() => setOpen(false)}>×</button></div>
      <textarea value={text} maxLength={8000} onChange={e => setText(e.target.value)} rows={3} aria-label={zh ? "补充要求" : "Research update"} placeholder={zh ? "补充要求、发现或实验方向…" : "Add requirements, findings or a new direction…"}/>
      {plan && <><input aria-label={zh ? "计划摘要" : "Plan summary"} value={plan.summary} onChange={e => setPlan({ ...plan, summary: e.target.value })}/>
        {plan.steps.map((step, index) => <div className="research-step-edit" key={step.id}><span>{index + 1}</span><div>
          <input required maxLength={2000} aria-label={`${zh ? "步骤" : "Step"} ${index + 1}`} value={step.title} onChange={e => setPlan({ ...plan, steps: plan.steps.map(s => s.id === step.id ? { ...s, title: e.target.value } : s) })}/>
          <textarea required maxLength={2000} aria-label={`${zh ? "验收标准" : "Acceptance"} ${index + 1}`} value={step.acceptance} rows={2} onChange={e => setPlan({ ...plan, steps: plan.steps.map(s => s.id === step.id ? { ...s, acceptance: e.target.value } : s) })}/>
        </div><button type="button" disabled={plan.steps.length === 1} aria-label={zh ? "删除步骤" : "Remove step"} onClick={() => setPlan({ ...plan, steps: plan.steps.filter(s => s.id !== step.id) })}>×</button></div>)}
        <button type="button" disabled={plan.steps.length >= 24} onClick={() => setPlan({ ...plan, steps: [...plan.steps, { id: crypto.randomUUID(), title: "", acceptance: "" }] })}>{zh ? "+ 添加步骤" : "+ Add step"}</button></>}
      <ResearchInputs ensureSession={async () => sessionId} value={attachments} onChange={setAttachments} onBusy={setUploading} disabled={busy}/>
      {error && <p role="alert" className="mp-error">{error}</p>}
      <button type="submit" className="research-save" disabled={busy || uploading || (!planChanged && !text.trim() && !attachments.length)}>{busy ? (zh ? "保存中…" : "Saving…") : goal.worker_alive ? (zh ? "下一轮应用" : "Apply next iteration") : (zh ? "保存并继续" : "Save and continue")}</button>
    </form>}
  </div>;
}
