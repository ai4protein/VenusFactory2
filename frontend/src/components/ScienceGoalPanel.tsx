import { submissionHistoryRow, type HistoryActivityCallback } from "../lib/optimisticHistory";
import { ScientistWelcome } from "./ScientistWelcome";
import { InteractionCard } from "./InteractionCard";
import type { UserResponse } from "../lib/interactions";
import { answerResearchInteraction } from "../lib/scienceGoals";
import { downloadResearchProject } from "../lib/scienceGoals";
import { submitsComposer } from "../lib/composerKeys";
import { ResearchWorkbench } from "./ResearchWorkbench";
import { isRunActive } from "../lib/runProgress";
import { RunSpinner } from "./RunSpinner";
import { useResearchDraft } from "../lib/useResearchDraft";
import { isDailyQuotaMessage } from "../lib/onlineQuota";
import { QueueState } from './QueueState';
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLang } from "../lib/i18n";
import { goalCatalog, listGoals, createGoal, controlGoal, provideResearchInput, type GoalBudget, type GoalCatalog, type ScienceGoal } from "../lib/scienceGoals";
import { ChatNotice, useDismissibleError } from "./ChatNotice";
import { executionModelItems } from "./ExecutionModelOptions";
import { ComposerSelect } from "./ComposerSelect";
import { useDismissibleDetails } from "../lib/useDismissibleDetails";
import { preferredModel } from "../lib/researchUi";
import "./ScienceGoalPanel.css";
import { ResearchInputs } from "./ResearchInputs";
import { ResearchRevision } from "./ResearchRevision";
import type { ResearchAttachment } from "../lib/scienceGoals";

const unfinished = new Set(["preparing", "active", "paused", "interrupted", "waiting_for_input"]);
const harnessNames: Record<string, string> = { "expert-kimi-code": "Kimi Code", "expert-deepseek-harness": "DeepSeek Harness", "expert-codex": "Codex", "expert-claude-code": "Claude Code", langgraph: "LangGraph" };
const statusNames: Record<string, [string, string]> = {
  preparing: ["Preparing", "准备中"], active: ["Researching", "研究中"], paused: ["Paused", "已暂停"],
  interrupted: ["Interrupted", "已中断"], waiting_for_input: ["Needs input", "待补充信息"],
  completed: ["Complete", "已完成"], budget_exhausted: ["Run limit reached", "已达到运行上限"],
  cancelled: ["Cancelled", "已取消"], prepare_failed: ["Could not start", "启动失败"],
  running: ["Running", "执行中"], pending: ["Pending", "待执行"], needs_revision: ["Revising", "调整中"], failed: ["Needs attention", "需处理"],
};

export function ScienceGoalPanel({ noticeTarget, online, toolbarTarget, composerTarget, actionsTarget, footerTarget, sessionId, draftRevision = 0, science, busy, ensureSession, onRunning, onChanged, onGoalStatus, onHistoryActivity }: {
  noticeTarget: HTMLElement | null; online: boolean; toolbarTarget: HTMLElement | null; composerTarget: HTMLElement | null; actionsTarget: HTMLElement | null; footerTarget: HTMLElement | null; sessionId: string; draftRevision?: number; science: boolean; busy: boolean; ensureSession: () => Promise<string>;
  onRunning: (running: boolean) => void; onChanged: (sid: string) => void | Promise<void>;
  onHistoryActivity: HistoryActivityCallback;
  onGoalStatus: (sid: string, status: string) => void;
}) {
  const { lang } = useLang(); const zh = lang === "zh";
  const formId = useId();
  const settingsRef = useRef<HTMLDetailsElement>(null);
  useDismissibleDetails(settingsRef, science);
  const say = (en: string, cn: string) => zh ? cn : en;
  const label = (value: string) => statusNames[value]?.[zh ? 1 : 0] || value;
  const [catalog, setCatalog] = useState<GoalCatalog | null>(null);
  const [goals, setGoals] = useState<ScienceGoal[]>([]);
  const [draft, setDraft] = useResearchDraft("scientist:" + (sessionId || "draft"), {
    objective: "", attachments: [] as ResearchAttachment[], reply: "",
    execution: { harness: "", model_id: "" },
    budget: { max_cycles: 8, model_calls: 192, tool_calls: 96, wall_seconds: 7200 } as GoalBudget,
  });
  const { objective, attachments, reply, execution, budget } = draft;
  const setObjective = (objective: string) => setDraft(d => ({ ...d, objective }));
  const setAttachments = (attachments: ResearchAttachment[]) => setDraft(d => ({ ...d, attachments }));
  const setReply = (reply: string) => setDraft(d => ({ ...d, reply }));
  const setExecution = (value: typeof execution | ((old: typeof execution) => typeof execution)) => setDraft(d => ({ ...d, execution: typeof value === "function" ? value(d.execution) : value }));
  const setBudget = (budget: GoalBudget) => setDraft(d => ({ ...d, budget }));
  const lastDraftRevision = useRef(draftRevision);
  useEffect(() => {
    if (lastDraftRevision.current === draftRevision) return;
    lastDraftRevision.current = draftRevision;
    setDraft(d => ({ ...d, objective: "", attachments: [], reply: "" })); setError("");
  }, [draftRevision]);
  const [uploading, setUploading] = useState(false);
  const [pending, setPending] = useState(false); const [loaded, setLoaded] = useState(false);
  const [error, setError, dismissError] = useDismissibleError();
  const startLock = useRef(false);
  const callbacks = useRef({ onRunning, onChanged, onGoalStatus, onHistoryActivity });
  callbacks.current = { onRunning, onChanged, onGoalStatus, onHistoryActivity };
  const currentSession = useRef(sessionId); currentSession.current = sessionId;

  function applyGoals(sid: string, next: ScienceGoal[]) {
    if (currentSession.current !== sid) return;
    setGoals(next); setLoaded(true);
    callbacks.current.onRunning(next.some(g => isRunActive(g.status)));
    callbacks.current.onGoalStatus(sid, (next.find(g => g.worker_alive || g.status === "active") || next[0])?.status || "");
  }
  useEffect(() => {
    if (!science || catalog) return;
    let alive = true;
    goalCatalog().then(c => {
      if (!alive) return; setCatalog(c);

    }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [science, catalog]);
  useEffect(() => {
    if (!catalog?.execution || execution.harness) return;
    const c = catalog.execution;
    setDraft(d => ({ ...d, execution: { harness: c.default.harness, model_id: preferredModel(c.combinations, c.default.harness, "", c.default.model_id) }, budget: catalog.default_budget || d.budget }));
  }, [catalog, sessionId, execution.harness]);
  useEffect(() => {
    let alive = true; let timer: ReturnType<typeof setTimeout>;
    setGoals([]); setLoaded(!sessionId); callbacks.current.onRunning(false);
    if (!sessionId) return;
    let polling = false; let hasRunningGoal = true;
    const poll = async () => {
      if (polling || !science || document.visibilityState === "hidden") return; polling = true;
      try { const result = await listGoals(sessionId); if (alive) { applyGoals(sessionId, result.goals); hasRunningGoal = result.goals.some(g => isRunActive(g.status)); } }
      catch (e) { if (alive) setError((e as Error).message); }
      polling = false; if (alive) timer = setTimeout(poll, hasRunningGoal ? 3000 : 15000);
    };
    const rejoin = () => { clearTimeout(timer); void poll(); };
    window.addEventListener("focus", rejoin); window.addEventListener("online", rejoin); document.addEventListener("visibilitychange", rejoin);
    void poll(); return () => { alive = false; clearTimeout(timer); window.removeEventListener("focus", rejoin); window.removeEventListener("online", rejoin); document.removeEventListener("visibilitychange", rejoin); };
  }, [sessionId, science]);

  useEffect(() => {
    const goal = goals.find(g => unfinished.has(g.status));
    const model = catalog?.execution?.models.find(m => m.model === goal?.execution?.model && m.provider === goal?.execution?.provider);
    if (goal?.execution && model) setExecution(previous => previous.harness === goal.execution!.harness && previous.model_id === model.id ? previous : { harness: goal.execution!.harness, model_id: model.id });
  }, [goals, catalog]);

  async function update(sid: string) { applyGoals(sid, (await listGoals(sid)).goals); await callbacks.current.onChanged(sid); }
  async function start() {
    if (pending || startLock.current) return;
    startLock.current = true; setPending(true); setError("");
    const activityKey = crypto.randomUUID(), activityDate = Date.now();
    const showActivity = (sid: string) => callbacks.current.onHistoryActivity(activityKey, submissionHistoryRow("science_expert", activityKey, objective, sid, activityDate));
    showActivity(sessionId);
    const spec = { objective: objective.trim(), budget, execution, attachments: attachments.map(f => f.id) };
    try {
      const sid = await ensureSession(); showActivity(sid);
      const key = `vf2_goal_pending_${sid}`; const serialized = JSON.stringify(spec);
      let previous: { spec: string; id: string } | null = null;
      try { previous = JSON.parse(sessionStorage.getItem(key) || "null"); } catch { /* New request. */ }
      const id = previous?.spec === serialized ? previous.id : crypto.randomUUID();
      sessionStorage.setItem(key, JSON.stringify({ spec: serialized, id }));
      const created = await createGoal(sid, spec, id); applyGoals(sid, [created, ...goals.filter(goal => goal.id !== created.id)]); callbacks.current.onGoalStatus(sid, created.status); sessionStorage.removeItem(key); if (currentSession.current === sid) { setObjective(""); setAttachments([]); } await update(sid);
    } catch (e) { setError((e as Error).message); } finally { callbacks.current.onHistoryActivity(activityKey, null); startLock.current = false; setPending(false); }
  }
  async function control(id: string, action: "pause" | "resume" | "cancel" | "publish") {
    setPending(true); setError("");
    try { await controlGoal(sessionId, id, action); await update(sessionId); }
    catch (e) { setError((e as Error).message); } finally { setPending(false); }
  }
  async function answerInteraction(id: string, response: UserResponse) {
    if (pending) return;
    const sid = sessionId; setPending(true); setError("");
    try { await answerResearchInteraction(sid, id, response); await update(sid); }
    catch (e) { if (currentSession.current === sid) setError((e as Error).message); }
    finally { if (currentSession.current === sid) setPending(false); }
  }
  async function answer(id: string) {
    setPending(true); setError("");
    try { const sid = sessionId; await provideResearchInput(sid, id, reply.trim(), undefined, attachments.map(f => f.id)); if (currentSession.current === sid) { setReply(""); setAttachments([]); } await update(sid); }
    catch (e) { setError((e as Error).message); } finally { setPending(false); }
  }
  const [downloading, setDownloading] = useState("");
  async function downloadProject(id: string, format: "zip" | "report") {
    setDownloading(id + format); setError("");
    try { await downloadResearchProject(sessionId, id, format); }
    catch (e) { setError((e as Error).message); }
    finally { setDownloading(""); }
  }
  const currentGoal = goals.find(g => g.worker_alive || g.status === "active") || goals.find(g => unfinished.has(g.status));
  const active = Boolean(currentGoal);
  const awaitingInteraction = Boolean(currentGoal?.research?.pending_interaction);

  if (!science) return null;
  const choices = catalog?.execution?.combinations || [];
  const ready = choices.some(c => c.harness === execution.harness && c.model_id === execution.model_id && c.ready);
  const canStart = !active && !uploading && !pending && !busy && loaded && ready && objective.trim().length >= 5;
  const executionControls = (
      <div className="scientist-execution-controls">
        <ComposerSelect label="Harness" ariaLabel="Scientist harness" value={execution.harness} disabled={pending || busy || active || !catalog}
          placeholder={say("Loading…", "加载中…")}
          items={(catalog?.execution?.harnesses || []).map(h => ({ value: h.id, label: harnessNames[h.id] || h.id, disabled: !choices.some(c => c.harness === h.id && c.ready), hint: say("Not configured", "未配置") }))}
          onChange={harness => setExecution({ harness, model_id: preferredModel(choices, harness, execution.model_id, catalog?.execution?.default.model_id) })}/>
        <ComposerSelect label={say("Model", "模型")} ariaLabel="Scientist model" value={execution.model_id} disabled={pending || busy || active || !catalog}
          placeholder={say("No models available", "暂无可用模型")}
          items={executionModelItems(catalog?.execution?.models || [], choices, execution.harness, zh, online)}
          onChange={model_id => setExecution({ ...execution, model_id })}/>
      </div>
  );
  const composerForm = <form id={formId} className="scientist-goal-form" onSubmit={e => { e.preventDefault(); if (currentGoal && (reply.trim() || attachments.length > 0) && !pending && !uploading) void answer(currentGoal.id); else if (canStart) void start(); }}>
      <div className="composer-textarea-wrap"><textarea aria-label={say("Research goal", "研究目标")} rows={4} onKeyDown={e => { if (submitsComposer({ key: e.key, shiftKey: e.shiftKey, altKey: e.altKey, isComposing: e.nativeEvent.isComposing, keyCode: e.nativeEvent.keyCode })) { e.preventDefault(); if (currentGoal && (reply.trim() || attachments.length > 0) && !pending && !uploading) void answer(currentGoal.id); else if (canStart) void start(); } }} minLength={active ? 0 : 5} maxLength={8000} value={active ? reply : objective} disabled={pending || busy || awaitingInteraction} onChange={e => active ? setReply(e.target.value) : setObjective(e.target.value)} placeholder={active ? say("Add a direction or adjust the plan…", "补充方向，或说明你想怎样调整计划…") : say("What would you like to develop? Add experimental data or a paper if available.", "你想研发什么？可以附上实验数据、论文或其他材料。")}/></div>
      {<ResearchInputs ensureSession={ensureSession} value={attachments} onChange={setAttachments} onBusy={setUploading} buttonTarget={actionsTarget} transferTarget={composerTarget?.closest<HTMLElement>(".composer")} disabled={pending || busy || !catalog}/>}
    </form>;
  const footer = !active ? <><details ref={settingsRef} className="research-settings"><summary>{say("Run settings", "运行设置")}</summary><div className="research-settings-popover"><div className="research-settings-heading"><strong>{say("Run settings", "运行设置")}</strong><button type="button" aria-label={say("Close run settings", "关闭运行设置")} onClick={() => { if (settingsRef.current) { settingsRef.current.open = false; settingsRef.current.querySelector("summary")?.focus(); } }}>×</button></div><div className="goal-options goal-options-pair"><label>{say("Maximum iterations", "最多迭代轮数")}<input type="number" min={1} max={catalog?.maximum_budget?.max_cycles || 32} value={budget.max_cycles} onChange={e => { const n = Math.max(1, Math.min(catalog?.maximum_budget?.max_cycles || 32, Number(e.target.value) || 1)); setBudget({ ...budget, max_cycles: n, model_calls: Math.min(n * 24, catalog?.maximum_budget?.model_calls || 768), tool_calls: Math.min(n * 12, catalog?.maximum_budget?.tool_calls || 384) }); }}/></label><label>{say("Time limit (hours)", "运行时长（小时）")}<input type="number" min={0.25} step={0.25} max={(catalog?.maximum_budget?.wall_seconds || 2419200) / 3600} value={budget.wall_seconds / 3600} onChange={e => setBudget({ ...budget, wall_seconds: Math.max(900, Math.min(catalog?.maximum_budget?.wall_seconds || 2419200, (Number(e.target.value) || 0.25) * 3600)) })}/></label></div></div></details><span>{say("Enter to send · Shift + Enter for a new line", "Enter 发送 · Shift + Enter 换行")}</span></> : <span>{currentGoal ? label(currentGoal.status) : ""}</span>;
  return <><section className="science-goal-panel" aria-label={say("Scientist research goals", "科学家研究目标")}>
    {error && !isDailyQuotaMessage(error) && noticeTarget && createPortal(<ChatNotice message={error} lang={lang} onDismiss={dismissError} />, noticeTarget)}
    {science && (!catalog || !loaded) && <p role="status">{say("Loading research…", "正在恢复研究…")}</p>}
    {(!sessionId || (loaded && !goals.length)) && <ScientistWelcome lang={lang} disabled={pending || busy || uploading} onExample={prompt => { setObjective(prompt); composerTarget?.querySelector<HTMLTextAreaElement>("textarea")?.focus(); }}/>}
    {goals.map(g => <article className="science-goal-card" key={g.id}>
      <div className="goal-heading"><span className={`goal-status goal-status-${g.status}`}>{isRunActive(g.status) && <RunSpinner lang={lang}/>} {g.queue?.status === "queued" ? say("Queued", "排队中") : label(g.status)}</span><small>{say("RSI iteration", "RSI 迭代")} {g.research?.live?.phase?.episode || g.cycles?.length || 0}</small></div>
      <QueueState queue={g.queue}/><h3>{g.objective.length > 120 ? (g.research?.plan?.summary || g.objective).slice(0,120) + "…" : g.objective}</h3>
      {g.objective.length > 120 && <details className="research-original-goal"><summary>{say("Original goal", "原始目标")}</summary><p>{g.objective}</p></details>}
      <ResearchWorkbench goal={g} sessionId={sessionId} lang={lang}/>
      {g.project && <section className="research-project" aria-label={say("Research project", "研究项目")}>
        <div><strong>{say("Research project", "研究项目")}</strong><span>{g.project.file_count} {say("files", "个文件")} · {label(g.status)}</span></div>
        <div className="research-project-actions">
          <button type="button" disabled={!!downloading || g.worker_alive} onClick={() => void downloadProject(g.id, "zip")}>{say("Download project (.zip)", "下载项目 (.zip)")}</button>
          <button type="button" disabled={!!downloading} onClick={() => void downloadProject(g.id, "report")}>{say("Research report", "研究报告")}</button>
        </div>

      </section>}
      {!g.research?.last_review && !!g.cycles?.length && g.cycles[g.cycles.length - 1].status === "failed" && <p className="goal-note">{g.cycles[g.cycles.length - 1].summary}</p>}
      {g.research?.last_review && <details className="research-review-summary"><summary>{say("Latest review", "最近一次审阅")}</summary><p>{g.research.last_review.summary}</p></details>}
      {!!g.research?.capabilities?.length && <div className="goal-capabilities"><h4>{say("Reusable capabilities", "可复用能力")}</h4>{g.research.capabilities.map(c => <p key={c.id}>{c.name}</p>)}</div>}
      {g.driver === "autonomous_research.v1" && unfinished.has(g.status) && <>{!g.research?.pending_interaction && !!g.research?.questions?.length && <ul className="research-questions">{g.research.questions.map((q, i) => <li key={i}>{q}</li>)}</ul>}{g.research?.pending_interaction ? <InteractionCard key={g.research.pending_interaction.id} interaction={g.research.pending_interaction} sessionId={sessionId} disabled={pending || g.worker_alive} onSubmit={response => answerInteraction(g.id, response)}/> : <ResearchRevision goal={g} sessionId={sessionId} onSaved={() => update(sessionId)}/>}</>}
      <div className="goal-actions">
        {g.status === "active" && <button type="button" disabled={pending} onClick={() => void control(g.id, "pause")}>{say("Pause", "暂停")}</button>}
        {["paused", "interrupted"].includes(g.status) && <button type="button" disabled={pending || g.worker_alive} onClick={() => void control(g.id, "resume")}>{say("Resume", "继续")}</button>}
        {unfinished.has(g.status) && <button type="button" disabled={pending} onClick={() => void control(g.id, "cancel")}>{say("End research", "结束研究")}</button>}
        {g.code_step?.registered && !g.code_step.published && g.status === "completed" && <button type="button" disabled={pending} onClick={() => void control(g.id, "publish")}>{say("Add to library", "加入能力库")}</button>}
      </div>
    </article>)}

  </section>
    {toolbarTarget && createPortal(executionControls, toolbarTarget)}
    {composerTarget && createPortal(composerForm || <div className="composer-textarea-wrap"><textarea rows={4} disabled aria-label={say("Research goal", "研究目标")} value={goals.find(g => unfinished.has(g.status) || g.worker_alive)?.objective || ""} placeholder={say("Loading research…", "正在加载研究…")}/></div>, composerTarget)}
    {footerTarget && createPortal(footer, footerTarget)}
    {actionsTarget && catalog && createPortal(
      <button type="submit" form={formId} className="composer-icon-btn composer-send-btn" disabled={active ? pending || uploading || awaitingInteraction || (!reply.trim() && !attachments.length) : !canStart}
        title={pending ? say("Starting…", "启动中…") : active ? say("Update research", "更新研究") : say("Start research", "开始研究")}
        aria-label={pending ? say("Starting…", "启动中…") : active ? say("Update research", "更新研究") : say("Start research", "开始研究")}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>
      </button>, actionsTarget)}
  </>;
}
