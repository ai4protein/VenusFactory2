import type { ScienceGoal } from "../lib/scienceGoals";
import { isRunActive, planProgress } from "../lib/runProgress";
import { RunSpinner } from "./RunSpinner";
import { openResearchFile } from "./SessionFilesPanel";
const roles: Record<string, [string, string]> = { PI: ["Defining direction", "明确研究方向"], CB: ["Planning experiments", "制定实验计划"], MLS: ["Running experiments", "执行代码与实验"], SC: ["Reviewing results", "审核实验结果"], REVIEW: ["Reviewing tool admission", "独立准入审核"] };
export function ResearchProgress({ goal: g, lang }: { goal: ScienceGoal; lang: string }) {
  const zh = lang === "zh", say = (en: string, cn: string) => zh ? cn : en;
  const live = g.research?.live, steps = g.research?.plan?.steps || [];
  const { complete, total, reviewing } = planProgress(steps, live?.checkpoint);
  const running = isRunActive(g.status), role = live?.phase?.role;
  const files = (live?.files || []).filter(f => !f.path.includes("__pycache__/") && !f.path.endsWith(".pyc")), metrics = live?.execution?.progress?.metrics || {};
  return <>
    <div className="research-now" role="status">
      <div className="research-now-title">{running && <RunSpinner lang={lang}/>}<strong>{g.queue?.status === "queued" ? say("Waiting for a worker", "等待开始") : running ? (roles[role || ""]?.[zh ? 1 : 0] || say("Research in progress", "研究进行中")) : g.status === "waiting_for_input" ? say("Waiting for your input", "等待补充信息") : say("Research progress", "研究进度")}</strong>
      {total > 0 && <span>{complete} / {total} {say("steps complete", "项已完成")}</span>}</div>
      {running && <div className="research-stage-row">{Object.entries(roles).map(([id, names]) => <span key={id} className={id === role ? "is-current" : ""} aria-current={id === role ? "step" : undefined}>{id} · {names[zh ? 1 : 0]}</span>)}</div>}
      {running && live?.tool && role === "MLS" && <p className="research-current-tool">{live.tool.tool.replace(/_/g, " ")}</p>}
      {total > 0 && <progress aria-label={say("Completed plan steps", "已完成计划项")} value={complete} max={total}/>}
    </div>
    {total > 0 && <details className="research-plan" open><summary>{say("Research plan", "研究计划")}</summary><ol>{steps.map(s => <li key={s.id} className={s.status === "completed" ? "is-complete" : reviewing.has(s.id) ? "is-reviewing" : ""}><span aria-hidden="true">{s.status === "completed" ? "✓" : reviewing.has(s.id) ? "◷" : "○"}</span><div><strong>{s.title}</strong>{reviewing.has(s.id) && <small className="plan-review-label">{say("Awaiting review", "待审核")}</small>}<p>{s.acceptance}</p></div></li>)}</ol></details>}
    {live?.checkpoint?.summary && <div className="research-result-summary"><strong>{say("Latest result", "最新中间结果")}</strong><p>{live.checkpoint.summary}</p></div>}
    {!!g.research?.publication_reviews?.length && <section className="research-result-summary" aria-label={say("Tool admission", "工具准入")}><strong>{say("Tool admission", "工具准入")}</strong>{g.research.publication_reviews.map(r => <details key={r.id} open={r.status !== 'approved'}><summary>{r.name} · {r.status === 'approved' ? say('Approved', '已通过') : say('Needs revision', '待修订')}</summary>{r.summary && <p>{r.summary}</p>}{r.issues.length > 0 && <ul>{r.issues.map((issue, i) => <li key={i}>{issue}</li>)}</ul>}</details>)}</section>}
    {!!Object.keys(metrics).length && <dl className="research-metrics">{Object.entries(metrics).slice(0, 12).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{typeof value === "number" ? Number(value.toPrecision(5)) : String(value).slice(0, 100)}</dd></div>)}</dl>}
    {!!files.length && <div className="research-result-files"><strong>{say("Result files", "中间产物")} · {files.length}</strong><div>{files.slice(-8).map(f => <button key={f.path} type="button" className="research-file-link" onClick={() => openResearchFile(g.id, f.path)} title={f.path}>{f.path.split('/').pop()}</button>)}</div></div>}
  </>;
}
