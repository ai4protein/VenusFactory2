import { useEffect, useMemo, useState } from "react";
import type { ScienceGoal } from "../lib/scienceGoals";
import { getChatSessionAuthHeaders } from "../lib/api";
import { renderMarkdown } from "../lib/markdown";
import { openResearchFile } from "./SessionFilesPanel";
import { ResearchProgress } from "./ResearchProgress";
import { ResearchEvolution } from "./ResearchEvolution";
import { RunSpinner } from "./RunSpinner";
import { isRunActive } from "../lib/runProgress";
import "./ResearchWorkbench.css";

type Tree = { files: string[]; folders: Record<string, Tree> };
function treeOf(paths: string[]) {
  const root: Tree = { files: [], folders: {} };
  for (const path of paths) {
    const parts = path.split("/").filter((part,i,all) => !(all[0] === 'inputs' && i === 1 && /^[a-f0-9-]{24,36}$/i.test(part))); let node = root;
    for (const part of parts.slice(0,-1)) node = node.folders[part] ||= { files: [], folders: {} };
    node.files.push(path);
  }
  return root;
}
function FileTree({ node, selected, choose }: { node: Tree; selected: string; choose: (p: string) => void }) {
  return <ul>{Object.entries(node.folders).sort(([a],[b]) => a.localeCompare(b)).map(([name, child]) => <li key={name}><details open><summary title={name}>{name}</summary><FileTree node={child} selected={selected} choose={choose}/></details></li>)}{node.files.map(path => <li key={path}><button type="button" title={path} aria-current={selected === path ? "true" : undefined} onClick={() => choose(path)}>{path.split('/').pop()}</button></li>)}</ul>;
}
function publicReport(content: string) {
  try { const value = JSON.parse(content); return [value.summary, ...(value.issues || []), ...(value.needs_input || [])].filter(Boolean).join("\n"); }
  catch { return content; }
}
export function ResearchWorkbench({ goal: g, sessionId, lang }: { goal: ScienceGoal; sessionId: string; lang: string }) {
  const zh = lang === "zh", say = (en: string, cn: string) => zh ? cn : en;
  const [tab, setTab] = useState("plan"), [stage, setStage] = useState("all"), [selected, setSelected] = useState(""), [search, setSearch] = useState("");
  const [preview, setPreview] = useState({ key: "", text: "", error: "" });
  const live = g.research?.live, running = isRunActive(g.status);
  const files = useMemo(() => [...new Set([...(g.project?.files || []), ...(live?.files || []).map(f => f.path)])].filter(p => !p.includes('__pycache__/') && !p.endsWith('.pyc')).sort(), [g.project?.files, live?.files]);
  const readme = files.find(p => /^readme\.md$/i.test(p)) || files.find(p => /(^|\/)readme\.md$/i.test(p));
  const chosen = selected && files.includes(selected) ? selected : readme || files[0] || "";
  const textFile = /\.(md|txt|csv|tsv|json|py|yaml|yml|toml|log|r|fasta|fa)$/i.test(chosen);
  const fileKey = `${sessionId}/${g.id}/${chosen}`;
  const revision = live?.files?.find(f => f.path === chosen)?.bytes;
  useEffect(() => {
    if (!chosen || !textFile || tab === 'logs') return;
    const abort = new AbortController();
    const load = async () => {
      try {
        const response = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/files/content?path=${encodeURIComponent(`research-${g.id}/${chosen}`)}`, { headers: getChatSessionAuthHeaders(sessionId), signal: abort.signal, cache: 'no-store' });
        if (!response.ok) throw new Error(say('Preview unavailable', '暂时无法预览'));
        if (Number(response.headers.get('content-length')) > 250000) throw new Error(say('Open this file for the full contents.', '打开文件查看完整内容。'));
        const reader = response.body?.getReader(); if (!reader) return;
        const decoder = new TextDecoder(); let text = '', size = 0;
        while (true) { const { done, value } = await reader.read(); if (done) break; size += value.length; if (size > 250000) { await reader.cancel(); throw new Error(say('Open this file for the full contents.', '打开文件查看完整内容。')); } text += decoder.decode(value, { stream: true }); }
        text += decoder.decode(); if (!abort.signal.aborted) setPreview({ key: fileKey, text, error: '' });
      } catch(e) { if (!abort.signal.aborted) setPreview({ key: fileKey, text: '', error: (e as Error).message }); }
    };
    void load(); return () => abort.abort();
  }, [fileKey, revision, tab, textFile, lang]);
  const roles = [{ id:'PI', name:'PI' }, { id:'CB', name:'Plan' }, { id:'MLS', name:'Implement' }, { id:'SC', name:'Critic' }, { id:'REVIEW', name:'Review' }];
  const matches = (role: string) => stage === 'all' || role === stage;
  const selectStage = (id: string) => { setStage(id); setTab('logs'); };
  return <section className="research-workbench">
    <nav className="research-phases" aria-label={say('Research stages', '研究阶段')}>
      {roles.map(role => {
        const rows = (live?.stages || []).filter(s => s.role === role.id);
        const current = live?.phase?.role === role.id;
        const recorded = !live?.stages?.length && (g.activity || []).some(a => a.role === role.id);
        const done = rows.length > 0 && rows.every(s => s.status === 'completed');
        const status = current && running ? 'running' : done ? 'completed' : rows.length ? 'interrupted' : 'pending';
        return <button type="button" key={role.id} className={`phase-${status}`} aria-current={current && running ? 'step' : undefined} onClick={() => selectStage(role.id)}><span>{status === 'running' ? <RunSpinner lang={lang}/> : done ? '✓' : '○'}</span><strong>{role.name}</strong><small>{status === 'running' ? say('In progress','进行中') : done ? say('Done','已完成') : recorded ? say('Recorded','有记录') : status === 'interrupted' ? say('Paused','待继续') : !live?.stages?.length && !running ? say('No record','无记录') : say('Next','待执行')}</small></button>;
      })}
    </nav>
    <div className="research-tabs" role="tablist" aria-label={say('Research workspace','研究工作区')}>
      {[["plan",say("Plan & results","计划与结果")],["project",say("Project","项目文件")],["logs",say("Activity","过程记录")]].map(([id,title]) => <button type="button" role="tab" aria-selected={tab===id} key={id} onClick={() => { setTab(id); if (id === 'logs') setStage('all'); }}>{title}{id==='project' && files.length ? ` · ${files.length}` : ''}</button>)}
      <span>{say('Iteration','第')} {live?.phase?.episode || (g.cycles?.length || 0) + (running ? 1 : 0)}{zh ? ' 轮' : ''}</span>
    </div>
    {tab === 'plan' && <div className="research-plan-view"><ResearchProgress goal={g} lang={lang}/><ResearchEvolution goal={g} lang={lang}/>{readme && <button className="research-open-readme" type="button" onClick={() => { setSelected(readme); setTab('project'); }}>README.md ↗</button>}</div>}
    {tab === 'project' && <div className="research-project-browser">
      <aside aria-label={say('Project directory','项目目录')}><input aria-label={say('Find a file','查找文件')} placeholder={say('Find a file…','查找文件…')} value={search} onChange={e => setSearch(e.target.value)}/><FileTree node={treeOf(files.filter(p => p.toLowerCase().includes(search.toLowerCase())))} selected={chosen} choose={setSelected}/>{!files.length && <p>{say('Files appear as research progresses.','研究产出的文件将在这里显示。')}</p>}</aside>
      <div className="research-file-preview"><header><strong title={chosen}>{chosen || say('Project','研究项目')}</strong>{chosen && <button type="button" onClick={() => openResearchFile(g.id, chosen)}>{say('Open','打开')} ↗</button>}</header>
        {!chosen ? <p>{say('Your README, code, models and report will appear here.','README、代码、模型和报告将在这里显示。')}</p> : !textFile ? <button type="button" onClick={() => openResearchFile(g.id,chosen)}>{say('Preview file','预览文件')}</button> : preview.key !== fileKey ? <p role="status">{say('Loading…','加载中…')}</p> : preview.error ? <p>{preview.error}</p> : /\.md$/i.test(chosen) ? <div className="markdown-body" dangerouslySetInnerHTML={{ __html: renderMarkdown(preview.text) }}/>: <pre>{preview.text}</pre>}
      </div>
    </div>}
    {tab === 'logs' && <div className="research-activity-view"><label>{say('Stage','阶段')} <select value={stage} onChange={e => setStage(e.target.value)}><option value="all">{say('All','全部')}</option>{roles.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select></label>
      {(g.activity || []).filter((a,i,all) => matches(a.role) && all.findIndex(b => b.role===a.role && b.episode===a.episode && publicReport(b.content)===publicReport(a.content))===i).map(a => <details key={a.id} className="research-role-report"><summary><strong>{a.role === 'CB' ? 'Plan' : a.role === 'MLS' ? 'Implement' : a.role === 'SC' ? 'Critic' : a.role}</strong><span>{say('Iteration','轮次')} {a.episode}</span>{a.timestamp && <time>{new Date(a.timestamp*1000).toLocaleTimeString()}</time>}</summary><p>{publicReport(a.content)}</p></details>)}
      <ol className="research-tool-log">{(live?.timeline || []).filter((e,i,all) => matches(e.role) && !(all[i+1]?.tool===e.tool && ['running','queued'].includes(e.status))).map((e,i) => <li key={`${e.timestamp}-${i}`}><time>{new Date(e.timestamp*1000).toLocaleTimeString()}</time><span>{e.tool.replace(/_/g,' ')}</span><small>{e.status === 'completed' ? '✓' : e.status === 'running' ? say('Running','运行中') : e.status === 'queued' ? say('Queued','排队中') : say('Review log','查看记录')}</small></li>)}</ol>
      {matches('MLS') && (live?.execution || live?.tool?.progress) && <div className="research-experiment-log"><strong>{say('Execution output','实验日志')}</strong>{live.tool?.progress?.progress && <progress max={live.tool.progress.progress.total} value={live.tool.progress.progress.step}/>}<pre>{live.tool?.progress?.stdout_tail || [live.execution?.stdout_tail, live.execution?.stderr_tail].filter(Boolean).join('\n') || say('Waiting for output…','等待执行输出…')}</pre></div>}
      {!g.activity?.length && !live?.timeline?.length && <p role="status">{say('Waiting for the first research update…','等待首条研究进展…')}</p>}
    </div>}
  </section>;
}
