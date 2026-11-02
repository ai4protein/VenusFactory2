export type Prediction = { id: string; task: string; label_zh?: string; model: string; dataset: string; status: string; job_number?: string; values?: Record<string, unknown>[] };
type Activity = { id: number; at?: number; kind: string; key: string; status: string; label: string; job_number?: string };
type ResearchState = { remote_services?: { label: string; label_zh: string; status: string }[]; status: string; phase?: string; activity?: Activity[]; predictions?: Prediction[]; sources?: unknown[]; chapters?: { id: string; title: string; status: string }[]; steps?: { id: string; status: string; message: string }[]; queue?: { position: number } };

function ResearchDog() {
  return <svg className="rr-research-dog" viewBox="0 0 64 40" fill="none" aria-hidden="true"><g className="rr-dog-body" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path className="rr-dog-tail" d="M16 21C6 20 5 13 8 10"/><path fill="white" d="M14 20Q14 15 22 15H38L42 9Q45 5 50 9L54 15L60 17V23H48L44 29H22Q14 29 14 20Z"/><path fill="#e8edf1" d="M42 10Q34 7 36 19Q38 23 43 18Z"/><path className="rr-dog-leg-a" d="M23 28L19 36H25M44 28L48 36H53"/><path className="rr-dog-leg-b" d="M18 27L24 35M40 28L36 35H41"/><path d="M58 18H60"/><circle cx="50" cy="14" r="1.2" fill="currentColor" stroke="none"/></g></svg>;
}

export function ResearchActivity({ job, zh, status, stepLabel }: { job: ResearchState; zh: boolean; status: (s: string) => string; stepLabel: (s: string) => string }) {
  const running = ['queued', 'running'].includes(job.status);
  const predictions = job.predictions || [], chapters = job.chapters || [];
  const liveModel = predictions.find(p => p.status === 'running');
  const liveService = job.remote_services?.find(s => s.status === 'running') || job.remote_services?.find(s => s.status === 'waiting');
  const liveChapter = chapters.find(c => ['writing', 'searching', 'revising'].includes(c.status));
  const phase = job.status === 'queued' ? (zh ? '等待开始研究' : 'Waiting to start') : liveModel ? `${zh ? liveModel.label_zh || liveModel.task : liveModel.task} · ${liveModel.model}` : liveChapter ? liveChapter.title : job.phase === 'reviewing' ? (zh ? '综合研究结果' : 'Bringing the research together') : running && liveService ? `${zh ? liveService.label_zh : liveService.label} · ${status(liveService.status)}` : running ? (zh ? '收集蛋白证据' : 'Gathering protein evidence') : (zh ? '研究记录' : 'Research activity');
  const events: Activity[] = job.activity || (job.steps || []).map((s, i) => ({ id: i, kind: 'step', key: s.id, status: s.status, label: stepLabel(s.id) }));
  const stage = job.status === 'completed' ? 3 : job.phase === 'reviewing' ? 3 : job.phase === 'chapters' ? 2 : predictions.length ? 1 : 0;
  const labels = zh ? ['检索', '预测', '撰写', '整合'] : ['Search', 'Predict', 'Write', 'Review'];
  return <section className={'rr-activity' + (running && job.status !== 'queued' ? ' is-running' : '')} aria-label={zh ? '研究进度' : 'Research progress'}>
    <div className="rr-activity-line">{running && <ResearchDog/>}<div><span className={running ? 'rr-thinking' : ''} role="status">{phase}</span><small>{job.sources?.length || 0} {zh ? '条来源' : 'sources'}{predictions.length > 0 && ` · ${predictions.filter(p => p.status === 'completed').length}/${predictions.length} ${zh ? '项预测' : 'predictions'}`}{chapters.length > 0 && ` · ${chapters.filter(c => c.status === 'completed').length}/${chapters.length} ${zh ? '章' : 'chapters'}`}{job.status === 'queued' && ` · ${zh ? '队列位置' : 'Queue position'} ${job.queue?.position ?? '—'}`}</small></div></div>
    {running && <ol className="rr-stage-track" aria-label={zh ? '研究阶段' : 'Research stages'}>{labels.map((label, i) => <li key={label} className={i <= stage && job.status !== 'queued' ? 'is-reached' : ''} aria-current={i === stage ? 'step' : undefined}><span/>{label}</li>)}</ol>}
    {events.length > 0 && <details className="rr-activity-log"><summary>{zh ? '查看研究日志' : 'View research log'}<span>{events.length}</span></summary><ol>{[...events].reverse().map(event => <li key={event.id}>{event.at && <time dateTime={new Date(event.at*1000).toISOString()}>{new Date(event.at*1000).toLocaleTimeString(zh ? 'zh-CN' : 'en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>}<span>{event.kind === 'phase' ? ({ collecting: zh ? '收集证据' : 'Gather evidence', chapters: zh ? '专题撰写' : 'Write chapters', reviewing: zh ? '综合评述' : 'Review findings', completed: zh ? '报告就绪' : 'Report ready', incomplete: zh ? '阶段性报告' : 'Interim report' }[event.status] || status(event.status)) : event.kind === 'step' ? stepLabel(event.key) : event.label}</span><small>{event.kind !== 'phase' && status(event.status)}</small></li>)}</ol></details>}
  </section>;
}

export function PredictionResults({ rows, zh, status }: { rows: Prediction[]; zh: boolean; status: (s: string) => string }) {
  if (!rows.length) return null;
  return <details className="rr-predictions"><summary>{zh ? 'AI 模型预测' : 'AI model predictions'}<span>{rows.filter(p => p.status === 'completed').length} / {rows.length}</span></summary><div className="rr-prediction-grid">{rows.map(p => <details key={p.id}><summary><span>{zh ? p.label_zh || p.task : p.task}<small>{p.model} · {p.dataset}</small></span><small>{status(p.status)}</small></summary><div>{p.job_number && <small>{p.job_number}</small>}{p.values?.map((row, i) => <dl key={i}>{Object.entries(row).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{typeof value === 'string' ? value : JSON.stringify(value)}</dd></div>)}</dl>)}</div></details>)}</div></details>;
}
