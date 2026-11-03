import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLang } from '../lib/i18n';
import { canRetryJob, isActiveJob, jobClient, type JobPage, type JobSummary } from '../lib/jobs';
import { JobDetails } from './JobDetails';
import { JobNumber } from './JobNumber';
import { QueueState, type QueueStatus } from './QueueState';
import './JobHistory.css';
import { JOBS_CHANGED, mergeJobUpdate, jobPollDelay, type JobUpdate } from '../lib/jobUpdates';

const statuses: Record<string, [string, string]> = { active: ['In progress', '进行中'], queued: ['Queued', '排队中'], running: ['Running', '运行中'], completed: ['Completed', '已完成'], failed: ['Failed', '失败'], interrupted: ['Interrupted', '已中断'], cancelled: ['Cancelled', '已取消'] };
const kinds: Record<string, [string, string]> = { train: ['Training', '训练'], training: ['Training', '训练'], predict: ['Prediction', '预测'], evaluation: ['Evaluation', '评估'], workflow: ['Workflow', '工作流'] };

export function JobHistory({ customTask, defer = false, presentation = 'inline' }: { customTask?: string; defer?: boolean; presentation?: 'inline' | 'dialog' }) {
  const { lang } = useLang(); const zh = lang === 'zh'; const tr = (pair: [string, string]) => pair[zh ? 1 : 0];
  const client = useMemo(() => jobClient(customTask), [customTask]);
  const [page, setPage] = useState<JobPage>(), [open, setOpen] = useState(false), [selected, setSelected] = useState('');
  const [query, setQuery] = useState(''), [status, setStatus] = useState(''), [kind, setKind] = useState(''), [offset, setOffset] = useState(0);
  const [error, setError] = useState(''), [loading, setLoading] = useState(false), [pending, setPending] = useState<string[]>([]), [revision, setRevision] = useState(0);

  const dialog = useRef<HTMLDialogElement>(null), trigger = useRef<HTMLButtonElement>(null);
  const polling = useRef<AbortController>();
  useEffect(() => {
    const refresh = () => { if (!document.hidden) { polling.current?.abort(); setRevision(v => v + 1); } };
    const changed = (event: Event) => {
      const update = (event as CustomEvent<JobUpdate>).detail;
      if (update.source !== (customTask ? 'custom' : 'platform')) return;
      polling.current?.abort();
      setPage(current => mergeJobUpdate(current, update.job));
      setRevision(v => v + 1);
    };
    window.addEventListener(JOBS_CHANGED, changed);
    window.addEventListener('focus', refresh);
    document.addEventListener('visibilitychange', refresh);
    return () => { window.removeEventListener(JOBS_CHANGED, changed); window.removeEventListener('focus', refresh); document.removeEventListener('visibilitychange', refresh); };
  }, [customTask]);
  useEffect(() => {
    if (presentation !== 'dialog') return;
    if (open && !dialog.current?.open) dialog.current?.showModal();
    else if (!open && dialog.current?.open) { dialog.current.close(); trigger.current?.focus(); }
  }, [open, presentation]);

  useEffect(() => {
    if (defer && !open) return;
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    polling.current = controller;
    async function poll() {
      let delay = 5000;
      try {
        const next = await client.list({ query, status, kind, offset }, controller.signal);
        if (controller.signal.aborted) return;
        setPage(next); setError(''); setLoading(false);
        delay = jobPollDelay(open, next.active > 0, document.hidden);
      } catch (e) { if (!controller.signal.aborted) { setError((e as Error).message); setLoading(false); } }
      if (!controller.signal.aborted) timer = setTimeout(poll, delay);
    }
    setLoading(true); timer = setTimeout(poll, query ? 250 : 0);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [client, query, status, kind, offset, revision, open, defer]);

  async function action(job: JobSummary, operation: 'cancel' | 'retry') {
    if (pending.includes(job.id)) return;
    setPending(ids => [...ids, job.id]); setError('');
    try {
      const next = await client.control(job.id, operation);
      if (operation === 'retry') { setSelected(next.id); setOffset(0); setStatus(''); setQuery(''); }
      setRevision(v => v + 1);
    } catch (e) { setError((e as Error).message); }
    finally { setPending(ids => ids.filter(id => id !== job.id)); }
  }
  function changed(next: JobSummary) {
    setPage(current => mergeJobUpdate(current, next));
    if (page?.items.find(item => item.id === next.id)?.status !== next.status) { polling.current?.abort(); setRevision(v => v + 1); }
    if (next.id !== selected) { setSelected(next.id); setOffset(0); setStatus(''); setQuery(''); setRevision(v => v + 1); }
  }
  const content = <>
    {open && <>
      <div className="job-history-filters">
        <input aria-label={zh ? '搜索我的作业' : 'Search my jobs'} placeholder={zh ? '作业号或模型名称' : 'Job number or model name'} value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }}/>
        <select aria-label={zh ? '作业状态' : 'Job status'} value={status} onChange={e => { setStatus(e.target.value); setOffset(0); }}><option value="">{zh ? '全部状态' : 'All statuses'}</option>{Object.entries(statuses).map(([value, label]) => <option key={value} value={value}>{tr(label)}</option>)}</select>
        {!customTask && <select aria-label={zh ? '作业类型' : 'Job type'} value={kind} onChange={e => { setKind(e.target.value); setOffset(0); }}><option value="">{zh ? '全部类型' : 'All types'}</option>{['predict', 'train', 'workflow'].map(value => <option key={value} value={value}>{tr(kinds[value])}</option>)}</select>}
      </div>
      {error && <p role="alert" className="mp-error">{error}</p>}
      <div className="job-history-list" aria-busy={loading}>{!!page?.items.length && <div className="job-history-columns" aria-hidden="true"><span>{zh ? '名称 / 作业号' : 'Name / job number'}</span><span>{zh ? '类型' : 'Type'}</span><span>{zh ? '状态' : 'Status'}</span><span>{zh ? '提交时间' : 'Submitted'}</span><span>{zh ? '操作' : 'Actions'}</span></div>}{page?.items.map(job => <article key={job.id}>
        <div className="job-history-main"><button className="job-history-title" onClick={() => setSelected(selected === job.id ? '' : job.id)}>{job.title || tr(kinds[job.kind.replace('custom:', '')] || ['Job', '作业'])}</button><JobNumber number={job.job_number}/></div>
        <span className="job-history-kind">{tr(kinds[job.kind.replace('custom:', '')] || ['Job', '作业'])}</span>
        <div className="job-history-status"><span className={'job-state ' + job.status}>{isActiveJob(job) && <span className="job-history-spinner" aria-hidden="true"/>}{tr(statuses[job.status] || [job.status, job.status])}</span>{isActiveJob(job) && <QueueState queue={job.queue || (job.resource ? job as QueueStatus : undefined)}/>}</div>
        <time className="job-history-time" dateTime={new Date(job.created * 1000).toISOString()} title={new Date(job.created * 1000).toLocaleString(zh ? 'zh-CN' : 'en-GB')}>{new Date(job.created * 1000).toLocaleString(zh ? 'zh-CN' : 'en-GB', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</time>
        <div className="job-history-actions"><button type="button" aria-expanded={selected === job.id} onClick={() => setSelected(selected === job.id ? '' : job.id)}>{selected === job.id ? (zh ? '收起' : 'Hide') : (zh ? '查看' : 'View')}</button>{isActiveJob(job) ? <button disabled={pending.includes(job.id) || Boolean(job.queue?.cancel_requested || job.cancel_requested)} onClick={() => void action(job, 'cancel')}>{zh ? '取消' : 'Cancel'}</button> : canRetryJob(job) && <button disabled={pending.includes(job.id)} onClick={() => void action(job, 'retry')}>{zh ? '重试' : 'Retry'}</button>}</div>
        {selected === job.id && <div className="job-history-detail"><button className="job-history-back" onClick={() => setSelected('')}>← {zh ? '返回作业列表' : 'Back to jobs'}</button><JobDetails id={job.id} customTask={customTask} onChange={changed}/></div>}
      </article>)}</div>
      {!page?.items.length && <p className="job-history-empty">{loading ? (zh ? '加载作业…' : 'Loading jobs…') : query || status || kind ? (zh ? '没有匹配的作业' : 'No matching jobs') : (zh ? '还没有提交作业' : 'No jobs yet')}</p>}
      {!!page?.total && <div className="job-history-pagination"><span>{offset + 1}–{Math.min(offset + page.limit, page.total)} / {page.total}</span><button disabled={!offset || loading} onClick={() => setOffset(v => Math.max(0, v - page.limit))}>{zh ? '上一页' : 'Previous'}</button><button disabled={offset + page.limit >= page.total || loading} onClick={() => setOffset(v => v + page.limit)}>{zh ? '下一页' : 'Next'}</button></div>}
    </>}
  </>;
  const active = !!page?.active && <span className="job-history-count">{page.active} {zh ? '项进行中' : 'in progress'}</span>;
  if (presentation === 'dialog') return <>
    <button ref={trigger} className="mp-button job-history-trigger" aria-haspopup="dialog" onClick={() => setOpen(true)}>{zh ? '我的作业' : 'My jobs'}{active}</button>
    {createPortal(<dialog ref={dialog} className="job-history job-history-panel" aria-labelledby="job-history-title" onClose={() => setOpen(false)} onClick={e => { if (e.target === e.currentTarget) setOpen(false); }}>
      <header className="job-history-heading"><div><h2 id="job-history-title">{zh ? '我的作业' : 'My jobs'}</h2>{active}</div><button className="job-history-close" aria-label={zh ? '关闭作业面板' : 'Close jobs'} onClick={() => setOpen(false)}>×</button></header>
      <div className="job-history-content">{content}</div>
    </dialog>, document.body)}
  </>;
  return <details className="job-history" open={open} onToggle={e => setOpen(e.currentTarget.open)}><summary>{zh ? '我的作业' : 'My jobs'}{active}</summary>{content}</details>;
}
