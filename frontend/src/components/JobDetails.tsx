import { useEffect, useState } from 'react';
import { useLang } from '../lib/i18n';
import { watchCustomJob } from '../lib/customModelApi';
import { jobClient, type JobSummary } from '../lib/jobs';
import type { Job } from '../lib/modelPlatform';
import { JobProgress } from './ModelPlatformShared';
import { QueueState, type QueueStatus } from './QueueState';

export function JobDetails({ id, customTask, onChange }: { id: string; customTask?: string; onChange: (job: JobSummary) => void }) {
  return customTask ? <CustomJobDetails key={id} id={id}/> : <ModelJobDetails key={id} id={id} onChange={onChange}/>;
}

function ModelJobDetails({ id, onChange }: { id: string; onChange: (job: JobSummary) => void }) {
  const { lang } = useLang();
  const [job, setJob] = useState<Job>(), [error, setError] = useState('');
  useEffect(() => { let alive = true; void jobClient().get(id).then(j => { if (alive) setJob(j); }).catch(e => { if (alive) setError(e.message); }); return () => { alive = false; }; }, [id]);
  if (error) return <p role="alert" className="mp-error">{error}</p>;
  return job ? <JobProgress initial={job} onChange={onChange}/> : <p>{lang === 'zh' ? '加载结果…' : 'Loading results…'}</p>;
}

function CustomJobDetails({ id }: { id: string }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [logs, setLogs] = useState<string[]>([]), [message, setMessage] = useState(''), [queue, setQueue] = useState<QueueStatus>(), [download, setDownload] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    void watchCustomJob(id, event => {
      if (controller.signal.aborted) return;
      if (event.type === 'log') setLogs(rows => [...rows, event.data.line].slice(-100));
      if (event.type === 'queue') setQueue(event.data);
      if (event.type === 'progress' || event.type === 'done' || event.type === 'error') setMessage(event.data.message);
      if (event.type === 'done' && event.data.csv_download_url) setDownload(event.data.csv_download_url);
    }, controller.signal).catch(e => { if (!controller.signal.aborted) setMessage(e.message); });
    return () => controller.abort();
  }, [id]);
  return <div><QueueState queue={queue}/><p>{message}</p>{download && <a href={download} download>{zh ? '下载结果' : 'Download results'}</a>}{logs.length > 0 && <details><summary>{zh ? '运行日志' : 'Execution log'}</summary><pre>{logs.join('\n')}</pre></details>}</div>;
}
