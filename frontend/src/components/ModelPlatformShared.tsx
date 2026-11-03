import { ADMETResults } from './ADMETResults';
import { useEffect, useRef, useState } from 'react';
import { JobNumber } from './JobNumber';
import { QueueState } from './QueueState';
import { Link } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { Dataset, Job, platform, post, uploadDataset } from '../lib/modelPlatform';
import { resultRows, tableCsv, type TableResult } from '../lib/biologicalInputs';
import { notifyJobs } from '../lib/jobUpdates';

export function ModelGlyph({ kind = 'function', size = 25 }: { kind?: string; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    {kind === 'workflow' ? <><rect x="2" y="3" width="9" height="8" rx="1"/><rect x="21" y="20" width="9" height="8" rx="1"/><path d="M6 11v13h15M11 7h15v13"/></> : (kind === 'data' || kind === 'processing') ? <><ellipse cx="16" cy="7" rx="10" ry="4"/><path d="M6 7v17c0 6 20 6 20 0V7M6 15c0 6 20 6 20 0"/></> : kind === 'output' ? <><rect x="5" y="3" width="22" height="26" rx="1"/><path d="M10 10h12M10 16h12M10 22h7"/></> : (kind === 'admet' || kind === 'docking') ? <><path d="m16 3 11 6v14l-11 6-11-6V9Z M16 3v10l11 10M16 13 5 23"/><circle cx="16" cy="13" r="3"/></> : kind === 'design' ? <><path d="m6 24 3-8L23 2l7 7-14 14-10 3Z M19 6l7 7M9 16l7 7"/></> : (kind === 'discovery' || kind === 'search') ? <><circle cx="14" cy="13" r="9"/><path d="m21 20 8 9M10 13h8M14 9v8"/></> : <><path d="M7 3c0 11 18 15 18 26M25 3C25 14 7 18 7 29M8 7h16M11 12h10M11 21h10M8 26h16"/><circle cx="16" cy="16.5" r="2"/></>}
  </svg>;
}

export function DataUpload({ onUpload, compact = false, disabled = false, accept = ".csv,.tsv,.json,.fasta,.fa,.faa,.pdb,.pdbqt,.sdf,.mol", formats = "CSV · TSV · JSON · FASTA · PDB · PDBQT · SDF" }: { onUpload: (d: Dataset) => void; compact?: boolean; disabled?: boolean; accept?: string; formats?: string }) {
  const { lang } = useLang(); const zh = lang === 'zh'; const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [over, setOver] = useState(false);
  async function upload(file?: File) { if (!file || disabled || busy) return; if (!accept.split(',').some(ext => file.name.toLowerCase().endsWith(ext))) { setError(zh ? `请选择 ${formats} 文件。` : `Choose a ${formats} file.`); return; } setBusy(true); setError(''); try { onUpload(await uploadDataset(file)); } catch (e) { setError(String((e as Error).message)); } finally { setBusy(false); if (input.current) input.current.value = ''; } }
  return <div><button type="button" className={`mp-upload ${compact ? 'compact' : ''} ${over ? 'over' : ''}`} disabled={busy || disabled} onClick={() => input.current?.click()} onDragOver={e => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={e => { e.preventDefault(); setOver(false); if (!busy && !disabled) void upload(e.dataTransfer.files[0]); }}>
    <ModelGlyph kind="data"/><strong>{busy ? (zh ? '正在上传…' : 'Uploading…') : zh ? '拖入数据，或选择文件' : 'Drop your data, or choose a file'}</strong><span>{formats}</span>
  </button><input ref={input} hidden type="file" accept={accept} disabled={disabled || busy} onChange={e => void upload(e.target.files?.[0])}/>{error && <p role="alert" className="mp-error">{error}</p>}</div>;
}

export function ResultTable({ rows, inputColumns = [] }: { rows: Record<string, unknown>[]; inputColumns?: string[] }) {
  if (!rows.length) return null;
  const all = Array.from(new Set(rows.flatMap(r => Object.keys(r))));
  const identifiers = all.filter(c => inputColumns.includes(c) && /^(name|id|sample|sample_id|protein_id|sequence_id|accession)$/i.test(c));
  const columns = [...identifiers, ...all.filter(c => !inputColumns.includes(c)), ...all.filter(c => inputColumns.includes(c) && !identifiers.includes(c))];
  return <div className="mp-table-scroll"><table><thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{columns.map(c => <td key={c} title={typeof r[c] === 'object' ? JSON.stringify(r[c]) : String(r[c] ?? '')}>{typeof r[c] === 'object' ? JSON.stringify(r[c]) : typeof r[c] === 'number' ? Number((r[c] as number).toPrecision(6)) : String(r[c] ?? '')}</td>)}</tr>)}</tbody></table></div>;
}

export function JobProgress({ initial, onChange, inputColumns }: { initial: Job; onChange?: (j: Job) => void; inputColumns?: string[] }) {
  const [job, setJob] = useState(initial), [error, setError] = useState('');
  const [downloading, setDownloading] = useState(false);
  const [changing,setChanging]=useState(false);
  async function control(action:'cancel'|'retry'){setChanging(true);try{const next=await post<Job>('/jobs/'+job.id+'/'+action,{});setJob(next);callback.current?.(next);}catch(e){setError((e as Error).message);}finally{setChanging(false);}}
  useEffect(()=>setJob(initial),[initial.id]);
  async function downloadCsv() { setDownloading(true); setError(''); try { const full = await platform<TableResult>('/jobs/' + job.id + '/result'); const url = URL.createObjectURL(new Blob([tableCsv(resultRows(full))], { type: 'text/csv;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = 'predictions.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); } catch (e) { setError((e as Error).message); } finally { setDownloading(false); } }
  const callback = useRef(onChange); callback.current = onChange;
  const { lang } = useLang(); const zh = lang === 'zh';
  useEffect(() => {
    let alive = true, timer: ReturnType<typeof setTimeout>, lastStatus = initial.status;
    setError('');
    async function poll() {
      try { const next = await platform<Job>('/jobs/' + job.id); if (!alive) return; setJob(next); callback.current?.(next); if (next.status !== lastStatus) { lastStatus = next.status; notifyJobs('platform', next); } setError(''); if (['queued', 'running'].includes(next.status)) timer = setTimeout(poll, 1000); }
      catch (e) { if (alive) { setError((e as Error).message); timer = setTimeout(poll, 4000); } }
    }
    void poll(); return () => { alive = false; clearTimeout(timer); };
  }, [job.id]);
  return <section className="mp-job" aria-live="polite"><div className="mp-section-heading"><strong>{job.status === 'completed' ? (zh ? '已完成' : 'Completed') : job.status === 'cancelled' || job.status === 'failed' || job.status === 'interrupted' ? (zh ? '未完成' : 'Run stopped') : job.status === 'queued' ? (zh ? '排队中' : 'Queued') : (zh ? '正在运行' : 'Running')}</strong><span>{job.progress}%</span></div>
    <JobNumber number={job.job_number}/>{['queued','running'].includes(job.status) && <QueueState queue={job.queue}/>}<progress max="100" value={job.progress} aria-label={zh ? '运行进度' : 'Run progress'}/>{job.message && job.status !== 'queued' && !['Running','Completed','Cancelled'].includes(job.message) && <p className={job.status === 'failed' ? 'mp-error' : 'mp-muted'}>{job.message}</p>}
    <div className="mp-job-controls">{['queued','running'].includes(job.status) && <button className="mp-button" disabled={changing||Boolean(job.queue?.cancel_requested)} onClick={()=>void control('cancel')}>{zh?'取消任务':'Cancel task'}</button>}{['failed','interrupted','cancelled'].includes(job.status) && <button className="mp-button" disabled={changing} onClick={()=>void control('retry')}>{zh?'重试':'Retry'}</button>}</div>
    {error && <p role="alert" className="mp-error">{error}</p>}
    {job.result?.model?.metrics && <div className="mp-metrics">{Object.entries(job.result.model.metrics).map(([k,v]) => <div key={k}><span>{k}</span><strong>{v.toFixed(4)}</strong></div>)}</div>}
    {!!job.result?.models?.length && <div className="mp-trained-models">{job.result.models.map(m => <div key={m.id}><strong>{m.name}</strong><span>{zh ? '已保存 · 仅自己' : 'Saved · Private'}</span>{m.metrics && <div className="mp-metrics">{Object.entries(m.metrics).map(([k,v]) => <div key={k}><span>{k}</span><strong>{v.toFixed(4)}</strong></div>)}</div>}<Link className="mp-text-link" to={`/${lang}/models?view=mine`}>{zh ? '在我的模型中查看' : 'View in My models'} →</Link></div>)}</div>}
    {job.result && <>{job.result.metadata?.engine==='ADMET-AI' ? <ADMETResults rows={job.result.preview || []} metadata={job.result.metadata} zh={zh}/> : <ResultTable rows={job.result.preview || []} inputColumns={inputColumns}/>}<div className="mp-result-downloads">{job.result.artifacts_download && <a className="mp-button" href={job.result.artifacts_download} download>{job.result.metadata?.engine==='ADMET-AI' ? (zh?"下载预测与指标说明":"Download predictions & endpoint details") : (zh?"下载结构与序列":"Download structures & sequences")}</a>}{job.result.rows > 0 && <button type="button" className="mp-button" disabled={downloading} onClick={() => void downloadCsv()}>{downloading ? (zh ? '正在准备…' : 'Preparing…') : (zh ? '下载 CSV' : 'Download CSV')}</button>}<a className="mp-text-link" href={job.result.download} download>{job.result.rows ? (zh ? `JSON · ${job.result.rows} 行` : `JSON · ${job.result.rows} rows`) : (zh ? "下载训练报告" : "Download training report")} ↗</a></div></>}
  </section>;
}
