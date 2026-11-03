import { CompletedExamples } from "./CompletedExamples";
import { submissionHistoryRow, type HistoryActivityCallback } from "../lib/optimisticHistory";
import { claimNewChatDraft, OPEN_NEW_CHAT, type ResearchHistoryRow } from "../lib/unifiedChatHistory";
import { shareRead } from "../lib/readRequests";
import { ResearchSelection } from "../lib/researchContinuity";
import { useResearchDraft } from "../lib/useResearchDraft";
import { useFileTransfer } from "../lib/useFileTransfer";
import { validateUploads } from "../lib/uploadLimits";
import { isDailyQuotaMessage } from "../lib/onlineQuota";
import { quotaHeaders, quotaError, QUOTA_CHANGED } from "../lib/onlineQuota";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { ChatNotice, useDismissibleError } from "./ChatNotice";
import { modelName } from "../lib/researchUi";
import { ComposerSelect } from './ComposerSelect';
import { ResearchReportReader } from './ResearchReportReader';
import { ResearchActivity, PredictionResults, type Prediction } from './ResearchActivity';

type Material = { id: string; name: string; bytes: number; url?: string; extraction: { format: string; truncated: boolean } };
type Protein = { id?: string; pdb_id?: string; structure_chain?: string; accession: string; name: string; organism: string; length?: number; reviewed?: boolean };
type Scope = { id: string; label: string; label_zh: string; available: boolean; default: boolean; remote?: boolean };
type Chapter = { id: string; title: string; status: string; length: number; draft?: string; source_ids: string[] };
type Options = { online?: boolean; chapters?: { id: string; label: string; label_zh: string }[]; minimum_chapters?: number; web_search?: { provider: string; tavily_configured: boolean }; models: { id: string; label: string; provider: string; ready: boolean; free: boolean }[]; default_model: string; analyses: Scope[] };
type Source = { id: string; title: string; url: string; text: string; kind: string; evidence_level: string; year?: string };
type RemoteService = { id: string; label: string; label_zh: string; status: string; url: string; count?: number; job_id?: string; summary?: string; data?: unknown };
type Report = { materials?: Material[]; resumable?: boolean; stop_reason?: string; predictions?: Prediction[]; activity?: { id: number; at: number; kind: string; key: string; status: string; label: string; job_number?: string }[]; remote_services?: RemoteService[]; chapters?: Chapter[]; phase?: string; minimum_met?: boolean; completed_chapters?: number; id: string; title: string; created: number; job_number: string; status: string; model_id?: string; partial?: boolean; report?: string; message?: string; focus?: string; sources?: Source[]; sections?: { id: string; title: string; data?: unknown; text?: string }[]; steps?: { id: string; status: string; message: string }[]; queries?: string[]; plan?: { id: string; label: string }[]; queue?: { position: number; cancel_requested: boolean } };
const proteinId = (protein?: Protein) => protein?.id || protein?.accession || '';
const proteinLabel = (protein: Protein) => protein.pdb_id ? `${protein.pdb_id} · ${protein.structure_chain}${protein.accession ? ' · ' + protein.accession : ''}` : protein.accession;
const pending = (r?: Report) => Boolean(r && ['queued', 'running'].includes(r.status));

function PublicEvidence({ service, zh }: { service: RemoteService; zh: boolean }) {
  const data = (service.data || {}) as Record<string, unknown>;
  const key = ['hits', 'entries', 'annotations', 'partners', 'pathways', 'models', 'structures'].find(k => Array.isArray(data[k]));
  const rows = key ? data[key] as Record<string, unknown>[] : [];
  const text = (value: unknown) => value === null || value === undefined ? '—' : String(value);
  const columnSets: Record<string, string[]> = {
    blast: ['Protein', 'Identity (%)', 'Query coverage (%)', 'E-value'],
    foldseek_remote: ['Structure / database', 'Identity (%)', 'E-value', 'Score'],
    interpro: ['Domain / family', 'Identifier', 'Type'], quickgo: ['GO term', 'Evidence', 'Reference'],
    string: ['Partner', 'Combined score', 'Experimental score'], reactome: ['Pathway', 'Identifier'], kegg: ['Pathway', 'Identifier', 'Genes'],
    alphafold: ['Predicted model', 'Mean pLDDT', 'Version'], pdbe: ['PDB / chain', 'Coverage', 'Resolution (Å)'],
  };
  const chinese: Record<string, string[]> = {
    blast: ['蛋白', '一致性 (%)', '查询覆盖率 (%)', 'E-value'], foldseek_remote: ['结构 / 数据库', '一致性 (%)', 'E-value', '得分'],
    interpro: ['结构域 / 家族', '编号', '类型'], quickgo: ['GO 注释', '证据代码', '参考来源'],
    string: ['关联蛋白', '综合得分', '实验得分'], reactome: ['通路', '编号'], kegg: ['通路', '编号', '基因'],
    alphafold: ['预测模型', '平均 pLDDT', '版本'], pdbe: ['PDB / 链', '覆盖率', '分辨率 (Å)'],
  };
  const cells = (row: Record<string, unknown>): unknown[] => {
    const meta = (row.metadata || {}) as Record<string, unknown>;
    switch (service.id) {
      case 'blast': return [row.accession, row.identity_pct, row.query_coverage_pct, row.evalue];
      case 'foldseek_remote': return [`${text(row.target)} · ${text(row.database)}`, row.seqId, row.eval, row.score];
      case 'interpro': return [meta.name, meta.accession, meta.type];
      case 'quickgo': return [row.goName || row.goId, row.goEvidence || row.evidenceCode, row.reference];
      case 'string': return [row.preferredName_B, row.score, row.escore];
      case 'kegg': return [row.name, row.id, (row.genes as string[] || []).join(', ')];
      case 'reactome': return [row.displayName, row.stId];
      case 'alphafold': return [row.modelEntityId, row.globalMetricValue, row.latestVersion];
      case 'pdbe': return [`${text(row.pdb_id)} / ${text(row.chain_id)}`, row.coverage, row.resolution];
      default: return [];
    }
  };
  return <>{rows.length > 0 && <div className="rr-service-table"><table><thead><tr>{(zh ? chinese : columnSets)[service.id]?.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{cells(row).map((cell, j) => <td key={j}>{text(cell)}</td>)}</tr>)}</tbody></table></div>}{typeof data.retention === 'string' && <p>{zh ? '每个数据库保留前 20 条命中，完整列表见服务结果。' : data.retention}</p>}<details className="rr-raw"><summary>{zh ? '原始数据' : 'Raw data'}</summary><pre>{JSON.stringify(service.data, null, 2)}</pre></details></>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api/report' + path, { credentials: 'same-origin', ...init, headers: { ...quotaHeaders(), ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...init?.headers } });
  const body = await response.json();
  if (!response.ok) throw new Error(quotaError(body) || (typeof body.detail === 'string' ? body.detail : body.detail?.message || 'Please check your input and try again.'));
  if (init?.method === 'POST') window.dispatchEvent(new Event(QUOTA_CHANGED));
  return body as T;
}

export type ResearcherPanelHandle = { deleteReport: (id: string) => Promise<boolean> };
export const ResearcherPanel = forwardRef<ResearcherPanelHandle, {
  freshDraftId?: string; onHistoryChange: (rows: ResearchHistoryRow[]) => void; onHistoryActivity: HistoryActivityCallback;
  routeJobId?: string; onJobChange?: (id: string, replace?: boolean) => void;
  noticeTarget: HTMLElement | null; online?: boolean; active: boolean; lang: string; toolbarTarget: HTMLElement | null; composerTarget: HTMLElement | null; actionsTarget: HTMLElement | null;
  footerTarget: HTMLElement | null; onRunning: (running: boolean) => void;
}>(function ResearcherPanel({ freshDraftId, onHistoryChange, onHistoryActivity, routeJobId, onJobChange, noticeTarget, online = false, active, lang, toolbarTarget, composerTarget, actionsTarget, footerTarget, onRunning }, ref) {
  const zh = lang === 'zh'; const tr = (en: string, cn: string) => zh ? cn : en;
  const [options, setOptions] = useState<Options>();
  const [draft, setDraft] = useResearchDraft('researcher:draft', {
    materials: [] as Material[], configured: false, model: '', scopes: [] as string[], query: '', protein: undefined as Protein | undefined,
    sequences: {} as Record<string, string>, chain: '', fileName: '', structurePdb: '', focus: '', paste: '', pasteOpen: false,
  });
  const materials = draft.materials || [];
  const { model, scopes, query, protein, sequences, chain, fileName, structurePdb, focus, paste, pasteOpen } = draft;
  function field<K extends keyof typeof draft>(key: K) { return (value: typeof draft[K] | ((old: typeof draft[K]) => typeof draft[K])) => setDraft(d => ({ ...d, [key]: typeof value === 'function' ? (value as (old: typeof draft[K]) => typeof draft[K])(d[key]) : value })); }
  const setMaterials = field('materials'), setModel = field('model'), setScopes = field('scopes'), setQuery = field('query'), setProtein = field('protein'), setSequences = field('sequences'), setChain = field('chain'), setFileName = field('fileName'), setStructurePdb = field('structurePdb'), setFocus = field('focus'), setPaste = field('paste'), setPasteOpen = field('pasteOpen');
  const [matches, setMatches] = useState<Protein[]>([]);
  const [job, setJob] = useState<Report>(), [history, setHistory] = useState<Report[]>([]);
  const [error, setError, dismissError] = useDismissibleError();
  const [busy, setBusy] = useState(false), [searched, setSearched] = useState(false), [view, setView] = useState<'report' | 'sources'>('report');
  const attachments = useRef<HTMLInputElement>(null), input = useRef<HTMLInputElement>(null), lock = useRef(false), lookupRevision = useRef(0), viewRevision = useRef(0);
  const selection = useRef(new ResearchSelection());
  const [loadingJob, setLoadingJob] = useState(false);
  const callbacks = useRef({ onJobChange, onHistoryActivity }); callbacks.current = { onJobChange, onHistoryActivity };
  const historyRevision = useRef(0);
  const jobRef = useRef(job); jobRef.current = job;
  const inputLock = useRef(false);
  const submission = useRef({ payload: "", id: crypto.randomUUID() });
  function reportPayload(value: Record<string, unknown>) {
    const encoded = JSON.stringify(value);
    if (submission.current.payload !== encoded) submission.current = { payload: encoded, id: crypto.randomUUID() };
    return JSON.stringify({ ...value, request_id: submission.current.id });
  }
  const running = pending(job);
  const status = (value: string) => ({ deferred: tr('Deferred', '稍后可运行'), collecting: tr('Gathering evidence', '收集证据'), chapters: tr('Writing chapters', '撰写专题'), not_applicable: tr('Input needed', '缺少对应输入'), waiting: tr('Server processing', '服务端处理中'), queued: tr('Queued', '排队中'), running: tr('Researching', '研究中'), completed: tr('Complete', '已完成'), incomplete: tr('Incomplete', '未完成'), pending: tr('Waiting', '等待中'), searching: tr('Searching', '检索中'), writing: tr('Writing', '撰写中'), revising: tr('Revising', '修订中'), reviewing: tr('Reviewing', '综合审阅'), partial: tr('Partial', '部分完成'), unavailable: tr('Unavailable', '暂不可用'), empty: tr('No evidence found', '未找到证据'), failed: tr('Stopped', '已停止'), cancelled: tr('Cancelled', '已取消'), interrupted: tr('Interrupted', '已中断') }[value] || value);
  async function loadHistory() { const revision = historyRevision.current; const h = await shareRead("research-history", () => request<{ items: Report[] }>('/jobs')); if (revision === historyRevision.current) setHistory(h.items); return h.items; }
  useEffect(() => {
    if (!active || options) return;
    let alive = true;
    void request<Options>('/options').then(o => {
      if (!alive) return; setOptions(o);
      setDraft(d => ({ ...d,
        model: o.models.some(m => m.id === d.model && m.ready && (!o.online || m.free)) ? d.model : o.models.find(m => m.id === o.default_model && m.ready && (!o.online || m.free))?.id || o.models.find(m => m.ready && (!o.online || m.free))?.id || '',
        scopes: d.configured ? d.scopes.filter(id => o.analyses.some(a => a.id === id && a.available)) : o.analyses.filter(a => a.default && a.available).map(a => a.id),
      }));
    }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [active, options]);
  useEffect(() => { onHistoryChange(history); }, [history, onHistoryChange]);
  useImperativeHandle(ref, () => ({ deleteReport }));
  useEffect(() => {
    if (!active) return;
    if (freshDraftId && !routeJobId) {
      if (claimNewChatDraft("researcher", freshDraftId, sessionStorage)) reset(false);
      else { selection.current.select(""); setJob(undefined); setLoadingJob(false); }
      void loadHistory().catch(e => setError(e.message)); return;
    }
    void loadHistory().catch(e => setError(e.message));
    const id = routeJobId || sessionStorage.getItem('venus.researcher.job') || '';
    if (id) void selectJob(id, true);
    else { selection.current.select(''); setJob(undefined); setLoadingJob(false); }
    const rejoin = () => { const id = selection.current.id; if (id) void selectJob(id, true); else void loadHistory().catch(() => {}); };
    window.addEventListener('focus', rejoin); window.addEventListener('online', rejoin);
    return () => { window.removeEventListener('focus', rejoin); window.removeEventListener('online', rejoin); };
  }, [active, routeJobId, freshDraftId]);
  useEffect(() => { onRunning(running || history.some(pending)); }, [running, history, onRunning]);
  useEffect(() => {
    let alive = true, fetching = false;
    const refresh = async () => {
      if (fetching) return; fetching = true; const revision = historyRevision.current;
      try { const h = await shareRead("research-history", () => request<{ items: Report[] }>('/jobs')); if (alive && revision === historyRevision.current) setHistory(h.items); }
      catch { /* History will catch up on reconnect. */ }
      finally { fetching = false; }
    };
    void refresh(); const timer = window.setInterval(() => void refresh(), active ? 5000 : 15000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [active]);
  useEffect(() => {
    if (!job || !running || !active) return;
    let alive = true; const id = job.id; const revision = selection.current.revision;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (document.visibilityState === "hidden") { timer = setTimeout(poll, 5000); return; }
        const next = await request<Report>('/jobs/' + id);
        if (!alive || !selection.current.current(id, revision)) return;
        setJob(next);
        setHistory(rows => rows.map(r => r.id === next.id ? { ...r, status: next.status } : r));
        if (!pending(next)) { await loadHistory(); return; }
      } catch (e) { if (alive) setError((e as Error).message); }
      if (alive) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 1200);
    return () => { alive = false; clearTimeout(timer); };
  }, [job?.id, running, loadingJob, active, selection.current.revision]);
  async function selectJob(id: string, replace = false) {
    const revision = selection.current.select(id); ++viewRevision.current;
    sessionStorage.setItem('venus.researcher.job', id);
    callbacks.current.onJobChange?.(id, replace);
    if (jobRef.current?.id !== id) { setJob(undefined); setLoadingJob(true); }
    setError('');
    try {
      const r = await request<Report>('/jobs/' + id);
      if (selection.current.current(id, revision)) { setJob(r); setView('report'); }
    } catch (e) { if (selection.current.current(id, revision)) setError((e as Error).message); }
    finally { if (selection.current.current(id, revision)) setLoadingJob(false); }
  }
  function reset(updateUrl = true) {
    submission.current = { payload: "", id: crypto.randomUUID() };
    selection.current.select(''); setLoadingJob(false); sessionStorage.setItem('venus.researcher.job', ''); if (updateUrl) callbacks.current.onJobChange?.('');
    viewRevision.current++; lookupRevision.current++; setJob(undefined); setMaterials([]); setFocus(''); setProtein(undefined); setSequences({}); setStructurePdb(''); setChain(''); setQuery(''); setMatches([]); setSearched(false); setError(''); setPaste(''); setFileName(''); setView('report'); sessionStorage.setItem('venus.researcher.job', '');
  }
  async function search() {
    if (!query.trim() || busy) return;
    const revision = ++lookupRevision.current;
    setBusy(true); setError(''); setProtein(undefined);
    try {
      const r = await request<{ items: Protein[] }>('/resolve?query=' + encodeURIComponent(query.trim()));
      if (revision !== lookupRevision.current) return;
      setMatches(r.items); setSearched(true);
      if (r.items.length === 1) { setProtein(r.items[0]); setSequences({}); setStructurePdb(''); }
    } catch (e) { if (revision === lookupRevision.current) setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function receiveFiles(files: File[]) {
    if (busy || inputLock.current || !files.length) return;
    const targets = files.filter(f => /\.(fa|fasta|faa|pdb|ent)$/i.test(f.name));
    const docs = files.filter(f => !targets.includes(f));
    if (targets.length > 1) { setError(tr('Choose one target protein file.', '请选择一个目标蛋白文件。')); return; }
    if (materials.length + docs.length > 12) { setError(tr('Attach up to 12 research materials.', '最多添加 12 份研究材料。')); return; }
    if (docs.some(f => !/\.(pdf|csv|tsv|xlsx|txt|md|json)$/i.test(f.name))) { setError(tr('Use PDF, CSV, TSV, XLSX, TXT, MD or JSON for research materials.', '研究材料支持 PDF、CSV、TSV、XLSX、TXT、MD 或 JSON。')); return; }
    const revision = viewRevision.current;
    inputLock.current = true; setBusy(true); setError('');
    try {
      await validateUploads(files);
      if (targets.length) {
        const file = targets[0], body = new FormData(); body.append('file', file);
        const parsed = await request<{ sequence_map: Record<string, string>; selected_chain: string; structure_pdb?: string }>('/upload', { method: 'POST', body });
        if (revision !== viewRevision.current) return;
        setStructurePdb(parsed.structure_pdb || ''); setSequences(parsed.sequence_map); setChain(parsed.selected_chain); setFileName(file.name); setProtein(undefined); setMatches([]); setPasteOpen(false); setSearched(false);
      }
      for (const file of docs) {
        const body = new FormData(); body.append('file', file);
        const row = await request<Material>('/materials', { method: 'POST', body });
        if (revision !== viewRevision.current) return;
        setMaterials(old => [...(old || []), row]);
      }
    } catch (e) { if (revision === viewRevision.current) setError((e as Error).message); }
    finally { inputLock.current = false; setBusy(false); }
  }
  useFileTransfer(composerTarget?.closest<HTMLElement>('.composer'), {
    disabled: loadingJob || !!selection.current.id || busy || running || !!job,
    label: tr('Drop protein files or research materials', '松开以上传蛋白文件或研究材料'),
    unavailable: tr('Start a new research to attach files.', '请在新建研究中添加文件。'),
    unreadable: tr('Choose a file; folders are not supported.', '请选择文件，暂不支持文件夹。'),
    onFiles: receiveFiles, onError: setError,
  }, active);
  async function parseInput(file?: File) {
    if (busy || inputLock.current) return;
    const revision = viewRevision.current;
    inputLock.current = true; setBusy(true); setError('');
    try {
      if (file) await validateUploads([file]);
      const body = new FormData(); if (file) body.append('file', file);
      const r = await request<{ sequence_map: Record<string, string>; selected_chain: string; structure_pdb?: string }>(file ? '/upload' : '/parse-input', { method: 'POST', body: file ? body : JSON.stringify({ content: paste }) });
      if (revision !== viewRevision.current) return;
      setStructurePdb(r.structure_pdb || ''); setSequences(r.sequence_map); setChain(r.selected_chain); setFileName(file?.name || tr('Pasted sequence', '粘贴序列')); setProtein(undefined); setMatches([]); setPasteOpen(false); setSearched(false);
    } catch (e) { setError((e as Error).message); } finally { inputLock.current = false; setBusy(false); }
  }
  async function start() {
    if (lock.current || inputLock.current || busy || running || !model || (!protein && !sequences[chain]) || !scopes.length) return;
    lock.current = true; setBusy(true); setError('');
    const activityKey = crypto.randomUUID(), activityDate = Date.now(); historyRevision.current++;
    const showActivity = (id = "") => callbacks.current.onHistoryActivity(activityKey, submissionHistoryRow("researcher", activityKey, protein?.name || fileName || chain, id, activityDate));
    showActivity();
    const revision = ++viewRevision.current;
    try {
      const r = await request<Report>('/jobs', { method: 'POST', body: reportPayload({ model_id: model, accession: protein?.pdb_id ? '' : protein?.accession || '', pdb_id: protein?.pdb_id || '', protein: protein?.name || chain, sequence: protein ? '' : sequences[chain], structure_pdb: protein ? '' : structurePdb, structure_chain: protein?.pdb_id ? protein.structure_chain || '' : !protein && structurePdb ? chain : '', focus, analyses: scopes, lang, material_ids: materials.map(m => m.id) }) });
      historyRevision.current++; showActivity(r.id);
      setHistory(rows => [r, ...rows.filter(row => row.id !== r.id)]);
      if (revision === viewRevision.current) { selection.current.select(r.id); setJob(r); setView('report'); sessionStorage.setItem('venus.researcher.job', r.id); callbacks.current.onJobChange?.(r.id, true); }
      await loadHistory();
    } catch (e) { setError((e as Error).message); } finally { callbacks.current.onHistoryActivity(activityKey, null); historyRevision.current++; setBusy(false); lock.current = false; }
  }
  async function resume() {
    if (!job || busy || !job.resumable) return;
    const id = job.id, revision = selection.current.revision;
    setBusy(true); setError('');
    try {
      const r = await request<Report>('/jobs/' + id + '/resume', { method: 'POST' });
      if (selection.current.current(id, revision)) setJob(r);
      await loadHistory();
    } catch (e) { if (selection.current.current(id, revision)) setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function cancel() {
    if (!job || busy) return;
    setBusy(true); try { setJob(await request<Report>('/jobs/' + job.id + '/cancel', { method: 'POST' })); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  const stepLabel = (id: string) => options?.analyses.find(a => a.id === id)?.[zh ? 'label_zh' : 'label'] || (id.startsWith('literature') ? tr('Literature search', '检索文献') : id.startsWith('web') ? tr('Web search', '网页检索') : id === 'structure_metadata' ? tr('PDB structure details', 'PDB 结构详情') : id === 'materials' ? tr('Reading your materials', '阅读研究材料') : id === 'plan' ? tr('Research plan', '研究计划') : id === 'synthesis' ? tr('Writing report', '撰写报告') : id);
  if (!active) return null;
  async function deleteReport(id: string) {
    if (busy || history.some(row => row.id === id && pending(row))) return false;
    if (!window.confirm(tr('Delete this research from your conversations?', '从对话中删除这项研究？'))) return false;
    setBusy(true); setError('');
    try {
      await request('/jobs/' + id, { method: 'DELETE' });
      setHistory(rows => rows.filter(row => row.id !== id));
      if (job?.id === id) reset();
      localStorage.setItem('vf_library_revision', String(Date.now()));
      window.dispatchEvent(new Event('vf:library-changed'));
      return true;
    } catch (e) { setError((e as Error).message); return false; }
    finally { setBusy(false); }
  }
  const visibleModels = options?.models.filter(m => m.ready && (!options.online || m.free)) || [];
  const label = protein ? `${proteinLabel(protein)} · ${protein.organism}` : sequences[chain] ? `${fileName} · ${sequences[chain].length} aa` : '';
  const visibleMatches = matches.length ? matches : protein ? [protein] : [];
  return <>
    {toolbarTarget && createPortal(<ComposerSelect label={tr('Model', '模型')} ariaLabel={tr('Research model', '研究模型')} helpText={tr('API response times vary. Please allow some time for a response.', '根据API反馈速度不一可能需要耐心等待')} value={job?.model_id || model}
      disabled={busy || !!job || !visibleModels.length} onChange={setModel}
      placeholder={!options ? tr('Loading…', '加载中…') : tr('No models available', '暂无可用模型')}
      items={visibleModels.map(m => ({ value: m.id, label: modelName(m) }))}/>, toolbarTarget)}
    {composerTarget && createPortal(<textarea className="rr-composer" aria-label={tr('Research focus', '研究重点')} placeholder={tr('What would you like to learn about this protein? (optional)', '希望重点了解这个蛋白的哪些方面？（可选）')} rows={4} maxLength={3000} value={job ? (job.focus || '') : focus} disabled={loadingJob || running || busy || !!job} onChange={e => setFocus(e.target.value)}/>, composerTarget)}
    {actionsTarget && createPortal(<div className="rr-actions"><input ref={input} type="file" hidden accept=".fa,.fasta,.faa,.pdb,.ent,.txt" onChange={e => { const files = Array.from(e.target.files || []); e.target.value = ''; if (files.length) void parseInput(files[0]); }}/><input ref={attachments} type="file" hidden multiple accept=".fa,.fasta,.faa,.pdb,.ent,.pdf,.csv,.tsv,.xlsx,.txt,.md,.json" onChange={e => { const files = Array.from(e.target.files || []); e.target.value = ''; void receiveFiles(files); }}/>{!job && <button type="button" className="rr-attach" aria-label={tr('Attach research files', '添加研究文件')} title="FASTA · PDB · PDF · CSV · XLSX" disabled={busy} onClick={() => attachments.current?.click()}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 12 6-6a3 3 0 0 1 4 4l-8 8a5 5 0 0 1-7-7l9-9M7 13l7-7"/></svg></button>}{running ? <button className="rr-primary" onClick={() => void cancel()} disabled={busy || job?.queue?.cancel_requested}>{job?.queue?.cancel_requested ? tr('Stopping…', '正在停止…') : tr('Stop', '停止')}</button> : job?.resumable ? <button className="rr-primary" disabled={busy} onClick={() => void resume()}>{tr('Continue research', '继续研究')}</button> : job ? <button className="rr-primary" disabled={busy} onClick={() => window.dispatchEvent(new Event(OPEN_NEW_CHAT))}>{tr('New Chat', '新对话')}</button> : <button className="rr-primary" disabled={loadingJob || !!selection.current.id || busy || !options || !model || !options.models.find(m => m.id === model)?.ready || (!protein && !sequences[chain]) || !scopes.length} onClick={() => void start()}>{busy ? tr('Preparing…', '准备中…') : tr('Research', '开始研究')} ↑</button>}</div>, actionsTarget)}
    {footerTarget && createPortal(<span className="rr-footer">{job ? `${job.job_number} · ${status(job.status)}${job.status === 'queued' ? ' · ' + tr('Position ', '队列位置 ') + (job.queue?.position ?? '—') : ''}` : label || tr('Choose a protein to get started.', '先选择要研究的蛋白。')}</span>, footerTarget)}
    <div className="rr-panel">
      {loadingJob ? <p role="status">{tr('Loading research…', '正在恢复研究…')}</p> : !job && selection.current.id ? <button type="button" onClick={() => void selectJob(selection.current.id, true)}>{tr('Reload research', '重新加载研究')}</button> : !job ? <div className="rr-start"><div className="rr-intro"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6M10 7v6"/></svg><h1>{tr('Understand your protein', '全面了解你的蛋白')}</h1><p>{tr('An in-depth report, researched and written by specialist agents.', '专题智能体协作检索与撰写，生成深入、全面的蛋白研究报告。')}</p></div>
        <form className="rr-search" onSubmit={e => { e.preventDefault(); void search(); }}><label htmlFor="rr-query">{tr('Protein name, UniProt ID or PDB ID', '蛋白名称、UniProt ID 或 PDB ID')}</label><div><input id="rr-query" placeholder={tr('e.g. P00520, 4HHB or ABL1 mouse', '例如 P00520、4HHB 或 ABL1 mouse')} value={query} maxLength={180} disabled={busy} onChange={e => { lookupRevision.current++; setQuery(e.target.value); setProtein(undefined); setMatches([]); setSearched(false); }}/><button disabled={busy || !query.trim()}>{tr('Find protein', '查找蛋白')}</button></div></form>
        {!label && <CompletedExamples mode="researcher" lang={lang}/>}
        {!!visibleMatches.length && <div className="rr-matches" aria-label={tr('Matching proteins', '匹配的蛋白')}>{visibleMatches.map(m => <button key={proteinId(m)} aria-pressed={proteinId(protein) === proteinId(m)} onClick={() => { setProtein(m); setSequences({}); setStructurePdb(''); }}><span><strong>{m.name}</strong><small>{proteinLabel(m)} · {m.organism} · {m.length} aa{m.reviewed ? ' · Swiss-Prot' : ''}</small></span><b>{proteinId(protein) === proteinId(m) ? '✓' : '＋'}</b></button>)}</div>}
        {searched && !matches.length && <p>{tr('No match found. Try a UniProt or PDB ID, or upload a sequence.', '没有找到匹配。请尝试 UniProt 或 PDB ID，或上传序列。')}</p>}
        {!!Object.keys(sequences).length && <div className="rr-sequence"><label>{fileName}<select aria-label={tr('Protein sequence or chain', '蛋白序列或链')} value={chain} onChange={e => setChain(e.target.value)}>{Object.entries(sequences).map(([id, seq]) => <option key={id} value={id}>{id} · {seq.length} aa</option>)}</select></label><button aria-label={tr('Remove sequence', '移除序列')} onClick={() => { setSequences({}); setStructurePdb(''); setChain(''); }}>×</button></div>}
        <div className="rr-input-links"><button disabled={busy} onClick={() => input.current?.click()}>{tr('Upload FASTA / PDB', '上传 FASTA / PDB')}</button><span>·</span><button onClick={() => setPasteOpen(!pasteOpen)} aria-expanded={pasteOpen}>{tr('Paste a sequence', '粘贴序列')}</button></div>
        {pasteOpen && <div className="rr-paste"><textarea aria-label={tr('Protein sequence', '蛋白序列')} value={paste} maxLength={30000} onChange={e => setPaste(e.target.value)} rows={4} placeholder={'>protein\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL'}/><button disabled={busy || !paste.trim()} onClick={() => void parseInput()}>{tr('Use sequence', '使用序列')}</button></div>}
        {!!materials.length && <section className="rr-materials" aria-label={tr('Attached files', '已上传文件')}><ul>{materials.map(m => <li key={m.id}><span><strong>{m.name}</strong><small>{m.extraction.format.toUpperCase()}{m.extraction.truncated ? tr(' · Partial extract', ' · 部分摘录') : ''}</small></span><button type="button" disabled={busy} aria-label={tr('Remove ', '移除 ') + m.name} onClick={() => setMaterials(old => old.filter(x => x.id !== m.id))}>×</button></li>)}</ul></section>}
        {!!options?.chapters?.length && <details className="rr-chapter-plan" open><summary>{tr('Report outline', '报告大纲')}<span>{options.chapters.length} {tr('chapters', '个专题')}</span></summary><ol>{options.chapters.map(c => <li key={c.id}>{zh ? c.label_zh : c.label}</li>)}</ol></details>}
        <details className="rr-topics"><summary>{tr('Sources & analyses', '检索与分析')}<span>{scopes.length} {tr('selected', '项已选')}</span></summary><div>{options?.analyses.map(s => <label className={!s.available ? 'unavailable' : ''} key={s.id}><input type="checkbox" checked={scopes.includes(s.id)} disabled={!s.available || busy} onChange={e => setScopes(old => e.target.checked ? [...old, s.id] : old.filter(x => x !== s.id))}/><span>{zh ? s.label_zh : s.label}</span>{!s.available && <small>{tr('Local only', '本地运行')}</small>}</label>)}</div></details>
      </div> : <div className="rr-results"><header><span className="rr-kicker">{job.job_number}</span><h1>{job.title}</h1>{job.focus && <p>{job.focus}</p>}<span className={'rr-status ' + job.status}>{status(job.status)}{job.partial && job.status === 'completed' ? tr(' · Some evidence unavailable', ' · 部分证据未获得') : ''}</span></header>
        {job.stop_reason && !running && <p role="status">{({ timeout: tr('The run reached its time limit.', '本次运行已达到时间上限。'), interrupted: tr('The worker was interrupted.', '研究进程已中断。'), unavailable: tr('A research service was temporarily unavailable.', '研究服务暂时不可用。'), incomplete: tr('Some chapters still need work.', '部分章节尚未完成。') }[job.stop_reason] || job.message)}{job.resumable ? tr(' Continue from the saved results.', ' 可从已保存结果继续。') : ''}</p>}
        {!!job.materials?.length && <details className="rr-materials"><summary>{tr('Research materials', '研究材料')} · {job.materials.length}</summary><ul>{job.materials.map(m => <li key={m.id}><a href={m.url}>{m.name}</a><small>{m.extraction.format.toUpperCase()}</small></li>)}</ul></details>}
        <ResearchActivity job={job} zh={zh} status={status} stepLabel={stepLabel}/>
        <PredictionResults rows={job.predictions || []} zh={zh} status={status}/>
        {!!job.chapters?.length && <section className="rr-chapters" aria-label={tr('Chapter progress', '章节进度')}><header><strong>{job.phase === 'reviewing' ? tr('Reviewing the full report', '正在综合审阅') : tr('Specialist chapters', '专题研究')}</strong><span>{job.chapters.filter(c => c.status === 'completed').length} / {job.chapters.length}</span></header><div>{job.chapters.map((c, i) => <details key={c.id} className={'rr-chapter ' + c.status}><summary><b>{String(i + 1).padStart(2, '0')}</b><span>{c.title}</span><small>{status(c.status)}</small></summary><div><p>{c.source_ids.length} {tr('sources', '条来源')}{c.length > 0 ? ` · ${c.length} ${tr('words', '字')}` : ''}</p>{c.draft ? <article className="rr-report" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(marked.parse(c.draft) as string) }}/> : <p>{tr('Evidence and draft will appear here.', '证据与章节草稿将在这里显示。')}</p>}</div></details>)}</div></section>}
        {!!job.remote_services?.length && <section className="rr-remote" aria-label={tr('Public protein analyses', '公共蛋白分析')}><header><strong>{tr('Public protein analyses', '公共蛋白分析')}</strong><span>{job.remote_services.filter(s => ['completed', 'empty'].includes(s.status)).length} / {job.remote_services.length}</span></header>{job.remote_services.map(s => <details key={s.id} className={s.status}><summary><span>{zh ? s.label_zh : s.label}</span><small>{s.count !== undefined ? `${s.count} ${tr('results', '条结果')} · ` : ''}{status(s.status)}</small></summary><div>{s.summary && <p>{s.summary}</p>}{s.job_id && <p>{tr('Job', '任务号')} <code>{s.job_id}</code></p>}<a href={s.url} target="_blank" rel="noreferrer">{tr('Open service result', '打开服务结果')} ↗</a>{s.data !== undefined && <PublicEvidence service={s} zh={zh}/>}</div></details>)}</section>}
        {(job.report || !!job.sources?.length) && <nav className="rr-result-nav"><div><button aria-pressed={view === 'report'} onClick={() => setView('report')}>{tr('Report', '报告')}</button><button aria-pressed={view === 'sources'} onClick={() => setView('sources')}>{tr('Sources', '来源')} {job.sources?.length || 0}</button></div>{job.report && <details className="rr-download"><summary>{tr('Download', '下载')} ↓</summary><div>{['md', 'html', 'json'].map(format => <a key={format} href={`/api/report/jobs/${job.id}/download?format=${format}`}>{format === 'json' ? tr('Evidence JSON', '证据 JSON') : format.toUpperCase()}</a>)}</div></details>}</nav>}
        {view === 'report' ? <>{job.report ? <ResearchReportReader markdown={job.report} identity={job.id} zh={zh} composer={composerTarget}/> : <p className={running ? 'rr-thinking' : ''}>{running ? tr('Collecting evidence and preparing your report…', '正在收集证据并整理报告…') : job.message || tr('No report was generated.', '尚未生成报告。')}</p>}{job.sections?.map(s => <details className="rr-calculation" key={s.id}><summary>{s.title}</summary><pre>{s.text || JSON.stringify(s.data, null, 2)}</pre></details>)}</> : <div className="rr-sources">{job.sources?.map(s => <article key={s.id}><span>{s.id} · {s.evidence_level.replace(/_/g, ' ')}{s.year ? ' · ' + s.year : ''}</span><h3>{s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title} ↗</a> : s.title}</h3><details><summary>{tr('Collected evidence', '已收集的证据')}</summary><pre>{s.text || tr('No abstract available.', '没有摘要。')}</pre></details></article>)}</div>}
      </div>}
      {error && !isDailyQuotaMessage(error) && noticeTarget && createPortal(<ChatNotice message={error} lang={lang} onDismiss={dismissError} />, noticeTarget)}
    </div>
  </>;
});
