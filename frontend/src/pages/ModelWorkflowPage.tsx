import { WorkflowRunTrace, WorkflowStepDetails, FlowData } from '../components/WorkflowRunTrace';
import { ModelExtraInputs } from '../components/ModelExtraInputs';
import { ModelParameters } from "../components/ModelParameters";
import { JobHistory } from '../components/JobHistory';
import { useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { ComponentCatalog, Dataset, Graph, GraphNode, Job, ModelCard, Workflow, categoryNames, connectGraph, removeGraphNode, platform, post, sourceNames } from '../lib/modelPlatform';
import { DataUpload, JobProgress, ModelGlyph } from '../components/ModelPlatformShared';
import { appendWorkflowStep, workflowIssue, workflowName, workflowSignature } from '../lib/workflowEditor';
import '../model-platform.css';
import '../workflow-studio.css';

const initialGraph: Graph = { nodes: [], edges: [] };
const draftKey = 'venus.model-workflow.draft.v1';
const pageSize = 24;
const types: Record<GraphNode['kind'], [string, string]> = {
  data: ['Input data', '输入数据'], model: ['Model', '模型'], tool: ['Tool', '工具'],
  train: ['Train model', '训练模型'], predict: ['Model prediction', '模型预测'], output: ['Results', '结果'],
};
type Parameter = { type?: string; title?: string; description?: string; default?: unknown; enum?: unknown[]; anyOf?: Parameter[] };

export function ModelWorkflowPage() {
  const { lang } = useLang(); const zh = lang === 'zh'; const tr = (en: string, cn: string) => zh ? cn : en;
  const [params, setParams] = useSearchParams();
  const [componentsOpen, setComponentsOpen] = useState(false);
  const [workflow, setWorkflow] = useState<Workflow>(() => {
    try { const w = JSON.parse(localStorage.getItem(draftKey) || ''); return typeof w.name === 'string' && Array.isArray(w.graph?.nodes) && Array.isArray(w.graph?.edges) ? w : { name: '', graph: initialGraph }; }
    catch { return { name: '', graph: initialGraph }; }
  });
  const [catalog, setCatalog] = useState<ComponentCatalog>(), [datasets, setDatasets] = useState<Dataset[]>([]), [saved, setSaved] = useState<Workflow[]>([]);
  const [query, setQuery] = useState(''), [kind, setKind] = useState(''), [category, setCategory] = useState(''), [source, setSource] = useState(''), [page, setPage] = useState(0), [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(''), [from, setFrom] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState(''), [busy, setBusy] = useState(false), [job, setJob] = useState<Job>();
  const [detail, setDetail] = useState<ModelCard>(), [configText, setConfigText] = useState('{}'), [history, setHistory] = useState<Graph[]>([]), [revision, setRevision] = useState(0);
  const [savedVersion, setSavedVersion] = useState(() => { try { return localStorage.getItem(draftKey + '.saved') || ''; } catch { return ''; } });
  const [loadingWorkflow, setLoadingWorkflow] = useState(Boolean(params.get('workflow')));
  const [restoringJob, setRestoringJob] = useState(Boolean(params.get('job')));
  const [pendingOpen, setPendingOpen] = useState<Workflow>();
  const [selectedEdge, setSelectedEdge] = useState('');
  const [inspectorView, setInspectorView] = useState<'settings'|'run'>('run');
  useEffect(() => { if (job?.id) setInspectorView('run'); }, [job?.id]);
  const requestedWorkflow = useRef('');
  const workflowRef = useRef(workflow); workflowRef.current = workflow;
  const canvas = useRef<HTMLDivElement>(null), drag = useRef<{ id: string; x: number; y: number; startX: number; startY: number }>();
  const requestedModel = useRef(''); const graph = workflow.graph, node = graph.nodes.find(n => n.id === selected);
  useEffect(() => {
    if (node && canvas.current && !drag.current) canvas.current.scrollTo({left: Math.max(0, node.x - (canvas.current.clientWidth - 224) / 2), top: Math.max(0, node.y - 40), behavior: 'smooth'});
  }, [selected, Boolean(job)]);
  const running = busy || loadingWorkflow || restoringJob || Boolean(job && ['running', 'queued'].includes(job.status));
  const dirty = workflowSignature(workflow) !== savedVersion;
  const undoRef = useRef(history); undoRef.current = history;
  function changeGraph(fn: (g: Graph) => Graph, remember = true) {
    if (running) return;
    if (remember) setHistory(h => [...h.slice(-29), graph]);
    setWorkflow(w => ({ ...w, graph: fn(w.graph) })); setNotice(''); setJob(undefined); setSelectedEdge('');
    if (params.has('job')) { const next = new URLSearchParams(params); next.delete('job'); setParams(next, {replace: true}); }
  }
  function undo() { const h = undoRef.current; if (running || !h.length) return; changeGraph(() => h[h.length - 1], false); setHistory(h.slice(0, -1)); setSelected(''); setFrom(''); setNotice(''); }
  function removeNode(id = selected) { if (!id || running) return; changeGraph(g => removeGraphNode(g, id)); setSelected(''); if (from === id) setFrom(''); }
  useEffect(() => { if (loadingWorkflow) return; try { localStorage.setItem(draftKey, JSON.stringify(workflow)); localStorage.setItem(draftKey + '.saved', savedVersion); } catch { /* Explicit server saving remains available. */ } }, [workflow, savedVersion, loadingWorkflow]);
  useEffect(() => {
    void platform<Dataset[]>('/datasets').then(setDatasets).catch(e => setError(e.message));
    void platform<Workflow[]>('/workflows').then(setSaved).catch(e => setError(e.message));
  }, []);
  useEffect(() => {
    if (params.has('job')) { setLoadingWorkflow(false); return; }
    const id = params.get('workflow'); if (!id || requestedWorkflow.current === id) return;
    let alive = true; requestedWorkflow.current = id; setLoadingWorkflow(true);
    void platform<Workflow>('/workflows/' + encodeURIComponent(id)).then(w => {
      if (!alive) return;
      // Preserve a local unsaved edit when refreshing this same saved workflow.
      if (workflowRef.current.id !== id) { setWorkflow(w); setSelected(''); setHistory([]); }
      setSavedVersion(workflowSignature(w)); setError('');
    }).catch(e => { if (alive) setError(e.message); }).finally(() => { if (alive) setLoadingWorkflow(false); });
    return () => { alive = false; if (requestedWorkflow.current === id) requestedWorkflow.current = ''; };
  }, [params.get('workflow'), params.get('job')]);
  useEffect(() => {
    const id = params.get('job'); if (!id) { setRestoringJob(false); return; }
    let alive = true; setRestoringJob(true);
    void platform<Job>('/jobs/' + encodeURIComponent(id)).then(value => { if (alive) { setJob(value); if (value.workflow) { setWorkflow(value.workflow); setSavedVersion(workflowSignature(value.workflow)); setHistory([]); } } }).catch(e => { if (alive) setError(e.message); }).finally(() => { if (alive) setRestoringJob(false); });
    return () => { alive = false; };
  }, [params.get('job')]);
  useEffect(() => {
    let alive = true; setLoading(true);
    const t = setTimeout(() => {
      const qs = new URLSearchParams({ query, category, source, kind, offset: String(page * pageSize), limit: String(pageSize) });
      void platform<ComponentCatalog>('/workflow-components?' + qs).then(c => { if (alive) { setCatalog(c); setLoading(false); } }).catch(e => { if (alive) { setError(e.message); setLoading(false); } });
    }, 180);
    return () => { alive = false; clearTimeout(t); };
  }, [query, category, source, kind, page, revision]);
  useEffect(() => {
    const id = params.get('model'); if (!id || loadingWorkflow || requestedModel.current === id) return; requestedModel.current = id;
    void platform<ModelCard>('/workflow-components/' + encodeURIComponent(id)).then(m => { add(m.node_kind || 'model', m); const next = new URLSearchParams(params); next.delete('model'); setParams(next, {replace: true}); }).catch(e => setError(e.message));
  }, [params, loadingWorkflow]);
  useEffect(() => {
    setDetail(undefined); setConfigText(JSON.stringify(node?.config || {}, null, 2));
    if (!node?.model_id) return;
    let alive = true;
    void platform<ModelCard>('/workflow-components/' + encodeURIComponent(node.model_id)).then(m => { if (alive) setDetail(m); }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [node?.id]);
  function add(type: GraphNode['kind'], model?: ModelCard, point?: { x: number; y: number }) {
    if (running || model?.available === false) return; const id = 'n_' + crypto.randomUUID().replace(/-/g, '');
    const label = model?.name || types[type][zh ? 1 : 0];
    const config = type === 'train' ? { method: 'forest', problem: 'regression', category: 'custom', features: [], target: '', sequence_column: '' } : {};
    const next: GraphNode = { id, kind: type, label, x: point?.x ?? 40, y: point?.y ?? 80 + graph.nodes.filter(n => n.kind === 'data').length * 190, config, ...(model ? { model_id: model.id, input_kinds: model.accepted_input_kinds || [model.input_kind], output_kind: model.output_kind || 'table', standalone: model.standalone } : {}) };
    const added = point ? { ...graph, nodes: [...graph.nodes, next] } : appendWorkflowStep(graph, next, selected, {data: types.data[zh ? 1 : 0], output: types.output[zh ? 1 : 0]}, model?.standalone);
    changeGraph(() => added); setSelected(!graph.nodes.length && added.nodes.some(n => n.id === id + '_data') ? id + '_data' : id);
  }
  function updateNode(patch: Partial<GraphNode>, identity = selected) { changeGraph(g => ({ ...g, nodes: g.nodes.map(n => n.id === identity ? { ...n, ...patch } : n) })); }
  function configure(patch: Record<string, unknown>) { const config = { ...node?.config, ...patch }; updateNode({ config }); setConfigText(JSON.stringify(config, null, 2)); }
  function connect(target: string, port?: string) {
    if (!from || running) return;
    try { const next = connectGraph(graph, from, target, port, datasets, detail ? [...(catalog?.items || []), detail] : catalog?.items); changeGraph(() => next); setError(''); } catch (e) { setError((e as Error).message); }
    setFrom('');
  }
  function open(w: Workflow) { requestedWorkflow.current = w.id || ''; setWorkflow(w); setSavedVersion(w.id ? workflowSignature(w) : ''); setParams(w.id ? { workflow: w.id } : {}, { replace: true }); setSelected(''); setSelectedEdge(''); setFrom(''); setHistory([]); setJob(undefined); setError(''); setNotice(''); setPendingOpen(undefined); }
  function requestOpen(w: Workflow) { if (running) return; if (dirty && (graph.nodes.length || workflow.name.trim())) setPendingOpen(w); else open(w); }
  async function persist() {
    const w = await post<Workflow>('/workflows', { ...workflowRef.current, name: workflowName(workflowRef.current, zh) });
    requestedWorkflow.current = w.id || ''; setWorkflow(w); setSavedVersion(workflowSignature(w));
    setSaved(rows => [w, ...rows.filter(row => row.id !== w.id)]);
    setParams({ workflow: w.id! }, { replace: true }); setNotice(tr('Saved', '已保存')); return w;
  }
  async function save(next?: Workflow) { if (running) return; setBusy(true); setError(''); try { await persist(); if (next) open(next); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }
  async function run() {
    if (running) return;
    const issue = workflowIssue(graph, zh, datasets, detail ? [...(catalog?.items || []), detail] : catalog?.items); if (issue) { setError(issue.message); setSelected(issue.id); return; }
    setBusy(true); setError('');
    try { const w = await persist(); const result = await post<Job>('/workflows/run', { graph: w.graph, name: w.name, workflow_id: w.id }); setJob(result); setParams({workflow: w.id!, job: result.id}, {replace: true}); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  function setInput(sourceId: string, port = 'data') {
    if (!node) return;
    try {
      if (sourceId) changeGraph(g => connectGraph(g, sourceId, node.id, port, datasets, detail ? [detail] : []));
      else changeGraph(g => ({ ...g, edges: g.edges.filter(e => e.target !== node.id || (node.kind === 'predict' && (e.target_port || (g.nodes.find(n => n.id === e.source)?.kind === 'train' ? 'model' : 'data')) !== port)) }));
      setError('');
    } catch (e) { setError((e as Error).message); }
  }
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); if (!running) void save(); } };
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler);
  });
  function point(e: { clientX: number; clientY: number }) { const rect = canvas.current!.getBoundingClientRect(); return { x: Math.max(10, e.clientX - rect.left + canvas.current!.scrollLeft - 100), y: Math.max(10, e.clientY - rect.top + canvas.current!.scrollTop - 40) }; }
  function columnsFor(id: string, seen = new Set<string>()): string[] {
    if (seen.has(id)) return []; seen.add(id); const n = graph.nodes.find(n => n.id === id);
    if (n?.kind === 'data') return datasets.find(d => d.id === n.dataset_id)?.columns || [];
    return [...new Set(graph.edges.filter(e => e.target === id).flatMap(e => columnsFor(e.source, seen)))];
  }
  const width = Math.max(1000, ...graph.nodes.map(n => n.x + 280)), height = Math.max(680, ...graph.nodes.map(n => n.y + 180));
  const columns = node ? columnsFor(node.id) : [];
  const toolProperties = (detail?.contract?.properties || {}) as Record<string, Parameter>;
  const toolParams = (node?.config?.parameters || {}) as Record<string, unknown>;
  const mapping = (node?.config?.columns || {}) as Record<string, string>;
  const updateTool = (key: string, value: unknown) => configure({ parameters: { ...toolParams, [key]: value } });
  return <main className="mp-studio">
    <header className="mp-studio-header"><div className="wf-document"><h1>{tr('Workflow', '工作流')}</h1><input aria-label={tr('Workflow name', '工作流名称')} placeholder={tr('Untitled workflow', '未命名工作流')} maxLength={120} value={workflow.name} disabled={running} onChange={e => { setWorkflow(w => ({ ...w, name: e.target.value })); setNotice(''); }}/><span className="wf-save-state" role="status">{loadingWorkflow ? tr('Opening…', '正在打开…') : busy ? tr('Saving…', '正在保存…') : dirty ? tr('Unsaved changes', '未保存') : notice || tr('Saved', '已保存')}</span></div>
      <div className="mp-studio-controls"><button className="mp-button" disabled={running} onClick={() => requestOpen({ name: '', graph: initialGraph })}>{tr('New', '新建')}</button><select aria-label={tr('Open workflow', '打开工作流')} disabled={running} value={saved.some(w => w.id === workflow.id) ? workflow.id : ''} onChange={e => { const w = saved.find(s => s.id === e.target.value); if (w) requestOpen(w); }}><option value="">{saved.length ? tr('Open saved workflow…', '打开已保存工作流…') : tr('No saved workflows yet', '暂无已保存工作流')}</option>{saved.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}</select><button className="mp-button mp-undo" disabled={running || !history.length} onClick={undo} title={tr('Undo last edit', '撤销上一步')}>↶ {tr('Undo', '撤销')}</button><button className="mp-button" disabled={running} onClick={() => void save()} title={tr('Save workflow · Ctrl / ⌘ S', '保存工作流 · Ctrl / ⌘ S')}>{tr('Save', '保存')}</button><button className="mp-button primary" disabled={running || !graph.nodes.length} onClick={() => void run()}>▷ {tr('Run', '运行')}</button></div>
    </header>
    {pendingOpen && <div className="wf-switch-confirm" role="alertdialog" aria-label={tr('Save your changes?', '保存当前修改？')}><strong>{tr('Save your changes before switching?', '切换前保存当前修改？')}</strong><button className="mp-button primary" disabled={busy} onClick={() => void save(pendingOpen)}>{tr('Save and continue', '保存并继续')}</button><button className="mp-button" disabled={busy} onClick={() => open(pendingOpen)}>{tr('Discard changes', '放弃修改')}</button><button className="mp-button" disabled={busy} onClick={() => setPendingOpen(undefined)}>{tr('Cancel', '取消')}</button></div>}
    <div className="wf-path"><span>1 · {tr('Add data', '添加数据')}</span><span aria-hidden="true">→</span><span>2 · {tr('Choose tools or models', '选择工具或模型')}</span><span aria-hidden="true">→</span><span>3 · {tr('Configure and run', '配置并运行')}</span><JobHistory/></div>
    <div className={`mp-studio-body${node ? ' has-inspector' : ''}`}><button type="button" className="wf-components-toggle" aria-expanded={componentsOpen} aria-controls="workflow-components" onClick={() => setComponentsOpen(value => !value)}>{tr('Components', '组件库')}<span aria-hidden="true">{componentsOpen ? '−' : '＋'}</span></button><aside id="workflow-components" className={`mp-palette${componentsOpen ? ' is-open' : ''}`}>
      <div className="mp-palette-top"><strong>{tr('Add a step', '添加步骤')}</strong></div>
      <div className="wf-primitives">{(['data', 'output', 'train', 'predict'] as const).map(type => <button key={type} draggable disabled={running} onDragStart={e => e.dataTransfer.setData('application/venus-node', type)} onClick={() => add(type)}><ModelGlyph kind={type === 'train' ? 'design' : type === 'predict' ? 'function' : type} size={17}/><span>{types[type][zh ? 1 : 0]}</span><b>＋</b></button>)}</div>
      <div className="wf-library-tabs" role="group" aria-label={tr('Component type', '组件类型')}>{[['', 'All', '全部'], ['model', 'Models', '模型'], ['tool', 'Tools', '工具']].map(([id,en,cn]) => <button key={id} aria-pressed={kind === id} onClick={() => { setKind(id); setPage(0); }}>{tr(en,cn)}</button>)}</div>
      <input aria-label={tr('Search components', '搜索组件')} value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} placeholder={tr('Search models or tools…', '搜索模型或工具…')}/>
      <div className="wf-library-filters"><select aria-label={tr('Component category', '组件分类')} value={category} onChange={e => { setCategory(e.target.value); setPage(0); }}><option value="">{tr('All categories', '全部分类')}</option>{Object.entries(catalog?.categories || {}).sort(([a],[b]) => a.localeCompare(b)).map(([c,count]) => <option key={c} value={c}>{categoryNames[c]?.[zh ? 1 : 0] || c} · {count}</option>)}</select><select aria-label={tr('Component source', '组件来源')} value={source} onChange={e => { setSource(e.target.value); setPage(0); }}><option value="">{tr('All sources', '全部来源')}</option>{Object.entries(sourceNames).map(([s,n]) => <option key={s} value={s}>{n[zh ? 1 : 0]}</option>)}</select></div>
      <div className="wf-library-count">{loading ? tr('Loading…', '加载中…') : `${catalog?.total || 0} ${tr('components', '个组件')}`}</div>
      <div className="mp-palette-models" aria-busy={loading}>{!loading && !catalog?.items.length && <p className="mp-muted">{tr('No matching components', '没有匹配的组件')}</p>}{catalog?.items.map(m => <button key={m.id} draggable={m.available !== false} disabled={running || loading || m.available === false} onDragStart={e => e.dataTransfer.setData('application/venus-model', m.id)} onClick={() => add(m.node_kind || 'model', m)} title={m.description}><ModelGlyph kind={m.node_kind === 'tool' ? 'discovery' : m.category} size={17}/><span>{m.name}{m.available === false ? (m.disabled_reason==="not_configured" ? tr(" · Not configured", " · 未配置") : tr(" · Local only", " · 本地运行")) : ""}<small>{tr(m.node_kind === 'tool' ? 'Tool' : 'Model', m.node_kind === 'tool' ? '工具' : '模型')} · {sourceNames[m.source]?.[zh ? 1 : 0] || m.source} · {m.input_kind === 'structure' ? (m.input_accept || '.pdb').replace('.','').toUpperCase() : tr('Table', '表格')}</small></span><b>＋</b></button>)}</div>
      <div className="wf-library-pages"><button aria-label={tr('Previous components', '上一页组件')} disabled={!page || loading} onClick={() => setPage(p => p - 1)}>←</button><span>{page + 1} / {Math.max(1, Math.ceil((catalog?.total || 0) / pageSize))}</span><button aria-label={tr('Next components', '下一页组件')} disabled={loading || (page + 1) * pageSize >= (catalog?.total || 0)} onClick={() => setPage(p => p + 1)}>→</button></div>
    </aside>
    <div className={`mp-canvas-area${job ? ' has-run' : ''}`}>{error && <div role="alert" className="mp-canvas-alert">{error}<button aria-label={tr('Dismiss error', '关闭错误')} onClick={() => setError('')}>×</button></div>}{from && <div className="mp-connect-hint">{tr('Select the next input · replaces its existing connection', '选择下一步输入 · 将替换该输入已有的连接')}<button aria-label={tr('Cancel connection', '取消连接')} onClick={() => setFrom('')}>×</button></div>}{selectedEdge && <div className="wf-connection-actions"><button className="mp-button" disabled={running} onClick={() => changeGraph(g => ({...g, edges: g.edges.filter(e => e.source + '→' + e.target !== selectedEdge)}))}>{tr('Disconnect', '断开连接')}</button><button className="mp-button" onClick={() => setSelectedEdge('')}>{tr('Cancel', '取消')}</button></div>}
      <div ref={canvas} className="mp-canvas" tabIndex={0} aria-label={tr('Workflow canvas', '工作流画布')} onKeyDown={e => {
        if ((e.target as HTMLElement).closest('input,textarea,select,button,[contenteditable=true]')) return;
        if (['Delete','Backspace'].includes(e.key) && selected && !selectedEdge) { e.preventDefault(); removeNode(); }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
        if (e.key === 'Escape') { setSelected(''); setSelectedEdge(''); setFrom(''); }
      }} onDragOver={e => e.preventDefault()} onDrop={e => {
        e.preventDefault(); if (running) return;
        const type = e.dataTransfer.getData('application/venus-node'), id = e.dataTransfer.getData('application/venus-model'), p = point(e);
        if (['data','output','train','predict'].includes(type)) add(type as GraphNode['kind'], undefined, p);
        else if (id) void platform<ModelCard>('/workflow-components/' + encodeURIComponent(id)).then(m => add(m.node_kind || 'model', m, p)).catch(err => setError(err.message));
      }} onPointerMove={e => {
        if (!drag.current || running) return; const d = drag.current;
        changeGraph(g => ({ ...g, nodes: g.nodes.map(n => n.id === d.id ? { ...n, x: Math.max(10, d.startX + e.clientX - d.x), y: Math.max(10, d.startY + e.clientY - d.y) } : n) }), false);
      }} onPointerUp={e => {
        drag.current = undefined;
        if (from) { const target = document.elementFromPoint(e.clientX, e.clientY)?.closest<HTMLElement>('[data-input-port]'); if (target?.dataset.inputPort) connect(target.dataset.inputPort, target.dataset.portKind); }
      }} onPointerCancel={() => { drag.current = undefined; setFrom(''); }}>
        <div className="mp-canvas-inner" style={{ width, height }} onClick={e => { if (e.target === e.currentTarget) { setSelected(''); setSelectedEdge(''); setFrom(''); } }}>
          <svg className="mp-edges" width={width} height={height} aria-label={tr('Workflow connections', '工作流连接')}>{graph.edges.map(edge => {
            const a = graph.nodes.find(n => n.id === edge.source), b = graph.nodes.find(n => n.id === edge.target); if (!a || !b) return null;
            const flowStatus = job?.steps.find(s => s.id === a.id)?.status === 'completed' ? job?.steps.find(s => s.id === b.id)?.status === 'running' ? 'flow-active' : job?.steps.find(s => s.id === b.id)?.started ? 'flow-complete' : '' : '';
            const modelPort = b.kind === 'predict' && a.kind === 'train', x = a.x + 224, y = a.y + 63, endX = b.x, endY = b.y + (modelPort ? 101 : 63);
            const path = `M${x},${y} C${x + 75},${y} ${endX - 75},${endY} ${endX},${endY}`;
            const remove = () => changeGraph(g => ({ ...g, edges: g.edges.filter(e => e !== edge) }));
            return <g key={edge.source + edge.target} role="button" tabIndex={0} aria-label={tr('Connection: ', '连接：') + a.label + ' → ' + b.label} onClick={() => { setSelectedEdge(edge.source + '→' + edge.target); setSelected(''); }} onKeyDown={e => { if (['Delete','Backspace'].includes(e.key)) { e.preventDefault(); e.stopPropagation(); remove(); } else if (e.key === 'Enter') { e.preventDefault(); setSelectedEdge(edge.source + '→' + edge.target); setSelected(''); } }}><path className="mp-edge-hit" d={path}/><path className={[flowStatus, modelPort ? 'wf-model-edge' : '', selectedEdge === edge.source + '→' + edge.target ? 'wf-selected-edge' : ''].join(' ')} d={path}/></g>;
          })}</svg>
          {!graph.nodes.length && <div className="mp-canvas-empty"><ModelGlyph kind="workflow" size={36}/><h2>{tr('Start with your data', '从你的数据开始')}</h2><p>{tr('Add a tool or model next. Steps connect automatically; click a card to configure it.', '再选择工具或模型，步骤会自动连接。点击卡片即可配置。')}</p><button className="mp-button" onClick={() => add('data')}>＋ {tr('Add input data', '添加输入数据')}</button></div>}
          {graph.nodes.map(n => {
            const step = job?.steps.find(s => s.id === n.id);
            return <div key={n.id} role="group" tabIndex={0} aria-label={tr('Step: ', '步骤：') + n.label} className={`mp-flow-node wf-kind-${n.kind} ${selected === n.id ? 'selected' : ''} ${step?.status || ''}`} style={{ left: n.x, top: n.y }} onClick={e => { e.currentTarget.focus(); setSelected(n.id); setSelectedEdge(''); }} onFocus={e => { if (e.target === e.currentTarget) setSelected(n.id); }}>
              <div className="mp-node-drag" onPointerDown={e => {
                if (running || e.button !== 0 || (e.target as HTMLElement).closest('button')) return;
                e.currentTarget.setPointerCapture(e.pointerId); setHistory(h => [...h.slice(-29), graph]); drag.current = { id: n.id, x: e.clientX, y: e.clientY, startX: n.x, startY: n.y }; setSelected(n.id);
              }}><ModelGlyph kind={n.kind === 'train' ? 'design' : n.kind === 'tool' ? 'discovery' : n.kind} size={17}/><span>{types[n.kind]?.[zh ? 1 : 0]}</span><span className="mp-drag-dots">⠿</span><button className="wf-delete-node" title={tr('Remove step', '删除步骤')} aria-label={tr('Remove step: ', '删除步骤：') + n.label} disabled={running} onPointerDown={e => e.stopPropagation()} onClick={e => { e.stopPropagation(); removeNode(n.id); }}>×</button></div>
              <strong title={n.label}>{n.label}</strong><small>{step ? `${({queued: tr('Waiting', '等待上游'), running: tr('Running', '处理中'), completed: tr('Completed', '已完成'), failed: tr('Stopped', '已停止'), skipped: tr('Not run', '未执行')} as Record<string, string>)[step.status] || step.status}${step.output?.rows !== undefined ? ' · ' + step.output.rows + tr(' rows', ' 行') : step.progress ? ' · ' + step.progress + '%' : ''}` : n.kind === 'data' ? datasets.find(d => d.id === n.dataset_id)?.name || tr('Choose data', '选择数据') : n.kind === 'train' ? tr('Train → validate → save', '训练 → 验证 → 保存') : n.kind === 'output' ? tr('Preview & export', '预览与导出') : tr('Click to configure', '点击配置')}</small>
              {n.kind !== 'data' && <button className="mp-port input" data-input-port={n.id} data-port-kind="data" aria-label={tr('Connect input: ', '连接输入：') + n.label} title={tr('Data input', '数据输入')} disabled={running} onClick={e => { e.stopPropagation(); connect(n.id, 'data'); }}/ >}
              {n.kind === 'predict' && <><button className="mp-port input wf-model-port" data-input-port={n.id} data-port-kind="model" aria-label={tr('Connect model input: ', '连接模型输入：') + n.label} title={tr('Trained model input', '训练模型输入')} disabled={running} onClick={e => { e.stopPropagation(); connect(n.id, 'model'); }}/><span className="wf-model-port-label">{tr('Model', '模型')}</span></>}
              {n.kind !== 'output' && <button className={`mp-port output ${from === n.id ? 'selected' : ''}`} aria-label={tr('Connect output: ', '连接输出：') + n.label} disabled={running} onPointerDown={e => { e.stopPropagation(); setFrom(n.id); }} onClick={e => { e.stopPropagation(); setFrom(n.id); }}/ >}
            </div>;
          })}
        </div>
      </div>
      <footer className="mp-canvas-footer"><span>{graph.nodes.length} {tr('steps', '个步骤')} · {graph.edges.length} {tr('connections', '条连接')}</span><span>{tr('Delete / Backspace to remove · Ctrl / ⌘ Z to undo', 'Delete / Backspace 删除 · Ctrl / ⌘ Z 撤销')}</span></footer>
      {job && <div className="mp-workflow-result"><WorkflowRunTrace job={job} graph={graph} selected={selected} onSelect={setSelected} zh={zh}/><JobProgress initial={job} onChange={j => { setJob(j); if (j.id !== job.id) { const next = new URLSearchParams(params); next.set('job', j.id); setParams(next, {replace:true}); } if (j.status === 'completed' && job.status !== 'completed') setRevision(r => r + 1); }}/></div>}
    </div>
    <aside className="mp-node-inspector" hidden={!node}>{node ? <>
      <div className="mp-palette-top"><strong>{tr('Step settings', '步骤设置')}</strong><button aria-label={tr('Close step settings', '关闭步骤设置')} onClick={() => setSelected('')}>×</button></div>
      {job?.steps.some(s => s.id === node.id) && <><div className="wf-inspector-tabs" role="group" aria-label={tr('Step view', '步骤视图')}><button aria-pressed={inspectorView === 'run'} onClick={()=>setInspectorView('run')}>{tr('Run details', '运行详情')}</button><button aria-pressed={inspectorView === 'settings'} onClick={()=>setInspectorView('settings')}>{tr('Settings', '设置')}</button></div>{inspectorView === 'run' && <WorkflowStepDetails step={job.steps.find(s=>s.id===node.id)} events={job.events || []} graph={graph} zh={zh}/>}</>}
      <div hidden={Boolean(job?.steps.some(s=>s.id===node.id) && inspectorView === 'run')}>
      <label className="mp-field">{tr('Name', '名称')}<input value={node.label} disabled={running} onChange={e => updateNode({ label: e.target.value })}/></label>
      {node.kind !== 'data' && (node.kind === 'predict' ? ['data','model'] : ['data']).map(port => {
        const current = graph.edges.find(e => e.target === node.id && (node.kind !== 'predict' || (e.target_port || (graph.nodes.find(n => n.id === e.source)?.kind === 'train' ? 'model' : 'data')) === port));
        const options = graph.nodes.filter(n => { try { connectGraph(graph, n.id, node.id, port, datasets, detail ? [detail] : []); return true; } catch { return false; } });
        return <label className="mp-field" key={port}>{port === 'model' ? tr('Trained model from', '训练模型来源') : tr('Input from', '输入来自')}<select value={current?.source || ''} disabled={running} onChange={e => setInput(e.target.value, port)}><option value="">{tr('Choose an input step', '选择输入步骤')}</option>{options.map(n => <option key={n.id} value={n.id}>{n.label}</option>)}</select></label>;
      })}
      {node.kind === 'data' && <><p className="wf-input-help">{tr('Tables contain one sample per row. FASTA becomes id + sequence columns. Structures and ligands are passed as files.', '表格每行是一条样本；FASTA 转为 id 和 sequence 两列。结构和小分子以文件传递。')}</p><label className="mp-field">{tr('Dataset', '数据集')}<select value={node.dataset_id || ''} disabled={running} onChange={e => updateNode({ dataset_id: e.target.value })}><option value="">{tr('Choose data', '选择数据')}</option>{datasets.map(d => <option value={d.id} key={d.id}>{d.name} · {d.kind === 'table' ? d.count + tr(' rows', ' 行') : d.kind === 'structure' ? tr('Structure', '结构') : tr('Ligand', '小分子')}</option>)}</select></label>{!running && <DataUpload compact onUpload={d => { setDatasets(v => [d, ...v]); updateNode({ dataset_id: d.id }, node.id); }}/>}<FlowData value={datasets.find(d => d.id === node.dataset_id)} zh={zh}/></>}
      {node.kind === 'train' && <TrainingSettings key={node.id} config={node.config || {}} columns={columns} disabled={running} zh={zh} onChange={configure}/>}
      {node.kind === 'predict' && <p className="mp-muted">{tr('Connect prediction data to the upper input and a training step to the lower model input.', '上方输入连接预测数据，下方模型输入连接训练步骤。')}</p>}
      {['model','tool'].includes(node.kind) && <><p className="mp-muted">{detail?.description}</p>{detail && <div className="mp-tags"><span>{sourceNames[detail.source]?.[zh ? 1 : 0] || detail.source}</span><span>{detail.standalone ? tr('Search query', '检索条件') : detail.input_format ? detail.input_format : detail.input_kind === 'structure' ? 'PDB' : tr('Table / sequence', '表格 / 序列')}</span></div>}{detail?.features && <p>{tr('Features', '特征')}: {detail.features.join(', ')}</p>}</>}
      {detail?.extra_inputs && <ModelExtraInputs model={detail} datasets={datasets} config={node.config || {}} onChange={configure} onUpload={d=>setDatasets(ds=>[d,...ds])} disabled={running} zh={zh}/>}
      {detail?.parameter_schema && <ModelParameters key={node.id} schema={detail.parameter_schema} values={toolParams} onChange={parameters=>configure({parameters})} disabled={running} zh={zh}/>}
      {node.kind === 'tool' && detail?.adapter === 'registered-tool' && Object.entries(toolProperties).map(([key,schema]) => {
        const type = schema.type || schema.anyOf?.find(v => v.type !== 'null')?.type;
        const labels: Record<string, [string,string]> = { query: ['Search query','检索条件'], max_results: ['Number of results','结果数量'], max_content_length: ['Abstract length','摘要长度'], days: ['Recent days','最近天数'], chain_id: ['Chain ID','链 ID'] };
        return <div className="wf-tool-field" key={key}><label className="mp-field">{labels[key]?.[zh ? 1 : 0] || schema.title || key}{(detail.contract?.required as string[] | undefined)?.includes(key) ? ' *' : ''}<input type={type === 'integer' || type === 'number' ? 'number' : 'text'} disabled={running || Boolean(mapping[key])} value={String(toolParams[key] ?? schema.default ?? '')} placeholder={schema.description} onChange={e => updateTool(key, type === 'integer' || type === 'number' ? (e.target.value === '' ? undefined : Number(e.target.value)) : e.target.value)}/></label><select aria-label={tr('Input column for ', '输入列：') + key} disabled={running} value={mapping[key] || ''} onChange={e => { const next = { ...mapping }; if (e.target.value) next[key] = e.target.value; else delete next[key]; configure({ columns: next }); }}><option value="">{tr('Use value above', '使用上方参数值')}</option>{columns.map(c => <option key={c} value={c}>{tr('Column: ', '数据列：')}{c}</option>)}</select></div>;
      })}
      {['model','tool','predict'].includes(node.kind) && <>
        {detail?.adapter !== 'external:admet_ai' && detail?.input_kind !== 'structure' && !detail?.standalone && node.kind !== 'predict' && <label className="mp-field">{tr('Sequence column', '序列列名')}<input disabled={running} value={String(node.config?.sequence_column || 'sequence')} onChange={e => configure({ sequence_column: e.target.value })}/></label>}
        {(node.kind === 'predict' || detail?.adapter === 'personal') && <label className="mp-field">{tr('Output column (optional)', '输出列名（可选）')}<input disabled={running} value={String(node.config?.output_column || '')} onChange={e => configure({ output_column: e.target.value })}/></label>}
        {detail?.contract && <details><summary>{tr('Input format', '输入格式')}</summary><pre className="mp-contract">{JSON.stringify(detail.contract, null, 2)}</pre></details>}
        <details><summary>{tr('Parameters & column mapping', '参数与字段映射')}</summary><textarea className="mp-config" aria-label={tr('Step parameters JSON', '步骤参数 JSON')} rows={8} disabled={running} value={configText} onChange={e => setConfigText(e.target.value)}/><button className="mp-button" disabled={running} onClick={() => { try { const config = JSON.parse(configText); if (!config || Array.isArray(config) || typeof config !== 'object') throw new Error(tr('Parameters must be an object', '参数必须是对象')); updateNode({ config }); setError(''); } catch (e) { setError((e as Error).message); } }}>{tr('Apply parameters', '应用参数')}</button></details>
      </>}
      <button className="mp-remove" disabled={running} onClick={() => removeNode()}>{tr('Remove step', '删除步骤')}</button>
      </div>
    </> : <div className="mp-inspector-empty"><ModelGlyph kind="workflow"/><p>{tr('Select a step to configure its inputs and parameters.', '选择步骤，配置输入与参数。')}</p></div>}</aside>
    </div>
  </main>;
}

function TrainingSettings({ config, columns, disabled, zh, onChange }: { config: Record<string, unknown>; columns: string[]; disabled: boolean; zh: boolean; onChange: (patch: Record<string, unknown>) => void }) {
  const tr = (en: string, cn: string) => zh ? cn : en;
  const [features, setFeatures] = useState((config.features as string[] || []).join(', '));
  return <fieldset className="wf-training-fields" disabled={disabled}>
    <label className="mp-field">{tr('Training method', '训练方法')}<select value={String(config.method || 'forest')} onChange={e => onChange({ method: e.target.value })}><option value="forest">{tr('Random forest', '随机森林')}</option><option value="extra_trees">{tr('Extra trees', '极端随机树')}</option><option value="linear">{tr('Linear / logistic regression', '线性 / 逻辑回归')}</option></select></label>
    <label className="mp-field">{tr('Prediction type', '预测类型')}<select value={String(config.problem || 'regression')} onChange={e => onChange({ problem: e.target.value })}><option value="regression">{tr('Regression', '回归')}</option><option value="classification">{tr('Classification', '分类')}</option></select></label>
    <datalist id="wf-input-columns">{columns.map(c => <option key={c} value={c}/>)}</datalist>
    <label className="mp-field">{tr('Target column', '目标列')}<input list="wf-input-columns" value={String(config.target || '')} onChange={e => onChange({ target: e.target.value })}/></label>
    <label className="mp-field">{tr('Numeric feature columns', '数值特征列')}<input value={features} placeholder="feature_a, feature_b" onChange={e => { setFeatures(e.target.value); onChange({ features: e.target.value.split(/[,，]/).map(v => v.trim()).filter(Boolean) }); }}/></label>
    <label className="mp-field">{tr('Sequence column (optional)', '序列列（可选）')}<input list="wf-input-columns" value={String(config.sequence_column || '')} onChange={e => onChange({ sequence_column: e.target.value })}/></label>
    <label className="mp-field">{tr('Validation group column (optional)', '验证分组列（可选）')}<input list="wf-input-columns" value={String(config.group_column || '')} onChange={e => onChange({ group_column: e.target.value })}/></label>
    <label className="mp-field">{tr('Model category', '模型分类')}<select value={String(config.category || 'custom')} onChange={e => onChange({ category: e.target.value })}>{Object.entries(categoryNames).filter(([c]) => !['foundation','search','processing'].includes(c)).map(([c,n]) => <option key={c} value={c}>{n[zh ? 1 : 0]}</option>)}</select></label>
    <p className="mp-muted">{tr('25% validation split. The trained model is saved to My models.', '25% 数据用于验证，训练完成后保存到我的模型。')}</p>
  </fieldset>;
}
