import { unavailableLabel } from '../lib/capabilityAvailability';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { Dataset, Graph, Job, ModelCard, categoryNames, platform, post } from '../lib/modelPlatform';
import { DataUpload, JobProgress, ModelGlyph } from './ModelPlatformShared';

type FieldSchema = { type?: string; anyOf?: { type: string }[]; title?: string; default?: unknown; enum?: string[]; minimum?: number; maximum?: number };
const fieldNames: Record<string, [string, string]> = {
  max_content_length: ['Content length limit', '内容长度上限'],
  query: ['Search query', '检索内容'], max_results: ['Maximum results', '结果数量'], email: ['Email', '邮箱'],
  include_nonstandard: ['Include nonstandard residues', '包含非标准残基'], chain_id: ['Chain', '链'], structure_chain: ['Protein chain (optional)', '蛋白质链（可选）'],
};

export function RegisteredToolDrawer({ tool, onClose }: { tool: ModelCard; onClose: () => void }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [datasets, setDatasets] = useState<Dataset[]>([]), [dataset, setDataset] = useState('');
  const [parameters, setParameters] = useState<Record<string, unknown>>({}), [sequence, setSequence] = useState('sequence');
  const [job, setJob] = useState<Job>(), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const fields = (tool.contract?.properties || {}) as Record<string, FieldSchema>;
  const required = (tool.contract?.required || []) as string[];
  const active = busy || Boolean(job && ['queued', 'running'].includes(job.status));
  useEffect(() => {
    let alive = true;
    void platform<Dataset[]>('/datasets').then(d => { if (alive) setDatasets(d); }).catch(e => { if (alive) setError(e.message); });
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', key);
    return () => { alive = false; document.removeEventListener('keydown', key); };
  }, [tool.id]);
  async function run() {
    setBusy(true); setError('');
    try {
      const graph: Graph = {
        nodes: [
          ...(dataset ? [{ id: 'input', kind: 'data' as const, label: 'Input', x: 0, y: 0, dataset_id: dataset }] : []),
          { id: 'tool', kind: 'tool', label: tool.name, x: 250, y: 0, model_id: tool.id, config: { parameters, sequence_column: sequence } },
          { id: 'result', kind: 'output', label: 'Result', x: 500, y: 0 },
        ],
        edges: [...(dataset ? [{ source: 'input', target: 'tool' }] : []), { source: 'tool', target: 'result' }],
      };
      setJob(await post<Job>('/workflows/run', { graph }));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  return <div className="mp-overlay" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
    <section className="mp-drawer" role="dialog" aria-modal="true" aria-labelledby="tool-detail-title">
      <header><span className="mp-eyebrow">{categoryNames[tool.category]?.[zh ? 1 : 0]}</span><button autoFocus className="mp-close" aria-label={zh ? '关闭' : 'Close'} onClick={onClose}>×</button></header>
      <span className={'mp-model-mark large ' + tool.category}><ModelGlyph kind={tool.category} size={32}/></span>
      <h2 id="tool-detail-title">{tool.name}</h2><p className="mp-muted">{tool.description.replace(/\.+$/, '.')}</p>
      {tool.available === false ? <p role="status">{unavailableLabel(tool,zh)}</p> : <form onSubmit={e => { e.preventDefault(); void run(); }}>
        {!tool.standalone && <><h3>{zh ? '输入数据' : 'Input data'}</h3>
          <label className="mp-field">{zh ? '数据集' : 'Dataset'}<select required disabled={active} value={dataset} onChange={e => setDataset(e.target.value)}>
            <option value="">{zh ? '选择数据集' : 'Choose a dataset'}</option>
            {datasets.filter(d => d.kind === tool.input_kind).map(d => <option key={d.id} value={d.id}>{d.name} · {d.count}</option>)}
          </select></label><DataUpload compact onUpload={d => {
            if (d.kind !== tool.input_kind) { setError(zh ? '请选择与工具匹配的数据类型。' : 'Choose data that matches this tool.'); return; }
            setDatasets(v => [d, ...v]); setDataset(d.id); setError('');
          }}/>
          {tool.input_kind === 'table' && <label className="mp-field">{zh ? '序列列名' : 'Sequence column'}<input required disabled={active} value={sequence} onChange={e => setSequence(e.target.value)}/></label>}
        </>}
        {Object.entries(fields).map(([key, schema]) => {
          const type = schema.type || schema.anyOf?.find(s => s.type !== 'null')?.type;
          const value = parameters[key] ?? schema.default ?? '';
          const update = (v: unknown) => setParameters(p => ({ ...p, [key]: v }));
          return <label className="mp-field" key={key}>{fieldNames[key]?.[zh ? 1 : 0] || schema.title || key}{required.includes(key) ? ' *' : ''}
            {schema.enum ? <select disabled={active} value={String(value)} onChange={e => update(e.target.value)}>{schema.enum.map(v => <option key={v} value={v}>{v}</option>)}</select>
              : type === 'boolean' ? <select disabled={active} value={String(Boolean(value))} onChange={e => update(e.target.value === 'true')}><option value="false">{zh ? '否' : 'No'}</option><option value="true">{zh ? '是' : 'Yes'}</option></select>
              : <input required={required.includes(key)} disabled={active} type={type === 'integer' || type === 'number' ? 'number' : 'text'} step={type === 'number' ? 'any' : undefined} min={schema.minimum} max={schema.maximum} value={String(value)} onChange={e => update(type === 'integer' || type === 'number' ? (e.target.value === '' ? undefined : Number(e.target.value)) : e.target.value)}/>}
          </label>;
        })}
        <div className="mp-actions"><button type="submit" className="mp-button primary" disabled={active || (!tool.standalone && !dataset)}>{zh ? '运行工具' : 'Run tool'}</button><Link className="mp-button" to={`/${lang}/workflows?model=${encodeURIComponent(tool.id)}`}>{zh ? '加入工作流' : 'Add to workflow'}</Link></div>
      </form>}
      {error && <p role="alert" className="mp-error">{error}</p>}
      {job && <JobProgress initial={job} onChange={setJob}/>}
    </section>
  </div>;
}
