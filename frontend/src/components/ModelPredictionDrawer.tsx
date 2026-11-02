import { unavailableLabel } from '../lib/capabilityAvailability';
import { ModelExtraInputs } from './ModelExtraInputs';
import { ModelParameters } from "./ModelParameters";
import { modelWorkspaceRoute } from '../lib/modelPlazaNavigation';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { Dataset, Job, ModelCard, platform, post, sourceNames, modelTypeNames, uploadDataset } from '../lib/modelPlatform';
import { parseProteinInput, sequenceColumn } from '../lib/biologicalInputs';
import { DataUpload, JobProgress, ModelGlyph, ResultTable } from './ModelPlatformShared';

export function ModelPredictionDrawer({ model, onClose, onUpdate }: { model: ModelCard; onClose: () => void; onUpdate: () => void }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [detail, setDetail] = useState(model), [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState(''), [sequence, setSequence] = useState(''), [mapping, setMapping] = useState('{}');
  const [mode, setMode] = useState('sequence'), [text, setText] = useState('');
  const [job, setJob] = useState<Job>(), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [ready, setReady] = useState(false);
  const [extraInputs,setExtraInputs]=useState<Record<string,unknown>>({});
  const [parameters, setParameters] = useState<Record<string,unknown>>({});
  useEffect(() => {
    let alive = true;
    void platform<ModelCard>('/models/' + encodeURIComponent(model.id)).then(m => { if (alive) { setDetail(m); setReady(true); } }).catch(e => { if (alive) setError(e.message); });
    void platform<Dataset[]>('/datasets').then(d => { if (alive) setDatasets(d); }).catch(e => { if (alive) setError(e.message); });
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', key);
    return () => { alive = false; document.removeEventListener('keydown', key); };
  }, [model.id]);
  const molecular = detail.adapter === 'external:admet_ai';
  const parseInput = (value:string) => molecular ? value.split(/\r?\n/).map(v=>v.trim()).filter(Boolean).map((smiles,i)=>({id:String(i+1),smiles})) : parseProteinInput(value, detail.sequence_column || 'sequence', zh);
  const canPaste = ready && detail.input_kind === 'table' && (
    ['external:admet_ai','descriptors', 'function', 'residue', 'evolution', 'checkpoint'].includes(detail.adapter) ||
    (detail.adapter === 'personal' && !!detail.sequence_column && !detail.features?.length));
  const paste = canPaste && mode === 'sequence';
  const selectedData = datasets.find(d => d.id === dataset);
  const needsSequence = !molecular && detail.input_kind === 'table' && !['personal', 'scientist', 'capability'].includes(detail.adapter);
  const requiredColumns = detail.adapter === 'personal' ? [...(detail.features || []), ...(detail.sequence_column ? [detail.sequence_column] : [])] : [];
  const missing = selectedData ? requiredColumns.filter(c => !selectedData.columns.includes(c)) : [];
  const active = Boolean(job && ['running', 'queued'].includes(job.status));
  let parsed: Record<string, string>[] = [], inputError = '';
  if (paste && text.trim()) { try { parsed = parseInput(text); } catch (e) { inputError = (e as Error).message; } }
  function choose(d: Dataset) { if(molecular && d.kind==='table')setParameters(v=>({...v,smiles_column:d.columns.find(c=>c.toLowerCase()==='smiles') || d.columns[0]})); setDataset(d.id); setSequence(sequenceColumn(d)); setError(''); }
  async function run() {
    setBusy(true); setError('');
    try {
      const columns = JSON.parse(mapping);
      let id = dataset;
      setInputColumns(selectedData?.columns || []);
      if (paste) {
        const rows = parseInput(text);
        const d = await uploadDataset(new File([JSON.stringify(rows)], molecular ? 'molecules.json' : 'protein-sequences.json', { type: 'application/json' }));
        setInputColumns(d.columns);
        setDatasets(ds => [d, ...ds]); choose(d); id = d.id;
      }
      setJob(await post<Job>('/predict', { model_id: model.id, dataset_id: id, config: { sequence_column: paste ? (detail.sequence_column || 'sequence') : sequence, columns, parameters: molecular && paste ? {...parameters,smiles_column:'smiles'} : parameters, ...extraInputs } }));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function publish() {
    setBusy(true);
    try { setDetail(await platform<ModelCard>('/models/' + encodeURIComponent(model.id) + '/visibility', { method: 'PATCH', body: JSON.stringify({ visibility: detail.visibility === 'public' ? 'private' : 'public' }) })); onUpdate(); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  const extraValid=(detail.extra_inputs || []).every(i=>!i.required || !!extraInputs[i.key]);
  const valid = extraValid && (paste ? parsed.length > 0 : !!selectedData && missing.length === 0 && (!needsSequence || !!sequence));
  return <div className="mp-overlay" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
    <section className="mp-drawer" role="dialog" aria-modal="true" aria-labelledby="model-detail-title">
      <header><span className="mp-eyebrow">{sourceNames[model.source]?.[zh ? 1 : 0]}</span><button autoFocus className="mp-close" aria-label={zh ? '关闭' : 'Close'} onClick={onClose}>×</button></header>
      <div className="mp-detail-heading"><ModelGlyph kind={model.category} size={24}/><h2 id="model-detail-title">{model.name}</h2></div>
      <p className="mp-muted">{model.description}</p>
      {detail.model_type && <dl className="mp-model-lineage">
        <div><dt>{zh ? '模型类型' : 'Model type'}</dt><dd>{modelTypeNames[detail.model_type]?.[zh ? 1 : 0]}</dd></div>
        {detail.backbone?.name && <div><dt>{zh ? '基础模型' : 'Backbone'}</dt><dd>{detail.backbone.name}</dd></div>}
        {detail.training_dataset && <div><dt>{zh ? '训练数据' : 'Training data'}</dt><dd>{detail.training_dataset}</dd></div>}
        {detail.training_method && <div><dt>{zh ? '训练方法' : 'Training method'}</dt><dd>{detail.training_method}</dd></div>}
      </dl>}
      {detail.target && <p><strong>{zh ? '预测性质' : 'Predicts'}:</strong> {detail.target}</p>}
      {detail.available === false ? <button className="mp-button" disabled>{unavailableLabel(detail,zh)}</button> : !model.workflow ? <Link className="mp-button primary" to={`/${lang}${modelWorkspaceRoute(model)}`}>{zh ? '打开模型工作台' : 'Open model workspace'} ↗</Link> : <>
        <h3>{detail.extra_inputs?.length ? (zh?'蛋白受体':'Protein receptor') : (zh ? '输入数据' : 'Input data')}</h3>
        <fieldset className="mp-input-fields" disabled={busy || active || !ready}>
          {canPaste && <div className="mp-tabs mp-input-tabs"><button type="button" aria-pressed={paste} className={paste ? 'active' : ''} onClick={() => setMode('sequence')}>{molecular ? (zh?'粘贴 SMILES':'Paste SMILES') : (zh ? '粘贴序列' : 'Paste sequence')}</button><button type="button" aria-pressed={!paste} className={!paste ? 'active' : ''} onClick={() => setMode('file')}>{zh ? '使用文件' : 'Use a file'}</button></div>}
          {paste ? <label className="mp-field">{molecular ? (zh?'每行一个分子 SMILES':'One molecule SMILES per line') : (zh ? '蛋白序列或 FASTA' : 'Protein sequence or FASTA')}<textarea className="mp-sequence-input" value={text} onChange={e => setText(e.target.value)} rows={6} placeholder={molecular ? 'CC(=O)Oc1ccccc1C(=O)O' : '>protein_1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL'}/>{inputError ? <span className="mp-error" role="alert">{inputError}</span> : parsed.length > 0 ? <span className="mp-muted">{parsed.length} {molecular ? (zh?'个分子':'molecules') : (zh ? '条蛋白序列' : 'protein sequences')}</span> : null}</label> : <>
            <DataUpload compact disabled={busy || active || !ready} accept={detail.input_accept || (detail.input_kind === 'structure' ? '.pdb' : canPaste ? '.csv,.tsv,.json,.fasta,.fa,.faa' : '.csv,.tsv,.json')} formats={detail.input_accept ? detail.input_accept.split(',').map(x=>x.replace('.','').toUpperCase()).join(' · ') : detail.input_kind === 'structure' ? 'PDB' : canPaste ? 'CSV · TSV · JSON · FASTA' : 'CSV · TSV · JSON'} onUpload={d => { setDatasets(ds => [d, ...ds]); choose(d); }}/>
            {!!datasets.filter(d => (detail.accepted_input_kinds || [model.input_kind]).includes(d.kind) && (!detail.input_accept || detail.input_accept.split(',').some(ext=>d.name.toLowerCase().endsWith(ext)))).length && <label className="mp-field">{zh ? '或选择已上传数据' : 'Or choose uploaded data'}<select value={dataset} onChange={e => { const d = datasets.find(x => x.id === e.target.value); if (d) choose(d); else setDataset(''); }}><option value="">{zh ? '选择数据' : 'Choose data'}</option>{datasets.filter(d => (detail.accepted_input_kinds || [model.input_kind]).includes(d.kind) && (!detail.input_accept || detail.input_accept.split(',').some(ext=>d.name.toLowerCase().endsWith(ext)))).map(d => <option key={d.id} value={d.id}>{d.name}{d.kind==='table' ? ` · ${d.count}` : ''}</option>)}</select></label>}
            {selectedData && <><p className="mp-muted">{selectedData.name}{selectedData.kind==='table' ? ` · ${selectedData.count} ${zh?'行':'rows'}` : ''}</p>{molecular && selectedData.kind==='table' && <label className="mp-field">{zh?'SMILES 所在列':'SMILES column'}<select value={String(parameters.smiles_column || 'smiles')} onChange={e=>setParameters(v=>({...v,smiles_column:e.target.value}))}>{selectedData.columns.map(c=><option key={c}>{c}</option>)}</select></label>}{needsSequence && <label className="mp-field">{zh ? '蛋白序列所在列' : 'Protein sequence column'}<select value={sequence} onChange={e => setSequence(e.target.value)}><option value="">{zh ? '选择序列列' : 'Select sequence column'}</option>{selectedData.columns.map(c => <option key={c}>{c}</option>)}</select></label>}<details className="mp-details"><summary>{zh ? '预览数据' : 'Preview data'}</summary><ResultTable rows={selectedData.preview}/></details></>}
            {missing.length > 0 && <p className="mp-error" role="alert">{zh ? '数据中缺少这些输入列：' : 'Required columns missing: '}{missing.join(', ')}</p>}
            {!selectedData && requiredColumns.length > 0 && <p className="mp-muted">{zh ? '需要包含：' : 'Include columns: '}{requiredColumns.join(', ')}</p>}
            {['scientist', 'capability'].includes(model.adapter) && <details className="mp-details"><summary>{zh ? '匹配输入列' : 'Match input columns'}</summary><p className="mp-muted">{zh ? '同名列自动匹配；名称不同时可在此指定。' : 'Matching names are used automatically. Map different names here.'}</p><label className="mp-field">{zh ? '参数名 → 数据列名' : 'Input → column'}<textarea value={mapping} onChange={e => setMapping(e.target.value)} rows={3}/></label></details>}
          </>}
          <ModelExtraInputs model={detail} datasets={datasets} config={extraInputs} onChange={setExtraInputs} onUpload={d=>setDatasets(ds=>[d,...ds])} disabled={busy || active || !ready} zh={zh}/>
          <ModelParameters key={model.id} schema={molecular && detail.parameter_schema ? {...detail.parameter_schema,properties:Object.fromEntries(Object.entries(detail.parameter_schema.properties || {}).filter(([k])=>k!=='smiles_column'))} : detail.parameter_schema} values={parameters} onChange={setParameters} disabled={busy || active || !ready} zh={zh}/>
        </fieldset>
        <div className="mp-actions"><button className="mp-button primary" disabled={!ready || !valid || busy || active} onClick={() => void run()}>{job?.status === 'queued' ? (zh ? '排队中…' : 'Queued…') : active ? (zh ? '正在运行…' : 'Running…') : (zh ? '开始预测' : 'Run prediction')}</button><Link className="mp-button" to={`/${lang}/workflows?model=${encodeURIComponent(model.id)}`}>{zh ? '加入工作流' : 'Add to workflow'}</Link></div>
      </>}
      {error && <p className="mp-error" role="alert">{error}</p>}
      {job && <JobProgress initial={job} onChange={setJob} inputColumns={inputColumns}/>}
      {(detail.contract || detail.metrics) && <details className="mp-details"><summary>{zh ? '模型信息与验证结果' : 'Model details & validation'}</summary>{detail.metrics && <div className="mp-metrics">{Object.entries(detail.metrics).map(([k,v]) => <div key={k}><span>{k}</span><strong>{v.toFixed(4)}</strong></div>)}</div>}{detail.contract && <pre className="mp-contract">{JSON.stringify(detail.contract, null, 2)}</pre>}</details>}
      {detail.owned && <div className="mp-visibility"><span>{detail.visibility === 'public' ? (zh ? '已发布到广场' : 'Published in the plaza') : (zh ? '仅自己可见' : 'Private model')}</span><button disabled={busy || active} onClick={() => void publish()}>{detail.visibility === 'public' ? (zh ? '设为私有' : 'Make private') : (zh ? '发布模型' : 'Publish model')}</button></div>}
    </section>
  </div>;
}
