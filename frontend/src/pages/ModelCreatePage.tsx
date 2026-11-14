import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { categories, categoryNames, Dataset, Job, ModelCard, platform, post } from '../lib/modelPlatform';
import { numericColumns, sequenceColumn } from '../lib/biologicalInputs';
import { DataUpload, JobProgress, ResultTable } from '../components/ModelPlatformShared';
import '../model-platform.css';

export function ModelCreatePage() {
  const { lang } = useLang(); const zh = lang === 'zh';
  const say = (en: string, cn: string) => zh ? cn : en;
  const [datasets, setDatasets] = useState<Dataset[]>([]), [data, setData] = useState<Dataset>();
  const [name, setName] = useState(''), [description, setDescription] = useState(''), [target, setTarget] = useState(''), [sequence, setSequence] = useState(''), [group, setGroup] = useState('');
  const [features, setFeatures] = useState<string[]>([]), [problem, setProblem] = useState('regression'), [method, setMethod] = useState('forest'), [category, setCategory] = useState('custom');
  const [job, setJob] = useState<Job>(), [model, setModel] = useState<ModelCard>(), [busy, setBusy] = useState(false), [error, setError] = useState('');
  useEffect(() => { void platform<Dataset[]>('/datasets').then(setDatasets).catch(e => setError(e.message)); }, []);
  function choose(d: Dataset) {
    setData(d); setTarget(''); setName(''); setFeatures([]); setSequence(sequenceColumn(d)); setGroup('');
    setJob(undefined); setModel(undefined); setError('');
  }
  const numbers = data ? numericColumns(data) : [];
  function selectTarget(value: string) {
    if (!name || name === `${target} model`) setName(value ? `${value} model` : '');
    setTarget(value); setFeatures(f => f.filter(x => x !== value));
    if (sequence === value) setSequence(''); if (group === value) setGroup('');
    setProblem(numbers.includes(value) ? 'regression' : 'classification');
  }
  async function train() {
    if (!data) return; setBusy(true); setError(''); setModel(undefined);
    try { setJob(await post<Job>('/train', { dataset_id: data.id, name, description, category, target, sequence_column: sequence, group_column: group, features, problem, method })); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function publish() {
    if (!model) return; setBusy(true);
    try { setModel(await platform<ModelCard>('/models/' + model.id + '/visibility', { method: 'PATCH', body: JSON.stringify({ visibility: model.visibility === 'public' ? 'private' : 'public' }) })); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  const running = Boolean(job && ['queued', 'running'].includes(job.status));
  const methods = [['forest', 'Random forest', '随机森林'], ['extra_trees', 'Extra trees', '极端随机树'], ['linear', 'Linear model', '线性模型']];
  const methodName = methods.find(m => m[0] === method)![zh ? 2 : 1];
  return <div className="mp-page mp-create">
    <header className="mp-page-header"><div><h2>{say('Train on experimental data', '用实验数据训练模型')}</h2><p>{say('Upload measurements and choose what to predict.', '上传实验数据，选择要预测的性质。')}</p></div></header>
    <div className={`mp-create-grid ${!data ? 'mp-create-empty' : ''}`}><div>
      <section className="mp-form-section"><div className="mp-section-title"><span>1</span><h2>{say('Your experimental data', '你的实验数据')}</h2></div>
        <DataUpload disabled={running || busy} accept=".csv,.tsv,.json" formats="CSV · TSV · JSON" onUpload={d => { setDatasets(ds => [d, ...ds]); choose(d); }}/>
        <p className="mp-muted">{say('At least 12 rows, with measured values and protein sequences or numeric inputs.', '至少 12 行，包含实验测量值，以及蛋白序列或数值输入。')}</p>
        {!!datasets.length && <label className="mp-field">{say('Or choose uploaded data', '或选择已上传数据')}<select value={data?.id || ''} disabled={running || busy} onChange={e => { const d = datasets.find(x => x.id === e.target.value); if (d) choose(d); else { setData(undefined); setModel(undefined); setJob(undefined); } }}><option value="">{say('Choose data', '选择数据')}</option>{datasets.filter(d => d.kind === 'table').map(d => <option key={d.id} value={d.id}>{d.name} · {d.count}</option>)}</select></label>}
        {data && <><div className="mp-data-meta"><strong>{data.name}</strong><span>{data.count} {say('rows', '行')} · {data.columns.length} {say('columns', '列')}</span></div><ResultTable rows={data.preview}/>{data.count < 12 && <p role="alert" className="mp-error">{say('Add more measurements: training needs at least 12 rows.', '请补充实验数据：训练至少需要 12 行。')}</p>}</>}
      </section>
      {data && <fieldset className="mp-form-section" disabled={running || busy}><div className="mp-section-title"><span>2</span><h2>{say('What do you want to predict?', '你想预测什么？')}</h2></div>
        <div className="mp-form-grid"><label className="mp-field">{say('Measured property', '实验测量性质')}<select value={target} onChange={e => selectTarget(e.target.value)}><option value="">{say('Choose a column', '选择数据列')}</option>{data.columns.filter(c => c !== sequence).map(c => <option key={c}>{c}</option>)}</select></label>
          <label className="mp-field">{say('Prediction type', '预测类型')}<select value={problem} onChange={e => setProblem(e.target.value)}><option value="regression">{say('Numerical value', '连续数值')}</option><option value="classification">{say('Category or label', '类别标签')}</option></select></label></div>
        <label className="mp-field">{say('Protein sequence column', '蛋白序列所在列')}<select value={sequence} onChange={e => { setSequence(e.target.value); setFeatures(f => f.filter(x => x !== e.target.value)); }}><option value="">{say('Use numeric inputs only', '仅使用数值输入')}</option>{data.columns.filter(c => c !== target && c !== group).map(c => <option key={c}>{c}</option>)}</select></label>
        {numbers.some(c => c !== target && c !== sequence && c !== group) && <><label className="mp-field">{say('Numeric inputs', '数值输入')}{sequence && <span className="mp-muted">{say('Optional', '可选')}</span>}</label><div className="mp-feature-list">{numbers.filter(c => c !== target && c !== sequence && c !== group).map(c => <label key={c}><input type="checkbox" checked={features.includes(c)} onChange={e => setFeatures(f => e.target.checked ? [...f, c] : f.filter(x => x !== c))}/>{c}</label>)}</div></>}
        <label className="mp-field">{say('Model name', '模型名称')}<input value={name} maxLength={120} onChange={e => setName(e.target.value)} placeholder={say('e.g. PETase activity', '例如：PETase 活性')}/></label>
        <details className="mp-details mp-training-options"><summary>{say('Training settings', '训练设置')}<span className="mp-muted">{methodName}</span></summary>
          <label className="mp-field">{say('Algorithm', '算法')}<select value={method} onChange={e => setMethod(e.target.value)}>{methods.map(([id,en,cn]) => <option key={id} value={id}>{zh ? cn : en}</option>)}</select></label>
          <label className="mp-field">{say('Keep related samples together', '将相关样本分在同一组')}<select value={group} onChange={e => { setGroup(e.target.value); setFeatures(f => f.filter(x => x !== e.target.value)); }}><option value="">{sequence ? say('Group by sequence', '按序列分组') : say('Random split', '随机划分')}</option>{data.columns.filter(c => c !== target && c !== sequence).map(c => <option key={c}>{c}</option>)}</select></label>
          <p className="mp-muted">{say('25% of rows are held out for evaluation. With at least five groups, related samples stay together. The saved model is then fitted on all rows.', '留出 25% 的数据评估。分组不少于 5 组时，相关样本保持在同一组；评估后使用全部数据拟合并保存模型。')}</p>
          {sequence && <p className="mp-muted">{say('Sequence inputs use length and amino-acid composition.', '序列输入使用长度与氨基酸组成。')}</p>}
          <label className="mp-field">{say('Category', '功能分类')}<select value={category} onChange={e => setCategory(e.target.value)}>{categories.filter(c => c !== 'foundation').map(c => <option key={c} value={c}>{categoryNames[c][zh ? 1 : 0]}</option>)}</select></label>
          <label className="mp-field">{say('Description', '模型简介')}<textarea value={description} maxLength={600} onChange={e => setDescription(e.target.value)} rows={2} placeholder={say('Protein, property and conditions', '蛋白、预测性质和适用条件')}/></label>
        </details>
        {!sequence && !features.length && target && <p className="mp-muted">{say('Choose a sequence column or at least one numeric input.', '请选择序列列，或至少一项数值输入。')}</p>}
        <button className="mp-button primary" disabled={data.count < 12 || !name.trim() || !target || (!sequence && !features.length) || running || busy} onClick={() => void train()}>{running ? say('Training…', '正在训练…') : say('Train model', '开始训练')} →</button>
      </fieldset>}
      {error && <p role="alert" className="mp-error">{error}</p>}
    </div>{data && <aside className="mp-create-summary"><div className="mp-eyebrow">{say('YOUR MODEL', '你的模型')}</div><h2>{name || say('Choose a property', '选择预测性质')}</h2><dl><dt>{say('Visibility', '可见性')}</dt><dd>{model?.visibility === 'public' ? say('Published', '已发布') : say('Private', '仅自己')}</dd><dt>{say('Predicts', '预测')}</dt><dd>{target || '—'}</dd><dt>{say('Inputs', '输入')}</dt><dd>{[sequence, ...features].filter(Boolean).join(', ') || '—'}</dd><dt>{say('Method', '方法')}</dt><dd>{methodName}</dd></dl>
      {model && <div className="mp-model-ready"><strong>{say('Your model is ready', '模型已保存')}</strong><Link className="mp-button primary" to={`/${lang}/models?model=${encodeURIComponent(model.id)}&view=mine`}>{say('Predict with new data', '预测新数据')} →</Link><button className="mp-button" disabled={busy} onClick={() => void publish()}>{model.visibility === 'public' ? say('Make private', '设为私有') : say('Publish to the plaza', '发布到模型广场')}</button></div>}
      {job && <JobProgress initial={job} onChange={j => { setJob(j); if (j.result?.model) setModel(m => m?.id === j.result?.model?.id ? m : j.result?.model); }}/>}
    </aside>}</div>
  </div>;
}
