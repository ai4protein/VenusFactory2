import { useState } from 'react';
import type { ADMETMetadata } from '../lib/modelPlatform';
import './admet-results.css';

const groups: Record<string,string> = {Absorption:'吸收',Distribution:'分布',Metabolism:'代谢',Excretion:'排泄',Toxicity:'毒性',Physicochemical:'理化性质',Other:'其他'};

export function ADMETResults({rows,metadata,zh}:{rows:Record<string,unknown>[];metadata:ADMETMetadata;zh:boolean}) {
  const [selected,setSelected]=useState(0),[group,setGroup]=useState('Absorption');
  const row=rows[selected] || rows[0];
  if(!row)return null;
  const available=Object.keys(groups).filter(g=>metadata.endpoints.some(e=>e.category===g));
  const active=available.includes(group)?group:available[0];
  const endpoints=metadata.endpoints.filter(e=>e.category===active);
  return <section className="admet-results" aria-label={zh?'ADMET 预测结果':'ADMET predictions'}>
    <div className="mp-section-heading"><strong>ADMET-AI <small>{metadata.version}</small></strong><span>{metadata.predicted} {zh?'个分子已预测':'molecules predicted'}</span></div>
    <div className="admet-filters">
      <label className="mp-field">{zh?'分子':'Molecule'}<select value={selected} onChange={e=>setSelected(Number(e.target.value))}>{rows.map((r,i)=><option key={i} value={i}>{String(r.admet_row || i+1)} · {String(r.name || r.id || r.admet_smiles || r.smiles || '').slice(0,64)}{r.admet_status!=='predicted'?(zh?' · 未预测':' · Not predicted'):''}</option>)}</select></label>
      <label className="mp-field">{zh?'性质分类':'Property group'}<select value={active} onChange={e=>setGroup(e.target.value)}>{available.map(g=><option key={g} value={g}>{zh?groups[g]:g} · {metadata.endpoints.filter(e=>e.category===g).length}</option>)}</select></label>
    </div>
    {row.admet_status!=='predicted' ? <p className="mp-muted">{String(row.admet_error || '')}</p> : <>
      <div className="mp-table-scroll"><table><thead><tr><th>{zh?'指标':'Endpoint'}</th><th>{zh?'预测值':'Prediction'}</th><th>{zh?'单位 / 类型':'Unit / type'}</th></tr></thead><tbody>{endpoints.map(e=><tr key={e.id}><td><a href={e.url || undefined} target="_blank" rel="noreferrer" title={`${e.id} · ${e.species}`}>{e.name}</a></td><td>{typeof row[e.id]==='number'?Number((row[e.id] as number).toPrecision(4)):String(row[e.id]??'—')}</td><td>{e.task_type==='classification'?(zh?'阳性类别概率':'Positive-class probability'):e.units}</td></tr>)}</tbody></table></div>
      <p className="admet-caption">{zh?'分类值为该指标的阳性概率，不是安全评分。':'Classification values are endpoint probabilities, not safety scores.'}</p>
    </>}
    {metadata.invalid>0 && <p className="admet-caption">{metadata.invalid} {zh?'个分子未预测；详情保留在结果文件中。':'molecules were not predicted; details are retained in the result files.'}</p>}
    {metadata.predicted+metadata.invalid>rows.length && <p className="admet-caption">{zh?'此处展示前 20 行，下载包含全部结果。':'Preview shows the first 20 rows; downloads contain all results.'}</p>}
  </section>;
}
