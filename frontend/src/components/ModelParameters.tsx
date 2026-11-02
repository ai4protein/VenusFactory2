import { useState } from 'react';
import { ParameterSchema, parseArrayParameter, parameterChoices, updateParameter } from '../lib/modelParameters';

const labels: Record<string,[string,string]> = {
  smiles_column:['SMILES column','SMILES 所在列'],include_physchem:['Include physicochemical properties','包含理化性质'],
  chains_to_design:['Chains to design','设计链'],design_positions:['Design residue positions','设计残基位置'],temperature:['Sampling temperature','采样温度'],number_of_batches:['Sampling batches','采样批次数'],
  fixed_residues:['Fixed residues (A12 A13)','固定残基（A12 A13）'],redesigned_residues:['Only redesign these residues','仅设计这些残基'],
  center:['Box centre X, Y, Z (Å)','对接盒中心 X、Y、Z（Å）'],size:['Box size X, Y, Z (Å)','对接盒尺寸 X、Y、Z（Å）'],
  exhaustiveness:['Search thoroughness','搜索充分程度'],num_modes:['Maximum binding poses','最大构象数量'],energy_range:['Energy window (kcal/mol)','能量窗口（kcal/mol）'],scoring:['Scoring function','评分函数'],
  chains:['Target chains','目标链'],target_hotspot_residues:['Target hotspots','目标结合热点'],min_length:['Minimum binder length','结合蛋白最小长度'],
  target_chain:['Target chain','目标链'],binder_chain:['Binder chain','结合蛋白链'],protocol:['Design protocol','设计流程'],num_designs:['Candidate designs','候选设计数'],budget:['Final designs','最终设计数'],
  number_of_final_designs:['Final designs','最终设计数'],binding_residues:['Target binding positions','目标结合位置'],diffusion_batch_size:['Diffusion batch size','扩散批大小'],
  ligand_smiles:['Ligand SMILES (or upload a file)','配体 SMILES（或上传文件）'],samples_per_complex:['Binding poses to sample','采样构象数'],inference_steps:['Denoising steps','去噪步数'],
  no_final_step_noise:['Disable noise in final step','最终步骤不加噪声'],save_visualisation:['Save diffusion trajectory','保存扩散轨迹'],
  omit_AA:['Exclude amino acids','排除的氨基酸'],bias_AA:['Amino acid bias (A:0.5,G:-0.5)','氨基酸偏好（A:0.5,G:-0.5）'],
  bias_AA_per_residue_text:['Residue-specific bias (JSON)','逐位氨基酸偏好（JSON）'],omit_AA_per_residue_text:['Residue exclusions (JSON)','逐位氨基酸排除（JSON）'],
  symmetry_residues:['Linked residues (A1,B1|A2,B2)','联动残基（A1,B1|A2,B2）'],symmetry_weights:['Symmetry weights','对称设计权重'],homo_oligomer:['Symmetric homomer','同源多聚体对称设计'],
  ligand_mpnn_use_atom_context:['Use ligand atom context','使用配体原子信息'],ligand_mpnn_use_side_chain_context:['Use fixed side-chain context','使用固定侧链信息'],ligand_mpnn_cutoff_for_score:['Ligand scoring distance (Å)','配体评分距离（Å）'],
  parse_these_chains_only:['Only read these chains','仅读取这些链'],parse_atoms_with_zero_occupancy:['Include zero-occupancy atoms','包含零占有率原子'],pack_side_chains:['Build side chains','构建侧链'],
  number_of_packs_per_design:['Side-chain conformers per design','每个设计的侧链构象数'],sc_num_denoising_steps:['Side-chain denoising steps','侧链去噪步数'],sc_num_samples:['Side-chain samples','侧链采样数'],repack_everything:['Rebuild all side chains','重新构建所有侧链'],pack_with_ligand_context:['Use ligand for side-chain packing','根据配体构建侧链'],
  model_family:['Model family','模型类型'], model_name:['Checkpoint','权重版本'], num_sequences:['Sequences to design','生成序列数'],
  temperatures:['Sampling temperatures','采样温度'], designed_chains:['Chains to design','设计链'], fixed_chains:['Fixed chains','固定链'],
  fixed_residues_text:['Fixed residues','固定残基'], homomer:['Symmetric homomer','同源多聚体对称设计'], omit_aas:['Exclude amino acids','排除的氨基酸'],
  backbone_noise:['Backbone noise (Å)','骨架噪声（Å）'], seed:['Random seed','随机种子'], batch_size:['Batch size','批大小'], max_length:['Maximum sequence length','最大序列长度'],
  tied_positions_text:['Linked residue positions','联动残基位置'], omit_aa_rules_text:['Residue-specific exclusions (JSON)','逐位氨基酸排除（JSON）'],
  aa_bias_text:['Amino acid bias','氨基酸偏好'], bias_by_residue_text:['Residue-specific bias (JSON)','逐位偏好（JSON）'],
  pssm_rules_text:['PSSM constraints (JSON)','PSSM 约束（JSON）'], pssm_multi:['PSSM mixing weight','PSSM 混合权重'], pssm_threshold:['PSSM threshold','PSSM 阈值'],
  pssm_log_odds_flag:['Use PSSM log odds (0/1)','启用 PSSM log odds（0/1）'], pssm_bias_flag:['Use PSSM bias (0/1)','启用 PSSM 偏好（0/1）'],
  protect_start:['Protected region start','保护区起点'], protect_end:['Protected region end','保护区终点'], mmseqs_threads:['Search threads','搜索线程'],
  mmseqs_iterations:['Search iterations','搜索迭代次数'], mmseqs_max_seqs:['Maximum search sequences','最大检索序列数'], cluster_min_seq_id:['Clustering sequence identity','聚类序列一致性'],
  cluster_threads:['Clustering threads','聚类线程'], top_n_threshold:['Top candidates','候选数量'], evalue_threshold:['E-value threshold','E-value 阈值'], function_selection:['Scoring function','评分方式'],
};
const common = new Set(['ligand_smiles','center','size','chains_to_design','design_positions','temperature','number_of_batches','chains','target_hotspot_residues','target_chain','binder_chain','min_length','max_length','protocol','num_designs','budget','number_of_final_designs','samples_per_complex','model_family','model_name','num_sequences','temperatures','designed_chains','fixed_chains','fixed_residues_text','homomer','omit_aas','protect_start','protect_end','top_n_threshold','function_selection']);

export function ModelParameters({schema,values,onChange,disabled=false,zh=false}:{schema?:ParameterSchema;values:Record<string,unknown>;onChange:(v:Record<string,unknown>)=>void;disabled?:boolean;zh?:boolean}) {
  const [drafts,setDrafts]=useState<Record<string,string>>({});
  if (!schema || !Object.keys(schema.properties).length) return null;
  const fields=Object.entries(schema.properties);
  function render([key,field]:typeof fields[number]) {
    const type=field.type || field.anyOf?.find(v=>v.type!=='null')?.type;
    const value=values[key] ?? field.default ?? '';
    const update=(v:unknown)=>onChange(updateParameter(schema!,values,key,v));
    const choices=parameterChoices(field,schema!,values);
    const label=labels[key]?.[zh?1:0] || field.title || key;
    return <label className="mp-field" key={key}>{label}
      {choices ? <select disabled={disabled} value={String(value)} onChange={e=>update(e.target.value)}>{choices.map(v=><option key={v} value={v}>{v}</option>)}</select>
       : type==='boolean' ? <select disabled={disabled} value={String(Boolean(value))} onChange={e=>update(e.target.value==='true')}><option value="false">{zh?'否':'No'}</option><option value="true">{zh?'是':'Yes'}</option></select>
       : type==='array' ? <input disabled={disabled} value={drafts[key] ?? (Array.isArray(value)?value.join(', '):String(value))} placeholder={key==='center'?'X, Y, Z':field.items?.type==='number'?'0.1, 0.2':'A, B'} onChange={e=>{const text=e.target.value;setDrafts(d=>({...d,[key]:text}));try{update(parseArrayParameter(text,['number','integer'].includes(field.items?.type || '')));}catch{update(text);}}}/>
       : key.endsWith('_text') ? <textarea disabled={disabled} rows={2} value={String(value)} onChange={e=>update(e.target.value)}/>
       : <input disabled={disabled} type={type==='number'||type==='integer'?'number':'text'} step={type==='number'?'any':undefined} min={field.minimum} max={field.maximum} value={String(value)} onChange={e=>update(type==='number'||type==='integer'?(e.target.value===''?undefined:Number(e.target.value)):e.target.value)}/>}
    </label>;
  }
  const extra=fields.filter(([key])=>!common.has(key));
  return <section className="mp-model-parameters"><h3>{zh?'模型参数':'Model parameters'}</h3>{fields.filter(([key])=>common.has(key)).map(render)}{extra.length>0&&<details className="mp-details"><summary>{zh?'更多参数':'More parameters'}</summary>{extra.map(render)}</details>}</section>;
}
