import { ModelCard, Dataset } from '../lib/modelPlatform';
import { DataUpload } from './ModelPlatformShared';

export function ModelExtraInputs({model,datasets,config,onChange,onUpload,disabled=false,zh=false}:{model:ModelCard;datasets:Dataset[];config:Record<string,unknown>;onChange:(v:Record<string,unknown>)=>void;onUpload:(d:Dataset)=>void;disabled?:boolean;zh?:boolean}) {
  return <>{model.extra_inputs?.map(input=><section key={input.key}>
    <h3>{zh?'配体':input.label}{!input.required && (zh?'（可选）':' (optional)')}</h3>
    <DataUpload compact accept={input.accept} formats={input.accept.replace(/\./g,'').toUpperCase().replace(/,/g,' · ')} disabled={disabled} onUpload={d=>{onUpload(d);onChange({...config,[input.key]:d.id});}}/>
    <label className="mp-field">{zh?'已上传配体':'Uploaded ligand'}<select disabled={disabled} value={String(config[input.key] || '')} onChange={e=>onChange({...config,[input.key]:e.target.value})}><option value="">{zh?'选择配体':'Choose ligand'}</option>{datasets.filter(d=>input.accept.split(',').some(ext=>d.name.toLowerCase().endsWith(ext))).map(d=><option key={d.id} value={d.id}>{d.name}</option>)}</select></label>
  </section>)}</>;
}
