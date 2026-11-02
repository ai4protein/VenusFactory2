import type {Dataset, FlowEvent, FlowStep, FlowSummary, Graph, Job} from '../lib/modelPlatform';
import {ResultTable} from './ModelPlatformShared';

const states: Record<string, [string, string]> = {queued: ['Waiting', '等待上游'], running: ['Running', '处理中'], completed: ['Completed', '已完成'], failed: ['Stopped', '已停止'], skipped: ['Not run', '未执行']};
export function FlowData({value, zh}: {value?: FlowSummary | Dataset; zh: boolean}) {
  if (!value) return null;
  const rows = 'count' in value ? value.count : value.rows;
  const model = 'model' in value ? value.model : undefined;
  const kind = ({table: zh ? '表格' : 'Table', structure: zh ? '蛋白结构' : 'Protein structure', ligand: zh ? '小分子' : 'Ligand', model: zh ? '已训练模型' : 'Trained model'} as Record<string,string>)[value.kind] || value.kind;
  return <div className="wf-data-preview"><div className="wf-data-caption"><strong>{kind}</strong><span>{value.kind === 'table' ? `${rows ?? 0} ${zh ? '行' : 'rows'} · ${('column_count' in value ? value.column_count : value.columns?.length) ?? 0} ${zh ? '列' : 'columns'}` : model?.name || (zh ? '1 个文件' : '1 file')}</span></div>
    {!!value.columns?.length && <p className="wf-column-names">{value.columns.join(' · ')}</p>}
    {!!value.preview?.length && <><ResultTable rows={value.preview.slice(0,3)}/><small>{zh ? '前 3 行预览' : 'Preview of first 3 rows'}</small></>}
    {model?.metrics && <dl className="wf-model-metrics">{Object.entries(model.metrics).map(([key,val]) => <div key={key}><dt>{key}</dt><dd>{Number(val).toPrecision(4)}</dd></div>)}</dl>}
  </div>;
}

export function WorkflowRunTrace({job, graph, selected, onSelect, zh}: {job: Job; graph: Graph; selected: string; onSelect: (id:string)=>void; zh:boolean}) {
  if (!job.steps.length) return null;
  return <section className="wf-run-trace" aria-label={zh ? '数据流转' : 'Data flow'}><strong>{zh ? '执行过程' : 'Execution trace'}</strong><ol>{job.steps.map((step,index) => <li key={step.id}><button type="button" className={`wf-trace-step ${step.status}`} aria-pressed={selected === step.id} onClick={()=>onSelect(step.id)}>
    <span className="wf-trace-index">{step.status === 'completed' ? '✓' : index+1}</span><span><b>{graph.nodes.find(n=>n.id===step.id)?.label || step.id}</b><small>{states[step.status]?.[zh ? 1 : 0] || step.status}{step.output?.rows !== undefined ? ` · ${step.output.rows} ${zh ? '行' : 'rows'}` : ''}</small></span>
  </button></li>)}</ol></section>;
}

export function WorkflowStepDetails({step, events, graph, zh}: {step?: FlowStep; events:FlowEvent[]; graph:Graph; zh:boolean}) {
  if (!step) return null;
  const logs = events.filter(e=>e.step_id===step.id).slice(-12);
  return <section className="wf-step-details"><h3>{zh ? '运行详情' : 'Run details'}</h3><p>{states[step.status]?.[zh?1:0] || step.status}{step.started && step.finished ? ` · ${(step.finished-step.started).toFixed(1)} s` : ''}</p>
    {step.status === 'running' && <progress aria-label={zh ? '步骤进度' : 'Step progress'} max={100} value={step.progress || undefined}/>}
    {step.message && <p className="wf-step-message">{step.message}</p>}
    {step.inputs?.map(input=><details key={input.source}><summary>{zh ? '输入来自：' : 'Input from: '}{graph.nodes.find(n=>n.id===input.source)?.label || input.source}</summary><FlowData value={input} zh={zh}/></details>)}
    {step.output && <details open><summary>{zh ? '输出' : 'Output'}</summary><FlowData value={step.output} zh={zh}/></details>}
    {!!logs.length && <details><summary>{zh ? '处理日志' : 'Processing log'}</summary><ol className="wf-processing-log">{logs.map(e=><li key={e.seq}><time>{new Date(e.time*1000).toLocaleTimeString(zh?'zh-CN':'en-GB')}</time><span>{e.message || states[e.status]?.[zh?1:0] || e.status}</span></li>)}</ol></details>}
  </section>;
}
