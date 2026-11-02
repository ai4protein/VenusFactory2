import { connectGraph } from './modelPlatform';
import type { Graph, GraphNode, Workflow, Dataset, ModelCard } from './modelPlatform';

export const workflowSignature = (w: Workflow) => JSON.stringify({ name: w.name, graph: w.graph });
export function workflowName(w: Workflow, zh: boolean): string {
  return w.name.trim() || w.graph.nodes.find(n => ['model', 'tool', 'train', 'predict'].includes(n.kind))?.label || (zh ? '未命名工作流' : 'Untitled workflow');
}

/** Add a normal dataflow step; preserve existing branches and model-input wires. */
export function appendWorkflowStep(graph: Graph, step: GraphNode, selected: string, labels: {data: string; output: string}, standalone = false): Graph {
  let g: Graph = { nodes: [...graph.nodes], edges: [...graph.edges] };
  const makeOutput = (source: GraphNode) => {
    const output: GraphNode = {id: step.id + '_result', kind: 'output', label: labels.output, x: source.x + 280, y: source.y};
    g.nodes.push(output); g = connectGraph(g, source.id, output.id);
  };
  if (!g.nodes.length) {
    if (!['data', 'output'].includes(step.kind) && !standalone) {
      const input: GraphNode = {id: step.id + '_data', kind: 'data', label: labels.data, x: 40, y: 80};
      step = {...step, x: 320, y: 80}; g.nodes.push(input, step);
      g = connectGraph(g, input.id, step.id, 'data');
    } else g.nodes.push(step);
    if (step.kind !== 'output') makeOutput(step);
    return g;
  }
  let anchor = g.nodes.find(n => n.id === selected);
  if (anchor?.kind === 'output') anchor = g.nodes.find(n => n.id === g.edges.find(e => e.target === selected)?.source);
  if (!anchor) anchor = [...g.nodes].reverse().find(n => n.kind !== 'output');
  if (step.kind === 'data' || standalone || !anchor || (anchor.kind === 'train' && !['predict','output'].includes(step.kind))) {
    g.nodes.push(step);
    if (standalone) makeOutput(step);
    return g;
  }
  step = {...step, x: anchor.x + 280, y: anchor.y};
  g.nodes.push(step);
  try { g = connectGraph(g, anchor.id, step.id); }
  catch { g.nodes = g.nodes.map(n => n.id === step.id ? {...n, y: Math.max(...g.nodes.map(n=>n.y)) + 190} : n); makeOutput(g.nodes.find(n=>n.id===step.id)!); return g; }
  if (step.kind === 'output') return g;
  const results = g.edges.filter(e => e.source === anchor!.id && g.nodes.find(n => n.id === e.target)?.kind === 'output');
  if (results.length) {
    for (const edge of results) g = connectGraph(g, step.id, edge.target);
    g.nodes = g.nodes.map(n => results.some(e => e.target === n.id) ? {...n, x: step.x + 280, y: step.y} : n);
  } else makeOutput(step);
  return g;
}

export function workflowIssue(graph: Graph, zh: boolean, datasets: Dataset[] = [], cards: ModelCard[] = []): {id: string; message: string} | undefined {
  const tr = (en: string, cn: string) => zh ? cn : en;
  const ids = new Set(graph.nodes.map(n => n.id));
  if (ids.size !== graph.nodes.length) return {id: '', message: tr('Step IDs must be unique', '步骤编号不能重复')};
  const seen = new Set<string>();
  for (const e of graph.edges) {
    const key = e.source + '→' + e.target;
    if (seen.has(key)) return {id: e.target, message: tr('Repeated connection', '连接重复')};
    seen.add(key);
    try { connectGraph(graph, e.source, e.target, e.target_port, datasets, cards); }
    catch (error) { return {id: e.target, message: (error as Error).message}; }
  }
  const ordered = new Set<string>();
  while (ordered.size < graph.nodes.length) {
    const ready = graph.nodes.filter(n => !ordered.has(n.id) && graph.edges.filter(e=>e.target===n.id).every(e=>ordered.has(e.source)));
    if (!ready.length) return {id: '', message: tr('Remove the cycle before running', '请先移除循环连接')};
    ready.forEach(n=>ordered.add(n.id));
  }
  for (const n of graph.nodes) {
    const incoming = graph.edges.filter(e => e.target === n.id);
    let problem = '';
    if (n.kind === 'data' && !n.dataset_id) problem = tr('Choose or upload data', '请选择或上传数据');
    else if (['model', 'tool'].includes(n.kind) && !n.model_id) problem = tr('Choose a model or tool', '请选择模型或工具');
    else if (n.kind === 'train' && (!n.config?.target || (!(n.config.features as unknown[])?.length && !n.config.sequence_column))) problem = tr('Set the target and feature columns', '请设置目标列和特征列');
    else if (n.kind === 'predict' && (incoming.length !== 2 || !incoming.some(e => graph.nodes.find(p => p.id === e.source)?.kind === 'train'))) problem = tr('Connect prediction data and a trained model', '请连接预测数据和训练模型');
    else if (['model', 'train', 'output'].includes(n.kind) && incoming.length !== 1) problem = tr('Choose the preceding input step', '请选择上游输入步骤');
    else if (n.kind === 'tool' && (incoming.length > 1 || (!incoming.length && (cards.find(c=>c.id===n.model_id)?.standalone ?? n.standalone) === false))) problem = tr('Connect one compatible input', '请连接一个兼容的输入');
    else if (n.kind !== 'output' && !graph.edges.some(e => e.source === n.id)) problem = tr('Connect this step to a result', '请将此步骤连接到结果');
    if (problem) return {id: n.id, message: n.label + ': ' + problem};
  }
  if (!graph.nodes.some(n => n.kind === 'output')) return {id: graph.nodes[0]?.id || '', message: tr('Add a Results step', '请添加结果步骤')};
}
