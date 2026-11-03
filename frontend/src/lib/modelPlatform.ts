import { validateUploads } from "./uploadLimits";
import type { ParameterSchema } from "./modelParameters";
export type ModelCard = { model_type?: string; runtime_id?: string; backbone?: {name:string;repository?:string}; training_dataset?: string; training_method?: string; provenance?: string; version?: string; artifact?: {sha256:string;bytes:number}; runtime_mode?: string; accepted_input_kinds?: string[]; input_format?: string; parameter_schema?: ParameterSchema; input_accept?: string; extra_inputs?: {key:string;label:string;accept:string;required:boolean}[]; available?: boolean; disabled_reason?: string; id: string; name: string; description: string; category: string; source: string; adapter: string; visibility: string; workflow: boolean; input_kind: string; node_kind?: 'model' | 'tool'; output_kind?: string; standalone?: boolean; owned?: boolean; route?: string; target?: string; features?: string[]; sequence_column?: string; method?: string; problem?: string; metrics?: Record<string, number>; split?: Record<string, unknown>; scope?: Record<string, unknown>; contract?: Record<string, unknown> };
export type ADMETEndpoint = {id:string;name:string;category:string;task_type:string;units:string;species:string;url:string};
export type ADMETMetadata = {engine:string;version:string;predicted:number;invalid:number;endpoints:ADMETEndpoint[]};
export type Dataset = { id: string; name: string; kind: string; columns: string[]; count: number; preview: Record<string, unknown>[] };
export type GraphNode = { id: string; kind: 'data' | 'model' | 'tool' | 'train' | 'predict' | 'output'; label: string; x: number; y: number; model_id?: string; dataset_id?: string; config?: Record<string, unknown>; input_kinds?: string[]; output_kind?: string; standalone?: boolean };
export type Graph = { nodes: GraphNode[]; edges: { source: string; target: string; target_port?: string }[] };
export type Workflow = { id?: string; name: string; graph: Graph };
export type FlowSummary = {kind: string; rows?: number; files?: number; columns?: string[]; column_count?: number; preview?: Record<string, unknown>[]; model?: {id?: string; name?: string; metrics?: Record<string, number>}};
export type FlowStep = {id: string; status: string; rows?: number; progress?: number; message?: string; model?: ModelCard; started?: number; finished?: number; inputs?: (FlowSummary & {source: string})[]; output?: FlowSummary};
export type FlowEvent = {seq: number; step_id: string; status: string; time: number; message: string};
export type Job = { job_number: string; title: string; created: number; queue?: { id:string; resource:string; status:string; position:number; cancel_requested:boolean }; id: string; kind: string; status: string; progress: number; message: string; steps: FlowStep[]; events?: FlowEvent[]; workflow?: Workflow; result?: { metadata?: ADMETMetadata; artifacts_download?: string; model?: ModelCard; models?: ModelCard[]; preview: Record<string, unknown>[]; rows: number; download: string } };
export type ComponentCatalog = { items: ModelCard[]; total: number; categories: Record<string, number>; kinds: Record<string, number>; unavailable_sources: string[]; unavailable_tools: string[] };
export type Catalog = { items: ModelCard[]; total: number; registered: number; categories: Record<string, number>; unavailable_sources: string[]; unavailable_tools?: string[] };
import { COMPUTE_QUOTA_CHANGED, computeQuotaError } from './computeQuota';
import { ensureJobSession } from './jobSession';
import { notifyJobs } from './jobUpdates';
export async function platform<T>(path: string, options?: RequestInit): Promise<T> {
  const request = async (): Promise<T> => {
  const res = await fetch('/api/model-platform' + path, { credentials: 'same-origin', ...options, headers: { ...(options?.method === 'POST' ? { 'Idempotency-Key': crypto.randomUUID() } : {}), ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options?.headers } });
  const data = await res.json();
  if (!res.ok) throw new Error(computeQuotaError(data) || (typeof data.detail === 'string' ? data.detail : data.detail?.message || 'Please check your input and try again.'));
  if (options?.method === 'POST' && (path === '/predict' || path === '/workflows/run' || path.endsWith('/retry'))) window.dispatchEvent(new Event(COMPUTE_QUOTA_CHANGED));
  if (options?.method === 'POST' && (path === '/predict' || path === '/train' || path === '/workflows/run' || /\/jobs\/[^/]+\/(cancel|retry)$/.test(path))) notifyJobs('platform', data);
  return data as T;
  };
  // The first read also establishes the workspace cookie. Other callers wait
  // on the same bootstrap, so parallel loads cannot create different owners.
  let initial: T | undefined;
  let initialized = false;
  await ensureJobSession('platform', path.startsWith('/catalog?') && (!options?.method || options.method === 'GET')
    ? async () => { initial = await request(); initialized = true; }
    : undefined);
  return initialized ? initial as T : request();
}
export const post = <T>(path: string, value: unknown) => platform<T>(path, { method: 'POST', body: JSON.stringify(value) });
export async function uploadDataset(file: File) {
  await validateUploads([file]);
  const body = new FormData(); body.append('file', file);
  return platform<Dataset>('/datasets', { method: 'POST', body });
}
export const categories = ['function', 'evolution', 'design', 'docking', 'admet', 'discovery', 'residue', 'properties', 'foundation', 'custom'];
export const plazaCategories = [...categories, 'processing', 'search'];
export const categoryNames: Record<string, [string, string]> = {
  admet: ['ADMET', '小分子 ADMET'], molecular_admet: ['ADMET', '小分子 ADMET'], docking: ['Molecular docking', '分子对接'], processing: ['Data processing', '数据处理'], search: ['Search & literature', '检索与文献'],
  function: ['Function & activity', '功能与活性'], evolution: ['Directed evolution', '定向进化'], design: ['Sequence design', '序列设计'], discovery: ['Protein discovery', '蛋白挖掘'], residue: ['Residue prediction', '残基预测'], properties: ['Physicochemical', '理化性质'], foundation: ['Foundation models', '基础模型'], custom: ['Custom models & tools', '自建模型与工具']
};
export const modelTypeNames: Record<string, [string, string]> = { foundation: ['Backbone', '基础模型'], pretrained: ['Pretrained model', '预训练模型'], finetuned: ['Fine-tuned model', '微调任务模型'], trained: ['Trained model', '训练模型'], tool: ['Tool', '计算工具'] };
export const sourceNames: Record<string, [string, string]> = { platform: ['Platform', '平台'], scientist: ['Scientist', 'Scientist'], community: ['Community', '社区'] };
export function connectGraph(graph: Graph, source: string, target: string, targetPort?: string, datasets: Dataset[] = [], cards: ModelCard[] = []): Graph {
  if (source === target) throw new Error('A step cannot connect to itself');
  const a = graph.nodes.find(n => n.id === source), b = graph.nodes.find(n => n.id === target);
  if (!a || !b || a.kind === 'output' || b.kind === 'data') throw new Error('Connect a step output to a compatible input');
  if (targetPort && !['data', 'model'].includes(targetPort)) throw new Error('Unknown input port');
  if (targetPort === 'model' && b.kind !== 'predict') throw new Error('Only prediction steps accept a model port');
  const port = b.kind === 'predict' ? (targetPort || (a.kind === 'train' ? 'model' : 'data')) : undefined;
  if (b.kind === 'predict' && (port === 'model') !== (a.kind === 'train')) throw new Error('Connect a training step to the model input, and data to the data input');
  if (a.kind === 'train' && !['predict', 'output'].includes(b.kind)) throw new Error('Connect a trained model to a prediction or result step');
  const sourceKind = a.kind === 'data' ? datasets.find(d => d.id === a.dataset_id)?.kind : a.kind === 'train' ? 'model' : a.output_kind || 'table';
  const card = cards.find(c => c.id === b.model_id);
  const accepted = b.kind === 'predict' ? [port === 'model' ? 'model' : 'table'] : b.kind === 'train' ? ['table'] : b.kind === 'output' ? undefined : card?.accepted_input_kinds || (card ? [card.input_kind] : b.input_kinds);
  if (sourceKind && accepted && !accepted.includes(sourceKind)) throw new Error(`${b.label || b.kind}: requires ${accepted.join(' / ')} input, received ${sourceKind}`);
  const edge = port ? { source, target, target_port: port } : { source, target };
  const edges = [...graph.edges.filter(e => e.target !== target || (b.kind === 'predict' && (e.target_port || (graph.nodes.find(n => n.id === e.source)?.kind === 'train' ? 'model' : 'data')) !== port)), edge];
  const reaches = (id: string, seen = new Set<string>()): boolean => {
    if (id === source) return true;
    if (seen.has(id)) return false;
    seen.add(id); return edges.filter(e => e.source === id).some(e => reaches(e.target, seen));
  };
  if (reaches(target)) throw new Error('This connection would create a cycle');
  return { ...graph, edges };
}

export function removeGraphNode(graph: Graph, id: string): Graph {
  return { nodes: graph.nodes.filter(n => n.id !== id), edges: graph.edges.filter(e => e.source !== id && e.target !== id) };
}
