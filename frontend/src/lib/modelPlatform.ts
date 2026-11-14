import { validateUploads } from "./uploadLimits";
import type { ParameterSchema } from "./modelParameters";
export type ModelCard = { author_email?: string; model_type?: string; runtime_id?: string; backbone?: {name:string;repository?:string}; training_dataset?: string; training_method?: string; provenance?: string; version?: string; artifact?: {sha256:string;bytes:number}; runtime_mode?: string; accepted_input_kinds?: string[]; input_format?: string; parameter_schema?: ParameterSchema; input_accept?: string; extra_inputs?: {key:string;label:string;accept:string;required:boolean}[]; available?: boolean; disabled_reason?: string; id: string; name: string; description: string; category: string; source: string; adapter: string; visibility: string; predictable: boolean; input_kind: string; node_kind?: 'model' | 'tool'; output_kind?: string; standalone?: boolean; owned?: boolean; route?: string; target?: string; features?: string[]; sequence_column?: string; method?: string; problem?: string; metrics?: Record<string, number>; split?: Record<string, unknown>; scope?: Record<string, unknown>; contract?: Record<string, unknown> };
export type ADMETEndpoint = {id:string;name:string;category:string;task_type:string;units:string;species:string;url:string};
export type ADMETMetadata = {engine:string;version:string;predicted:number;invalid:number;endpoints:ADMETEndpoint[]};
export type Dataset = { id: string; name: string; kind: string; columns: string[]; count: number; preview: Record<string, unknown>[] };
export type FlowSummary = {kind: string; rows?: number; files?: number; columns?: string[]; column_count?: number; preview?: Record<string, unknown>[]; model?: {id?: string; name?: string; metrics?: Record<string, number>}};
export type FlowStep = {id: string; status: string; rows?: number; progress?: number; message?: string; model?: ModelCard; started?: number; finished?: number; inputs?: (FlowSummary & {source: string})[]; output?: FlowSummary};
export type FlowEvent = {seq: number; step_id: string; status: string; time: number; message: string};
export type Job = { job_number: string; title: string; created: number; queue?: { id:string; resource:string; status:string; position:number; cancel_requested:boolean }; id: string; kind: string; status: string; progress: number; message: string; steps: FlowStep[]; events?: FlowEvent[]; result?: { metadata?: ADMETMetadata; artifacts_download?: string; model?: ModelCard; models?: ModelCard[]; preview: Record<string, unknown>[]; rows: number; download: string } };
export type Catalog = { items: ModelCard[]; total: number; registered: number; categories: Record<string, number>; unavailable_sources: string[]; unavailable_tools?: string[] };
import { COMPUTE_QUOTA_CHANGED, computeQuotaError } from './computeQuota';
import { ensureJobSession } from './jobSession';
import { notifyJobs } from './jobUpdates';
export async function platform<T>(path: string, options?: RequestInit): Promise<T> {
  const request = async (): Promise<T> => {
  const res = await fetch('/api/model-platform' + path, { credentials: 'same-origin', ...options, headers: { ...(options?.method === 'POST' ? { 'Idempotency-Key': crypto.randomUUID() } : {}), ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options?.headers } });
  const data = await res.json();
  if (!res.ok) throw new Error(computeQuotaError(data) || (typeof data.detail === 'string' ? data.detail : data.detail?.message || 'Please check your input and try again.'));
  if (options?.method === 'POST' && (path === '/predict' || path === '/tools/run' || path.endsWith('/retry'))) window.dispatchEvent(new Event(COMPUTE_QUOTA_CHANGED));
  if (options?.method === 'POST' && (path === '/predict' || path === '/train' || path === '/tools/run' || /\/jobs\/[^/]+\/(cancel|retry)$/.test(path))) notifyJobs('platform', data);
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
export const sourceNames: Record<string, [string, string]> = { platform: ['Platform', '平台'], community: ['Community', '社区'] };
