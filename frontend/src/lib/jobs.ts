import { ensureJobSession } from './jobSession';
import { platform, type Job } from './modelPlatform';
import type { QueueStatus } from '../components/QueueState';

export type JobSummary = {
  id: string; job_number: string; title: string; kind: string; created: number;
  status: string; progress?: number; queue?: QueueStatus; resource?: string;
  position?: number; cancel_requested?: boolean;
};
export type JobPage = { items: JobSummary[]; total: number; active: number; offset: number; limit: number };
export type JobFilters = { query: string; status: string; kind: string; offset: number };
export const isActiveJob = (job: Pick<JobSummary, 'status'>) => ['queued', 'running'].includes(job.status);
export const canRetryJob = (job: Pick<JobSummary, 'status'>) => ['failed', 'interrupted', 'cancelled'].includes(job.status);

export function jobClient(customTask?: string) {
  async function request<T>(path: string, options?: RequestInit): Promise<T> {
    if (!customTask) return platform<T>(path, options);
    await ensureJobSession('custom');
    const response = await fetch('/api/custom-model' + path, { credentials: 'same-origin', ...options });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : 'Could not load jobs');
    return value as T;
  }
  return {
    list: (filters: JobFilters, signal?: AbortSignal) => request<JobPage>('/job-history?' + new URLSearchParams({
      ...filters, offset: String(filters.offset), limit: '10', ...(customTask ? { task: customTask } : {}),
    }), { signal }),
    get: (id: string) => request<Job>('/jobs/' + encodeURIComponent(id)),
    control: (id: string, action: 'cancel' | 'retry') => request<JobSummary>('/jobs/' + encodeURIComponent(id) + '/' + action, { method: 'POST' }),
  };
}
