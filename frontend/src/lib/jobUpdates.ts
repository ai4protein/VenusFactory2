import type { JobPage, JobSummary } from './jobs';

export const JOBS_CHANGED = 'venus:jobs-changed';
export type JobUpdate = { source: 'platform' | 'custom'; job?: JobSummary };

export function notifyJobs(source: JobUpdate['source'], job?: JobSummary) {
  window.dispatchEvent(new CustomEvent<JobUpdate>(JOBS_CHANGED, { detail: { source, job } }));
}

export function mergeJobUpdate(page: JobPage | undefined, job?: JobSummary): JobPage | undefined {
  if (!page || !job) return page;
  const old = page.items.find(item => item.id === job.id);
  if (!old) return page;
  const active = (status: string) => Number(['queued', 'running'].includes(status));
  return { ...page, active: Math.max(0, page.active + active(job.status) - active(old.status)),
    items: page.items.map(item => item.id === job.id ? { ...item, ...job } : item) };
}

export function jobPollDelay(open: boolean, active: boolean, hidden: boolean) {
  return hidden ? 30000 : active ? (open ? 1000 : 3000) : (open ? 5000 : 15000);
}
