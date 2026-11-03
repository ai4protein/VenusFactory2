export const QUOTA_EVENT = 'vf:quota-dialog';
export const QUOTA_CHANGED = 'vf:quota-changed';
const STORAGE = 'vf_online_quota_v1';
export type Quota = { compute?: { prediction: { remaining: number; limit: number } }; mode: string; enforced: boolean; limit: number | null; used: number; remaining: number | null; extra_limit?: number; email?: string | null; resets_at?: string; timezone?: string };
export function savedQuota(): { email: string; key: string } {
  try { const value = JSON.parse(localStorage.getItem(STORAGE) || '{}'); return { email: typeof value.email === 'string' ? value.email : '', key: typeof value.key === 'string' ? value.key : '' }; }
  catch { return { email: '', key: '' }; }
}
export function saveQuota(value: { email: string; key: string } | null) {
  if (value) localStorage.setItem(STORAGE, JSON.stringify(value)); else localStorage.removeItem(STORAGE);
  window.dispatchEvent(new Event(QUOTA_CHANGED));
}
export function quotaHeaders(): Record<string, string> {
  const { email, key } = savedQuota();
  return email && key ? { 'X-Venus-Quota-Email': email, 'X-Venus-Quota-Key': key } : {};
}
export function openQuota(exhausted = false, quota?: Quota) {
  window.dispatchEvent(new CustomEvent(QUOTA_EVENT, { detail: { exhausted, quota } }));
  if (exhausted) window.dispatchEvent(new Event(QUOTA_CHANGED));
}
export function openQuotaRequest() {
  window.dispatchEvent(new CustomEvent(QUOTA_EVENT, { detail: { tab: 'request' } }));
}
export function quotaError(body: unknown): string | null {
  const detail = (body as { detail?: { code?: string; message?: string; quota?: Quota } })?.detail;
  if (detail?.code === 'CHAT_DAILY_LIMIT_REACHED' || detail?.code === 'QUOTA_KEY_INVALID') {
    openQuota(detail.code === 'CHAT_DAILY_LIMIT_REACHED', detail.quota);
    return detail.message || 'Check your daily allowance in Settings.';
  }
  return null;
}
export async function fetchQuota(): Promise<Quota> {
  const response = await fetch('/api/quota', { headers: quotaHeaders(), cache: 'no-store' });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : body.detail?.message || 'Could not load allowance');
  return body;
}

export function quotaRetryLabel(quota: Quota | null | undefined, lang: string): string {
  const date = new Date(quota?.resets_at || '');
  if (!Number.isFinite(date.getTime())) return lang === 'zh' ? '请稍后重试' : 'Please try again later';
  const format: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' };
  let time: string;
  try { time = date.toLocaleTimeString('en-GB', { ...format, timeZone: quota?.timezone }); }
  catch { time = date.toLocaleTimeString('en-GB', format); }
  return lang === 'zh' ? `请于 ${time} 后重试` : `Try again at ${time}`;
}
export function isDailyQuotaMessage(message: string): boolean {
  return /new conversation allowance|CHAT_DAILY_LIMIT_REACHED/.test(message);
}
