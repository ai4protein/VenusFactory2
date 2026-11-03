export const COMPUTE_QUOTA_CHANGED = 'vf:compute-quota-changed';
export type ComputeQuota = {
  enforced: boolean; mode: string; resets_at?: string; timezone?: string;
  workflow?: { limit: number; used: number; remaining: number };
  prediction?: { limit: number; used: number; remaining: number };
};

export function computeQuotaError(body: unknown): string | null {
  const detail = (body as { detail?: { code?: string; resource?: string; required?: number; quota?: ComputeQuota; message?: string } })?.detail;
  if (detail?.code !== 'COMPUTE_DAILY_LIMIT_REACHED') return null;
  window.dispatchEvent(new CustomEvent(COMPUTE_QUOTA_CHANGED, { detail: detail.quota }));
  const zh = window.location.pathname.startsWith('/zh/');
  const name = detail.resource === 'workflow' ? (zh ? '工作流' : 'workflow') : (zh ? '模型推理' : 'prediction');
  const remaining = detail.quota?.[detail.resource === 'workflow' ? 'workflow' : 'prediction']?.remaining ?? 0;
  const date = new Date(detail.quota?.resets_at || '');
  const time = Number.isFinite(date.getTime()) ? date.toLocaleTimeString(zh ? 'zh-CN' : 'en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: detail.quota?.timezone }) : '';
  return zh ? `本次需要 ${detail.required || 1} 次${name}，今日剩余 ${remaining} 次。${time ? `额度将于 ${time} 重置。` : ''}`
    : `This job needs ${detail.required || 1} ${name} runs; ${remaining} remain today.${time ? ` Resets at ${time}.` : ''}`;
}
