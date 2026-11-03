import { useEffect, useState } from 'react';
import { openQuota, QUOTA_CHANGED, QUOTA_EVENT, quotaRetryLabel, type Quota } from '../lib/onlineQuota';
import './access.css';

export function QuotaNotice({ quota, lang }: { quota: Quota | null; lang: string }) {
  const [current, setCurrent] = useState(quota);
  const [dismissed, setDismissed] = useState<string | null>(null);
  useEffect(() => { setCurrent(quota); }, [quota]);
  useEffect(() => {
    const show = (event: Event) => {
      const data = (event as CustomEvent<{ exhausted?: boolean; quota?: Quota }>).detail;
      if (!data?.exhausted) return;
      if (data.quota) setCurrent(data.quota);
      setDismissed(null);
    };
    window.addEventListener(QUOTA_EVENT, show);
    return () => window.removeEventListener(QUOTA_EVENT, show);
  }, []);
  useEffect(() => {
    if (!current?.enforced) return;
    const reset = Date.parse(current.resets_at || '');
    if (!Number.isFinite(reset)) return;
    const timer = window.setTimeout(() => window.dispatchEvent(new Event(QUOTA_CHANGED)),
      Math.min(2_147_483_647, Math.max(1000, reset - Date.now() + 500)));
    return () => window.clearTimeout(timer);
  }, [current?.resets_at, current?.enforced]);

  const identity = `${current?.resets_at}:${current?.limit}`;
  if (!current?.enforced || current.remaining == null || current.remaining > 0 || dismissed === identity) return null;
  const zh = lang === 'zh';
  return <div className="chat-quota-notice-slot">
    <div className="chat-quota-notice">
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
      <span className="chat-quota-notice-text" role="status" aria-live="polite" aria-atomic="true" title={current.timezone}>{quotaRetryLabel(current, lang)}</span>
      <button type="button" className="chat-quota-extra" onClick={() => openQuota()}>{zh ? '额外额度（反馈）' : 'Extra allowance (Feedback)'}</button>
      <button type="button" className="chat-quota-dismiss" aria-label={zh ? '关闭额度提示' : 'Dismiss quota notice'} onClick={() => setDismissed(identity)}>×</button>
    </div>
  </div>;
}
