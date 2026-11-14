import { useEffect, useState } from 'react';
import { useLang } from '../lib/i18n';
import { platform } from '../lib/modelPlatform';
import { COMPUTE_QUOTA_CHANGED, type ComputeQuota } from '../lib/computeQuota';
import './ComputeAllowance.css';

export function ComputeAllowance({ only }: { only?: 'prediction' }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [quota, setQuota] = useState<ComputeQuota>();
  useEffect(() => {
    let alive = true;
    const refresh = () => { void platform<ComputeQuota>('/allowance').then(value => { if (alive) setQuota(value); }).catch(() => { /* Admission remains authoritative if the status request fails. */ }); };
    refresh();
    window.addEventListener(COMPUTE_QUOTA_CHANGED, refresh);
    window.addEventListener('focus', refresh);
    const timer = setInterval(refresh, 60000);
    return () => { alive = false; clearInterval(timer); window.removeEventListener(COMPUTE_QUOTA_CHANGED, refresh); window.removeEventListener('focus', refresh); };
  }, []);
  if (!quota?.enforced) return null;
  return <div className="compute-allowance" aria-label={zh ? '今日剩余额度' : 'Remaining today'}>
    {(['prediction'] as const).filter(kind => !only || only === kind).map(kind => {
      const value = quota[kind]; if (!value) return null;
      return <span key={kind} className={value.remaining ? '' : 'is-empty'}><span>{zh ? '模型推理' : 'Predictions'}</span><strong>{value.remaining}<small> / {value.limit}</small></strong></span>;
    })}<span className="compute-allowance-label">{zh ? '今日剩余' : 'left today'}</span>
  </div>;
}
