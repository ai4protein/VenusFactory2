import { useState } from 'react';
import { useLang } from '../lib/i18n';

export function JobNumber({ number }: { number?: string }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [copied, setCopied] = useState(false);
  if (!number) return null;
  async function copy() {
    try { await navigator.clipboard.writeText(number!); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch { setCopied(false); }
  }
  return <span className="job-number"><code>{number}</code><button type="button" onClick={() => void copy()} aria-label={`${zh ? '复制作业号' : 'Copy job number'} ${number}`} title={zh ? '复制作业号' : 'Copy job number'}>{copied ? '✓' : '⧉'}</button></span>;
}
