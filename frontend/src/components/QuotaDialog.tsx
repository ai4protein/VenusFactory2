import { useEffect, useRef, useState } from 'react';
import { useLang } from '../lib/i18n';
import { fetchQuota, QUOTA_EVENT, savedQuota, saveQuota, type Quota } from '../lib/onlineQuota';
import { allowanceEmailLink } from '../lib/allowanceEmail';
import './access.css';

export function QuotaDialog() {
  const { lang } = useLang(); const zh = lang === 'zh'; const t = (a: string, b: string) => zh ? b : a;
  const ref = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState(false), [quota, setQuota] = useState<Quota>();
  const [tab, setTab] = useState<'key' | 'request'>('key');
  const [email, setEmail] = useState(''), [key, setKey] = useState(''), [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [name, setName] = useState(''), [institution, setInstitution] = useState(''), [contact, setContact] = useState(''), [purpose, setPurpose] = useState(''), [amount, setAmount] = useState(10), [predictions, setPredictions] = useState(20);
  useEffect(() => {
    const show = (event: Event) => {
      const data = (event as CustomEvent).detail || {};
      if (data.exhausted) { setOpen(false); return; }
      const saved = savedQuota(); setEmail(saved.email); setKey(saved.key);
      setQuota(data.quota); setTab(data.tab === 'request' ? 'request' : 'key'); setError(''); setNotice(''); setOpen(true);
      void fetchQuota().then(setQuota).catch(e => setError(e.message));
    };
    window.addEventListener(QUOTA_EVENT, show); return () => window.removeEventListener(QUOTA_EVENT, show);
  }, []);
  useEffect(() => { if (open) ref.current?.showModal(); else ref.current?.close(); }, [open]);
  async function bind(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError(''); setNotice('');
    try {
      const value = { email: email.trim().toLowerCase(), key: key.trim() };
      const response = await fetch('/api/quota/validate', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Venus-Access': '1' }, body: JSON.stringify(value) });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : body.detail?.message || t('Could not validate key', '额度 key 验证失败'));
      saveQuota(value); setQuota(body); setNotice(t('Saved on this browser. Your allowance is ready.', '已在此浏览器保存，额度已生效。'));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  function request(event: React.FormEvent) {
    event.preventDefault(); setError(''); setNotice('');
    if (![name, institution, email, contact, purpose].every(value => value.trim())) {
      setError(t('Please complete your details and describe a bug or suggestion.', '请填写完整信息和 Bug 或建议。'));
      return;
    }
    window.location.href = allowanceEmailLink({
      name, institution, email, contact, feedback: purpose, conversations: amount, predictions,
    });
    setNotice(t('Complete and send the request in your email app.', '请在邮件客户端中确认并发送申请。'));
  }
  return <dialog ref={ref} className="access-dialog" aria-labelledby="quota-dialog-title" onCancel={() => setOpen(false)} onClose={() => setOpen(false)} onClick={e => { if (e.target === ref.current) setOpen(false); }}>
    <div className="access-dialog-inner">
      <header><h2 id="quota-dialog-title">{t('Extra allowance (Feedback)', '额外额度（反馈）')}</h2><button type="button" aria-label={t('Close', '关闭')} onClick={() => setOpen(false)}>×</button></header>
      <p>{t('Each visitor has 3 new conversations and 20 model predictions per day across the online service. Existing conversations can continue.', '线上每位访客每天拥有 3 个新对话和 20 次模型推理额度，各入口共享。已有对话可以继续。')}</p>
      {quota?.enforced && <div className="access-quota-line"><strong>{quota.remaining} / {quota.limit}</strong><span>{t('new conversations remaining today', '个新对话今日可用')}{quota.resets_at ? ` · ${t('Resets', '重置')} ${new Date(quota.resets_at).toLocaleString(zh ? 'zh-CN' : 'en-GB', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}` : ''}</span></div>}
      {quota?.enforced && quota.compute && <div className="access-quota-line">
        <strong>{quota.compute.prediction.remaining} / {quota.compute.prediction.limit}</strong>
        <span>{t('model predictions remaining today', '次模型推理今日可用')}</span>
      </div>}
      <div className="access-tabs"><button type="button" aria-pressed={tab === 'key'} onClick={() => { setTab('key'); setNotice(''); }}>{t('Use a key', '填写额度 key')}</button><button type="button" aria-pressed={tab === 'request'} onClick={() => { setTab('request'); setNotice(''); }}>{t('Request allowance', '申请额度')}</button></div>
      {tab === 'key' ? <form onSubmit={bind}>
        <label>{t('Email', '邮箱')}<input type="email" autoComplete="email" required value={email} onChange={e => setEmail(e.target.value)}/></label>
        <label>{t('Quota key', '额度 key')}<input type="password" autoComplete="off" required value={key} onChange={e => setKey(e.target.value)} placeholder="vfq_…"/></label>
        <p className="access-hint">{t('Saved in this browser. The same key shares its daily allowance across devices.', '保存在此浏览器；同一 key 在不同设备上共享每日额度。')}</p>
        <div className="access-actions"><button className="access-primary" disabled={busy}>{busy ? t('Checking…', '验证中…') : t('Save key', '保存并使用')}</button>{savedQuota().key && <button type="button" disabled={busy} onClick={async () => { try { const removed = await fetch("/api/quota/disconnect", {method:"POST",headers:{"X-Venus-Access":"1"}}); if (!removed.ok) throw new Error("Could not remove key"); saveQuota(null); setKey(''); setEmail(''); setError(''); setNotice(t('Key removed from this browser.', '已清除此浏览器保存的 key。')); void fetchQuota().then(setQuota); } catch { setError(t('Browser storage is unavailable.', '浏览器不允许保存设置。')); } }}>{t('Remove saved key', '清除保存')}</button>}</div>
      </form> : <form onSubmit={request}>
        <div className="access-form-grid"><label>{t('Name', '姓名')}<input required maxLength={100} value={name} onChange={e => setName(e.target.value)}/></label><label>{t('Institution', '单位')}<input required maxLength={200} value={institution} onChange={e => setInstitution(e.target.value)}/></label></div>
        <label>{t('Email', '邮箱')}<input type="email" required value={email} onChange={e => setEmail(e.target.value)}/></label>
        <label>{t('Contact', '联系方式')}<input required maxLength={200} value={contact} onChange={e => setContact(e.target.value)} placeholder={t('Phone or another contact method', '电话或其他联系方式')}/></label>
        <div className="access-form-grid"><label>{t('Extra conversations per day', '申请额外每日对话数')}<input type="number" min={0} max={10000} required value={amount} onChange={e => setAmount(Number(e.target.value))}/></label>
        <label>{t('Extra model predictions per day', '申请额外每日模型推理数')}<input type="number" min={0} max={10000} required value={predictions} onChange={e => setPredictions(Number(e.target.value))}/></label></div>
        <p className="access-hint">{t('More specific, useful feedback can improve your chances of approval and the allowance awarded.', '更多具体、有价值的 Bug 或建议，有助于提高额度申请的获批机会和数额。')}</p>
        <label>{t('Bugs or suggestions', 'Bug 或改进建议')}<textarea required maxLength={5000} rows={5} value={purpose} onChange={e => setPurpose(e.target.value)}/></label>
        <button className="access-primary" disabled={busy}>{t('Request by email','通过邮件申请')}</button><p className="access-hint">tanyang.august@sjtu.edu.cn</p>
      </form>}
      {error && <p role="alert" className="access-error">{error}</p>}{notice && <p role="status" className="access-notice">{notice}</p>}
    </div>
  </dialog>;
}
