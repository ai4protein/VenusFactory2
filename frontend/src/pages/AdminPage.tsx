import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import '../components/access.css';
import { AdminCapabilities } from '../components/AdminCapabilities';
import { AdminUsage } from '../components/AdminUsage';
import { AdminAllowances } from '../components/AdminAllowances';
import { SettingsPage } from './SettingsPage';

type Admin = { id: string; email: string; active?: boolean; created?: number };
type Grant = { id: string; email: string; extra: number; extra_predictions: number; active: boolean; expires: number | null; name: string; institution: string; contact: string; note: string; used_today: number };
type GrantForm = Omit<Grant, 'id' | 'used_today'> & { id?: string };
type AdminForm = { id?: string; email: string; password: string; active: boolean };
const emptyGrant = (): GrantForm => ({ email: '', extra: 3, extra_predictions: 0, active: true, expires: null, name: '', institution: '', contact: '', note: '' });
const adminTabs = ['usage', 'capabilities', 'environment', 'grants', 'admins', 'audit'] as const;
type AdminTab = typeof adminTabs[number];
async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch('/api/admin' + path, { method, cache: 'no-store', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-Venus-Access': '1' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401 && path !== '/login') window.dispatchEvent(new Event('vf:admin-expired'));
    throw new Error(typeof result.detail === 'string' ? result.detail : result.detail?.message || `Request failed (${response.status})`);
  }
  return result;
}
export function AdminPage() {
  const { lang } = useLang(); const zh = lang === 'zh'; const t = (a: string, b: string) => zh ? b : a;
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab') as AdminTab;
  const tab = adminTabs.includes(requestedTab) ? requestedTab : 'usage';
  const setTab = (next: AdminTab) => setSearchParams({ tab: next }, { replace: true });
  const backToChat = <Link className="access-admin-back" to={`/${lang}/agent/chat`}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m12 5-7 7 7 7M5 12h14"/></svg><span>{t('Back to chat', '返回对话')}</span></Link>;
  const [usageRevision, setUsageRevision] = useState(0);
  const refreshUsage = () => setUsageRevision(n => n + 1);
  const [me, setMe] = useState<Admin>(), [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [address, setAddress] = useState(''), [password, setPassword] = useState('');
  const [grants, setGrants] = useState<Grant[]>([]), [admins, setAdmins] = useState<Admin[]>([]);
  const [audit, setAudit] = useState<{ id: number; at: number; actor: string; action: string; target: string }[]>([]);
  const [query, setQuery] = useState(''), [grantForm, setGrantForm] = useState<GrantForm>(), [adminForm, setAdminForm] = useState<AdminForm>();
  const [issued, setIssued] = useState<{ email: string; key: string }>(), [confirm, setConfirm] = useState<{ kind: 'grant' | 'admin'; id: string; email: string }>();
  useEffect(() => {
    const expired = () => { setMe(undefined); setIssued(undefined); setGrants([]); setAdmins([]); setAudit([]); };
    window.addEventListener('vf:admin-expired', expired);
    void api<Admin>('/me').then(setMe).catch(() => {}).finally(() => setLoading(false));
    return () => window.removeEventListener('vf:admin-expired', expired);
  }, []);
  async function refresh() {
    const [g, a, log] = await Promise.all([api<{ items: Grant[] }>('/grants'), api<{ items: Admin[] }>('/admins'), api<{ items: typeof audit }>('/audit')]);
    setGrants(g.items); setAdmins(a.items); setAudit(log.items);
  }
  useEffect(() => { if (me) void refresh().catch(e => setError(e.message)); }, [me?.id]);
  async function act(fn: () => Promise<void>) { setBusy(true); setError(''); try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }
  const date = (value: number | null | undefined) => value ? new Date(value * 1000).toLocaleString(zh ? 'zh-CN' : 'en-GB') : t('No expiry', '长期有效');
  if (loading) return <main className="access-admin" aria-busy="true">{backToChat}<p>{t('Loading…', '加载中…')}</p></main>;
  if (!me) return <main className="access-admin">{backToChat}<section className="access-admin-login"><h1>{t('Administration', '管理后台')}</h1><p>VenusFactory</p><form onSubmit={e => { e.preventDefault(); void act(async () => { const user = await api<Admin>('/login', 'POST', { email: address, password }); setPassword(''); setMe(user); }); }}>
    <label>{t('Email', '邮箱')}<input type="email" autoComplete="username" value={address} required onChange={e => setAddress(e.target.value)}/></label>
    <label>{t('Password', '密码')}<input type="password" autoComplete="current-password" value={password} required onChange={e => setPassword(e.target.value)}/></label>
    <button className="access-primary" disabled={busy}>{t('Sign in', '登录')}</button>{error && <p className="access-error" role="alert">{error}</p>}
  </form></section></main>;
  const rows = grants.filter(g => [g.email, g.name, g.institution].join(' ').toLowerCase().includes(query.toLowerCase()));
  return <main className="access-admin">
    {backToChat}
    <header><div><h1>{t('Administration', '管理后台')}</h1><p>{me.email}</p></div><button onClick={() => void act(async () => { await api('/logout', 'POST', {}); setMe(undefined); setIssued(undefined); })}>{t('Sign out', '退出登录')}</button></header>
    <nav className="access-tabs">{adminTabs.map(key => <button key={key} aria-pressed={tab === key} onClick={() => { setTab(key); setGrantForm(undefined); setAdminForm(undefined); setConfirm(undefined); }}>{key === 'usage' ? t('Usage & runs', '用量与运行') : key === 'capabilities' ? t('Tools & models', '工具与模型') : key === 'environment' ? t('Environment', '环境配置') : key === 'grants' ? t('Quota keys', '额度 key') : key === 'admins' ? t('Administrators', '管理员') : t('Activity', '操作记录')}</button>)}</nav>
    {error && <p className="access-error" role="alert">{error}</p>}
    {issued && <div className="access-key" role="status"><strong>{t('New key', '新额度 key')} · {issued.email}</strong><code>{issued.key}</code><div className="access-actions"><button onClick={() => { void navigator.clipboard.writeText(issued.key).catch(() => setError(t('Select the key to copy it.', '请选择上方 key 手动复制。'))); }}>{t('Copy', '复制')}</button><button onClick={() => setIssued(undefined)}>{t('Done', '完成')}</button><span>{t('Copy now; it will not be shown again.', '请复制保存，关闭后不再显示。')}</span></div></div>}
    {confirm && <section className="access-editor"><p>{confirm.kind === 'grant' ? t('Revoke this key?', '停用并撤销此 key？') : t('Delete this administrator?', '删除此管理员？')} <strong>{confirm.email}</strong></p><div className="access-actions"><button className="access-danger" disabled={busy} onClick={() => void act(async () => { await api(`/${confirm.kind === 'grant' ? 'grants' : 'admins'}/${confirm.id}`, 'DELETE'); setConfirm(undefined); await refresh(); })}>{t('Confirm', '确认')}</button><button onClick={() => setConfirm(undefined)}>{t('Cancel', '取消')}</button></div></section>}
    {tab === 'capabilities' && <AdminCapabilities api={api}/>}
    {tab === 'usage' && <><AdminAllowances api={api} refreshKey={usageRevision} onRefresh={refreshUsage}/><AdminUsage api={api} refreshKey={usageRevision} onRefresh={refreshUsage}/></>}
    {tab === 'environment' && <SettingsPage admin/>}
    {tab === 'grants' && <>
      <AdminAllowances api={api} refreshKey={usageRevision} onRefresh={refreshUsage}/>
      <div className="access-admin-top"><input aria-label={t('Search quota keys', '搜索额度 key')} value={query} onChange={e => setQuery(e.target.value)} placeholder={t('Search email, name or institution', '搜索邮箱、姓名或单位')}/><button className="access-primary" onClick={() => { setGrantForm(emptyGrant()); setIssued(undefined); }}>{t('Create key', '生成 key')}</button></div>
      {grantForm && <section className="access-editor"><h2>{grantForm.id ? t('Edit allowance', '修改额度') : t('Create allowance', '创建额度')}</h2><form onSubmit={e => { e.preventDefault(); void act(async () => { const { id, ...payload } = grantForm; const result = await api<{ email: string; key: string | null }>('/grants' + (id ? '/' + id : ''), id ? 'PUT' : 'POST', payload); if (result.key) setIssued({ email: result.email, key: result.key }); setGrantForm(undefined); await refresh(); }); }}>
        <label>{t('Email', '邮箱')}<input type="email" required value={grantForm.email} onChange={e => setGrantForm({ ...grantForm, email: e.target.value })}/></label>
        <div className="access-form-grid"><label>{t('Extra conversations per day', '额外每日对话数')}<input type="number" min={0} max={10000} required value={grantForm.extra} onChange={e => setGrantForm({ ...grantForm, extra: Number(e.target.value) })}/></label><label>{t('Extra predictions per day', '额外每日模型推理数')}<input type="number" min={0} max={10000} required value={grantForm.extra_predictions} onChange={e => setGrantForm({...grantForm,extra_predictions:Number(e.target.value)})}/></label><label>{t('Expiry date (optional)', '到期时间（可选）')}<input type="datetime-local" value={grantForm.expires ? new Date(grantForm.expires * 1000 - new Date(grantForm.expires * 1000).getTimezoneOffset() * 60000).toISOString().slice(0, 16) : ''} onChange={e => setGrantForm({ ...grantForm, expires: e.target.value ? new Date(e.target.value).getTime() / 1000 : null })}/></label></div>
        {(['name', 'institution', 'contact', 'note'] as const).map((key, i) => <label key={key}>{[t('Name', '姓名'), t('Institution', '单位'), t('Contact', '联系方式'), t('Note', '备注')][i]}<input value={grantForm[key]} maxLength={key === 'note' ? 2000 : 200} onChange={e => setGrantForm({ ...grantForm, [key]: e.target.value })}/></label>)}
        <label>{t('Status', '状态')}<select value={grantForm.active ? '1' : '0'} onChange={e => setGrantForm({ ...grantForm, active: e.target.value === '1' })}><option value="1">{t('Active', '启用')}</option><option value="0">{t('Disabled', '停用')}</option></select></label>
        <p className="access-hint">{t(`Daily total: 3 free + ${grantForm.extra} extra.`, `每日合计：3 个免费 + ${grantForm.extra} 个额外额度。`)}</p><div className="access-actions"><button className="access-primary" disabled={busy}>{t('Save', '保存')}</button><button type="button" onClick={() => setGrantForm(undefined)}>{t('Cancel', '取消')}</button>{grantForm.id && <button type="button" disabled={busy} onClick={() => void act(async () => { const { id, ...payload } = grantForm; const result = await api<{ email: string; key: string }>(`/grants/${id}/rotate`, 'POST', payload); setIssued(result); setGrantForm(undefined); await refresh(); })}>{t('Replace key', '重发 key')}</button>}</div>
      </form></section>}
      <div className="access-admin-table"><table><thead><tr>{[t('Recipient', '申请人'), t('Allowance / day', '每日额度'), t('Used today', '今日已用'), t('Expires', '到期'), t('Status', '状态'), ''].map((x, i) => <th key={i}>{x}</th>)}</tr></thead><tbody>{rows.map(g => <tr key={g.id} className={g.active ? '' : 'access-muted'}><td>{g.email}<small>{[g.name, g.institution].filter(Boolean).join(' · ')}</small></td><td>{t('Chats','对话')}: +{g.extra}<small>{t('Predictions','模型推理')}: +{g.extra_predictions}</small></td><td>{g.used_today}</td><td>{date(g.expires)}</td><td>{!g.active ? t('Disabled', '已停用') : g.expires && g.expires < Date.now() / 1000 ? t('Expired', '已到期') : t('Active', '使用中')}</td><td><button onClick={() => { const { used_today: _, ...form } = g; setGrantForm({ ...form, active: Boolean(g.active) }); setIssued(undefined); }}>{t('Edit', '编辑')}</button><button className="access-danger" onClick={() => setConfirm({ kind: 'grant', id: g.id, email: g.email })}>{t('Revoke', '撤销')}</button></td></tr>)}</tbody></table>{!rows.length && <p className="access-admin-empty">{t('No quota keys yet', '暂无额度 key')}</p>}</div>
    </>}
    {tab === 'admins' && <>
      <div className="access-admin-top"><span>{admins.length} {t('administrators', '位管理员')}</span><button className="access-primary" onClick={() => setAdminForm({ email: '', password: '', active: true })}>{t('Add administrator', '添加管理员')}</button></div>
      {adminForm && <section className="access-editor"><h2>{t('Administrator', '管理员')}</h2><form onSubmit={e => { e.preventDefault(); void act(async () => { const { id, ...payload } = adminForm; await api('/admins' + (id ? '/' + id : ''), id ? 'PUT' : 'POST', payload); setAdminForm(undefined); if (id === me.id) { setMe(undefined); return; } await refresh(); }); }}>
        <label>{t('Email', '邮箱')}<input type="email" required value={adminForm.email} onChange={e => setAdminForm({ ...adminForm, email: e.target.value })}/></label>
        <label>{adminForm.id ? t('New password (leave blank to keep)', '新密码（留空保留）') : t('Password', '密码')}<input type="password" autoComplete="new-password" minLength={8} maxLength={256} required={!adminForm.id} value={adminForm.password} onChange={e => setAdminForm({ ...adminForm, password: e.target.value })}/></label>
        <label>{t('Status', '状态')}<select value={adminForm.active ? '1' : '0'} disabled={adminForm.id === me.id} onChange={e => setAdminForm({ ...adminForm, active: e.target.value === '1' })}><option value="1">{t('Active', '启用')}</option><option value="0">{t('Disabled', '停用')}</option></select></label>
        <div className="access-actions"><button className="access-primary" disabled={busy}>{t('Save', '保存')}</button><button type="button" onClick={() => setAdminForm(undefined)}>{t('Cancel', '取消')}</button></div>
      </form></section>}
      <div className="access-admin-table"><table><thead><tr><th>{t('Email', '邮箱')}</th><th>{t('Status', '状态')}</th><th>{t('Created', '创建时间')}</th><th/></tr></thead><tbody>{admins.map(a => <tr key={a.id}><td>{a.email}{a.id === me.id && <small>{t('You', '当前账户')}</small>}</td><td>{a.active ? t('Active', '启用') : t('Disabled', '停用')}</td><td>{date(a.created)}</td><td><button onClick={() => setAdminForm({ id: a.id, email: a.email, password: '', active: Boolean(a.active) })}>{t('Edit', '编辑')}</button><button className="access-danger" disabled={a.id === me.id} onClick={() => setConfirm({ kind: 'admin', id: a.id, email: a.email })}>{t('Delete', '删除')}</button></td></tr>)}</tbody></table></div>
    </>}
    {tab === 'audit' && <div className="access-admin-table"><table><thead><tr>{[t('Time', '时间'), t('Administrator', '管理员'), t('Action', '操作'), t('Record', '对象')].map(x => <th key={x}>{x}</th>)}</tr></thead><tbody>{audit.map(a => <tr key={a.id}><td>{date(a.at)}</td><td>{a.actor}</td><td>{a.action}</td><td>{a.target}</td></tr>)}</tbody></table></div>}
  </main>;
}
