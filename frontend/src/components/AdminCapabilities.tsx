import { useEffect, useMemo, useState } from 'react';
import { useLang } from '../lib/i18n';
import { categoryNames } from '../lib/modelPlatform';

type Row = { key: string; name: string; kind: string; category: string; category_label?: string; category_label_zh?: string; description: string; enabled: boolean; available: boolean; reason?: string; revision: number };
type Catalogue = { mode: string; items: Row[]; unavailable_sources: string[] };
type API = <T>(path: string, method?: string, body?: unknown) => Promise<T>;

export function AdminCapabilities({ api }: { api: API }) {
  const { lang } = useLang(); const zh = lang === 'zh';
  const [data,setData] = useState<Catalogue>(), [query,setQuery] = useState(''), [kind,setKind] = useState(''), [category,setCategory] = useState(''), [state,setState] = useState('');
  const [page,setPage] = useState(0), [busy,setBusy] = useState(''), [loading,setLoading] = useState(true), [error,setError] = useState('');
  async function load() { setLoading(true); try { setData(await api<Catalogue>('/capabilities')); setError(''); } catch(e) {setError((e as Error).message);} finally {setLoading(false);} }
  useEffect(() => { void load(); }, []);
  useEffect(() => setPage(0), [query,kind,category,state]);
  const label = (row: Row) => categoryNames[row.category]?.[zh?1:0] || (zh ? row.category_label_zh : row.category_label) || row.category;
  const categories = useMemo(() => [...new Map((data?.items || []).filter(r=>!kind || r.kind===kind).map(r=>[r.category,label(r)])).entries()].sort((a,b)=>a[1].localeCompare(b[1])), [data,kind,zh]);
  const rows = useMemo(() => (data?.items || []).filter(r=>(!kind || r.kind===kind) && (!category || r.category===category) && (!state || (state==='enabled' ? r.enabled : !r.enabled)) && `${r.name} ${r.description}`.toLowerCase().includes(query.trim().toLowerCase())), [data,kind,category,state,query]);
  async function toggle(row: Row) {
    setBusy(row.key); setError('');
    try {
      await api('/capabilities','PUT',{key:row.key,enabled:!row.enabled,revision:row.revision});
      // Refresh dependent tools/model variants together.
      await load();
    } catch(e) {setError((e as Error).message);} finally {setBusy('');}
  }
  return <section className="admin-capabilities" aria-busy={loading}>
    <div className="admin-cap-heading"><div><h2>{zh?'工具与模型':'Tools & models'}</h2><p>{data?.mode==='online' ? (zh?'线上环境':'Online deployment') : (zh?'本地环境':'Local deployment')} · {zh?'开关即时生效':'Changes apply immediately'}</p></div><button disabled={loading || !!busy} onClick={()=>void load()}>{zh?'刷新':'Refresh'}</button></div>
    <div className="admin-cap-filters"><input type="search" aria-label={zh?'搜索工具和模型':'Search tools and models'} placeholder={zh?'搜索名称或功能…':'Search name or function…'} value={query} onChange={e=>setQuery(e.target.value)}/><select aria-label={zh?'类型':'Type'} value={kind} onChange={e=>{setKind(e.target.value);setCategory('');}}><option value="">{zh?'所有类型':'All types'}</option><option value="tool">MCP {zh?'工具':'tools'}</option><option value="model">{zh?'模型':'Models'}</option></select><select aria-label={zh?'分类':'Category'} value={category} onChange={e=>setCategory(e.target.value)}><option value="">{zh?'所有功能':'All functions'}</option>{categories.map(([key,value])=><option key={key} value={key}>{value}</option>)}</select><select aria-label={zh?'开关状态':'Switch status'} value={state} onChange={e=>setState(e.target.value)}><option value="">{zh?'所有状态':'All states'}</option><option value="enabled">{zh?'已开启':'Enabled'}</option><option value="disabled">{zh?'已关闭':'Disabled'}</option></select></div>
    {error && <p className="access-error" role="alert">{error}</p>}
    {!!data?.unavailable_sources.length && <p role="status">{zh?'部分注册来源暂未加载，请刷新。':'Some registry sources could not be loaded. Refresh to retry.'}</p>}
    <div className="admin-cap-count">{rows.length} {zh?'项':'items'}</div>
    <div className="admin-cap-list">{rows.slice(page*40,(page+1)*40).map(row=><div className="admin-cap-row" key={row.key}>
      <div className="admin-cap-name"><strong>{row.name}</strong><small>{label(row)} · {row.kind==='tool'?'MCP':(zh?'模型':'Model')}</small></div>
      <span className={`admin-cap-state${row.available?' is-ready':''}`}>{!row.enabled ? (zh?'已关闭':'Disabled') : row.available ? (zh?'可用':'Available') : row.reason==='not_configured' ? (zh?'待配置':'Setup required') : (zh?'受运行策略限制':'Restricted by runtime policy')}</span>
      <button type="button" role="switch" aria-checked={row.enabled} aria-label={`${zh?'启用':'Enable'} ${row.name}`} className="admin-cap-switch" disabled={!!busy || loading} onClick={()=>void toggle(row)}><span/></button>
    </div>)}</div>
    {!loading && !rows.length && <p className="access-admin-empty">{zh?'没有匹配的工具或模型':'No matching tools or models'}</p>}
    {rows.length>40 && <div className="admin-cap-pages"><button disabled={page===0} onClick={()=>setPage(v=>v-1)}>← {zh?'上一页':'Previous'}</button><span>{page+1} / {Math.ceil(rows.length/40)}</span><button disabled={(page+1)*40>=rows.length} onClick={()=>setPage(v=>v+1)}>{zh?'下一页':'Next'} →</button></div>}
  </section>;
}
