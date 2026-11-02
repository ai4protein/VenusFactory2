import { useEffect, useState } from 'react';
import { useLang } from '../lib/i18n';
import './admin-usage.css';

type Counters = { model_calls: number; tool_calls: number; input_tokens: number | null; output_tokens: number | null; total_tokens: number | null; unreported_calls: number; reported_calls: number; failed: number };
type Group = { name: string; provider?: string; harness?: string; calls: number; failed: number; reported_calls?: number; total_tokens?: number | null; latency_ms: number | null };
type Overview = { deployment: string; at: number; recording_since: number | null; summary: Counters; models: Group[]; tools: Group[] };
type Session = { session_id: string; title: string; mode: string; status: string; messages: number | null; runs: number; updated: number; model_calls?: number; tool_calls?: number; total_tokens?: number | null };
type Event = { id: string; created: number; kind: string; mode: string; session_id: string; run_id: string; name: string; provider: string; harness: string; status: string; latency_ms: number | null; input_tokens: number | null; output_tokens: number | null; total_tokens: number | null; error_type: string };
type Job = { id: string; kind: string; resource: string; status: string; created: number; mode: string; session_id: string; latency_ms: number | null };
type Page<T> = { items: T[]; total: number; active?: number; offset: number; limit: number };
type Api = <T>(path: string, method?: string, body?: unknown) => Promise<T>;

export function AdminUsage({ api }: { api: Api }) {
  const { lang } = useLang(); const zh = lang === 'zh'; const t = (en: string, cn: string) => zh ? cn : en;
  const [days,setDays]=useState(7), [mode,setMode]=useState(''), [session,setSession]=useState('');
  const [tab,setTab]=useState('overview'), [page,setPage]=useState(0), [refresh,setRefresh]=useState(0);
  const [overview,setOverview]=useState<Overview>(), [sessions,setSessions]=useState<Page<Session>>(), [events,setEvents]=useState<Page<Event>>();
  const [jobs,setJobs]=useState<Page<Job>>();
  const [loading,setLoading]=useState(true), [error,setError]=useState('');
  useEffect(() => {
    let alive=true; setLoading(true); setError('');
    const query=new URLSearchParams({days:String(days),mode,session_id:session});
    const offset=page*25;
    void Promise.all([
      api<Overview>('/usage?'+query),
      api<Page<Session>>('/usage/sessions?'+query+`&offset=${tab==='sessions'?offset:0}`),
      tab==='model' || tab==='tool' ? api<Page<Event>>('/usage/events?'+query+`&kind=${tab}&offset=${offset}`) : Promise.resolve(undefined),
      tab==='jobs' ? api<Page<Job>>('/usage/jobs?'+query+`&offset=${offset}`) : Promise.resolve(undefined),
    ]).then(([o,s,e,j])=>{ if(alive){setOverview(o);setSessions(s);setEvents(e);setJobs(j);} }).catch(e=>{if(alive)setError(e.message);}).finally(()=>{if(alive)setLoading(false);});
    return ()=>{alive=false;};
  },[api,days,mode,session,tab,page,refresh]);
  useEffect(()=>{const id=window.setInterval(()=>{if(document.visibilityState==='visible')setRefresh(n=>n+1);},30000);return()=>clearInterval(id);},[]);
  const number=(n: number | null | undefined)=> n==null ? '—' : n.toLocaleString(zh?'zh-CN':'en-US');
  const time=(n: number)=>new Date(n*1000).toLocaleString(zh?'zh-CN':'en-GB');
  const duration=(n: number | null)=>n==null?'—':`${(n/1000).toFixed(1)} s`;
  const labels: Record<string,string>={running:t('Running','运行中'),queued:t('Queued','排队中'),completed:t('Completed','已完成'),failed:t('Failed','失败'),cancelled:t('Cancelled','已取消'),interrupted:t('Interrupted','已中断'),paused:t('Paused','已暂停'),waiting_for_input:t('Waiting','等待输入'),stopped:t('Stopped','已停止')};
  const status=(value:string)=><span className={`usage-status usage-status--${['running','queued','completed','failed','interrupted'].includes(value)?value:'other'}`}>{labels[value]||value||'—'}</span>;
  const inspect=(id:string)=>{setSession(id);setTab('model');setPage(0);};
  const pagerTotal=tab==='sessions'?sessions?.total:tab==='jobs'?jobs?.total:events?.total;
  const groups=(rows:Group[],models:boolean)=><div className="access-admin-table"><table><thead><tr><th>{models?t('Model','模型'):t('Tool','工具')}</th><th>{t('Calls','调用次数')}</th><th>{t('Failures','失败')}</th>{models&&<th>Tokens</th>}<th>{t('Avg. duration','平均耗时')}</th></tr></thead><tbody>{rows.map((r,i)=><tr key={`${r.name}-${i}`}><td>{r.name}{models&&<small>{[r.provider,r.harness].filter(Boolean).join(' · ')}</small>}</td><td>{number(r.calls)}</td><td>{number(r.failed)}</td>{models&&<td>{number(r.total_tokens)}<small>{r.reported_calls}/{r.calls} {t('reported','已上报')}</small></td>}<td>{duration(r.latency_ms)}</td></tr>)}</tbody></table>{!rows.length&&<p className="access-admin-empty">{t('No records in this period','此时间范围内暂无记录')}</p>}</div>;
  return <section className="admin-usage" aria-label={t('Platform usage','平台用量')} aria-busy={loading}>
    <div className="usage-filters"><label>{t('Period','时间范围')}<select value={days} onChange={e=>{setDays(Number(e.target.value));setPage(0);}}>{[1,7,30,90].map(n=><option key={n} value={n}>{zh?`最近 ${n} 天`:`Last ${n} ${n===1?'day':'days'}`}</option>)}</select></label><label>{t('Mode','模式')}<select value={mode} onChange={e=>{setMode(e.target.value);setPage(0);}}><option value="">{t('All modes','所有模式')}</option>{['agent','scientist','researcher','tools'].map(m=><option key={m} value={m}>{m==='tools'?t('Other tools','其他工具'):m[0].toUpperCase()+m.slice(1)}</option>)}</select></label><button disabled={loading} onClick={()=>setRefresh(n=>n+1)}>{loading?t('Updating…','更新中…'):t('Refresh','刷新')}</button><span className="usage-updated">{overview?(overview.deployment==='online'?'Online':'Local'):''}{overview&&` · ${t('Updated','更新于')} ${time(overview.at)}`}</span></div>
    {session&&<div className="usage-session-filter"><span>{t('Conversation','会话')} <code>{session}</code></span><button onClick={()=>{setSession('');setPage(0);}} aria-label={t('Clear conversation filter','清除会话筛选')}>×</button></div>}
    {error&&<p role="alert" className="access-error">{error}</p>}
    {overview&&<><div className="usage-cards">
      <article><span>{t('Reported tokens','已上报 Token')}</span><strong>{number(overview.summary.total_tokens)}</strong><small>{t('Input','输入')} {number(overview.summary.input_tokens)} · {t('Output','输出')} {number(overview.summary.output_tokens)}</small></article>
      <article><span>{t('Model requests','模型请求')}</span><strong>{number(overview.summary.model_calls||0)}</strong><small>{number(overview.summary.unreported_calls||0)} {t('without token counts','次未上报 Token')}</small></article>
      <article><span>{t('Conversations','会话')}</span><strong>{number(sessions?.total)}</strong><small>{number(sessions?.active||0)} {t('active','个运行中')}</small></article>
      <article><span>{t('Tool calls','工具调用')}</span><strong>{number(overview.summary.tool_calls||0)}</strong><small>{number(overview.summary.failed||0)} {t('failed requests / calls','次请求／调用失败')}</small></article>
    </div><p className="usage-note">{overview.recording_since ? t('Usage recording since ','用量记录始于 ')+time(overview.recording_since) : t('Usage will appear after the next model request or tool call.','下一次模型请求或工具调用后开始显示用量。')} · {t('— means not reported. Provider totals are used without adding cache or reasoning breakdowns again.','— 表示未上报。采用供应商总量，不重复累加缓存或推理明细。')}</p></>}
    <nav className="usage-tabs" aria-label={t('Usage views','用量视图')}>{[['overview',t('Overview','概览')],['sessions',t('Conversations','会话')],['model',t('Model requests','模型请求')],['tool',t('Tool calls','工具调用')],['jobs',t('Queue history','排队任务')]].map(([id,label])=><button key={id} aria-pressed={tab===id} onClick={()=>{setTab(id);setPage(0);}}>{label}</button>)}</nav>
    {tab==='overview'&&overview&&<div className="usage-breakdown"><div><h2>{t('Models · top 100','模型 · 前 100 项')}</h2>{groups(overview.models,true)}</div><div><h2>{t('Tools · top 100','工具 · 前 100 项')}</h2>{groups(overview.tools,false)}</div></div>}
    {tab==='sessions'&&sessions&&<div className="access-admin-table"><table><thead><tr>{[t('Conversation','会话'),t('Mode','模式'),t('Status','状态'),t('Messages / runs','消息／运行次数'),'Tokens',t('Tools','工具'),t('Updated','更新时间')].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{sessions.items.map(s=><tr key={s.session_id}><td><button className="usage-session-link" onClick={()=>inspect(s.session_id)}>{s.title||t('Conversation','会话')}</button><small><code>{s.session_id}</code></small></td><td>{s.mode}</td><td>{status(s.status)}</td><td>{number(s.messages)} / {s.runs}</td><td>{number(s.total_tokens)}</td><td>{number(s.tool_calls||0)}</td><td>{time(s.updated)}</td></tr>)}</tbody></table>{!sessions.items.length&&<p className="access-admin-empty">{t('No conversations in this period','此时间范围内暂无会话')}</p>}</div>}
    {(tab==='model'||tab==='tool')&&events&&<div className="access-admin-table"><table><thead><tr>{[t('Time','时间'),tab==='model'?t('Model','模型'):t('Tool','工具'),t('Mode / harness','模式／Harness'),t('Status','状态'),t('Duration','耗时'),...(tab==='model'?[t('Input / output','输入／输出'),'Tokens']:[]),t('Conversation / run','会话／运行编号')].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{events.items.map(e=><tr key={e.id}><td>{time(e.created)}</td><td>{e.name}<small>{e.provider}</small></td><td>{e.mode}<small>{e.harness}</small></td><td>{status(e.status)}<small>{e.error_type}</small></td><td>{duration(e.latency_ms)}</td>{tab==='model'&&<><td>{number(e.input_tokens)} / {number(e.output_tokens)}</td><td>{number(e.total_tokens)}</td></>}<td>{e.session_id&&<button className="usage-session-link" onClick={()=>inspect(e.session_id)}><code>{e.session_id}</code></button>}<small><code>{e.run_id||'—'}</code></small></td></tr>)}</tbody></table>{!events.items.length&&<p className="access-admin-empty">{t('No records in this period','此时间范围内暂无记录')}</p>}</div>}
    {tab==='jobs'&&jobs&&<div className="access-admin-table"><table><thead><tr>{[t('Created','提交时间'),t('Task','任务'),t('Resource','资源'),t('Status','状态'),t('Duration','耗时'),t('Conversation','会话')].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{jobs.items.map(j=><tr key={j.id}><td>{time(j.created)}</td><td>{j.kind}<small><code>{j.id}</code></small></td><td>{j.resource}</td><td>{status(j.status)}</td><td>{duration(j.latency_ms)}</td><td>{j.mode}{j.session_id&&<small><button className="usage-session-link" onClick={()=>inspect(j.session_id)}><code>{j.session_id}</code></button></small>}</td></tr>)}</tbody></table>{!jobs.items.length&&<p className="access-admin-empty">{t('No queued tasks in this period','此时间范围内暂无排队任务')}</p>}</div>}
    {tab!=='overview'&&<div className="usage-pagination"><span>{number(pagerTotal||0)} {t('records','条记录')}</span><button disabled={loading||page===0} onClick={()=>setPage(n=>n-1)}>{t('Previous','上一页')}</button><span>{page+1}</span><button disabled={loading||(page+1)*25>=(pagerTotal||0)} onClick={()=>setPage(n=>n+1)}>{t('Next','下一页')}</button></div>}
  </section>;
}
