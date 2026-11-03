import { useEffect, useState } from 'react';
import { useLang } from '../lib/i18n';
export type Feedback = {id:string;created:number;email:string;name:string;institution:string;contact:string;feedback:string;conversations:number;predictions:number};
type Overview = {day:string;visitors_today:number;visits_today:number;visitors_total:number;recent_visitors:number;conversations_used:number;predictions_used:number;workflows_used:number;active_grants:number;last_reset:number;mode_calls:Record<string,{model_calls:number;tool_calls:number;tokens:number|null}>;modes:Record<string,number>};
type Props = {api:<T>(path:string,method?:string,body?:unknown)=>Promise<T>;onReview?:(request:Feedback)=>void};
export function AdminAllowances({api,onReview}:Props) {
 const {lang}=useLang();const t=(en:string,zh:string)=>lang==='zh'?zh:en;
 const [data,setData]=useState<{overview:Overview;requests:Feedback[]}>();
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[confirm,setConfirm]=useState(false);
 const refresh=()=>api<{overview:Overview;requests:Feedback[]}>('/allowances').then(value=>{setData(value);setError('');}).catch(e=>setError(e.message));
 useEffect(()=>{void refresh();const timer=setInterval(()=>{if(!document.hidden)void refresh();},15000);return()=>clearInterval(timer);},[]);
 const s=data?.overview;
 return <section className="admin-allowances">
 <div className="access-admin-top"><div><h2>{t('Visitors & allowances','访客与额度')}</h2><small>{s?.day} · {t('Current deployment','当前部署')}</small></div>
 <button className="access-danger allowance-reset" onClick={()=>setConfirm(true)} disabled={busy}>{t('Reset all allowances','重置全部额度')}</button></div>
 {confirm&&<div className="access-editor" role="alert"><p>{t('Restore today’s conversation, prediction and Workflow allowances for every visitor in this deployment? History and running jobs are preserved.','恢复当前部署所有访客今日的对话、模型推理和 Workflow 额度？历史记录和运行任务将保留。')}</p>
 <div className="access-actions"><button className="access-danger" disabled={busy} onClick={async()=>{setBusy(true);setError('');try{await api('/allowances/reset','POST',{confirm:'RESET'});setConfirm(false);await refresh();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}}>{t('Confirm reset','确认重置')}</button><button disabled={busy} onClick={()=>setConfirm(false)}>{t('Cancel','取消')}</button></div></div>}
 {error&&<p role="alert" className="access-error">{error}</p>}
 {s&&<><div className="allowance-metrics">{[
 [t('Visitors today','今日访客'),s.visitors_today],[t('Recorded visitors','累计记录访客'),s.visitors_total],
 [t('Page visits today','今日页面访问'),s.visits_today],[t('Seen in last 5 min','近 5 分钟访问'),s.recent_visitors],
 [t('Conversation credits used','已用对话额度'),s.conversations_used],[t('Prediction credits used','已用模型推理额度'),s.predictions_used],
 [t('Workflow credits used','已用 Workflow 额度'),s.workflows_used],[t('Active allowance keys','有效额度 key'),s.active_grants]
 ].map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
 <div className="allowance-modes">{['agent','scientist','researcher'].map(mode=><div key={mode}><strong>{mode[0].toUpperCase()+mode.slice(1)}</strong><span>{s.modes[mode]||0} {t('conversations started today','个今日新对话')}</span><small>LLM {s.mode_calls[mode]?.model_calls || 0} · {t('Tools','工具')} {s.mode_calls[mode]?.tool_calls || 0} · {t('Reported tokens','已报告 tokens')} {s.mode_calls[mode]?.tokens ?? '—'}</small></div>)}</div>
 {s.last_reset>0&&<p className="access-hint">{t('Last reset','上次重置')} · {new Date(s.last_reset*1000).toLocaleString()}</p>}</>}
 {onReview&&<><h3>{t('Feedback & requests','反馈与申请')}</h3>{data?.requests.length ? data.requests.map(request=><article className="allowance-feedback" key={request.id}>
 <header><strong>{request.name} · {request.institution}</strong><small>{new Date(request.created*1000).toLocaleString()}</small></header>
 <p>{request.email} · {request.contact}</p><p style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{request.feedback}</p>
 <div className="access-actions"><span>{t('Requested per day','申请每日额外额度')}: {request.conversations} {t('chats','对话')} · {request.predictions} {t('predictions','模型推理')}</span>
 <button onClick={()=>onReview(request)}>{t('Review allowance','审核额度')}</button></div></article>):<p>{t('No feedback requests yet.','暂无反馈申请。')}</p>}</>}
 </section>;
}
