import { useEffect, useState } from "react";
import { useLang } from "../lib/i18n";
type Row = { key: string; model: string; provider: string; free: boolean; enabled: boolean; can_publish: boolean; reason: string | null; revision: number };
type Catalog = { mode: string; items: Row[] };
export function AdminLLMs({ refreshKey = 0, query = "" }: { refreshKey?: number; query?: string }) {
  const { lang } = useLang(); const zh=lang==="zh";
  const [data,setData]=useState<Catalog>(), [busy,setBusy]=useState(""), [error,setError]=useState(""), [loading,setLoading]=useState(true);
  async function request(method="GET", body?: unknown) {
    const res=await fetch("/api/admin/llms",{method,cache:"no-store",credentials:"same-origin",
      headers:{"Content-Type":"application/json","X-Venus-Access":"1"},...(body ? {body:JSON.stringify(body)} : {})});
    if (!res.ok) throw new Error(res.status===409 ? (zh?"配置已变化，请刷新。":"Settings changed. Refresh and try again.") : (zh?"操作失败，请刷新重试。":"Could not update LLMs. Refresh and try again."));
    return res.json();
  }
  async function load() {
    setLoading(true); setError("");
    try {setData(await request());} catch(e) {setError((e as Error).message);} finally {setLoading(false);}
  }
  useEffect(()=>{let alive=true; setLoading(true); void request().then(d=>{if(alive){setData(d);setError("");}}).catch(e=>{if(alive)setError(e.message);}).finally(()=>{if(alive)setLoading(false);}); return()=>{alive=false;};},[refreshKey]);
  async function toggle(row: Row) {
    setBusy(row.key); setError("");
    try {await request("PUT",{key:row.key,enabled:!row.enabled,revision:row.revision}); await load();}
    catch(e) {setError((e as Error).message);} finally {setBusy("");}
  }
  const rows=(data?.items || []).filter(r=>(r.model+" "+r.provider+(r.free?" free":"")).toLowerCase().includes(query.trim().toLowerCase()));
  return <section className="prefs-llms" aria-busy={loading}>
    <div className="prefs-llms-heading"><div><h3>{zh?"模型上架":"LLM availability"}</h3><p>{data?.mode==="online" ? (zh?"线上环境 · 仅免费模型":"Online deployment · Free models only") : (zh?"本地环境":"Local deployment")}</p></div>
      <button type="button" className="prefs-button" disabled={loading || !!busy} onClick={()=>void load()}>{zh?"刷新模型":"Refresh LLMs"}</button></div>
    {error && <p role="alert" className="prefs-error">{error}</p>}
    <div className="prefs-llms-list">{rows.map(row=><div className="prefs-llm-row" key={row.key}>
      <div className="prefs-llm-name"><strong>{row.model}</strong><small>{row.provider}</small></div>
      <div className="prefs-llm-badges">{row.free && <span className="prefs-free">Free</span>}
        <span>{row.reason==="credential_unavailable" ? (zh?"待配置 API key":"API key required") : row.reason==="free_model_required" ? (zh?"仅限本地":"Local only") : row.enabled ? (zh?"已上架":"Published") : (zh?"未上架":"Unpublished")}</span></div>
      <button type="button" role="switch" aria-checked={row.enabled && row.can_publish} aria-label={`${zh?"上架":"Publish"} ${row.model}`}
        className="admin-cap-switch" disabled={loading || !!busy || !row.can_publish}
        onClick={()=>void toggle(row)}><span/></button>
    </div>)}</div>
    {!loading && !rows.length && <p className="prefs-empty">{zh?"暂无匹配模型。请配置 API key 和模型 ID。":"No matching LLMs. Configure an API key and model ID."}</p>}
  </section>;
}
