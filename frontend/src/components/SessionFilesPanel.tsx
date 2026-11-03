import { useCallback, useEffect, useId, useMemo, useRef, useState, type ReactNode, type CSSProperties } from "react";
import { useLang } from "../lib/i18n";
import { FilePreviewModal, type PreviewableFile } from "./FilePreviewModal";
import "./SessionFilesPanel.css";
import { MolstarViewer } from "./MolstarViewer";
import { isStructureFile, matchSessionFile, structureExtension, structurePathFromLink, type StructureRequest } from "../lib/structurePreview";
import { shareRead } from "../lib/readRequests";

interface SessionFile { key: string; name: string; rel: string; scope: string; size: number; mtime: number; ext: string }
interface Props { sessionId: string; authHeaders?: Record<string, string>; liveRefresh?: boolean; pollMs?: number; quota?: ReactNode; actions?: ReactNode }
const kinds: Record<string, string[]> = {
  structure: ["pdb", "cif", "mmcif", "ent", "sdf", "mol", "mol2"],
  data: ["csv", "tsv", "json", "jsonl", "xlsx", "xls"],
  sequence: ["fasta", "fa", "faa", "fna", "ffn", "frn"],
  document: ["pdf", "txt", "md", "markdown", "html", "log"],
  image: ["png", "jpg", "jpeg", "svg", "gif", "webp", "bmp", "tif", "tiff"],
  code: ["py", "r", "sh", "js", "ts", "tsx", "jsx", "css"],
};
const formatSize = (n: number) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
export function openResearchFile(goalId: string, path: string) {
  window.dispatchEvent(new CustomEvent("venus:preview-file", { detail: { key: `research-${goalId}/${path}` } }));
}

function FileIcon({ name }: { name: "folder" | "refresh" | "close" | "download" }) {
  const paths = {
    folder: "M3 7V5a1 1 0 0 1 1-1h6l2 3h8a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1Z",
    refresh: "M20 7v5h-5M19 12a7 7 0 1 0-2 5M20 7l-3-3",
    close: "m6 6 12 12M18 6 6 18",
    download: "M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4",
  };
  return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]}/></svg>;
}

export function SessionFilesPanel({ sessionId, authHeaders, liveRefresh = false, pollMs = 5000, quota, actions }: Props) {
  const { lang } = useLang();
  const zh = lang === "zh";
  const panelId = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  const selectAll = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState<SessionFile[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [opening, setOpening] = useState("");
  const [selecting, setSelecting] = useState(false);
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [date, setDate] = useState("");
  const [order, setOrder] = useState("newest");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<PreviewableFile | null>(null);
  const [structureWidth, setStructureWidth] = useState(620);
  const [wide, setWide] = useState(false);
  const structure = preview && isStructureFile(`file.${preview.ext}`) ? preview : null;
  const requestVersion = useRef(0);
  const [truncated, setTruncated] = useState(false);
  const generation = useRef(0);
  const blob = useRef("");
  const headers = JSON.stringify(authHeaders || {});
  const root = `/api/chat/sessions/${encodeURIComponent(sessionId)}/files`;

  const refresh = useCallback(async () => {
    if (!sessionId) return [] as SessionFile[];
    const epoch = generation.current;
    setLoading(true);
    try {
      const j = await shareRead(`session-files:${root}:${headers}`, async () => {
        const r = await fetch(root, { headers: JSON.parse(headers), cache: "no-store" });
        if (!r.ok) throw new Error();
        return r.json();
      });
      const rows: SessionFile[] = j.files || [];
      if (epoch === generation.current) {
        setFiles(rows); setTruncated(!!j.truncated); setError("");
        setSelected(previous => new Set([...previous].filter(key => rows.some(file => file.key === key))));
      }
      return rows;
    } catch {
      if (epoch === generation.current) setError(zh ? "文件加载失败，请重试" : "Could not load files. Try refreshing.");
      return [] as SessionFile[];
    } finally {
      if (epoch === generation.current) setLoading(false);
    }
  }, [sessionId, root, headers, zh]);

  function closePreview() {
    requestVersion.current += 1;
    setPreview(null);
    if (blob.current) URL.revokeObjectURL(blob.current);
    blob.current = "";
  }
  function closePanel() {
    closePreview(); setWide(false);
    setOpen(false);
    trigger.current?.focus({ preventScroll: true });
  }
  const show = useCallback(async (file: SessionFile) => {
    if (file.size > 30 * 1024 * 1024) {
      setError(zh ? "文件较大，请下载查看" : "Download this file to view it"); return;
    }
    const epoch = generation.current;
    const request = ++requestVersion.current;
    setOpening(file.key);
    try {
      const r = await fetch(`${root}/content?path=${encodeURIComponent(file.key)}`, { headers: JSON.parse(headers) });
      if (!r.ok) throw new Error();
      const data = await r.blob();
      if (epoch !== generation.current || request !== requestVersion.current) return;
      if (blob.current) URL.revokeObjectURL(blob.current);
      blob.current = URL.createObjectURL(data);
      setPreview({ ...file, abs: file.key, url: blob.current }); setOpen(true); setError("");
    } catch {
      if (epoch === generation.current && request === requestVersion.current) setError(zh ? "暂时无法预览此文件，请重试" : "Could not open this file. Try again.");
    } finally {
      if (epoch === generation.current && request === requestVersion.current) setOpening("");
    }
  }, [root, headers, zh]);

  useEffect(() => {
    generation.current += 1;
    setFiles([]); setSelected(new Set()); setError(""); setOpening(""); setDownloading(false); setLoading(false);
    setQuery(""); setKind(""); setDate(""); setOrder("newest"); setSelecting(false);
    closePreview();
    return () => { generation.current += 1; if (blob.current) URL.revokeObjectURL(blob.current); };
  }, [refresh]);
  useEffect(() => {
    if (open && sessionId) void refresh();
  }, [open, sessionId, refresh]);
  useEffect(() => {
    if (!open || !liveRefresh || !sessionId) return;
    const timer = window.setInterval(() => void refresh(), pollMs);
    return () => clearInterval(timer);
  }, [open, liveRefresh, sessionId, pollMs, refresh]);
  useEffect(() => {
    if (!open || preview) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) { e.preventDefault(); closePanel(); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [open, preview]);
  useEffect(() => {
    const handler = async (e: Event) => {
      const epoch = generation.current;
      setOpen(true);
      const rows = await refresh();
      if (epoch !== generation.current) return;
      const file = rows.find(row => row.key === (e as CustomEvent).detail?.key);
      if (file) await show(file);
    };
    window.addEventListener("venus:preview-file", handler);
    return () => window.removeEventListener("venus:preview-file", handler);
  }, [refresh, show]);

  useEffect(() => {
    const handler = async (event: Event) => {
      const request = (event as CustomEvent<StructureRequest>).detail;
      if (!request?.path || (!isStructureFile(request.path) && !isStructureFile(request.name || ""))) return;
      const epoch = generation.current;
      closePreview();
      const ticket = ++requestVersion.current;
      setOpen(true);
      if (request.url?.startsWith("blob:")) {
        closePreview(); blob.current = request.url; setPreview({ name: request.name || request.path, abs: request.path, url: request.url, size: 0, ext: request.ext || structureExtension(request.path) }); return;
      }
      const path = structurePathFromLink(request.path);
      const rows = await refresh();
      if (epoch !== generation.current || ticket !== requestVersion.current) return;
      const file = matchSessionFile(rows, path);
      if (file) { await show(file); return; }
      if (/^(https?:)?\/\//i.test(path) && new URL(path, window.location.href).origin !== window.location.origin) {
        closePreview(); setPreview({ name: request.name || path.split("/").pop() || path, abs: path, url: new URL(path, window.location.href).href, size: 0, ext: structureExtension(path) }); return;
      }
      setError(zh ? "未找到此会话中的文件，请从文件列表选择" : "File not found in this conversation. Choose it from Files.");
    };
    window.addEventListener("venus:open-structure", handler);
    return () => window.removeEventListener("venus:open-structure", handler);
  }, [refresh, show, zh]);

  const filtered = useMemo(() => files.filter(f =>
    (!query.trim() || f.rel.toLowerCase().includes(query.trim().toLowerCase())) &&
    (!kind || kinds[kind]?.includes(f.ext.toLowerCase())) &&
    (!date || new Date(f.mtime * 1000).toLocaleDateString("sv-SE") >= date)
  ).sort((a, b) => order === "name" ? a.name.localeCompare(b.name) : order === "size" ? b.size - a.size : b.mtime - a.mtime), [files, query, kind, date, order]);
  useEffect(() => { setSelected(new Set()); }, [query, kind, date]);
  useEffect(() => {
    if (selectAll.current) selectAll.current.indeterminate = selected.size > 0 && selected.size < filtered.length;
  }, [selected, filtered.length, selecting]);
  const hasFilter = !!(query || kind || date);
  function clearFilters() { setQuery(""); setKind(""); setDate(""); }
  async function download() {
    const paths = (selecting ? filtered.filter(f => selected.has(f.key)) : filtered).map(f => f.key);
    if (!paths.length) return;
    const epoch = generation.current;
    setDownloading(true); setError("");
    try {
      const r = await fetch(`${root}/archive`, { method: "POST", headers: { ...JSON.parse(headers), "Content-Type": "application/json" }, body: JSON.stringify({ paths }) });
      if (!r.ok) throw new Error();
      const data = await r.blob();
      if (epoch !== generation.current) return;
      const url = URL.createObjectURL(data);
      const a = document.createElement("a"); a.href = url; a.download = "research-files.zip"; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch {
      if (epoch === generation.current) setError(zh ? "下载失败，请重试" : "Could not download files. Try again.");
    } finally {
      if (epoch === generation.current) setDownloading(false);
    }
  }

  return <>
    <div className="chat-file-actions">{quota}{actions}
      <button ref={trigger} className="session-files-toggle" type="button" onClick={() => { if (structure) { closePreview(); setWide(false); } else setOpen(!open); }} aria-expanded={open} aria-controls={panelId} aria-label={zh ? "研究文件" : "Research files"} title={zh ? "文件" : "Files"}><FileIcon name="folder"/></button>
    </div>
    {open && <aside ref={panel} id={panelId} style={{ "--structure-width": `${structureWidth}px` } as CSSProperties} className={`session-files-sidebar${structure ? " is-structure" : ""}${wide && structure ? " is-wide" : ""}`} aria-labelledby={`${panelId}-title`}>
      {structure ? <>
        <div className="structure-panel-resizer" role="separator" tabIndex={0} aria-label={zh ? "调整结构面板宽度" : "Resize structure panel"} aria-orientation="vertical" aria-valuenow={structureWidth} aria-valuemin={420} aria-valuemax={1000}
          onKeyDown={e => { if (["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); setStructureWidth(w => Math.max(420, Math.min(1000, w + (e.key === "ArrowLeft" ? 40 : -40)))); } }}
          onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); }}
          onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId) && panel.current) { const right = panel.current.getBoundingClientRect().right; setStructureWidth(Math.max(420, Math.min(1000, right - e.clientX))); } }}
          onPointerUp={e => e.currentTarget.releasePointerCapture(e.pointerId)}/>
        <header className="session-files-heading structure-panel-heading">
          <button className="sf-icon-button" onClick={() => { closePreview(); setWide(false); }} aria-label={zh ? "返回文件" : "Back to files"} title={zh ? "返回文件" : "Back to files"}>←</button>
          <h2 id={`${panelId}-title`} title={structure.name}>{structure.name}</h2>
          <div><a className="sf-icon-button" href={structure.url} download={structure.name} title={zh ? "下载结构" : "Download structure"}><FileIcon name="download"/></a>
            <button className="sf-icon-button" aria-label={wide ? (zh ? "缩小面板" : "Restore panel") : (zh ? "展开面板" : "Expand panel")} title={zh ? "展开 / 缩小" : "Expand / restore"} onClick={() => setWide(!wide)}>⛶</button>
            <button className="sf-icon-button" onClick={closePanel} aria-label={zh ? "关闭结构" : "Close structure"}><FileIcon name="close"/></button></div>
        </header>
        <MolstarViewer key={structure.url || structure.abs} filePath={structure.abs} sourceUrl={structure.url} format={structure.ext} label={structure.name} embedded/>
      </> : <>
      <header className="session-files-heading">
        <h2 id={`${panelId}-title`}>{zh ? "文件" : "Files"}{!!files.length && <span>{files.length}</span>}</h2>
        <div>
          <button className="sf-icon-button" type="button" aria-label={zh ? "刷新文件" : "Refresh files"} title={zh ? "刷新" : "Refresh"} disabled={loading} onClick={() => void refresh()}><FileIcon name="refresh"/></button>
          <button className="sf-icon-button" type="button" aria-label={zh ? "关闭文件面板" : "Close files"} title={zh ? "关闭" : "Close"} onClick={closePanel}><FileIcon name="close"/></button>
        </div>
      </header>
      {!!files.length && <>
        <div className="session-files-search">
          <input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder={zh ? "搜索文件" : "Search files"} aria-label={zh ? "搜索文件" : "Search files"}/>

        </div>
        <div className="session-files-filter-fields">
          <label>{zh ? "文件类型" : "File type"}<select value={kind} onChange={e => setKind(e.target.value)}><option value="">{zh ? "全部" : "All types"}</option>{Object.keys(kinds).map((k, i) => <option key={k} value={k}>{zh ? ["结构 / 小分子", "表格 / 数据", "序列", "文档", "图片", "代码"][i] : ["Structures", "Data", "Sequences", "Documents", "Images", "Code"][i]}</option>)}</select></label>
          <label>{zh ? "起始日期" : "From date"}<input type="date" value={date} onChange={e => setDate(e.target.value)}/></label>
          <label>{zh ? "排序" : "Sort"}<select value={order} onChange={e => setOrder(e.target.value)}><option value="newest">{zh ? "最新" : "Newest"}</option><option value="name">{zh ? "名称" : "Name"}</option><option value="size">{zh ? "大小" : "Size"}</option></select></label>
          {hasFilter && <button className="sf-text-button" type="button" onClick={clearFilters}>{zh ? "清除筛选" : "Clear filters"}</button>}
        </div>
      </>}
      {error && <p role="alert" className="session-files-notice">{error}</p>}
      {selecting && !!filtered.length && <label className="session-files-select-all"><input ref={selectAll} type="checkbox" checked={filtered.every(f => selected.has(f.key))} onChange={e => setSelected(e.target.checked ? new Set(filtered.map(f => f.key)) : new Set())}/>{selected.size ? `${selected.size} ${zh ? "已选" : "selected"}` : (zh ? "全选" : "Select all")}</label>}
      <div className="session-files-results" aria-busy={loading}>
        {filtered.map(f => <div className="session-files-entry" key={f.key}>
          {selecting && <input type="checkbox" aria-label={(zh ? "选择 " : "Select ") + f.name} checked={selected.has(f.key)} onChange={e => setSelected(previous => { const next = new Set(previous); if (e.target.checked) next.add(f.key); else next.delete(f.key); return next; })}/>}
          <button type="button" className="session-file-preview" onClick={() => void show(f)} title={f.rel} disabled={!!opening} aria-busy={opening === f.key}>
            <span className="session-file-extension">{f.ext.slice(0, 5) || "FILE"}</span>
            <span className="session-file-info"><strong>{f.name}</strong><small>{new Date(f.mtime * 1000).toLocaleDateString(zh ? "zh-CN" : "en", { month: "short", day: "numeric" })} · {formatSize(f.size)} · {f.scope.startsWith("research-") ? "Scientist" : f.scope === "uploads" ? (zh ? "上传" : "Upload") : "Agent"}</small></span>
            <span className="session-file-action">{opening === f.key ? "…" : "↗"}</span>
          </button>
        </div>)}
        {!filtered.length && <div className="session-files-empty-state" role="status">
          <FileIcon name="folder"/>
          <p>{loading ? (zh ? "加载中…" : "Loading…") : error ? (zh ? "暂时无法加载文件" : "Files unavailable") : files.length ? (zh ? "没有匹配的文件" : "No matching files") : (zh ? "暂无文件" : "No files yet")}</p>
          {hasFilter ? <button className="sf-text-button" type="button" onClick={clearFilters}>{zh ? "清除筛选" : "Clear filters"}</button> : !error && !loading && <small>{zh ? "上传的文件和研究结果会显示在这里" : "Uploads and research results appear here."}</small>}
        </div>}
      </div>
      {!!files.length && <footer className="session-files-footer">
        {truncated && <small>{zh ? "显示最新 2,000 个文件" : "Latest 2,000 files"}</small>}
        <div><button className="sf-text-button" type="button" onClick={() => { setSelecting(!selecting); setSelected(new Set()); }} disabled={downloading || (!selecting && !filtered.length)}>{selecting ? (zh ? "取消选择" : "Cancel selection") : (zh ? "选择文件" : "Select files")}</button>
          <button className="sf-download-button" type="button" disabled={downloading || (selecting ? !selected.size : !filtered.length)} onClick={() => void download()}><FileIcon name="download"/>{downloading ? (zh ? "打包中…" : "Preparing…") : selecting ? `${zh ? "下载" : "Download"} (${selected.size})` : (zh ? "打包下载" : "Download ZIP")}</button></div>
      </footer>}
    </>}</aside>}
    <FilePreviewModal file={structure ? null : preview} onClose={closePreview}/>
  </>;
}
