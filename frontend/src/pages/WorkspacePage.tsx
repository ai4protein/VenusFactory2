import { LangLink } from "../components/LangLink";
import { useEffect, useRef, useState } from "react";
import { FilePreviewModal, type PreviewableFile } from "../components/FilePreviewModal";
import {
  deleteWorkspaceFile,
  workspaceContentUrl,
  downloadWorkspaceArchive,
  listWorkspaceFiles,
  replaceWorkspaceFile,
  uploadWorkspaceFile,
  type WorkspaceFile
} from "../lib/workspaceApi";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

type WorkspacePageProps = {
  workspaceEnabled: boolean;
};

const STRINGS = {
  en: {
    docTitle: "Library — VenusFactory2",
    docDescription: "Files from your chats and uploads.",
    library: "Library",
    onlyLocal: "Available when running locally.",
    searchPlaceholder: "Search by name or path...",
    refreshing: "Refreshing...",
    search: "Search",
    allSources: "All Sources",
    allCategories: "All Categories",
    catSequence: "Sequence",
    catStructure: "Structure",
    catTableText: "Table/Text",
    catOther: "Other",
    sortNewest: "Newest First",
    sortNameAsc: "Name A-Z",
    sortLargest: "Largest First",
    allBuckets: "All Buckets",
    bucketUser: "Uploaded",
    bucketChat: "Chat files",
    bucketTool: "Tool inputs",
    uploading: "Uploading...",
    uploadBtn: "Upload",
    filesHeading: "Files",
    browseDisabled: "Workspace browsing is disabled in Online mode.",
    noFiles: "No files found. Upload files or clear filters.",
    colName: "Name",
    colBucket: "Bucket",
    colSource: "Source",
    colCategory: "Category",
    colSize: "Size",
    colUpdated: "Updated",
    colActions: "Actions",
    working: "Working...",
    replace: "Replace",
    deleteBtn: "Delete",
    readonly: "Readonly",
    confirmDelete: "Confirm Delete",
    confirmDeletePrefix: "Delete",
    confirmDeleteSuffix: "? This cannot be undone.",
    cancel: "Cancel",
    deleting: "Deleting...",
    dismissNotif: "Dismiss notification",
    errLoad: "Failed to load library files.",
    errUpload: "Upload failed.",
    errReplace: "Replace failed.",
    errDelete: "Delete failed.",
    replacedToast: "Replaced %s successfully."
  },
  zh: {
    docTitle: "文件库 — VenusFactory2",
    docDescription: "对话生成的文件和上传文件。",
    library: "文件库",
    onlyLocal: "本地运行时可用。",
    searchPlaceholder: "按名称或路径搜索…",
    refreshing: "刷新中…",
    search: "搜索",
    allSources: "全部来源",
    allCategories: "全部类别",
    catSequence: "序列",
    catStructure: "结构",
    catTableText: "表格 / 文本",
    catOther: "其他",
    sortNewest: "最新优先",
    sortNameAsc: "按名称 A-Z",
    sortLargest: "最大优先",
    allBuckets: "全部存储桶",
    bucketUser: "上传文件",
    bucketChat: "对话文件",
    bucketTool: "工具输入",
    uploading: "上传中…",
    uploadBtn: "上传",
    filesHeading: "文件",
    browseDisabled: "在线模式下工作区浏览功能不可用。",
    noFiles: "未找到文件。上传文件或清除筛选条件。",
    colName: "名称",
    colBucket: "存储桶",
    colSource: "来源",
    colCategory: "类别",
    colSize: "大小",
    colUpdated: "更新时间",
    colActions: "操作",
    working: "处理中…",
    replace: "替换",
    deleteBtn: "删除",
    readonly: "只读",
    confirmDelete: "确认删除",
    confirmDeletePrefix: "删除",
    confirmDeleteSuffix: " 吗？此操作不可撤销。",
    cancel: "取消",
    deleting: "删除中…",
    dismissNotif: "关闭通知",
    errLoad: "加载文件库失败。",
    errUpload: "上传失败。",
    errReplace: "替换失败。",
    errDelete: "删除失败。",
    replacedToast: "已成功替换 %s。"
  }
};

export function WorkspacePage({ workspaceEnabled }: WorkspacePageProps) {
  const {lang,t:translate}=useLang(); const t=translate(STRINGS); const zh=lang==='zh';
  const tr=(en:string,cn:string)=>zh?cn:en;
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const [items, setItems] = useState<WorkspaceFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [q, setQ] = useState("");
  const [source, setSource] = useState("");
  const [fileType, setFileType] = useState("");
  const [sort, setSort] = useState<"created_desc" | "name_asc" | "size_desc">("created_desc");
  const [bucket, setBucket] = useState("");
  const [actingId, setActingId] = useState("");
  const [pendingDeleteItem, setPendingDeleteItem] = useState<WorkspaceFile | null>(null);
  const [toastMessage, setToastMessage] = useState("");
  const [deleteTriggerEl, setDeleteTriggerEl] = useState<HTMLButtonElement | null>(null);
  const cancelDeleteBtnRef = useRef<HTMLButtonElement | null>(null);
  const [page,setPage]=useState(1); const [total,setTotal]=useState(0);
  const [totalSize,setTotalSize]=useState(0); const [sources,setSources]=useState<string[]>([]);
  const [dateRange,setDateRange]=useState(''); const [selected,setSelected]=useState<Set<string>>(new Set());
  const [packing,setPacking]=useState(false); const [preview,setPreview]=useState<PreviewableFile|null>(null);
  const requestId=useRef(0); const uploadRef=useRef<HTMLInputElement>(null);
  const formatSize=(n:number)=>n<1024?`${n} B`:n<1048576?`${(n/1024).toFixed(1)} KB`:`${(n/1048576).toFixed(1)} MB`;
  function resetFilters(){setQ('');setSource('');setFileType('');setBucket('');setDateRange('');setPage(1);}
  async function downloadSelected(){setPacking(true);setError('');try{await downloadWorkspaceArchive([...selected]);}catch(e){setError(String(e));}finally{setPacking(false);}}
  useEffect(()=>()=>{if(preview?.url?.startsWith('blob:'))URL.revokeObjectURL(preview.url);},[preview]);
  async function showPreview(item:WorkspaceFile){
    if(item.size>20*1024*1024){setError(tr('Download this file to view it; the preview limit is 20 MB.','请下载后查看，预览上限为 20 MB。'));return;}
    try {
      const res=await fetch(workspaceContentUrl(item.storage_path));
      if(!res.ok)throw new Error(`${tr('Preview failed','预览失败')} (${res.status})`);
      const url=URL.createObjectURL(await res.blob());
      setPreview({name:item.display_name,abs:item.storage_path,url,size:item.size,ext:item.suffix.replace(/^\./,'')});
    }catch(err){setError(err instanceof Error?err.message:String(err));}
  }

  async function refresh() {
    if (!workspaceEnabled) return;
    const id=++requestId.current;
    setLoading(true);
    setError("");
    try {
      const data = await listWorkspaceFiles({
        q,
        source,
        fileType,
        sort,
        includeSessions: true, page, pageSize:50, bucket,
        modifiedAfter:dateRange ? new Date(Date.now()-Number(dateRange)*86400000).toISOString() : undefined
      });
      if(id!==requestId.current)return;
      setItems(data.items);setTotal(data.total);setTotalSize(data.total_size||0);setSources(data.sources||[]);setSelected(new Set());
      if(data.page!==page)setPage(data.page);
    } catch (err) {
      if(id===requestId.current)setError(err instanceof Error ? err.message : t.errLoad);
    } finally {
      if(id===requestId.current)setLoading(false);
    }
  }

  useEffect(() => {
    const timer=window.setTimeout(()=>void refresh(),200);
    return ()=>{window.clearTimeout(timer);requestId.current++;};
  }, [workspaceEnabled, q, source, fileType, sort, bucket, dateRange, page]);

  useEffect(() => {
    if (!toastMessage) return;
    const timer = window.setTimeout(() => setToastMessage(""), 2500);
    return () => window.clearTimeout(timer);
  }, [toastMessage]);

  useEffect(() => {
    if (!pendingDeleteItem) return;
    cancelDeleteBtnRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [pendingDeleteItem]);

  useEffect(() => {
    if (!pendingDeleteItem) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      if (Boolean(actingId)) return;
      setPendingDeleteItem(null);
      deleteTriggerEl?.focus();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [pendingDeleteItem, actingId, deleteTriggerEl]);

  async function onUpload(files: File[]) {
    if (!files.length || !workspaceEnabled) return;
    setUploading(true);setError('');let completed=0;const failures:string[]=[];
    try {
      for(const file of files){try{await uploadWorkspaceFile(file);completed++;}catch(err){failures.push(`${file.name}: ${err instanceof Error?err.message:String(err)}`);}}
      await refresh();
      if(completed)setToastMessage(tr(`${completed} files uploaded`, `已上传 ${completed} 个文件`));
      if(failures.length)setError(failures.join('\n'));
    }finally{setUploading(false);}
  }

  async function onReplace(item: WorkspaceFile, file: File | null) {
    if (!workspaceEnabled || !file) return;
    setActingId(item.id);
    setError("");
    try {
      await replaceWorkspaceFile(item.storage_path, file);
      await refresh();
      setToastMessage(t.replacedToast.replace("%s", item.display_name));
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errReplace);
    } finally {
      setActingId("");
    }
  }

  function onDelete(item: WorkspaceFile, trigger?: HTMLButtonElement | null) {
    if (!workspaceEnabled) return;
    setDeleteTriggerEl(trigger || null);
    setPendingDeleteItem(item);
  }

  function cancelDelete() {
    if (Boolean(actingId)) return;
    setPendingDeleteItem(null);
    deleteTriggerEl?.focus();
  }

  async function confirmDelete() {
    if (!workspaceEnabled || !pendingDeleteItem) return;
    const item = pendingDeleteItem;
    setActingId(item.id);
    setError("");
    try {
      await deleteWorkspaceFile(item.storage_path);
      await refresh();
      setPendingDeleteItem(null);
      deleteTriggerEl?.focus();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errDelete);
    } finally {
      setActingId("");
    }
  }

  const pages=Math.max(1,Math.ceil(total/50));
  const categoryLabel=(item:WorkspaceFile)=>({sequence:t.catSequence,structure:t.catStructure,table_or_text:t.catTableText,other:t.catOther}[item.category]||item.category);
  const bucketLabel=(item:WorkspaceFile)=>({user_upload:t.bucketUser,chat_session:t.bucketChat,tool_upload:t.bucketTool,generated:tr('AI outputs','AI 生成'),other:t.catOther}[item.bucket]);
  if (!workspaceEnabled) return <div className="library-unavailable"><h1>{tr('Library','文件库')}</h1><p>{t.onlyLocal}</p><LangLink to="/agent/chat">{tr('Back to Chat','返回对话')} →</LangLink></div>;
  return (
    <div className="research-page workspace-library">
      <header className="research-page-header"><div><h1>{tr('Library','文件库')}</h1><p>{tr('Files from your chats and uploads.','对话生成的文件和上传文件。')}</p></div><div className="research-inline-actions"><button disabled={loading||!workspaceEnabled} onClick={()=>void refresh()}>{tr('Refresh','刷新')}</button><button className="research-primary" disabled={uploading||!workspaceEnabled} onClick={()=>uploadRef.current?.click()}>＋ {uploading?t.uploading:tr('Upload files','上传文件')}</button><input ref={uploadRef} hidden type="file" multiple onChange={e=>{void onUpload(Array.from(e.target.files||[]));e.target.value='';}}/></div></header>
      {toastMessage&&<div className="research-toast" role="status">{toastMessage}<button aria-label={t.dismissNotif} onClick={()=>setToastMessage('')}>×</button></div>}
      {error&&<div className="error-box" role="alert">{error}</div>}
      <div className="research-toolbar">
        <input className="research-search" aria-label={t.search} value={q} onChange={e=>{setQ(e.target.value);setPage(1);}} placeholder={tr('Search files…','搜索文件…')}/>
        <select aria-label={t.colCategory} value={fileType} onChange={e=>{setFileType(e.target.value);setPage(1);}}><option value="">{t.allCategories}</option><option value="sequence">{t.catSequence}</option><option value="structure">{t.catStructure}</option><option value="table_or_text">{t.catTableText}</option><option value="other">{t.catOther}</option></select>
        <select aria-label={tr('Location','位置')} value={bucket} onChange={e=>{setBucket(e.target.value);setPage(1);}}><option value="">{tr('All locations','全部位置')}</option><option value="user_upload">{t.bucketUser}</option><option value="chat_session">{t.bucketChat}</option><option value="generated">{tr('AI outputs','AI 生成')}</option><option value="tool_upload">{t.bucketTool}</option></select>
        <select aria-label={t.colSource} value={source} onChange={e=>{setSource(e.target.value);setPage(1);}}><option value="">{t.allSources}</option>{sources.map(x=><option key={x}>{x}</option>)}</select>
        <select aria-label={tr('Date','日期')} value={dateRange} onChange={e=>{setDateRange(e.target.value);setPage(1);}}><option value="">{tr('Any date','全部日期')}</option><option value="1">{tr('Last 24 hours','最近 24 小时')}</option><option value="7">{tr('Last 7 days','最近 7 天')}</option><option value="30">{tr('Last 30 days','最近 30 天')}</option></select>
        <select aria-label={tr('Sort','排序')} value={sort} onChange={e=>{setSort(e.target.value as typeof sort);setPage(1);}}><option value="created_desc">{t.sortNewest}</option><option value="name_asc">{t.sortNameAsc}</option><option value="size_desc">{t.sortLargest}</option></select>
        {(q||source||fileType||bucket||dateRange)&&<button onClick={resetFilters}>{tr('Clear filters','清除筛选')}</button>}
      </div>
      <section className="research-card workspace-files-card" aria-busy={loading}>
        <div className="workspace-list-heading"><span>{loading?tr('Loading files…','加载中…'):`${total} ${tr('files','个文件')} · ${formatSize(totalSize)}`}</span><div className="research-inline-actions">{selected.size>0&&<><span>{selected.size} {tr('selected','项已选')}</span><button disabled={packing||loading} onClick={()=>void downloadSelected()}>{packing?tr('Preparing…','准备中…'):tr('Download ZIP','打包下载')} ↓</button><button onClick={()=>setSelected(new Set())}>{tr('Clear','清除')}</button></>}</div></div>
        {!workspaceEnabled?<div className="research-empty"><h3>{t.onlyLocal}</h3></div>:items.length===0?<div className="research-empty"><span>▱</span><h3>{loading?tr('Loading your library','加载资源库'):tr('No files here yet','暂无文件')}</h3><p>{tr('Upload a file or adjust your filters.','上传文件或调整筛选条件。')}</p></div>:<div className="research-table-scroll"><table className="research-file-table"><thead><tr><th><input type="checkbox" aria-label={tr('Select this page','选择当前页')} disabled={loading} checked={items.length>0&&items.every(i=>selected.has(i.storage_path))} onChange={e=>setSelected(e.target.checked?new Set(items.map(i=>i.storage_path)):new Set())}/></th><th>{t.colName}</th><th>{tr('Location','位置')}</th><th>{t.colCategory}</th><th>{t.colSize}</th><th>{t.colUpdated}</th><th><span className="research-sr-only">{t.colActions}</span></th></tr></thead><tbody>{items.map(item=><tr key={item.id} className={selected.has(item.storage_path)?'selected':''}>
          <td><input type="checkbox" disabled={loading} aria-label={`${tr('Select','选择')} ${item.display_name}`} checked={selected.has(item.storage_path)} onChange={e=>setSelected(prev=>{const next=new Set(prev);e.target.checked?next.add(item.storage_path):next.delete(item.storage_path);return next;})}/></td>
          <td><button className="workspace-file-name" onClick={()=>showPreview(item)} title={item.storage_path}><span className="workspace-file-ext">{item.suffix.replace('.','').slice(0,5)||'FILE'}</span><span>{item.display_name}</span></button></td><td><span>{bucketLabel(item)}</span><small className="workspace-file-source">{item.bucket==='chat_session'?tr('Chat','对话'):item.source}</small></td><td>{categoryLabel(item)}</td><td>{formatSize(item.size)}</td><td>{new Date(item.created_at).toLocaleDateString(zh?'zh-CN':'en-GB',{year:'numeric',month:'short',day:'numeric'})}</td>
          <td><div className="workspace-file-actions"><a href={workspaceContentUrl(item.storage_path)} download title={tr('Download','下载')} aria-label={`${tr('Download','下载')} ${item.display_name}`}>↓</a>{item.bucket==='user_upload'&&<details><summary aria-label={`${tr('More actions','更多操作')} ${item.display_name}`}>⋯</summary><div className="workspace-file-menu"><label>{t.replace}<input type="file" disabled={Boolean(actingId)} onChange={e=>void onReplace(item,e.target.files?.[0]||null)}/></label><button className="danger" disabled={Boolean(actingId)} onClick={e=>onDelete(item,e.currentTarget)}>{t.deleteBtn}</button></div></details>}</div></td>
        </tr>)}</tbody></table></div>}
        <div className="research-pagination"><span>{total?`${(page-1)*50+1}–${Math.min(page*50,total)} / ${total}`:'0'}</span><div><button disabled={page<=1||loading} onClick={()=>setPage(p=>p-1)}>← {tr('Previous','上一页')}</button><span>{page} / {pages}</span><button disabled={page>=pages||loading} onClick={()=>setPage(p=>p+1)}>{tr('Next','下一页')} →</button></div></div>
      </section>
      {preview&&<FilePreviewModal file={preview} onClose={()=>setPreview(null)}/>}
      {pendingDeleteItem && (
        <div className="workspace-modal-backdrop" role="presentation" onClick={cancelDelete}>
          <div
            className="workspace-modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="workspace-delete-title"
            aria-describedby="workspace-delete-desc"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 id="workspace-delete-title">{t.confirmDelete}</h4>
            <p id="workspace-delete-desc">
              {t.confirmDeletePrefix} <strong>{pendingDeleteItem.display_name}</strong>{t.confirmDeleteSuffix}
            </p>
            <div className="workspace-modal-actions">
              <button
                ref={cancelDeleteBtnRef}
                type="button"
                className="workspace-row-btn"
                disabled={Boolean(actingId)}
                onClick={cancelDelete}
              >
                {t.cancel}
              </button>
              <button
                type="button"
                className="workspace-row-btn danger"
                disabled={Boolean(actingId)}
                onClick={() => void confirmDelete()}
              >
                {Boolean(actingId) ? t.deleting : t.deleteBtn}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
