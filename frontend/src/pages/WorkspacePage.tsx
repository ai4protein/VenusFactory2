import { useEffect, useMemo, useRef, useState } from "react";
import { PageFooter } from "../components/PageFooter";
import {
  deleteWorkspaceFile,
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
    docTitle: "Workspace — VenusFactory2",
    docDescription: "Manage your local workspace files.",
    library: "Workspace Library",
    onlyLocal: "Workspace is available in Local mode only.",
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
    bucketUser: "User Upload",
    bucketChat: "Chat Session",
    bucketTool: "Tool Upload",
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
    errLoad: "Failed to load workspace files.",
    errUpload: "Upload failed.",
    errReplace: "Replace failed.",
    errDelete: "Delete failed.",
    replacedToast: "Replaced %s successfully."
  },
  zh: {
    docTitle: "工作区 — VenusFactory2",
    docDescription: "管理本地工作区文件。",
    library: "工作区资源库",
    onlyLocal: "工作区仅在本地模式下可用。",
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
    bucketUser: "用户上传",
    bucketChat: "对话会话",
    bucketTool: "工具上传",
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
    errLoad: "加载工作区文件失败。",
    errUpload: "上传失败。",
    errReplace: "替换失败。",
    errDelete: "删除失败。",
    replacedToast: "已成功替换 %s。"
  }
};

export function WorkspacePage({ workspaceEnabled }: WorkspacePageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });
  const [items, setItems] = useState<WorkspaceFile[]>([]);
  const [loading, setLoading] = useState(false);
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

  async function refresh() {
    if (!workspaceEnabled) return;
    setLoading(true);
    setError("");
    try {
      const data = await listWorkspaceFiles({
        q,
        source,
        fileType,
        sort,
        includeSessions: true
      });
      setItems(data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errLoad);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, [workspaceEnabled, source, fileType, sort]);

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

  async function onUpload(file: File | null) {
    if (!file || !workspaceEnabled) return;
    setUploading(true);
    setError("");
    try {
      await uploadWorkspaceFile(file);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errUpload);
    } finally {
      setUploading(false);
    }
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

  const sourceOptions = useMemo(() => {
    const set = new Set<string>();
    items.forEach((item) => set.add(item.source));
    return Array.from(set).sort();
  }, [items]);
  const bucketFilteredItems = useMemo(() => {
    if (!bucket) return items;
    return items.filter((item) => item.bucket === bucket);
  }, [items, bucket]);

  return (
    <div className="workspace-page">
      {toastMessage && (
        <div className="workspace-toast" role="status" aria-live="polite">
          <span>{toastMessage}</span>
          <button
            type="button"
            className="workspace-toast-close"
            onClick={() => setToastMessage("")}
            aria-label={t.dismissNotif}
          >
            ×
          </button>
        </div>
      )}
      <section className="chat-panel workspace-control-panel">
        <h3>{t.library}</h3>
        {!workspaceEnabled && (
          <div className="readonly-banner">{t.onlyLocal}</div>
        )}
        {workspaceEnabled && (
          <>
            <div className="workspace-filter-row">
              <input
                type="text"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder={t.searchPlaceholder}
              />
              <button type="button" onClick={() => void refresh()} disabled={loading}>
                {loading ? t.refreshing : t.search}
              </button>
            </div>
            <div className="workspace-filter-row">
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="">{t.allSources}</option>
                {sourceOptions.map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
              <select value={fileType} onChange={(e) => setFileType(e.target.value)}>
                <option value="">{t.allCategories}</option>
                <option value="sequence">{t.catSequence}</option>
                <option value="structure">{t.catStructure}</option>
                <option value="table_or_text">{t.catTableText}</option>
                <option value="other">{t.catOther}</option>
              </select>
              <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}>
                <option value="created_desc">{t.sortNewest}</option>
                <option value="name_asc">{t.sortNameAsc}</option>
                <option value="size_desc">{t.sortLargest}</option>
              </select>
              <select value={bucket} onChange={(e) => setBucket(e.target.value)}>
                <option value="">{t.allBuckets}</option>
                <option value="user_upload">{t.bucketUser}</option>
                <option value="chat_session">{t.bucketChat}</option>
                <option value="tool_upload">{t.bucketTool}</option>
              </select>
              <label className="workspace-upload-btn">
                {uploading ? t.uploading : t.uploadBtn}
                <input
                  type="file"
                  disabled={uploading}
                  onChange={(e) => void onUpload(e.target.files?.[0] || null)}
                />
              </label>
            </div>
          </>
        )}
        {error && <div className="error-box">{error}</div>}
      </section>

      <section className="chat-panel workspace-list-panel">
        <h3>{t.filesHeading} ({bucketFilteredItems.length})</h3>
        {!workspaceEnabled ? (
          <div className="session-empty">{t.browseDisabled}</div>
        ) : bucketFilteredItems.length === 0 ? (
          <div className="session-empty">{t.noFiles}</div>
        ) : (
          <div className="workspace-table">
            <div className="workspace-row workspace-header-row">
              <span>{t.colName}</span>
              <span>{t.colBucket}</span>
              <span>{t.colSource}</span>
              <span>{t.colCategory}</span>
              <span>{t.colSize}</span>
              <span>{t.colUpdated}</span>
              <span>{t.colActions}</span>
            </div>
            {bucketFilteredItems.map((item) => (
              <div key={item.id} className="workspace-row">
                <span title={item.storage_path}>{item.display_name}</span>
                <span>{item.bucket}</span>
                <span>{item.source}</span>
                <span>{item.category}</span>
                <span>{(item.size / 1024).toFixed(1)} KB</span>
                <span>{new Date(item.created_at).toLocaleString()}</span>
                <span className="workspace-row-actions">
                  {item.bucket === "user_upload" ? (
                    <>
                      <label className="workspace-row-btn">
                        {actingId === item.id ? t.working : t.replace}
                        <input
                          type="file"
                          disabled={Boolean(actingId)}
                          onChange={(e) => void onReplace(item, e.target.files?.[0] || null)}
                        />
                      </label>
                      <button
                        type="button"
                        className="workspace-row-btn danger"
                        disabled={Boolean(actingId)}
                        onClick={(e) => void onDelete(item, e.currentTarget)}
                      >
                        {t.deleteBtn}
                      </button>
                    </>
                  ) : (
                    <span className="workspace-row-readonly">{t.readonly}</span>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
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
      <PageFooter />
    </div>
  );
}
