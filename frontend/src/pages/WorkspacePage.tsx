import { useEffect, useMemo, useRef, useState } from "react";
import { PageFooter } from "../components/PageFooter";
import {
  deleteWorkspaceFile,
  listWorkspaceFiles,
  replaceWorkspaceFile,
  uploadWorkspaceFile,
  type WorkspaceFile
} from "../lib/workspaceApi";

type WorkspacePageProps = {
  workspaceEnabled: boolean;
};

export function WorkspacePage({ workspaceEnabled }: WorkspacePageProps) {
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
      setError(err instanceof Error ? err.message : "Failed to load workspace files.");
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
      setError(err instanceof Error ? err.message : "Upload failed.");
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
      setToastMessage(`Replaced ${item.display_name} successfully.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Replace failed.");
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
      setError(err instanceof Error ? err.message : "Delete failed.");
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
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      )}
      <section className="chat-panel workspace-control-panel">
        <h3>Workspace Library</h3>
        {!workspaceEnabled && (
          <div className="readonly-banner">Workspace is available in Local mode only.</div>
        )}
        {workspaceEnabled && (
          <>
            <div className="workspace-filter-row">
              <input
                type="text"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search by name or path..."
              />
              <button type="button" onClick={() => void refresh()} disabled={loading}>
                {loading ? "Refreshing..." : "Search"}
              </button>
            </div>
            <div className="workspace-filter-row">
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="">All Sources</option>
                {sourceOptions.map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
              <select value={fileType} onChange={(e) => setFileType(e.target.value)}>
                <option value="">All Categories</option>
                <option value="sequence">Sequence</option>
                <option value="structure">Structure</option>
                <option value="table_or_text">Table/Text</option>
                <option value="other">Other</option>
              </select>
              <select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}>
                <option value="created_desc">Newest First</option>
                <option value="name_asc">Name A-Z</option>
                <option value="size_desc">Largest First</option>
              </select>
              <select value={bucket} onChange={(e) => setBucket(e.target.value)}>
                <option value="">All Buckets</option>
                <option value="user_upload">User Upload</option>
                <option value="chat_session">Chat Session</option>
                <option value="tool_upload">Tool Upload</option>
              </select>
              <label className="workspace-upload-btn">
                {uploading ? "Uploading..." : "Upload"}
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
        <h3>Files ({bucketFilteredItems.length})</h3>
        {!workspaceEnabled ? (
          <div className="session-empty">
            Workspace browsing is disabled in Online mode.
          </div>
        ) : bucketFilteredItems.length === 0 ? (
          <div className="session-empty">No files found. Upload files or clear filters.</div>
        ) : (
          <div className="workspace-table">
            <div className="workspace-row workspace-header-row">
              <span>Name</span>
              <span>Bucket</span>
              <span>Source</span>
              <span>Category</span>
              <span>Size</span>
              <span>Updated</span>
              <span>Actions</span>
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
                        {actingId === item.id ? "Working..." : "Replace"}
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
                        Delete
                      </button>
                    </>
                  ) : (
                    <span className="workspace-row-readonly">Readonly</span>
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
            <h4 id="workspace-delete-title">Confirm Delete</h4>
            <p id="workspace-delete-desc">
              Delete <strong>{pendingDeleteItem.display_name}</strong>? This cannot be undone.
            </p>
            <div className="workspace-modal-actions">
              <button
                ref={cancelDeleteBtnRef}
                type="button"
                className="workspace-row-btn"
                disabled={Boolean(actingId)}
                onClick={cancelDelete}
              >
                Cancel
              </button>
              <button
                type="button"
                className="workspace-row-btn danger"
                disabled={Boolean(actingId)}
                onClick={() => void confirmDelete()}
              >
                {Boolean(actingId) ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
      <PageFooter />
    </div>
  );
}
