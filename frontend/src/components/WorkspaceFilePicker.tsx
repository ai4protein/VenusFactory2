import { useEffect, useMemo, useState } from "react";
import { listWorkspaceFiles, type WorkspaceFile } from "../lib/workspaceApi";
import { useLang } from "../lib/i18n";

type WorkspaceFilePickerProps = {
  workspaceEnabled: boolean;
  disabled?: boolean;
  allowMultiple?: boolean;
  acceptedCategories?: Array<"sequence" | "structure" | "table_or_text" | "other">;
  buttonLabel?: string;
  /** Hide the built-in trigger button (use with controlled open from parent). */
  hideTrigger?: boolean;
  /** Controlled open state. When omitted, component manages its own open state. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onPick: (files: WorkspaceFile[]) => void;
};

const STRINGS = {
  en: {
    trigger: "Pick from Workspace",
    enabledHint: "Select existing local file.",
    disabledHint: "Workspace is available in Local mode only. Deploy VenusFactory2 locally to use this.",
    searchPlaceholder: "Search workspace files...",
    loading: "Loading...",
    refresh: "Refresh",
    empty: "No files available.",
    useSelected: "Use Selected",
    loadFailed: "Failed to load workspace files.",
    close: "Close",
  },
  zh: {
    trigger: "从工作区选择",
    enabledHint: "选择已有的本地文件。",
    disabledHint: "工作区仅在本地部署模式下可用，请本地部署 VenusFactory2 后使用。",
    searchPlaceholder: "搜索工作区文件…",
    loading: "加载中…",
    refresh: "刷新",
    empty: "暂无可选文件。",
    useSelected: "使用所选",
    loadFailed: "工作区文件加载失败。",
    close: "关闭",
  }
};

export function WorkspaceFilePicker({
  workspaceEnabled,
  disabled = false,
  allowMultiple = false,
  acceptedCategories,
  buttonLabel,
  hideTrigger = false,
  open: openProp,
  onOpenChange,
  onPick
}: WorkspaceFilePickerProps) {
  const t = useLang().t(STRINGS);
  const triggerLabel = buttonLabel ?? t.trigger;
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const controlled = typeof openProp === "boolean";
  const open = controlled ? Boolean(openProp) : uncontrolledOpen;

  function setOpen(next: boolean) {
    if (!controlled) setUncontrolledOpen(next);
    onOpenChange?.(next);
  }

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<WorkspaceFile[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  async function load() {
    if (!workspaceEnabled || disabled) return;
    setLoading(true);
    setError("");
    try {
      const data = await listWorkspaceFiles({
        q: query,
        includeSessions: false
      });
      setItems(data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadFailed);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open && workspaceEnabled && !disabled) {
      void load();
    }
  }, [open, workspaceEnabled, disabled]);

  const visibleItems = useMemo(() => {
    if (!acceptedCategories || acceptedCategories.length === 0) {
      return items;
    }
    const allow = new Set(acceptedCategories);
    return items.filter((item) => allow.has(item.category as "sequence" | "structure" | "table_or_text" | "other"));
  }, [items, acceptedCategories]);

  function toggleSelect(id: string) {
    setSelectedIds((prev) => {
      if (!allowMultiple) {
        return prev[0] === id ? [] : [id];
      }
      return prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id];
    });
  }

  function confirmPick() {
    const set = new Set(selectedIds);
    const picked = visibleItems.filter((item) => set.has(item.id));
    onPick(picked);
    setOpen(false);
    setSelectedIds([]);
  }

  const blocked = disabled || !workspaceEnabled;

  return (
    <div className={`workspace-picker${hideTrigger ? " workspace-picker--external" : ""}`}>
      {!hideTrigger && (
        <button
          type="button"
          className="workspace-picker-trigger"
          onClick={() => setOpen(!open)}
          disabled={blocked}
          title={workspaceEnabled ? t.enabledHint : t.disabledHint}
        >
          {triggerLabel}
        </button>
      )}
      {open && !blocked && (
        <div className="workspace-picker-popover" role="dialog" aria-label={triggerLabel}>
          <div className="workspace-picker-controls">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t.searchPlaceholder}
            />
            <button type="button" onClick={() => void load()} disabled={loading}>
              {loading ? t.loading : t.refresh}
            </button>
            {hideTrigger && (
              <button type="button" className="workspace-picker-close" onClick={() => setOpen(false)}>
                {t.close}
              </button>
            )}
          </div>
          {error && <div className="error-box">{error}</div>}
          <div className="workspace-picker-list">
            {visibleItems.length === 0 ? (
              <div className="session-empty">{t.empty}</div>
            ) : (
              visibleItems.slice(0, 100).map((item) => {
                const checked = selectedIds.includes(item.id);
                return (
                  <label key={item.id} className="workspace-picker-item">
                    <input
                      type={allowMultiple ? "checkbox" : "radio"}
                      name="workspace-select"
                      checked={checked}
                      onChange={() => toggleSelect(item.id)}
                    />
                    <span className="workspace-picker-item-name">{item.display_name}</span>
                    <small>{item.source}</small>
                  </label>
                );
              })
            )}
          </div>
          <div className="workspace-picker-actions">
            <button type="button" onClick={confirmPick} disabled={selectedIds.length === 0}>
              {t.useSelected}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
