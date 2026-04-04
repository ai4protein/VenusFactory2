import { useEffect, useMemo, useState } from "react";
import { listWorkspaceFiles, type WorkspaceFile } from "../lib/workspaceApi";

type WorkspaceFilePickerProps = {
  workspaceEnabled: boolean;
  disabled?: boolean;
  allowMultiple?: boolean;
  acceptedCategories?: Array<"sequence" | "structure" | "table_or_text" | "other">;
  buttonLabel?: string;
  onPick: (files: WorkspaceFile[]) => void;
};

export function WorkspaceFilePicker({
  workspaceEnabled,
  disabled = false,
  allowMultiple = false,
  acceptedCategories,
  buttonLabel = "Pick from Workspace",
  onPick
}: WorkspaceFilePickerProps) {
  const [open, setOpen] = useState(false);
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
      setError(err instanceof Error ? err.message : "Failed to load workspace files.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open) {
      void load();
    }
  }, [open]);

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
  }

  const blocked = disabled || !workspaceEnabled;

  return (
    <div className="workspace-picker">
      <button
        type="button"
        className="workspace-picker-trigger"
        onClick={() => setOpen((v) => !v)}
        disabled={blocked}
        title={workspaceEnabled ? "Select existing local file." : "Workspace is available in Local mode only."}
      >
        {buttonLabel}
      </button>
      {open && !blocked && (
        <div className="workspace-picker-popover">
          <div className="workspace-picker-controls">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search workspace files..."
            />
            <button type="button" onClick={() => void load()} disabled={loading}>
              {loading ? "Loading..." : "Refresh"}
            </button>
          </div>
          {error && <div className="error-box">{error}</div>}
          <div className="workspace-picker-list">
            {visibleItems.length === 0 ? (
              <div className="session-empty">No files available.</div>
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
              Use Selected
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
