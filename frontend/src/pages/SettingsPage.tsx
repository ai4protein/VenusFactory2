import { useEffect, useMemo, useState } from "react";
import { PageFooter } from "../components/PageFooter";

type EnvEntry = {
  key: string;
  value: string;
};

type ImportanceLevel = "sensitive" | "important" | "normal";

function isPlatformKey(key: string) {
  const k = key.toUpperCase();
  return (
    k.includes("KEY") ||
    k.includes("TOKEN") ||
    k.includes("SECRET") ||
    k.includes("PASSWORD")
  );
}

function isImportantConfigKey(key: string) {
  const k = key.toUpperCase();
  return (
    k.includes("LIMIT") ||
    k.includes("MAX") ||
    k.includes("MIN") ||
    k.includes("TIMEOUT") ||
    k.includes("RETRY") ||
    k.includes("THREAD") ||
    k.includes("WORKER") ||
    k.includes("BATCH") ||
    k.includes("PORT")
  );
}

function getImportance(key: string): ImportanceLevel {
  if (isPlatformKey(key)) return "sensitive";
  if (isImportantConfigKey(key)) return "important";
  return "normal";
}

function parseBooleanLiteral(value: string): boolean | null {
  const normalized = value.trim().toLowerCase();
  if (normalized === "true") return true;
  if (normalized === "false") return false;
  return null;
}

function isNotFoundLikeError(message: string): boolean {
  const text = String(message || "").toLowerCase();
  return text.includes("404") || text.includes("not found") || text.includes('{"detail":"not found"}');
}

type SettingsPageProps = {
  readonly?: boolean;
};

export function SettingsPage({ readonly = false }: SettingsPageProps) {
  const [entries, setEntries] = useState<EnvEntry[]>([]);
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  const [path, setPath] = useState(".env");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [typeFilter, setTypeFilter] = useState<"all" | "config" | "key">("all");
  const [importanceFilter, setImportanceFilter] = useState<"all" | ImportanceLevel>("all");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (readonly) return;
    void loadEnv();
  }, [readonly]);

  async function loadEnv() {
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/settings/env");
      if (!res.ok) throw new Error(`Load failed (${res.status})`);
      const data = (await res.json()) as {
        entries: EnvEntry[];
        path: string;
        created_from_example?: boolean;
        source?: string;
      };
      setEntries(data.entries || []);
      setVisibility((prev) => {
        const next: Record<string, boolean> = {};
        (data.entries || []).forEach((entry) => {
          const isKey = isPlatformKey(entry.key);
          next[entry.key] = isKey ? prev[entry.key] || false : true;
        });
        return next;
      });
      setPath(data.path || ".env");
      if (data.created_from_example) {
        setMessage(`.env not found, created from ${data.source || ".env.example"}.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load .env settings.");
    } finally {
      setLoading(false);
    }
  }

  function updateEntryValue(index: number, value: string) {
    setEntries((prev) => prev.map((item, idx) => (idx === index ? { ...item, value } : item)));
  }

  function toggleVisibility(key: string) {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function updateBooleanEntry(index: number, checked: boolean) {
    updateEntryValue(index, checked ? "true" : "false");
  }

  async function saveEnv() {
    setError("");
    setMessage("");
    setSaving(true);
    try {
      const res = await fetch("/api/settings/env", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || `Save failed (${res.status})`);
      }
      setMessage(`Saved ${data.count ?? entries.length} entries to ${data.path || ".env"}.`);
      await loadEnv();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save .env settings.");
    } finally {
      setSaving(false);
    }
  }

  const stats = useMemo(() => {
    const nonEmpty = entries.filter((e) => e.value.trim()).length;
    const keyCount = entries.filter((e) => isPlatformKey(e.key)).length;
    return { total: entries.length, nonEmpty, keyCount, configCount: entries.length - keyCount };
  }, [entries]);

  const groupedEntries = useMemo(() => {
    const config: Array<{ entry: EnvEntry; idx: number }> = [];
    const key: Array<{ entry: EnvEntry; idx: number }> = [];
    const query = searchText.trim().toUpperCase();
    entries.forEach((entry, idx) => {
      const isKey = isPlatformKey(entry.key);
      const importance = getImportance(entry.key);
      const typeMatched = typeFilter === "all" || (typeFilter === "key" ? isKey : !isKey);
      const importanceMatched = importanceFilter === "all" || importance === importanceFilter;
      const searchMatched = !query || entry.key.toUpperCase().includes(query);
      if (!(typeMatched && importanceMatched && searchMatched)) return;
      if (isKey) key.push({ entry, idx });
      else config.push({ entry, idx });
    });
    return { config, key };
  }, [entries, importanceFilter, searchText, typeFilter]);

  const allVisible = useMemo(() => {
    const keyEntries = entries.filter((entry) => isPlatformKey(entry.key));
    if (keyEntries.length === 0) return false;
    return keyEntries.every((entry) => visibility[entry.key]);
  }, [entries, visibility]);

  function toggleShowAll() {
    setVisibility((prev) => {
      const next: Record<string, boolean> = { ...prev };
      entries.forEach((entry) => {
        if (!isPlatformKey(entry.key)) return;
        next[entry.key] = !allVisible;
      });
      return next;
    });
  }

  const visibleError = readonly && isNotFoundLikeError(error) ? "" : error;

  return (
    <div className={`settings-page ${readonly ? "readonly-mode" : ""}`}>
      <header className="chat-header">
        <div>
          <h2>Settings</h2>
          <p>View and edit runtime environment variables in a form-based editor.</p>
        </div>
      </header>
      {readonly && (
        <div className="readonly-banner" role="status" aria-live="polite">
          Online mode: settings are view-only in this deployment.
        </div>
      )}

      <section className="settings-stack">
        <section className="chat-panel settings-help-card">
          <h3>How It Works</h3>
          <p className="settings-help-lead">
            Variables are fixed from <code>.env.example</code>. This page loads values from <code>{path}</code> and
            lets you edit only predefined keys.
          </p>
          <div className="settings-help-grid">
            <div className="settings-help-item">
              <div className="settings-help-item-title">Display Rules</div>
              <ul>
                <li>Config values are visible by default.</li>
                <li>Key values can be hidden/shown.</li>
                <li>Boolean values use on/off switches.</li>
              </ul>
            </div>
            <div className="settings-help-item">
              <div className="settings-help-item-title">Variable Types</div>
              <ul>
                <li><strong>Config</strong>: system/runtime configuration and limits.</li>
                <li><strong>Key</strong>: platform API credentials and access tokens.</li>
                <li>Filters support type, importance, and key search.</li>
              </ul>
            </div>
            <div className="settings-help-item">
              <div className="settings-help-item-title">Save Behavior</div>
              <ul>
                <li>Only current form values are written to <code>{path}</code>.</li>
                <li>Changes apply after restarting related services/processes.</li>
              </ul>
            </div>
          </div>
          <div className="settings-meta">
            <div>Total rows: {stats.total}</div>
            <div>Configured values: {stats.nonEmpty}</div>
            <div>Config rows: {stats.configCount}</div>
            <div>Key rows: {stats.keyCount}</div>
          </div>
        </section>

        <section className="chat-panel settings-editor">
          <fieldset className="readonly-fieldset" disabled={readonly}>
          <div className="settings-toolbar">
            <button type="button" onClick={() => void loadEnv()} disabled={loading || saving}>
              {loading ? "Loading..." : "Reload .env"}
            </button>
            <button type="button" onClick={toggleShowAll} disabled={stats.keyCount === 0}>
              {allVisible ? "Hide all keys" : "Show all keys"}
            </button>
            <input
              className="settings-filter-input"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              placeholder="Search by key..."
            />
            <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as "all" | "config" | "key")}>
              <option value="all">Type: All</option>
              <option value="config">Type: Config</option>
              <option value="key">Type: Key</option>
            </select>
            <select
              value={importanceFilter}
              onChange={(e) => setImportanceFilter(e.target.value as "all" | ImportanceLevel)}
            >
              <option value="all">Importance: All</option>
              <option value="sensitive">Importance: Sensitive</option>
              <option value="important">Importance: Important</option>
              <option value="normal">Importance: Normal</option>
            </select>
            <button type="button" className="report-btn-primary" onClick={() => void saveEnv()} disabled={saving}>
              {saving ? "Saving..." : "Save .env"}
            </button>
          </div>

          <div className="settings-rows">
            {entries.length === 0 && <div className="chat-empty">No variables found in .env.example / .env.</div>}
            {entries.length > 0 && groupedEntries.config.length === 0 && groupedEntries.key.length === 0 && (
              <div className="chat-empty">No variables match current search/filter.</div>
            )}
            {groupedEntries.config.length > 0 && <h4 className="settings-group-title">Config</h4>}
            {groupedEntries.config.map(({ entry, idx }) => (
              <div className="settings-row settings-row-inline" key={`${idx}-${entry.key}`}>
                <div className="settings-key-block">
                  <label className="settings-key-label">{entry.key}</label>
                  <span className="settings-key-tag normal">Config</span>
                </div>
                <div className="settings-row-value">
                  {parseBooleanLiteral(entry.value) == null ? (
                    <input
                      className="settings-value-input"
                      type="text"
                      value={entry.value}
                      placeholder="value"
                      onChange={(e) => updateEntryValue(idx, e.target.value)}
                    />
                  ) : (
                    <label className="settings-bool-toggle">
                      <input
                        type="checkbox"
                        checked={Boolean(parseBooleanLiteral(entry.value))}
                        onChange={(e) => updateBooleanEntry(idx, e.target.checked)}
                      />
                      <span className="settings-bool-slider" aria-hidden="true" />
                      <span className="settings-bool-label">{parseBooleanLiteral(entry.value) ? "True" : "False"}</span>
                    </label>
                  )}
                </div>
              </div>
            ))}
            {groupedEntries.key.length > 0 && <h4 className="settings-group-title">Key</h4>}
            {groupedEntries.key.map(({ entry, idx }) => (
              <div className="settings-row settings-row-inline" key={`${idx}-${entry.key}`}>
                <div className="settings-key-block">
                  <label className="settings-key-label">{entry.key}</label>
                  <span className="settings-key-tag secret">Key</span>
                </div>
                <div className="settings-row-value">
                  {parseBooleanLiteral(entry.value) == null ? (
                    <input
                      className="settings-value-input"
                      type={visibility[entry.key] ? "text" : "password"}
                      value={entry.value}
                      placeholder="value"
                      onChange={(e) => updateEntryValue(idx, e.target.value)}
                    />
                  ) : (
                    <label className="settings-bool-toggle">
                      <input
                        type="checkbox"
                        checked={Boolean(parseBooleanLiteral(entry.value))}
                        onChange={(e) => updateBooleanEntry(idx, e.target.checked)}
                      />
                      <span className="settings-bool-slider" aria-hidden="true" />
                      <span className="settings-bool-label">{parseBooleanLiteral(entry.value) ? "True" : "False"}</span>
                    </label>
                  )}
                </div>
                <button
                  type="button"
                  className="settings-eye-btn"
                  onClick={() => toggleVisibility(entry.key)}
                  title={visibility[entry.key] ? "Hide value" : "Show value"}
                >
                  {visibility[entry.key] ? "Hide" : "Show"}
                </button>
              </div>
            ))}
          </div>

          {message && <div className="settings-success">{message}</div>}
          {visibleError && <div className="error-box">{visibleError}</div>}
          </fieldset>
        </section>
      </section>
      <PageFooter />
    </div>
  );
}
