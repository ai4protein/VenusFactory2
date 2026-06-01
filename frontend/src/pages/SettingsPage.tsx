import { useEffect, useMemo, useRef, useState } from "react";
import { PageFooter } from "../components/PageFooter";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
  en: {
    title: "Env Settings",
    subtitle: "View and edit runtime environment variables in a form-based editor.",
    onlineReadonly: "Online mode: settings are view-only in this deployment.",
    howItWorks: "How It Works",
    helpLeadA: "Variables are fixed from",
    helpLeadB: ". This page loads values from",
    helpLeadC: "and lets you edit only predefined keys.",
    displayRules: "Display Rules",
    rule1: "Config values are visible by default.",
    rule2: "Key values can be hidden/shown.",
    rule3: "Boolean values use on/off switches.",
    varTypes: "Variable Types",
    typeConfigStrong: "Config",
    typeConfigDesc: ": system/runtime configuration and limits.",
    typeKeyStrong: "Key",
    typeKeyDesc: ": platform API credentials and access tokens.",
    sectionsNote: "Sections follow",
    sectionsNote2: "heading groups.",
    filtersNote: "Filters support type, importance, and key search.",
    saveBehavior: "Save Behavior",
    save1Pre: "Only current form values are written to",
    save2: "Changes apply after restarting related services/processes.",
    metaTotal: "Total rows:",
    metaConfigured: "Configured values:",
    metaConfigRows: "Config rows:",
    metaKeyRows: "Key rows:",
    loadingBtn: "Loading...",
    reload: "Reload .env",
    hideAllKeys: "Hide all keys",
    showAllKeys: "Show all keys",
    searchKeyPlaceholder: "Search by key...",
    typeAll: "Type: All",
    typeConfig: "Type: Config",
    typeKey: "Type: Key",
    impAll: "Importance: All",
    impSensitive: "Importance: Sensitive",
    impImportant: "Importance: Important",
    impNormal: "Importance: Normal",
    savingBtn: "Saving...",
    saveBtn: "Save .env",
    noVars: "No variables found in .env.example / .env.",
    noMatch: "No variables match current search/filter.",
    sections: "Sections",
    sectionsAria: "Settings sections",
    tagKey: "Key",
    tagConfig: "Config",
    valuePlaceholder: "value",
    boolTrue: "True",
    boolFalse: "False",
    hide: "Hide",
    show: "Show",
    hideValue: "Hide value",
    showValue: "Show value",
    errLoad: "Failed to load .env settings.",
    errSave: "Failed to save .env settings.",
    errLoadStatus: "Load failed",
    errSaveStatus: "Save failed",
    createdFromEx: ".env not found, created from",
    savedPrefix: "Saved",
    savedMidEntries: "entries to"
  },
  zh: {
    title: "环境变量设置",
    subtitle: "通过表单方式查看与编辑运行时环境变量。",
    onlineReadonly: "在线模式下，设置在当前部署中仅可查看。",
    howItWorks: "工作机制",
    helpLeadA: "变量条目以",
    helpLeadB: "为准。此页面从",
    helpLeadC: "读取值，仅允许编辑预定义的 key。",
    displayRules: "展示规则",
    rule1: "Config 类型的值默认显示。",
    rule2: "Key 类型的值可隐藏 / 显示。",
    rule3: "布尔值使用开关切换。",
    varTypes: "变量类型",
    typeConfigStrong: "Config",
    typeConfigDesc: "：系统 / 运行时配置与限制。",
    typeKeyStrong: "Key",
    typeKeyDesc: "：平台 API 凭据与访问令牌。",
    sectionsNote: "分组沿用",
    sectionsNote2: "中的标题分组。",
    filtersNote: "筛选支持按类型、重要程度和 key 搜索。",
    saveBehavior: "保存行为",
    save1Pre: "只会将当前表单的值写入",
    save2: "变更需在相关服务 / 进程重启后生效。",
    metaTotal: "总行数：",
    metaConfigured: "已配置值数：",
    metaConfigRows: "Config 行数：",
    metaKeyRows: "Key 行数：",
    loadingBtn: "加载中…",
    reload: "重新加载 .env",
    hideAllKeys: "全部隐藏 Key",
    showAllKeys: "全部显示 Key",
    searchKeyPlaceholder: "按 key 搜索…",
    typeAll: "类型：全部",
    typeConfig: "类型：Config",
    typeKey: "类型：Key",
    impAll: "重要程度：全部",
    impSensitive: "重要程度：敏感",
    impImportant: "重要程度：重要",
    impNormal: "重要程度：普通",
    savingBtn: "保存中…",
    saveBtn: "保存 .env",
    noVars: "未在 .env.example / .env 中找到变量。",
    noMatch: "没有匹配当前搜索 / 筛选条件的变量。",
    sections: "分组",
    sectionsAria: "设置分组导航",
    tagKey: "Key",
    tagConfig: "Config",
    valuePlaceholder: "值",
    boolTrue: "True",
    boolFalse: "False",
    hide: "隐藏",
    show: "显示",
    hideValue: "隐藏值",
    showValue: "显示值",
    errLoad: "加载 .env 设置失败。",
    errSave: "保存 .env 设置失败。",
    errLoadStatus: "加载失败",
    errSaveStatus: "保存失败",
    createdFromEx: "未找到 .env，已从",
    savedPrefix: "已保存",
    savedMidEntries: "条变量到"
  }
};

type EnvEntry = {
  key: string;
  value: string;
  section?: string | null;
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
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
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
  const [activeSection, setActiveSection] = useState("");
  const rowsContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (readonly) return;
    void loadEnv();
  }, [readonly]);

  async function loadEnv() {
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/settings/env");
      if (!res.ok) throw new Error(`${t.errLoadStatus} (${res.status})`);
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
        setMessage(`${t.createdFromEx} ${data.source || ".env.example"}.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errLoad);
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
        throw new Error(data?.detail || `${t.errSaveStatus} (${res.status})`);
      }
      setMessage(`${t.savedPrefix} ${data.count ?? entries.length} ${t.savedMidEntries} ${data.path || ".env"}.`);
      await loadEnv();
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errSave);
    } finally {
      setSaving(false);
    }
  }

  const stats = useMemo(() => {
    const nonEmpty = entries.filter((e) => e.value.trim()).length;
    const keyCount = entries.filter((e) => isPlatformKey(e.key)).length;
    return { total: entries.length, nonEmpty, keyCount, configCount: entries.length - keyCount };
  }, [entries]);

  const filteredEntries = useMemo(() => {
    const rows: Array<{ entry: EnvEntry; idx: number; isKey: boolean }> = [];
    const query = searchText.trim().toUpperCase();
    entries.forEach((entry, idx) => {
      const isKey = isPlatformKey(entry.key);
      const importance = getImportance(entry.key);
      const typeMatched = typeFilter === "all" || (typeFilter === "key" ? isKey : !isKey);
      const importanceMatched = importanceFilter === "all" || importance === importanceFilter;
      const searchMatched = !query || entry.key.toUpperCase().includes(query);
      if (!(typeMatched && importanceMatched && searchMatched)) return;
      rows.push({ entry, idx, isKey });
    });
    return rows;
  }, [entries, importanceFilter, searchText, typeFilter]);

  const sectionedEntries = useMemo(() => {
    const order: string[] = [];
    const buckets = new Map<string, Array<{ entry: EnvEntry; idx: number; isKey: boolean }>>();
    filteredEntries.forEach((row) => {
      const section = (row.entry.section || "General").trim() || "General";
      if (!buckets.has(section)) {
        buckets.set(section, []);
        order.push(section);
      }
      buckets.get(section)?.push(row);
    });
    return order.map((section) => ({ section, rows: buckets.get(section) || [] }));
  }, [filteredEntries]);

  const allVisible = useMemo(() => {
    const keyEntries = entries.filter((entry) => isPlatformKey(entry.key));
    if (keyEntries.length === 0) return false;
    return keyEntries.every((entry) => visibility[entry.key]);
  }, [entries, visibility]);

  useEffect(() => {
    if (sectionedEntries.length === 0) {
      setActiveSection("");
      return;
    }
    if (!activeSection || !sectionedEntries.some((s) => s.section === activeSection)) {
      setActiveSection(sectionedEntries[0].section);
    }
  }, [activeSection, sectionedEntries]);

  function toSectionId(section: string) {
    return `settings-section-${section.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "")}`;
  }

  function jumpToSection(section: string) {
    setActiveSection(section);
    const el = document.getElementById(toSectionId(section));
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  useEffect(() => {
    const container = rowsContainerRef.current;
    if (!container || sectionedEntries.length === 0) return;

    const syncActiveSectionByScroll = () => {
      const containerTop = container.getBoundingClientRect().top;
      const threshold = 42;
      let next = sectionedEntries[0].section;

      for (const group of sectionedEntries) {
        const el = document.getElementById(toSectionId(group.section));
        if (!el) continue;
        const relativeTop = el.getBoundingClientRect().top - containerTop;
        if (relativeTop <= threshold) {
          next = group.section;
        } else {
          break;
        }
      }

      setActiveSection((prev) => (prev === next ? prev : next));
    };

    syncActiveSectionByScroll();
    container.addEventListener("scroll", syncActiveSectionByScroll, { passive: true });
    window.addEventListener("resize", syncActiveSectionByScroll);
    return () => {
      container.removeEventListener("scroll", syncActiveSectionByScroll);
      window.removeEventListener("resize", syncActiveSectionByScroll);
    };
  }, [sectionedEntries]);

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
          <h2>{t.title}</h2>
          <p>{t.subtitle}</p>
        </div>
      </header>
      {readonly && (
        <div className="readonly-banner" role="status" aria-live="polite">
          {t.onlineReadonly}
        </div>
      )}

      <section className="settings-stack">
        <section className="chat-panel settings-help-card">
          <h3>{t.howItWorks}</h3>
          <p className="settings-help-lead">
            {t.helpLeadA} <code>.env.example</code>{t.helpLeadB} <code>{path}</code> {t.helpLeadC}
          </p>
          <div className="settings-help-grid">
            <div className="settings-help-item">
              <div className="settings-help-item-title">{t.displayRules}</div>
              <ul>
                <li>{t.rule1}</li>
                <li>{t.rule2}</li>
                <li>{t.rule3}</li>
              </ul>
            </div>
            <div className="settings-help-item">
              <div className="settings-help-item-title">{t.varTypes}</div>
              <ul>
                <li><strong>{t.typeConfigStrong}</strong>{t.typeConfigDesc}</li>
                <li><strong>{t.typeKeyStrong}</strong>{t.typeKeyDesc}</li>
                <li>{t.sectionsNote} <code>.env.example</code> {t.sectionsNote2}</li>
                <li>{t.filtersNote}</li>
              </ul>
            </div>
            <div className="settings-help-item">
              <div className="settings-help-item-title">{t.saveBehavior}</div>
              <ul>
                <li>{t.save1Pre} <code>{path}</code>.</li>
                <li>{t.save2}</li>
              </ul>
            </div>
          </div>
          <div className="settings-meta">
            <div>{t.metaTotal} {stats.total}</div>
            <div>{t.metaConfigured} {stats.nonEmpty}</div>
            <div>{t.metaConfigRows} {stats.configCount}</div>
            <div>{t.metaKeyRows} {stats.keyCount}</div>
          </div>
        </section>

        <section className="chat-panel settings-editor">
          <fieldset className="readonly-fieldset" disabled={readonly}>
          <div className="settings-toolbar">
            <button type="button" onClick={() => void loadEnv()} disabled={loading || saving}>
              {loading ? t.loadingBtn : t.reload}
            </button>
            <button type="button" onClick={toggleShowAll} disabled={stats.keyCount === 0}>
              {allVisible ? t.hideAllKeys : t.showAllKeys}
            </button>
            <input
              className="settings-filter-input"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              placeholder={t.searchKeyPlaceholder}
            />
            <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as "all" | "config" | "key")}>
              <option value="all">{t.typeAll}</option>
              <option value="config">{t.typeConfig}</option>
              <option value="key">{t.typeKey}</option>
            </select>
            <select
              value={importanceFilter}
              onChange={(e) => setImportanceFilter(e.target.value as "all" | ImportanceLevel)}
            >
              <option value="all">{t.impAll}</option>
              <option value="sensitive">{t.impSensitive}</option>
              <option value="important">{t.impImportant}</option>
              <option value="normal">{t.impNormal}</option>
            </select>
            <button type="button" className="report-btn-primary" onClick={() => void saveEnv()} disabled={saving}>
              {saving ? t.savingBtn : t.saveBtn}
            </button>
          </div>

          <div className="settings-rows" ref={rowsContainerRef}>
            {entries.length === 0 && <div className="chat-empty">{t.noVars}</div>}
            {entries.length > 0 && filteredEntries.length === 0 && (
              <div className="chat-empty">{t.noMatch}</div>
            )}
            {sectionedEntries.length > 0 && (
              <div className="settings-layout">
                <aside className="settings-sections-nav" aria-label={t.sectionsAria}>
                  <h4 className="settings-group-title">{t.sections}</h4>
                  {sectionedEntries.map(({ section, rows }) => (
                    <button
                      key={section}
                      type="button"
                      className={`settings-section-nav-btn ${activeSection === section ? "active" : ""}`}
                      onClick={() => jumpToSection(section)}
                    >
                      <span>{section}</span>
                      <span className="settings-section-count">{rows.length}</span>
                    </button>
                  ))}
                </aside>
                <div className="settings-sections-list">
                  {sectionedEntries.map(({ section, rows }) => (
                    <section id={toSectionId(section)} className="settings-section-block" key={section}>
                      <h5 className="settings-subgroup-title">{section}</h5>
                      {rows.map(({ entry, idx, isKey }) => (
                        <div className="settings-row settings-row-inline" key={`${idx}-${entry.key}`}>
                          <div className="settings-key-block">
                            <label className="settings-key-label">{entry.key}</label>
                            <span className={`settings-key-tag ${isKey ? "secret" : "normal"}`}>{isKey ? t.tagKey : t.tagConfig}</span>
                          </div>
                          <div className="settings-row-value">
                            {parseBooleanLiteral(entry.value) == null ? (
                              <input
                                className="settings-value-input"
                                type={isKey && !visibility[entry.key] ? "password" : "text"}
                                value={entry.value}
                                placeholder={t.valuePlaceholder}
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
                                <span className="settings-bool-label">{parseBooleanLiteral(entry.value) ? t.boolTrue : t.boolFalse}</span>
                              </label>
                            )}
                          </div>
                          {isKey ? (
                            <button
                              type="button"
                              className="settings-eye-btn"
                              onClick={() => toggleVisibility(entry.key)}
                              title={visibility[entry.key] ? t.hideValue : t.showValue}
                            >
                              {visibility[entry.key] ? t.hide : t.show}
                            </button>
                          ) : (
                            <div />
                          )}
                        </div>
                      ))}
                    </section>
                  ))}
                </div>
              </div>
            )}
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
