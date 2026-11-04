import { useEffect, useMemo, useState } from "react";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";
import "../settings.css";
import { AdminLLMs } from "../components/AdminLLMs";
import { LLM_PROVIDERS, llmProvider, llmSettingLabel } from "../lib/llmSettings";

const STRINGS = {
  en: {
    title: "Environment variables", subtitle: "Model connections and runtime configuration.", search: "Search settings…",
    models: "LLMs", searchGroup: "Web search", limits: "Usage limits", advanced: "Advanced",
    providers: "Model providers", responses: "Response preferences", searchServices: "Search services",
    limitsGroup: "Task and conversation limits", advancedGroup: "System settings",
    modelHint: "API connections used by Agent, Scientist and Researcher.",
    searchHint: "Services for finding papers and protein information.",
    limitsHint: "Control task size, execution time and daily usage.",
    advancedHint: "Storage, tools and server configuration.",
    reload: "Refresh", save: "Save changes", saving: "Saving…", reset: "Discard changes",
    loading: "Loading settings…", loadError: "Could not load settings. Try refreshing.",
    saveError: "Could not save changes. Please try again.", conflict: "Settings changed in another window. Reload the page before editing again.",
    saved: "Saved. Restart the relevant service to apply changes.", changes: "unsaved changes",
    configured: "Configured", notSet: "Not set", useDefault: "Default", on: "On", off: "Off",
    show: "Show", hide: "Hide", noMatch: "No matching settings", clear: "Clear search",
    localOnly: "Connection settings are managed on the local installation.",
    availableLocal: "Available locally", local: "Local", online: "Online", results: "Search results"
  },
  zh: {
    title: "环境变量", subtitle: "模型连接与运行配置。", search: "搜索设置…",
    models: "LLMs", searchGroup: "网络检索", limits: "使用限额", advanced: "高级设置",
    providers: "模型服务", responses: "回答偏好", searchServices: "检索服务",
    limitsGroup: "任务与对话限额", advancedGroup: "系统设置",
    modelHint: "Agent、Scientist 和 Researcher 使用的 API 连接。",
    searchHint: "用于查询文献和蛋白质信息的服务。",
    limitsHint: "设置任务规模、执行时间和每日使用量。",
    advancedHint: "存储、工具和服务器配置。",
    reload: "刷新", save: "保存修改", saving: "保存中…", reset: "撤销修改",
    loading: "正在加载设置…", loadError: "设置加载失败，请刷新重试。",
    saveError: "修改未保存，请重试。", conflict: "设置已在其他窗口更新，请重新加载页面后再编辑。",
    saved: "已保存，重启相关服务后生效。", changes: "项未保存修改",
    configured: "已配置", notSet: "未配置", useDefault: "默认", on: "开启", off: "关闭",
    show: "显示", hide: "隐藏", noMatch: "没有匹配的设置", clear: "清除搜索",
    localOnly: "连接设置在本地运行时管理。", availableLocal: "本地可用", local: "本地", online: "在线", results: "搜索结果"
  }
};

type Group = "models" | "search" | "limits" | "advanced";
type EnvEntry = { key: string; value: string; section?: string | null };
const LABELS: Record<string, [string, string]> = {
  DMXAPI_API_KEY: ["DMX API key", "DMX API 密钥"], OPENROUTER_API_KEY: ["OpenRouter API key", "OpenRouter API 密钥"],
  CHAT_MODEL_NAME: ["Default model", "默认模型"], CHAT_TEMPERATURE: ["Temperature", "采样温度"],
  CHAT_MAX_TOKENS: ["Maximum response tokens", "最大回答长度（token）"], CHAT_CODE_MAX_TOKENS: ["Maximum code tokens", "最大代码长度（token）"],
  VENUS_EXECUTION_CATALOG: ["Model and harness catalog", "模型与 Harness 配置文件"],
  TAVILY_API_KEY: ["Tavily API key", "Tavily API 密钥"], SERP_API_KEY: ["SerpAPI key", "SerpAPI 密钥"],
  SEMANTIC_SCHOLAR_API_KEY: ["Semantic Scholar API key", "Semantic Scholar API 密钥"], FDA_API_KEY: ["FDA API key", "FDA API 密钥"],
  BRENDA_EMAIL: ["BRENDA email", "BRENDA 邮箱"], BRENDA_PASSWORD: ["BRENDA password", "BRENDA 密码"],
  SEARCH_MAX_RESULTS: ["Results per search", "每次检索结果数"], MAX_STEP_RETRIES: ["Retries per step", "每步重试次数"],
  TOOL_EXECUTION_TIMEOUT: ["Tool timeout (seconds)", "工具超时（秒）"],
  WEBUI_V2_AGENT_CONTEXT_TOKENS: ["Conversation context budget (tokens)", "对话上下文预算（token）"],
  WEBUI_V2_AGENT_RECENT_TURNS: ["Recent turns kept in full", "保留原文的最近轮数"],
  WEBUI_V2_AGENT_COMPACT_PERCENT: ["Compress at usage (%)", "自动压缩阈值（%）"],
  WEBUI_V2_AGENT_SUMMARY_TOKENS: ["Memory summary budget (tokens)", "历史摘要预算（token）"],
  WEBUI_V2_AGENT_MAX_TURNS: ["Maximum turns per conversation", "每段对话最多轮数"],
  WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT: ["Daily new conversations per visitor", "每位访客每日新对话次数"],
  VENUS_ONLINE_IP_STARTS_PER_MINUTE: ["New conversations per IP per minute", "每个 IP 每分钟新对话上限"],
  VENUS_QUOTA_TIMEZONE: ["Daily credit reset timezone", "每日额度重置时区"],
  VENUS_TRUSTED_PROXIES: ["Trusted reverse proxies", "可信反向代理"],
  WEBUI_V2_ONLINE_FASTA_LIMIT: ["Sequences per online task", "在线任务序列数"],
  WEBUI_V2_ONLINE_SEQUENCE_DESIGN_LIMIT: ["Designs per online task", "在线任务设计数量"],
  WEBUI_V2_LOCAL_DOWNLOAD_BATCH_LIMIT: ["Local download batch size", "本地批量下载数量"],
  WEBUI_V2_ONLINE_DOWNLOAD_BATCH_LIMIT: ["Online download batch size", "在线批量下载数量"],
  WEBUI_V2_SESSION_TOKEN_TTL_HOURS: ["Session duration (hours)", "会话有效期（小时）"],
  WEBUI_V2_SESSION_TOKEN_SECRET: ["Session signing secret", "会话签名密钥"],
  WEBUI_V2_MODE: ["Run mode", "运行模式"], WEBUI_V2_HOST: ["Listen address", "监听地址"], WEBUI_V2_PORT: ["Web port", "网页端口"],
  WEBUI_V2_ALLOW_REMOTE_SETTINGS: ["Remote settings access", "远程设置访问"], WEBUI_V2_DEV_MODE: ["Development mode", "开发模式"],
  WEBUI_V2_FRONTEND_DIST: ["Frontend directory", "前端目录"], WEBUI_V2_FRONTEND_DEV_URL: ["Development server", "前端开发地址"],
  WEBUI_V2_CORS_ORIGINS: ["Allowed origins", "允许的访问来源"], WEBUI_V2_LOG_LEVEL: ["Web log level", "网页日志级别"],
  LANGSMITH_TRACING: ["LangSmith tracing", "LangSmith 追踪"], LANGSMITH_ENDPOINT: ["LangSmith endpoint", "LangSmith 地址"],
  LANGSMITH_API_KEY: ["LangSmith API key", "LangSmith API 密钥"], LANGSMITH_PROJECT: ["LangSmith project", "LangSmith 项目"],
  TEMP_OUTPUTS_DIR: ["File storage directory", "文件存储目录"], TEMP_OUTPUTS_ENABLE_TTL_CLEANUP: ["Automatic file cleanup", "自动清理文件"],
  MCP_HTTP_HOST: ["MCP listen address", "MCP 监听地址"], MCP_HTTP_PORT: ["MCP port", "MCP 端口"],
  SCP_HUB_API_KEY: ["SCP Hub API key", "SCP Hub API 密钥"], ESMFOLD_BACKEND: ["ESMFold service", "ESMFold 服务"],
  SCP_WORKFLOW_SERVER_URL: ["Workflow server", "工作流服务地址"], DRUGSDA_MODEL_SERVER_URL: ["DrugSDA model server", "DrugSDA 模型服务地址"],
  DRUGSDA_TOOL_SERVER_URL: ["DrugSDA tool server", "DrugSDA 工具服务地址"], VENUSMINE_MMSEQS_DATABASE_PATH: ["VenusMine database", "VenusMine 数据库"],
  MMSEQS_DATABASE_PATH: ["MMseqs database", "MMseqs 数据库"], HF_ENDPOINT: ["Hugging Face endpoint", "Hugging Face 地址"],
  PT_DATA_DIR: ["Protein data directory", "蛋白质数据目录"], VENUS_CKPT_REPO_ID: ["Checkpoint repository", "模型权重仓库"],
  VENUS_CKPT_DIR: ["Checkpoint directory", "模型权重目录"], VENUS_CKPT_REVISION: ["Checkpoint version", "模型权重版本"],
  VENUS_CKPT_AUTO_DOWNLOAD: ["Automatic checkpoint downloads", "自动下载模型权重"],
  VENUS_LOG_LEVEL: ["Log level", "日志级别"], VENUS_LOG_FORMAT: ["Log format", "日志格式"],
  WEBUI_V2_ANALYTICS_DB_PATH: ["Usage database", "使用统计数据库"], WEBUI_V2_GEOIP_MMDB_PATH: ["IP location database", "IP 位置数据库"],
  WEBUI_V2_TOKEN_PRICE_JSON: ["Token pricing configuration", "Token 价格配置"],
  CUSTOM_MODEL_DEBUG_LOG_PATH: ["Training debug log", "训练调试日志"], TRAIN_ARGS_DEBUG_LOG_PATH: ["Training arguments log", "训练参数日志"],
  ADAPTER_MODEL_DEBUG_LOG_PATH: ["Adapter debug log", "适配器调试日志"], BRENDA_DEBUG: ["BRENDA debugging", "BRENDA 调试"],
  VENUS_FACTORY_VERSION_NUMBER: ["Build version", "构建版本"], VENUS_FACTORY_CUDA_VARIANT: ["CUDA version", "CUDA 版本"],
  VENUS_FEEDBACK_WEBHOOK_URL: ["Feedback webhook", "反馈 Webhook 地址"], VENUS_FEEDBACK_WEBHOOK_SECRET: ["Feedback webhook secret", "反馈 Webhook 密钥"]
};
const CONTEXT_DEFAULTS: Record<string, string> = {
  WEBUI_V2_AGENT_CONTEXT_TOKENS: "32000", WEBUI_V2_AGENT_RECENT_TURNS: "8", WEBUI_V2_AGENT_COMPACT_PERCENT: "80",
  WEBUI_V2_AGENT_SUMMARY_TOKENS: "2000", WEBUI_V2_AGENT_MAX_TURNS: "0"
};
const CONTEXT_HINTS: Record<string, [string, string]> = {
  WEBUI_V2_AGENT_CONTEXT_TOKENS: ["Estimated text budget. Original chat records are retained.", "按文本估算，原始聊天记录保留。"],
  WEBUI_V2_AGENT_RECENT_TURNS: ["Older turns become a summary; long turns may be compressed earlier.", "更早的内容转为摘要；长消息会提前压缩。"],
  WEBUI_V2_AGENT_COMPACT_PERCENT: ["Automatic compression starts at this percentage of the budget.", "达到预算的这一比例时自动压缩。"],
  WEBUI_V2_AGENT_SUMMARY_TOKENS: ["Preserves goals, results, file paths and pending work.", "保留目标、结果、文件路径和待办事项。"],
  WEBUI_V2_AGENT_MAX_TURNS: ["0 means no turn limit. This is separate from the daily chat allowance.", "0 表示不限制轮数，与每日对话额度分开。"]
};
const SEARCH_KEYS = new Set(["TAVILY_API_KEY", "SERP_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "FDA_API_KEY", "BRENDA_EMAIL", "BRENDA_PASSWORD"]);
const BOOL_KEYS = new Set(["WEBUI_V2_DEV_MODE", "WEBUI_V2_ALLOW_REMOTE_SETTINGS", "LANGSMITH_TRACING", "TEMP_OUTPUTS_ENABLE_TTL_CLEANUP", "VENUS_CKPT_AUTO_DOWNLOAD", "BRENDA_DEBUG"]);
const SECTION_LABELS: Record<string, [string, string]> = {
  "Core Runtime (required/recommended)": ["Server", "服务器"], "LLM / AI Keys": ["Model routing and tracing", "模型路由与追踪"],
  "WebUI V2 Limits (mode-based)": ["Sessions", "会话"], "Storage / Temp Outputs": ["File storage", "文件存储"],
  "MCP / Remote / DrugSDA": ["Tools and MCP", "工具与 MCP"], "Hugging Face / Data": ["Data sources", "数据源"],
  "Checkpoint Hub": ["Model weights", "模型权重"], "Logging": ["Logs", "日志"], "Analytics (optional)": ["Usage analytics", "使用统计"],
  "Advanced Debug (optional)": ["Debugging", "调试"], "Docker / Build": ["Build and feedback", "构建与反馈"]
};
export function isSecretSetting(key: string) {
  return /(?:^|_)(?:API_KEY|ACCESS_KEY|PRIVATE_KEY|TOKEN|SECRET|PASSWORD)$/.test(key) || /(?:^|_)SECRET(?:_|$)/.test(key);
}
function groupFor(key: string): Group {
  if (key.startsWith("WEBUI_V2_AGENT_")) return "limits";
  if (key.startsWith("CHAT_") || llmProvider(key) || key === "VENUS_EXECUTION_CATALOG") return "models";
  if (SEARCH_KEYS.has(key)) return "search";
  if (key.endsWith("_LIMIT") || ["SEARCH_MAX_RESULTS", "MAX_STEP_RETRIES", "TOOL_EXECUTION_TIMEOUT"].includes(key)) return "limits";
  return "advanced";
}
function boolValue(value: string) {
  if (["true", "1"].includes(value.toLowerCase())) return "true";
  if (["false", "0"].includes(value.toLowerCase())) return "false";
  return value;
}

export function SettingsPage({ readonly = false, admin = false }: { readonly?: boolean; admin?: boolean }) {
  const { lang, t: translate } = useLang();
  const t = translate(STRINGS);
  const zh = lang === "zh";
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [entries, setEntries] = useState<EnvEntry[]>([]);
  const [original, setOriginal] = useState<EnvEntry[]>([]);
  const [revision, setRevision] = useState("");
  const [llmRevision, setLlmRevision] = useState(0);
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(!readonly);
  const [saving, setSaving] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState<Group>("models");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const originalValues = useMemo(() => new Map(original.map(e => [e.key, e.value])), [original]);
  const changed = entries.filter(e => e.value !== originalValues.get(e.key));
  const dirty = changed.length > 0;
  const groups: { id: Group; label: string; hint: string }[] = [
    { id: "models", label: t.models, hint: t.modelHint }, { id: "search", label: t.searchGroup, hint: t.searchHint },
    { id: "limits", label: t.limits, hint: t.limitsHint }, { id: "advanced", label: t.advanced, hint: t.advancedHint }
  ];
  function label(key: string) {
    const providerLabel = llmSettingLabel(key, zh);
    if (providerLabel) return providerLabel;
    if (LABELS[key]) return LABELS[key][zh ? 1 : 0];
    const retention = key.match(/^TEMP_OUTPUTS_(.+)_TTL_DAYS$/);
    if (retention) {
      const types: Record<string, string> = {UPLOADS: "上传文件", WORK: "工作文件", RESULTS: "结果文件", SESSIONS: "会话文件", MANIFESTS: "索引", CACHE: "缓存", AGENT: "智能体文件"};
      return zh ? `${types[retention[1]] || retention[1]}保留天数` : `${retention[1][0]}${retention[1].slice(1).toLowerCase()} retention (days)`;
    }
    return key;
  }
  async function load() {
    setLoading(true); setError(""); setMessage("");
    try {
      const response = await fetch(admin ? "/api/admin/env" : "/api/settings/env", {cache:"no-store", credentials:"same-origin"});
      if (!response.ok) throw new Error(t.loadError);
      const data = await response.json();
      setEntries(data.entries || []); setOriginal(data.entries || []); setRevision(data.revision || ""); setVisibility({}); setLlmRevision(v => v + 1);
    } catch { setError(t.loadError); }
    finally { setLoading(false); }
  }
  useEffect(() => { if (!readonly) void load(); }, [readonly]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  function update(key: string, value: string) {
    setEntries(prev => prev.map(e => e.key === key ? { ...e, value } : e)); setMessage(""); setError("");
  }
  async function save() {
    if (!dirty || saving) return;
    setSaving(true); setError(""); setMessage("");
    try {
      const response = await fetch(admin ? "/api/admin/env" : "/api/settings/env", { method: "POST", credentials:"same-origin", headers: { "Content-Type": "application/json", ...(admin ? {"X-Venus-Access":"1"} : {}) }, body: JSON.stringify({ entries: changed, revision }) });
      if (!response.ok) throw new Error(response.status === 409 ? t.conflict : t.saveError);
      const data = await response.json();
      const savedEntries = entries.map(e => admin && isSecretSetting(e.key) && e.value ? {...e,value:"••••••••"} : e); setEntries(savedEntries); setOriginal(savedEntries); setRevision(data.revision || ""); setVisibility({}); setLlmRevision(v => v + 1); setMessage(t.saved);
    } catch (err) { setError(err instanceof Error ? err.message : t.saveError); }
    finally { setSaving(false); }
  }
  const search = query.trim().toLowerCase();
  const visible = entries.filter(e => search ? `${e.key} ${label(e.key)} ${llmProvider(e.key)?.[1] || ""} ${e.section || ""}`.toLowerCase().includes(search) : groupFor(e.key) === active);
  const sections = new Map<string, EnvEntry[]>();
  visible.forEach(entry => {
    const group = groupFor(entry.key);
    const section = group === "models" ? (llmProvider(entry.key)?.[1] || t.responses) : group === "search" ? t.searchServices : group === "limits" ? (entry.key.startsWith("WEBUI_V2_AGENT_") ? (zh ? "Agent 上下文" : "Agent context") : t.limitsGroup) : (SECTION_LABELS[entry.section || ""]?.[zh ? 1 : 0] || entry.section || t.advancedGroup);
    sections.set(section, [...(sections.get(section) || []), entry]);
  });
  function row(entry: EnvEntry) {
    const secret = isSecretSetting(entry.key);
    const id = `pref-${entry.key}`;
    const boolean = BOOL_KEYS.has(entry.key);
    const numeric = !secret && /(?:_PORT|_LIMIT|_TOKENS|_TEMPERATURE|_RESULTS|_RETRIES|_TIMEOUT|_HOURS|_DAYS|_TURNS|_PERCENT)$/.test(entry.key);
    const options = entry.key === "WEBUI_V2_MODE" ? ["local", "online"] : entry.key.endsWith("LOG_LEVEL") ? ["debug", "info", "warning", "error", "critical"] : null;
    return <div className="prefs-row" key={entry.key}>
      <div className="prefs-label"><div className="prefs-name"><label htmlFor={id}>{label(entry.key)}{entry.value !== originalValues.get(entry.key) && <span className="prefs-dot" aria-label={t.changes}/>}</label>
        {secret && <small className={`prefs-key-status ${entry.value ? "is-configured" : ""}`}>{entry.value ? t.configured : t.notSet}</small>}</div>
        <code>{entry.key}</code>
        {CONTEXT_HINTS[entry.key] && <small style={{ display: "block", marginTop: 4, fontSize: 12, lineHeight: 1.45, color: "var(--muted, #667085)" }}>{CONTEXT_HINTS[entry.key][zh ? 1 : 0]}</small>}
      </div>
      <div className="prefs-field">
        {boolean ? <select id={id} value={boolValue(entry.value)} onChange={e => update(entry.key, e.target.value)} disabled={readonly || saving}>
          <option value="">{t.useDefault}</option><option value="true">{t.on}</option><option value="false">{t.off}</option>
          {!["", "true", "false"].includes(boolValue(entry.value)) && <option value={entry.value}>{entry.value}</option>}
        </select> : options ? <select id={id} value={entry.value} onChange={e => update(entry.key, e.target.value)} disabled={readonly || saving}>
          <option value="">{t.useDefault}</option>{options.map(option => <option key={option} value={option}>{option === "local" ? t.local : option === "online" ? t.online : option}</option>)}
          {entry.value && !options.includes(entry.value) && <option value={entry.value}>{entry.value}</option>}
        </select> : <div className={`prefs-input-wrap ${secret ? "is-secret" : ""}`}>
          <input id={id} type={secret && !visibility[entry.key] ? "password" : numeric ? "number" : "text"} step={numeric ? "any" : undefined}
            value={entry.value} onChange={e => update(entry.key, e.target.value)} onFocus={e => { if (admin && secret && entry.value === "••••••••") e.target.select(); }} disabled={readonly || saving}
            placeholder={secret ? t.notSet : CONTEXT_DEFAULTS[entry.key] || t.useDefault} autoComplete="off" spellCheck={false}/>
          {secret && <button className="prefs-reveal" type="button" onClick={() => setVisibility(prev => ({ ...prev, [entry.key]: !prev[entry.key] }))}
            disabled={!entry.value || readonly || saving} aria-pressed={!!visibility[entry.key]} aria-label={`${visibility[entry.key] ? t.hide : t.show} ${label(entry.key)}`}>
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M2 10s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z"/><circle cx="10" cy="10" r="2"/>{visibility[entry.key] && <path d="m3 3 14 14"/>}</svg>
          </button>}
        </div>}
      </div>
    </div>;
  }
  return <div className={`prefs-page${admin ? " prefs-admin" : ""}`}>
    <header className="prefs-header"><div><h1>{t.title}</h1><p>{t.subtitle}</p></div>
      {!readonly && <button className="prefs-button" onClick={() => void load()} disabled={loading || saving || dirty}>{t.reload}</button>}
    </header>
    {readonly ? <div className="prefs-empty"><h2>{t.availableLocal}</h2><p>{t.localOnly}</p></div> : <>
      <div className="prefs-navigation"><nav aria-label={t.title}>{groups.map(group => <button type="button" key={group.id} aria-current={!search && active === group.id ? "page" : undefined}
        onClick={() => { setActive(group.id); setQuery(""); }}>{group.label}</button>)}</nav>
        <input type="search" aria-label={t.search} placeholder={t.search} value={query} onChange={e => setQuery(e.target.value)}/>
      </div>
      <div className="prefs-body" aria-busy={loading}>
        <div className="prefs-intro"><h2>{search ? t.results : groups.find(g => g.id === active)?.label}</h2>
          {!search && <p>{groups.find(g => g.id === active)?.hint}</p>}</div>
        {admin && !loading && (active === "models" || search) && <AdminLLMs refreshKey={llmRevision} query={search}/>}
        {loading ? <p className="prefs-empty" role="status">{t.loading}</p> : !visible.length ? <div className="prefs-empty"><p>{error || t.noMatch}</p>{search && <button className="prefs-button" onClick={() => setQuery("")}>{t.clear}</button>}</div> :
          Array.from(sections).sort(([a], [b]) => { const rank = (name: string) => { const i = LLM_PROVIDERS.findIndex(p => p[1] === name); return i < 0 ? 99 : i; }; return rank(a) - rank(b); }).map(([section, rows]) => !search && (active === "advanced" || active === "models") ? <details className="prefs-section prefs-disclosure" key={section}>
            <summary><strong>{section}</strong><div className="prefs-provider-state">{rows.some(e => isSecretSetting(e.key) && !!e.value) && <small>{t.configured}</small>}<span aria-hidden="true">＋</span></div></summary><div>{rows.map(row)}</div></details> :
            <section className="prefs-section" key={section}><h3>{section}</h3>{rows.map(row)}</section>)}
      </div>
      <footer className="prefs-savebar"><div aria-live="polite">{error ? <span className="prefs-error" role="alert">{error}</span> : message || (dirty ? `${changed.length} ${!zh && changed.length === 1 ? "unsaved change" : t.changes}` : "")}</div>
        <div className="prefs-actions">{dirty && <button type="button" className="prefs-button" disabled={saving} onClick={() => { setEntries(original); setVisibility({}); setMessage(""); setError(""); }}>{t.reset}</button>}
          <button type="button" className="prefs-button prefs-primary" disabled={!dirty || loading || saving} onClick={() => void save()}>{saving ? t.saving : t.save}</button></div>
      </footer>
    </>}
  </div>;
}
