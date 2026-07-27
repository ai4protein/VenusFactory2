import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { LangNavLink, LangNavigate } from "./components/LangLink";
import { readDefaultLang, langFromPath } from "./lib/i18n";
import { ChatPage } from "./pages/ChatPage";
import { ReportPage } from "./pages/ReportPage";
import { ModuleShellPage } from "./pages/ModuleShellPage";
import { CustomModelTrainingPage } from "./pages/CustomModelTrainingPage";
import { CustomModelEvaluationPage } from "./pages/CustomModelEvaluationPage";
import { CustomModelPredictPage } from "./pages/CustomModelPredictPage";
import { DirectedEvolutionPage } from "./pages/quick-tools/DirectedEvolutionPage";
import { ProteinFunctionPage } from "./pages/quick-tools/ProteinFunctionPage";
import { FunctionalResiduePage } from "./pages/quick-tools/FunctionalResiduePage";
import { PhysicochemicalPropertyPage } from "./pages/quick-tools/PhysicochemicalPropertyPage";
import { SequenceDesignPage } from "./pages/quick-tools/SequenceDesignPage";
import { ProteinDiscoveryPage } from "./pages/quick-tools/ProteinDiscoveryPage";
import { AdvancedDirectedEvolutionPage } from "./pages/advanced-tools/AdvancedDirectedEvolutionPage";
import { AdvancedProteinDiscoveryPage } from "./pages/advanced-tools/AdvancedProteinDiscoveryPage";
import { AdvancedProteinFunctionPage } from "./pages/advanced-tools/AdvancedProteinFunctionPage";
import { AdvancedFunctionalResiduePage } from "./pages/advanced-tools/AdvancedFunctionalResiduePage";
import { AdvancedSequenceDesignPage } from "./pages/advanced-tools/AdvancedSequenceDesignPage";
import { UniProtDownloadPage } from "./pages/download/UniProtDownloadPage";
import { NcbiDownloadPage } from "./pages/download/NcbiDownloadPage";
import { RcsbStructureDownloadPage } from "./pages/download/RcsbStructureDownloadPage";
import { AlphaFoldDownloadPage } from "./pages/download/AlphaFoldDownloadPage";
import { RcsbMetadataDownloadPage } from "./pages/download/RcsbMetadataDownloadPage";
import { InterProDownloadPage } from "./pages/download/InterProDownloadPage";
import { SettingsPage } from "./pages/SettingsPage";
import { SettingsInsightsPage } from "./pages/SettingsInsightsPage";
import { ManualLayout } from "./pages/manual/ManualLayout";
import { ManualIndexPage } from "./pages/manual/ManualIndexPage";
import { ManualDocPage } from "./pages/manual/ManualDocPage";
import { RuntimeModeBadge } from "./components/RuntimeModeBadge";
import { AgentShellPage } from "./pages/AgentShellPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { HomePage } from "./pages/HomePage";
import { LeaderboardsPage } from "./pages/LeaderboardsPage";
import { useLang } from "./lib/i18n";
import { LanguageSwitch } from "./components/LanguageSwitch";

const SIDEBAR_STRINGS = {
  en: {
    agent: "Agent",
    report: "Report",
    leaderboards: "Leaderboards",
    quickTools: "Quick Tools",
    advancedTools: "Advanced Tools",
    settings: "Settings",
    download: "Download",
    manual: "Manual",
    customModel: "Custom Model",
    chat: "Chat",
    workspace: "Workspace",
    train: "Train",
    evaluate: "Evaluate",
    predict: "Predict",
    directedEvolution: "Directed Evolution",
    sequenceDesign: "Sequence Design",
    proteinDiscovery: "Protein Discovery",
    proteinFunction: "Protein Function",
    functionalResidue: "Functional Residue",
    physicochemicalProperty: "Physicochemical Property",
    uniprot: "UniProt",
    ncbi: "NCBI",
    rcsbStructure: "RCSB Structure",
    alphafold: "AlphaFold",
    rcsbMetadata: "RCSB Metadata",
    interproMetadata: "InterPro Metadata",
    index: "Index",
    prediction: "Prediction",
    faq: "FAQ",
    envSettings: "Env Settings",
    insights: "Insights",
    conference: "conf",
    arxiv: "Arxiv",
    hf: "Hugging Face",
    github: "GitHub",
    expandSidebar: "Expand sidebar",
    collapseSidebar: "Collapse sidebar",
    brandHome: "VenusFactory2 — Home"
  },
  zh: {
    agent: "智能体",
    report: "报告",
    leaderboards: "排行榜",
    quickTools: "快速工具",
    advancedTools: "高级工具",
    settings: "设置",
    download: "下载",
    manual: "手册",
    customModel: "自定义模型",
    chat: "对话",
    workspace: "工作区",
    train: "训练",
    evaluate: "评估",
    predict: "预测",
    directedEvolution: "定向进化",
    sequenceDesign: "序列设计",
    proteinDiscovery: "蛋白发现",
    proteinFunction: "蛋白功能",
    functionalResidue: "功能残基",
    physicochemicalProperty: "理化性质",
    uniprot: "UniProt",
    ncbi: "NCBI",
    rcsbStructure: "RCSB 结构",
    alphafold: "AlphaFold",
    rcsbMetadata: "RCSB 元数据",
    interproMetadata: "InterPro 元数据",
    index: "目录",
    prediction: "预测",
    faq: "常见问题",
    envSettings: "环境设置",
    insights: "洞察",
    conference: "会议",
    arxiv: "Arxiv",
    hf: "Hugging Face",
    github: "GitHub",
    expandSidebar: "展开侧边栏",
    collapseSidebar: "收起侧边栏",
    brandHome: "VenusFactory2 — 首页"
  }
};

type LabelKey = keyof typeof SIDEBAR_STRINGS["en"];

/** Compact labels for the ~84px collapsed left rail. Full text stays in `title`. */
const SIDEBAR_SHORT: Record<"en" | "zh", Partial<Record<LabelKey, string>>> = {
  en: {
    agent: "Agent",
    report: "Report",
    leaderboards: "Ranks",
    quickTools: "Quick",
    advancedTools: "Adv",
    settings: "Set",
    download: "DL",
    manual: "Docs",
    customModel: "Model",
  },
  zh: {
    agent: "智能",
    report: "报告",
    leaderboards: "排行",
    quickTools: "快速",
    advancedTools: "高级",
    settings: "设置",
    download: "下载",
    manual: "手册",
    customModel: "模型",
  },
};

const MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/agent", labelKey: "agent", status: "Available" },
  { path: "/report", labelKey: "report", status: "Available" },
  { path: "/leaderboards", labelKey: "leaderboards", status: "Available" },
  { path: "/quick-tools", labelKey: "quickTools", status: "Available" },
  { path: "/advanced-tools", labelKey: "advancedTools", status: "Available" },
  { path: "/settings", labelKey: "settings", status: "Available" },
  { path: "/download", labelKey: "download", status: "Available" },
  { path: "/manual", labelKey: "manual", status: "Available" }
];

const AGENT_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/agent/chat", labelKey: "chat", status: "Available" },
  { path: "/report", labelKey: "report", status: "Available" },
  { path: "/agent/workspace", labelKey: "workspace", status: "Available" }
];

const CUSTOM_MODEL_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/custom-model/training", labelKey: "train", status: "Available" },
  { path: "/custom-model/evaluation", labelKey: "evaluate", status: "Available" },
  { path: "/custom-model/predict", labelKey: "predict", status: "Available" }
];

const QUICK_TOOL_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/quick-tools/directed-evolution", labelKey: "directedEvolution", status: "Available" },
  { path: "/quick-tools/sequence-design", labelKey: "sequenceDesign", status: "Available" },
  { path: "/quick-tools/protein-discovery", labelKey: "proteinDiscovery", status: "Available" },
  { path: "/quick-tools/protein-function", labelKey: "proteinFunction", status: "Available" },
  { path: "/quick-tools/functional-residue", labelKey: "functionalResidue", status: "Available" },
  { path: "/quick-tools/physicochemical-property", labelKey: "physicochemicalProperty", status: "Available" }
];

const ADVANCED_TOOL_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/advanced-tools/directed-evolution", labelKey: "directedEvolution", status: "Available" },
  { path: "/advanced-tools/sequence-design", labelKey: "sequenceDesign", status: "Available" },
  { path: "/advanced-tools/protein-discovery", labelKey: "proteinDiscovery", status: "Available" },
  { path: "/advanced-tools/protein-function", labelKey: "proteinFunction", status: "Available" },
  { path: "/advanced-tools/functional-residue", labelKey: "functionalResidue", status: "Available" }
];

const DOWNLOAD_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/download/uniprot", labelKey: "uniprot", status: "Available" },
  { path: "/download/ncbi", labelKey: "ncbi", status: "Available" },
  { path: "/download/rcsb-structure", labelKey: "rcsbStructure", status: "Available" },
  { path: "/download/alphafold", labelKey: "alphafold", status: "Available" },
  { path: "/download/rcsb-metadata", labelKey: "rcsbMetadata", status: "Available" },
  { path: "/download/interpro", labelKey: "interproMetadata", status: "Available" }
];

const MANUAL_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/manual/index", labelKey: "index", status: "Available" },
  { path: "/manual/report", labelKey: "report", status: "Available" },
  { path: "/manual/agent", labelKey: "agent", status: "Available" },
  { path: "/manual/training", labelKey: "train", status: "Available" },
  { path: "/manual/prediction", labelKey: "prediction", status: "Available" },
  { path: "/manual/evaluation", labelKey: "evaluate", status: "Available" },
  { path: "/manual/quick-tools", labelKey: "quickTools", status: "Available" },
  { path: "/manual/advanced-tools", labelKey: "advancedTools", status: "Available" },
  { path: "/manual/download", labelKey: "download", status: "Available" },
  { path: "/manual/faq", labelKey: "faq", status: "Available" }
];

const SETTINGS_MODULES: Array<{ path: string; labelKey: LabelKey; status: string }> = [
  { path: "/settings/env", labelKey: "envSettings", status: "Available" },
  { path: "/settings/insights", labelKey: "insights", status: "Available" }
];

/** Outer router: pins every URL into a `/${lang}/...` shape so that
 *  search engines and hreflang see two genuinely distinct documents
 *  (one per language) instead of a single SPA that swaps language
 *  client-side. */
export default function App() {
  return (
    <Routes>
      {/* Bare root → redirect to the default-language root. */}
      <Route path="/" element={<DefaultLangRedirect />} />
      {/* Localized app shell. All real content lives here. */}
      <Route path="/:lang/*" element={<LocalizedApp />} />
      {/* Legacy / external links without lang prefix → prepend default lang. */}
      <Route path="/*" element={<LegacyToLangRedirect />} />
    </Routes>
  );
}

function DefaultLangRedirect() {
  return <Navigate to={`/${readDefaultLang()}`} replace />;
}

// Path prefixes that are NEVER React-Router routes — they're served by
// the backend (API) or as static files. Direct navigation to these paths
// must NOT be rewritten to /:lang/... otherwise the resulting URL is
// meaningless (e.g. /en/api/foo doesn't exist on the backend).
const NON_SPA_PREFIXES = [
  "/api/",
  "/manual-docs/",
  "/img/",
  "/assets/",
  "/static/"
];

// Filenames at the root that should also bypass the redirect.
const NON_SPA_FILES = new Set([
  "/favicon.svg",
  "/favicon.ico",
  "/robots.txt",
  "/sitemap.xml",
  "/site.webmanifest",
  "/manifest.json"
]);

function isNonSpaPath(pathname: string): boolean {
  if (NON_SPA_FILES.has(pathname)) return true;
  return NON_SPA_PREFIXES.some((p) => pathname.startsWith(p));
}

function LegacyToLangRedirect() {
  const loc = useLocation();
  // Don't infinite-loop: only prepend if first segment isn't already a known lang.
  if (langFromPath(loc.pathname)) return null;
  // Don't hijack API / static-asset paths. If a user (or external link)
  // hits /api/foo or /sitemap.xml, the SPA must NOT prepend /en/ — that
  // would produce a URL the backend doesn't serve. Render null and let
  // the request 404 the normal way (in production these paths are
  // dispatched to the backend by nginx/proxy before reaching the SPA).
  if (isNonSpaPath(loc.pathname)) return null;
  const target = `/${readDefaultLang()}${loc.pathname}${loc.search}${loc.hash}`;
  return <Navigate to={target} replace />;
}

function LocalizedApp() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [customModelExpanded, setCustomModelExpanded] = useState(false);
  const [quickToolsExpanded, setQuickToolsExpanded] = useState(false);
  const [advancedToolsExpanded, setAdvancedToolsExpanded] = useState(false);
  const [downloadExpanded, setDownloadExpanded] = useState(false);
  const [manualExpanded, setManualExpanded] = useState(false);
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  const [agentExpanded, setAgentExpanded] = useState(true);
  const [runtimeMode, setRuntimeMode] = useState<"unknown" | "local" | "online">("unknown");
  const location = useLocation();
  const navigate = useNavigate();
  // Path beyond the `:lang` prefix, e.g. "/en/agent/chat" → "/agent/chat".
  // Used for route-active comparisons that were authored against the
  // pre-prefix URLs.
  const langSeg = langFromPath(location.pathname);
  const pathBeyondLang = langSeg
    ? location.pathname.slice(`/${langSeg}`.length) || "/"
    : location.pathname;
  const agentRouteActive =
    AGENT_MODULES.some((m) => pathBeyondLang.startsWith(m.path)) || pathBeyondLang === "/chat";
  const customModelRouteActive = CUSTOM_MODEL_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const quickToolsRouteActive = QUICK_TOOL_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const advancedToolsRouteActive = ADVANCED_TOOL_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const downloadRouteActive = DOWNLOAD_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const manualRouteActive = MANUAL_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const settingsRouteActive = SETTINGS_MODULES.some((m) =>
    pathBeyondLang.startsWith(m.path)
  );
  const showCustomModelChildren =
    !sidebarCollapsed && (customModelExpanded || customModelRouteActive);
  const showQuickToolChildren =
    !sidebarCollapsed && (quickToolsExpanded || quickToolsRouteActive);
  const showAdvancedToolChildren =
    !sidebarCollapsed && (advancedToolsExpanded || advancedToolsRouteActive);
  const showDownloadChildren =
    !sidebarCollapsed && (downloadExpanded || downloadRouteActive);
  const showManualChildren =
    !sidebarCollapsed && (manualExpanded || manualRouteActive);
  const showSettingsChildren =
    !sidebarCollapsed && (settingsExpanded || settingsRouteActive);
  const showAgentChildren =
    !sidebarCollapsed && (agentExpanded || agentRouteActive || pathBeyondLang.startsWith("/report"));
  const localFeaturesEnabled = runtimeMode === "local";
  const workspaceEnabled = localFeaturesEnabled;
  // Landing page = the localized root `/en` or `/zh` (no further path).
  const isLanding = pathBeyondLang === "/" || pathBeyondLang === "";
  const { lang, t: tFn } = useLang();
  const t = tFn(SIDEBAR_STRINGS);
  const short = SIDEBAR_SHORT[lang === "zh" ? "zh" : "en"];
  const navLabel = (key: LabelKey) =>
    sidebarCollapsed ? short[key] ?? t[key] : t[key];
  const go = (path: string) => navigate(`/${langSeg ?? lang}${path}`);

  // NOTE: every hook must be called above the `if (isLanding)` early return,
  // otherwise navigating between `/` and any inner route changes the hook
  // count and React throws "Rendered more hooks than during the previous
  // render", blanking the page.
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const res = await fetch("/api/runtime-config");
        if (!res.ok) {
          if (!alive) return;
          setRuntimeMode("online");
          return;
        }
        const data = (await res.json()) as { mode?: string };
        if (!alive) return;
        setRuntimeMode(data.mode === "online" ? "online" : "local");
      } catch {
        if (!alive) return;
        // Fail closed when mode cannot be determined.
        setRuntimeMode("online");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (isLanding) {
    return (
      <Routes>
        <Route path="" element={<HomePage />} />
      </Routes>
    );
  }

  return (
    <div className={`vf2-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`vf2-sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="vf2-sidebar-top">
          <LangNavLink to="/" className="vf2-brand vf2-brand-link" title={t.brandHome}>
            <h1>{sidebarCollapsed ? "VF2" : "VenusFactory2"}</h1>
          </LangNavLink>
          <button
            className="vf2-sidebar-toggle"
            type="button"
            onClick={() => setSidebarCollapsed((v) => !v)}
            aria-label={sidebarCollapsed ? t.expandSidebar : t.collapseSidebar}
          >
            {sidebarCollapsed ? "›" : "‹"}
          </button>
        </div>
        {!sidebarCollapsed && (
          <div className="vf2-sidebar-links">
            <a
              className="vf2-sidebar-link"
              href="https://aclanthology.org/2025.acl-demo.23/"
              target="_blank"
              rel="noreferrer"
              title={t.conference}
            >
              {t.conference}
            </a>
            <a
              className="vf2-sidebar-link"
              href="http://arxiv.org/abs/2603.27303"
              target="_blank"
              rel="noreferrer"
              title={t.arxiv}
            >
              {t.arxiv}
            </a>
            <a
              className="vf2-sidebar-link"
              href="https://huggingface.co/AI4Protein"
              target="_blank"
              rel="noreferrer"
              title={t.hf}
            >
              HF
            </a>
            <a
              className="vf2-sidebar-link"
              href="https://github.com/ai4protein/VenusFactory2"
              target="_blank"
              rel="noreferrer"
              title={t.github}
            >
              {t.github}
            </a>
          </div>
        )}
        <nav>
          {MODULES.filter(
            (item) =>
              item.path !== "/agent" &&
              item.path !== "/report" &&
              item.path !== "/quick-tools" &&
              item.path !== "/advanced-tools" &&
              item.path !== "/download" &&
              item.path !== "/manual" &&
              item.path !== "/settings"
          ).map((item) => (
            <LangNavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `vf2-nav-item ${isActive ? "active" : ""}`}
              title={t[item.labelKey]}
            >
              <span className="vf2-nav-label">{navLabel(item.labelKey)}</span>
            </LangNavLink>
          ))}

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${agentRouteActive || pathBeyondLang.startsWith("/report") ? "active" : ""}`}
              title={t.agent}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/agent/chat");
                  return;
                }
                setAgentExpanded((v) => !v);
              }}
              aria-expanded={showAgentChildren}
            >
              <span className="vf2-nav-label">{navLabel("agent")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showAgentChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>
            {showAgentChildren &&
              AGENT_MODULES.map((item) => {
                return (
                  <LangNavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                    title={t[item.labelKey]}
                  >
                    <span className="vf2-nav-label">{t[item.labelKey]}</span>
                  </LangNavLink>
                );
              })}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${quickToolsRouteActive ? "active" : ""}`}
              title={t.quickTools}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/quick-tools/directed-evolution");
                  return;
                }
                setQuickToolsExpanded((v) => !v);
              }}
              aria-expanded={showQuickToolChildren}
            >
              <span className="vf2-nav-label">{navLabel("quickTools")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showQuickToolChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showQuickToolChildren &&
              QUICK_TOOL_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${advancedToolsRouteActive ? "active" : ""}`}
              title={t.advancedTools}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/advanced-tools/directed-evolution");
                  return;
                }
                setAdvancedToolsExpanded((v) => !v);
              }}
              aria-expanded={showAdvancedToolChildren}
            >
              <span className="vf2-nav-label">{navLabel("advancedTools")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showAdvancedToolChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showAdvancedToolChildren &&
              ADVANCED_TOOL_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${customModelRouteActive ? "active" : ""}`}
              title={t.customModel}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/custom-model/training");
                  return;
                }
                setCustomModelExpanded((v) => !v);
              }}
              aria-expanded={showCustomModelChildren}
            >
              <span className="vf2-nav-label">{navLabel("customModel")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showCustomModelChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showCustomModelChildren &&
              CUSTOM_MODEL_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${downloadRouteActive ? "active" : ""}`}
              title={t.download}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/download/uniprot");
                  return;
                }
                setDownloadExpanded((v) => !v);
              }}
              aria-expanded={showDownloadChildren}
            >
              <span className="vf2-nav-label">{navLabel("download")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showDownloadChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showDownloadChildren &&
              DOWNLOAD_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${manualRouteActive ? "active" : ""}`}
              title={t.manual}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/manual/index");
                  return;
                }
                setManualExpanded((v) => !v);
              }}
              aria-expanded={showManualChildren}
            >
              <span className="vf2-nav-label">{navLabel("manual")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showManualChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showManualChildren &&
              MANUAL_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${settingsRouteActive ? "active" : ""}`}
              title={t.settings}
              onClick={() => {
                if (sidebarCollapsed) {
                  go("/settings/env");
                  return;
                }
                setSettingsExpanded((v) => !v);
              }}
              aria-expanded={showSettingsChildren}
            >
              <span className="vf2-nav-label">{navLabel("settings")}</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showSettingsChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showSettingsChildren &&
              SETTINGS_MODULES.map((item) => (
                <LangNavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={t[item.labelKey]}
                >
                  <span className="vf2-nav-label">{t[item.labelKey]}</span>
                </LangNavLink>
              ))}
          </div>
          {sidebarCollapsed ? (
            <RuntimeModeBadge runtimeMode={runtimeMode} placement="sidebar" />
          ) : (
            <div className="vf2-sidebar-foot-row">
              <RuntimeModeBadge runtimeMode={runtimeMode} placement="sidebar" />
              <LanguageSwitch variant="compact" />
            </div>
          )}
        </nav>
      </aside>
      <main className="vf2-main">
        <Routes>
          <Route path="" element={<LangNavigate to="/agent/chat" replace />} />
          <Route path="chat" element={<LangNavigate to="/agent/chat" replace />} />
          <Route path="agent" element={<AgentShellPage />}>
            <Route index element={<LangNavigate to="/agent/chat" replace />} />
            <Route path="chat" element={<ChatPage workspaceEnabled={workspaceEnabled} />} />
            <Route path="workspace" element={<WorkspacePage workspaceEnabled={workspaceEnabled} />} />
          </Route>
          <Route path="report" element={<ReportPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="leaderboards" element={<LeaderboardsPage />} />
          <Route path="settings" element={<LangNavigate to="/settings/env" replace />} />
          <Route path="settings/env" element={<SettingsPage readonly={!localFeaturesEnabled} />} />
          <Route path="settings/insights" element={<SettingsInsightsPage />} />
          <Route path="quick-tools" element={<LangNavigate to="/quick-tools/directed-evolution" replace />} />
          <Route path="quick-tools/directed-evolution" element={<DirectedEvolutionPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/sequence-design" element={<SequenceDesignPage workspaceEnabled={workspaceEnabled} />} />
          <Route
            path="/quick-tools/protein-discovery"
            element={<ProteinDiscoveryPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
          <Route path="quick-tools/protein-function" element={<ProteinFunctionPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/functional-residue" element={<FunctionalResiduePage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/physicochemical-property" element={<PhysicochemicalPropertyPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="advanced-tools" element={<LangNavigate to="/advanced-tools/directed-evolution" replace />} />
          <Route
            path="/advanced-tools/directed-evolution"
            element={<AdvancedDirectedEvolutionPage workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/advanced-tools/sequence-design"
            element={<AdvancedSequenceDesignPage workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/advanced-tools/protein-discovery"
            element={<AdvancedProteinDiscoveryPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/advanced-tools/protein-function"
            element={<AdvancedProteinFunctionPage workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/advanced-tools/functional-residue"
            element={<AdvancedFunctionalResiduePage workspaceEnabled={workspaceEnabled} />}
          />
          <Route path="download" element={<LangNavigate to="/download/uniprot" replace />} />
          <Route path="download/uniprot" element={<UniProtDownloadPage />} />
          <Route path="download/ncbi" element={<NcbiDownloadPage />} />
          <Route path="download/rcsb-structure" element={<RcsbStructureDownloadPage />} />
          <Route path="download/alphafold" element={<AlphaFoldDownloadPage />} />
          <Route path="download/rcsb-metadata" element={<RcsbMetadataDownloadPage />} />
          <Route path="download/interpro" element={<InterProDownloadPage />} />
          <Route path="manual" element={<LangNavigate to="/manual/index" replace />} />
          <Route path="manual/*" element={<ManualLayout />}>
            <Route path="index" element={<ManualIndexPage />} />
            <Route path=":section" element={<ManualDocPage />} />
          </Route>
          {MODULES.filter((m) => m.path !== "/agent" && m.path !== "/report" && m.path !== "/leaderboards" && m.path !== "/quick-tools" && m.path !== "/advanced-tools" && m.path !== "/download" && m.path !== "/manual" && m.path !== "/settings").map((module) => (
            <Route
              key={module.path}
              path={module.path}
              element={<ModuleShellPage title={t[module.labelKey]} status={module.status} />}
            />
          ))}
          <Route
            path="/custom-model/training"
            element={<CustomModelTrainingPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/custom-model/evaluation"
            element={<CustomModelEvaluationPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
          <Route
            path="/custom-model/predict"
            element={<CustomModelPredictPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
        </Routes>
      </main>
    </div>
  );
}
