import { pageLoaders } from "./lib/pageLoaders";
import { PageLoading } from "./components/PageLoading";
import { initializeGoogleAnalytics, type AnalyticsRuntimeConfig } from "./lib/googleAnalytics";
const OnlineLibraryPage = lazy(() => pageLoaders.OnlineLibraryPage().then(m => ({ default: m.OnlineLibraryPage })));
const AdminPage = lazy(() => pageLoaders.AdminPage().then(m => ({ default: m.AdminPage })));
import { QuotaDialog } from "./components/QuotaDialog";
import { ModelPlazaLayout, LegacyModelRedirect } from "./components/ModelPlazaLayout";
const ModelPlazaPage = lazy(() => pageLoaders.ModelPlazaPage().then(m => ({ default: m.ModelPlazaPage })));
const ModelCreatePage = lazy(() => pageLoaders.ModelCreatePage().then(m => ({ default: m.ModelCreatePage })));
const ModelWorkflowPage = lazy(() => pageLoaders.ModelWorkflowPage().then(m => ({ default: m.ModelWorkflowPage })));
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { lazy, Suspense, useEffect, useState } from "react";
import { LangNavigate } from "./components/LangLink";
import { readDefaultLang, langFromPath } from "./lib/i18n";
const ChatPage = lazy(() => pageLoaders.ChatPage().then(m => ({ default: m.ChatPage })));
const ReportPage = lazy(() => pageLoaders.ReportPage().then(m => ({ default: m.ReportPage })));
const CustomModelTrainingPage = lazy(() => pageLoaders.CustomModelTrainingPage().then(m => ({ default: m.CustomModelTrainingPage })));
const CustomModelEvaluationPage = lazy(() => pageLoaders.CustomModelEvaluationPage().then(m => ({ default: m.CustomModelEvaluationPage })));
const CustomModelPredictPage = lazy(() => pageLoaders.CustomModelPredictPage().then(m => ({ default: m.CustomModelPredictPage })));
const DirectedEvolutionPage = lazy(() => pageLoaders.DirectedEvolutionPage().then(m => ({ default: m.DirectedEvolutionPage })));
const ProteinFunctionPage = lazy(() => pageLoaders.ProteinFunctionPage().then(m => ({ default: m.ProteinFunctionPage })));
const FunctionalResiduePage = lazy(() => pageLoaders.FunctionalResiduePage().then(m => ({ default: m.FunctionalResiduePage })));
const PhysicochemicalPropertyPage = lazy(() => pageLoaders.PhysicochemicalPropertyPage().then(m => ({ default: m.PhysicochemicalPropertyPage })));
const SequenceDesignPage = lazy(() => pageLoaders.SequenceDesignPage().then(m => ({ default: m.SequenceDesignPage })));
const ProteinDiscoveryPage = lazy(() => pageLoaders.ProteinDiscoveryPage().then(m => ({ default: m.ProteinDiscoveryPage })));
const AdvancedDirectedEvolutionPage = lazy(() => pageLoaders.AdvancedDirectedEvolutionPage().then(m => ({ default: m.AdvancedDirectedEvolutionPage })));
const AdvancedProteinDiscoveryPage = lazy(() => pageLoaders.AdvancedProteinDiscoveryPage().then(m => ({ default: m.AdvancedProteinDiscoveryPage })));
const AdvancedProteinFunctionPage = lazy(() => pageLoaders.AdvancedProteinFunctionPage().then(m => ({ default: m.AdvancedProteinFunctionPage })));
const AdvancedFunctionalResiduePage = lazy(() => pageLoaders.AdvancedFunctionalResiduePage().then(m => ({ default: m.AdvancedFunctionalResiduePage })));
const AdvancedSequenceDesignPage = lazy(() => pageLoaders.AdvancedSequenceDesignPage().then(m => ({ default: m.AdvancedSequenceDesignPage })));
const SettingsInsightsPage = lazy(() => pageLoaders.SettingsInsightsPage().then(m => ({ default: m.SettingsInsightsPage })));
const ManualLayout = lazy(() => pageLoaders.ManualLayout().then(m => ({ default: m.ManualLayout })));
const ManualIndexPage = lazy(() => pageLoaders.ManualIndexPage().then(m => ({ default: m.ManualIndexPage })));
const ManualDocPage = lazy(() => pageLoaders.ManualDocPage().then(m => ({ default: m.ManualDocPage })));
import { AppSidebar, SidebarContext } from "./components/AppSidebar";
const AgentShellPage = lazy(() => pageLoaders.AgentShellPage().then(m => ({ default: m.AgentShellPage })));
const WorkspacePage = lazy(() => pageLoaders.WorkspacePage().then(m => ({ default: m.WorkspacePage })));
const HomePage = lazy(() => pageLoaders.HomePage().then(m => ({ default: m.HomePage })));
const LeaderboardsPage = lazy(() => pageLoaders.LeaderboardsPage().then(m => ({ default: m.LeaderboardsPage })));


/** Outer router: pins every URL into a `/${lang}/...` shape so that
 *  search engines and hreflang see two genuinely distinct documents
 *  (one per language) instead of a single SPA that swaps language
 *  client-side. */
export default function App() {
  return (
    <Suspense fallback={<PageLoading/>}><Routes>
      <Route path="/admin" element={<AdminPage />} />
      <Route path="/en/admin" element={<AdminPage />} />
      <Route path="/zh/admin" element={<AdminPage />} />
      {/* Bare root → redirect to the default-language root. */}
      <Route path="/" element={<DefaultLangRedirect />} />
      {/* Localized app shell. All real content lives here. */}
      <Route path="/:lang/*" element={<LocalizedApp />} />
      {/* Legacy / external links without lang prefix → prepend default lang. */}
      <Route path="/*" element={<LegacyToLangRedirect />} />
    </Routes></Suspense>
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
  "/favicon-16.png",
  "/favicon-32.png",
  "/favicon-venus.png",
  "/apple-touch-icon.png",
  "/logo-venus.png",
  "/logo-venus-192.png",
  "/logo-venus-256.png",
  "/logo-venus-512.png",
  "/logo-venus-avatar.png",
  "/og-image.png",
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.matchMedia("(max-width: 900px)").matches);
  const [sidebarTarget, setSidebarTarget] = useState<HTMLDivElement | null>(null);
  const [runtimeMode, setRuntimeMode] = useState<"unknown" | "local" | "online">("unknown");
  const location = useLocation();
  const langSeg = langFromPath(location.pathname);
  const pathBeyondLang = langSeg ? location.pathname.slice(`/${langSeg}`.length) || "/" : location.pathname;
  const localFeaturesEnabled = runtimeMode === "local";
  const workspaceEnabled = localFeaturesEnabled;
  const isLanding = pathBeyondLang === "/" || pathBeyondLang === "";
  const closeOnNarrow = () => { if (window.matchMedia("(max-width: 900px)").matches) setSidebarCollapsed(true); };
  useEffect(() => {
    const query = window.matchMedia("(max-width: 900px)");
    const update = () => { if (query.matches) setSidebarCollapsed(true); };
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") closeOnNarrow(); };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, []);

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
        const data = (await res.json()) as AnalyticsRuntimeConfig;
        if (!alive) return;
        setRuntimeMode(data.mode === "local" ? "local" : "online");
        initializeGoogleAnalytics(data);
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

  if (!langSeg) return <LegacyToLangRedirect />;

  if (isLanding) {
    return (
      <Routes>
        <Route path="" element={<HomePage />} />
      </Routes>
    );
  }

  return (
    <SidebarContext.Provider value={{ target: sidebarTarget, closeOnNarrow }}>
    <QuotaDialog/>
    <div className={`vf2-layout chat-first-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <AppSidebar collapsed={sidebarCollapsed} toggle={() => setSidebarCollapsed(v => !v)} mode={runtimeMode}
        historyRef={setSidebarTarget} onNavigate={closeOnNarrow}/>
      {!sidebarCollapsed && <button className="app-sidebar-backdrop" aria-label="Close sidebar" onClick={() => setSidebarCollapsed(true)}/>}
      <main className="vf2-main">
        <Suspense fallback={<PageLoading/>}><Routes>
          <Route path="" element={<LangNavigate to="/agent/chat" replace />} />
          <Route path="chat" element={<LangNavigate to="/agent/chat" replace />} />
          <Route path="agent" element={<AgentShellPage />}>
            <Route index element={<LangNavigate to="/agent/chat" replace />} />
            <Route path="chat/:sessionId?" element={<ChatPage workspaceEnabled={workspaceEnabled} />} />
            <Route path="workspace" element={<LangNavigate to="/library" replace />} />
          </Route>
          <Route path="library" element={runtimeMode === "online" ? <OnlineLibraryPage /> : <WorkspacePage workspaceEnabled={workspaceEnabled} />} />
          <Route path="workspace" element={<LangNavigate to="/library" replace />} />
          <Route path="report" element={<ReportPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="leaderboards" element={<LegacyModelRedirect to="/models/leaderboards" />} />
          <Route path="settings" element={runtimeMode === "unknown" ? null : <LangNavigate to={localFeaturesEnabled ? "/admin" : "/agent/chat"} replace />} />
          <Route path="settings/env" element={runtimeMode === "unknown" ? null : <LangNavigate to={localFeaturesEnabled ? "/admin?tab=environment" : "/agent/chat"} replace />} />
          <Route path="settings/insights" element={<SettingsInsightsPage />} />
          <Route path="quick-tools" element={<LangNavigate to="/models" replace />} />
          <Route path="models" element={<ModelPlazaLayout />}>
            <Route index element={<ModelPlazaPage />} />
            <Route path="new" element={<ModelCreatePage />} />
            <Route path="training" element={<CustomModelTrainingPage readonly={runtimeMode === "unknown"} workspaceEnabled={workspaceEnabled} />} />
            <Route path="evaluation" element={<CustomModelEvaluationPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />} />
            <Route path="predict" element={<CustomModelPredictPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />} />
            <Route path="leaderboards" element={<LeaderboardsPage />} />
          </Route>
              <Route path="workflows" element={<ModelWorkflowPage />} />
              <Route path="quick-tools/directed-evolution" element={<DirectedEvolutionPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/sequence-design" element={<SequenceDesignPage workspaceEnabled={workspaceEnabled} />} />
          <Route
            path="/quick-tools/protein-discovery"
            element={<ProteinDiscoveryPage readonly={!localFeaturesEnabled} workspaceEnabled={workspaceEnabled} />}
          />
          <Route path="quick-tools/protein-function" element={<ProteinFunctionPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/functional-residue" element={<FunctionalResiduePage workspaceEnabled={workspaceEnabled} />} />
          <Route path="quick-tools/physicochemical-property" element={<PhysicochemicalPropertyPage workspaceEnabled={workspaceEnabled} />} />
          <Route path="advanced-tools" element={<LangNavigate to="/models" replace />} />
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
          <Route path="download/*" element={<LangNavigate to="/agent/chat" replace />} />
          <Route path="manual" element={<ManualLayout />}>
            <Route index element={<LangNavigate to="/manual/index" replace />} />
            <Route path="index" element={<ManualIndexPage />} />
            <Route path=":section" element={<ManualDocPage />} />
            <Route path="*" element={<ManualDocPage />} />
          </Route>
          <Route
            path="/custom-model/training"
            element={<LegacyModelRedirect to="/models/training" />}
          />
          <Route
            path="/custom-model/evaluation"
            element={<LegacyModelRedirect to="/models/evaluation" />}
          />
          <Route
            path="/custom-model/predict"
            element={<LegacyModelRedirect to="/models/predict" />}
          />
        </Routes></Suspense>
      </main>
    </div>
    </SidebarContext.Provider>
  );
}
