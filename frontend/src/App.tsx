import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
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
import { AdvancedDirectedEvolutionPage } from "./pages/advanced-tools/AdvancedDirectedEvolutionPage";
import { AdvancedProteinDiscoveryPage } from "./pages/advanced-tools/AdvancedProteinDiscoveryPage";
import { AdvancedProteinFunctionPage } from "./pages/advanced-tools/AdvancedProteinFunctionPage";
import { AdvancedFunctionalResiduePage } from "./pages/advanced-tools/AdvancedFunctionalResiduePage";
import { UniProtDownloadPage } from "./pages/download/UniProtDownloadPage";
import { NcbiDownloadPage } from "./pages/download/NcbiDownloadPage";
import { RcsbStructureDownloadPage } from "./pages/download/RcsbStructureDownloadPage";
import { AlphaFoldDownloadPage } from "./pages/download/AlphaFoldDownloadPage";
import { RcsbMetadataDownloadPage } from "./pages/download/RcsbMetadataDownloadPage";
import { InterProDownloadPage } from "./pages/download/InterProDownloadPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ManualLayout } from "./pages/manual/ManualLayout";
import { ManualIndexPage } from "./pages/manual/ManualIndexPage";
import { ManualDocPage } from "./pages/manual/ManualDocPage";
import { RuntimeModeBadge } from "./components/RuntimeModeBadge";

const MODULES = [
  { path: "/chat", label: "Agent", status: "Available" },
  { path: "/report", label: "Report", status: "Available" },
  { path: "/quick-tools", label: "Quick Tools", status: "Available" },
  { path: "/advanced-tools", label: "Advanced Tools", status: "Available" },
  { path: "/settings", label: "Settings", status: "Available" },
  { path: "/download", label: "Download", status: "Available" },
  { path: "/manual", label: "Manual", status: "Available" }
];

const CUSTOM_MODEL_MODULES = [
  { path: "/custom-model/training", label: "Train", status: "Available" },
  { path: "/custom-model/evaluation", label: "Evaluate", status: "Available" },
  { path: "/custom-model/predict", label: "Predict", status: "Available" }
];

const QUICK_TOOL_MODULES = [
  { path: "/quick-tools/directed-evolution", label: "Directed Evolution", status: "Available" },
  { path: "/quick-tools/protein-function", label: "Protein Function", status: "Available" },
  { path: "/quick-tools/functional-residue", label: "Functional Residue", status: "Available" },
  { path: "/quick-tools/physicochemical-property", label: "Physicochemical Property", status: "Available" }
];

const ADVANCED_TOOL_MODULES = [
  { path: "/advanced-tools/directed-evolution", label: "Directed Evolution", status: "Available" },
  { path: "/advanced-tools/protein-discovery", label: "Protein Discovery", status: "Available" },
  { path: "/advanced-tools/protein-function", label: "Protein Function", status: "Available" },
  { path: "/advanced-tools/functional-residue", label: "Functional Residue", status: "Available" }
];

const DOWNLOAD_MODULES = [
  { path: "/download/uniprot", label: "UniProt", status: "Available" },
  { path: "/download/ncbi", label: "NCBI", status: "Available" },
  { path: "/download/rcsb-structure", label: "RCSB Structure", status: "Available" },
  { path: "/download/alphafold", label: "AlphaFold", status: "Available" },
  { path: "/download/rcsb-metadata", label: "RCSB Metadata", status: "Available" },
  { path: "/download/interpro", label: "InterPro Metadata", status: "Available" }
];

const MANUAL_MODULES = [
  { path: "/manual/index", label: "Index", status: "Available" },
  { path: "/manual/report", label: "Report", status: "Available" },
  { path: "/manual/agent", label: "Agent", status: "Available" },
  { path: "/manual/training", label: "Train", status: "Available" },
  { path: "/manual/prediction", label: "Prediction", status: "Available" },
  { path: "/manual/evaluation", label: "Evaluate", status: "Available" },
  { path: "/manual/quick-tools", label: "Quick Tools", status: "Available" },
  { path: "/manual/advanced-tools", label: "Advanced Tools", status: "Available" },
  { path: "/manual/download", label: "Download", status: "Available" },
  { path: "/manual/faq", label: "FAQ", status: "Available" }
];

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [customModelExpanded, setCustomModelExpanded] = useState(false);
  const [quickToolsExpanded, setQuickToolsExpanded] = useState(false);
  const [advancedToolsExpanded, setAdvancedToolsExpanded] = useState(false);
  const [downloadExpanded, setDownloadExpanded] = useState(false);
  const [manualExpanded, setManualExpanded] = useState(false);
  const [runtimeMode, setRuntimeMode] = useState<"unknown" | "local" | "online">("unknown");
  const location = useLocation();
  const customModelRouteActive = CUSTOM_MODEL_MODULES.some((m) =>
    location.pathname.startsWith(m.path)
  );
  const quickToolsRouteActive = QUICK_TOOL_MODULES.some((m) =>
    location.pathname.startsWith(m.path)
  );
  const advancedToolsRouteActive = ADVANCED_TOOL_MODULES.some((m) =>
    location.pathname.startsWith(m.path)
  );
  const downloadRouteActive = DOWNLOAD_MODULES.some((m) =>
    location.pathname.startsWith(m.path)
  );
  const manualRouteActive = MANUAL_MODULES.some((m) =>
    location.pathname.startsWith(m.path)
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
  const localFeaturesEnabled = runtimeMode === "local";

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

  return (
    <div className={`vf2-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`vf2-sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="vf2-sidebar-top">
          <div className="vf2-brand">
            <h1>{sidebarCollapsed ? "VF2" : "VenusFactory2"}</h1>
          </div>
          <button
            className="vf2-sidebar-toggle"
            type="button"
            onClick={() => setSidebarCollapsed((v) => !v)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
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
              title="Conference"
            >
              📄 Conference
            </a>
            <a
              className="vf2-sidebar-link"
              href="http://arxiv.org/abs/2603.27303"
              target="_blank"
              rel="noreferrer"
              title="Arxiv"
            >
              🆕 Arxiv
            </a>
            <a
              className="vf2-sidebar-link"
              href="https://huggingface.co/AI4Protein"
              target="_blank"
              rel="noreferrer"
              title="Hugging Face"
            >
              🤗 HF
            </a>
            <a
              className="vf2-sidebar-link"
              href="https://github.com/ai4protein/VenusFactory2"
              target="_blank"
              rel="noreferrer"
              title="GitHub"
            >
              🐙 GitHub
            </a>
          </div>
        )}
        <nav>
          {MODULES.filter(
            (item) =>
              item.path !== "/quick-tools" &&
              item.path !== "/advanced-tools" &&
              item.path !== "/download" &&
              item.path !== "/manual" &&
              item.path !== "/settings"
          ).map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `vf2-nav-item ${isActive ? "active" : ""}`}
              title={item.label}
            >
              <span className="vf2-nav-label">{item.label}</span>
            </NavLink>
          ))}

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${quickToolsRouteActive ? "active" : ""}`}
              title="Quick Tools"
              onClick={() => setQuickToolsExpanded((v) => !v)}
              aria-expanded={showQuickToolChildren}
            >
              <span className="vf2-nav-label">Quick Tools</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showQuickToolChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showQuickToolChildren &&
              QUICK_TOOL_MODULES.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={item.label}
                >
                  <span className="vf2-nav-label">{item.label}</span>
                </NavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${advancedToolsRouteActive ? "active" : ""}`}
              title="Advanced Tools"
              onClick={() => setAdvancedToolsExpanded((v) => !v)}
              aria-expanded={showAdvancedToolChildren}
            >
              <span className="vf2-nav-label">Advanced Tools</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showAdvancedToolChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showAdvancedToolChildren &&
              ADVANCED_TOOL_MODULES.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={item.label}
                >
                  <span className="vf2-nav-label">{item.label}</span>
                </NavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${customModelRouteActive ? "active" : ""}`}
              title="Custom Model"
              onClick={() => setCustomModelExpanded((v) => !v)}
              aria-expanded={showCustomModelChildren}
            >
              <span className="vf2-nav-label">Custom Model</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showCustomModelChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showCustomModelChildren &&
              CUSTOM_MODEL_MODULES.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={item.label}
                >
                  <span className="vf2-nav-label">{item.label}</span>
                </NavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${downloadRouteActive ? "active" : ""}`}
              title="Download"
              onClick={() => setDownloadExpanded((v) => !v)}
              aria-expanded={showDownloadChildren}
            >
              <span className="vf2-nav-label">Download</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showDownloadChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showDownloadChildren &&
              DOWNLOAD_MODULES.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={item.label}
                >
                  <span className="vf2-nav-label">{item.label}</span>
                </NavLink>
              ))}
          </div>

          <div className="vf2-nav-group">
            <button
              type="button"
              className={`vf2-nav-item vf2-nav-parent ${manualRouteActive ? "active" : ""}`}
              title="Manual"
              onClick={() => setManualExpanded((v) => !v)}
              aria-expanded={showManualChildren}
            >
              <span className="vf2-nav-label">Manual</span>
              {!sidebarCollapsed && (
                <span className={`vf2-nav-caret ${showManualChildren ? "expanded" : ""}`}>
                  ▾
                </span>
              )}
            </button>

            {showManualChildren &&
              MANUAL_MODULES.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => `vf2-nav-item vf2-nav-subitem ${isActive ? "active" : ""}`}
                  title={item.label}
                >
                  <span className="vf2-nav-label">{item.label}</span>
                </NavLink>
              ))}
          </div>

          <NavLink
            to="/settings"
            className={({ isActive }) => `vf2-nav-item ${isActive ? "active" : ""}`}
            title="Settings"
          >
            <span className="vf2-nav-label">Settings</span>
          </NavLink>
          <RuntimeModeBadge runtimeMode={runtimeMode} placement="sidebar" />
        </nav>
      </aside>
      <main className="vf2-main">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/report" element={<ReportPage />} />
          <Route path="/settings" element={<SettingsPage readonly={!localFeaturesEnabled} />} />
          <Route path="/quick-tools" element={<Navigate to="/quick-tools/directed-evolution" replace />} />
          <Route path="/quick-tools/directed-evolution" element={<DirectedEvolutionPage />} />
          <Route path="/quick-tools/protein-function" element={<ProteinFunctionPage />} />
          <Route path="/quick-tools/functional-residue" element={<FunctionalResiduePage />} />
          <Route path="/quick-tools/physicochemical-property" element={<PhysicochemicalPropertyPage />} />
          <Route path="/advanced-tools" element={<Navigate to="/advanced-tools/directed-evolution" replace />} />
          <Route path="/advanced-tools/directed-evolution" element={<AdvancedDirectedEvolutionPage />} />
          <Route
            path="/advanced-tools/protein-discovery"
            element={<AdvancedProteinDiscoveryPage readonly={!localFeaturesEnabled} />}
          />
          <Route path="/advanced-tools/protein-function" element={<AdvancedProteinFunctionPage />} />
          <Route path="/advanced-tools/functional-residue" element={<AdvancedFunctionalResiduePage />} />
          <Route path="/download" element={<Navigate to="/download/uniprot" replace />} />
          <Route path="/download/uniprot" element={<UniProtDownloadPage />} />
          <Route path="/download/ncbi" element={<NcbiDownloadPage />} />
          <Route path="/download/rcsb-structure" element={<RcsbStructureDownloadPage />} />
          <Route path="/download/alphafold" element={<AlphaFoldDownloadPage />} />
          <Route path="/download/rcsb-metadata" element={<RcsbMetadataDownloadPage />} />
          <Route path="/download/interpro" element={<InterProDownloadPage />} />
          <Route path="/manual" element={<Navigate to="/manual/index" replace />} />
          <Route path="/manual/*" element={<ManualLayout />}>
            <Route path="index" element={<ManualIndexPage />} />
            <Route path=":section" element={<ManualDocPage />} />
          </Route>
          {MODULES.filter((m) => m.path !== "/chat" && m.path !== "/report" && m.path !== "/quick-tools" && m.path !== "/advanced-tools" && m.path !== "/download" && m.path !== "/manual" && m.path !== "/settings").map((module) => (
            <Route
              key={module.path}
              path={module.path}
              element={<ModuleShellPage title={module.label} status={module.status} />}
            />
          ))}
          <Route
            path="/custom-model/training"
            element={<CustomModelTrainingPage readonly={!localFeaturesEnabled} />}
          />
          <Route
            path="/custom-model/evaluation"
            element={<CustomModelEvaluationPage readonly={!localFeaturesEnabled} />}
          />
          <Route
            path="/custom-model/predict"
            element={<CustomModelPredictPage readonly={!localFeaturesEnabled} />}
          />
        </Routes>
      </main>
    </div>
  );
}
