import { NewChatButton } from "./NewChatButton";
import { warmPage } from "../lib/pageLoaders";
import { openQuota } from "../lib/onlineQuota";
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useLocation } from "react-router-dom";
import { LangNavLink } from "./LangLink";
import { LanguageSwitch } from "./LanguageSwitch";
import { LOGO_VENUS_URL } from "../lib/brandAssets";
import { useLang } from "../lib/i18n";

export const SidebarContext = createContext<{ target: HTMLElement | null; closeOnNarrow: () => void }>({ target: null, closeOnNarrow: () => {} });
export function SidebarContent({ children }: { children: ReactNode }) {
  const { target } = useContext(SidebarContext);
  return target ? createPortal(children, target) : null;
}
export function useSidebar() { return useContext(SidebarContext); }

function Icon({ name }: { name: "chat" | "library" | "settings" | "panel" | "language" | "manual" | "environment" | "models" | "workflow" }) {
  const paths = {
    chat: <path d="M19 11.5a7.5 7.5 0 0 1-7.5 7.5H5l-3 3V11.5A7.5 7.5 0 0 1 9.5 4h2a7.5 7.5 0 0 1 7.5 7.5Z" />,
    library: <><rect x="3" y="4" width="4" height="16" rx="1"/><rect x="9" y="4" width="4" height="16" rx="1"/><path d="m16 5 4-1 3 15-4 1z"/></>,
    settings: <><path d="m10 3-.7 2.2-2 .9L5 5.6 3 9l1.7 1.7v2.6L3 15l2 3.4 2.3-.5 2 .9L10 21h4l.7-2.2 2-.9 2.3.5 2-3.4-1.7-1.7v-2.6L21 9l-2-3.4-2.3.5-2-.9L14 3Z"/><circle cx="12" cy="12" r="3"/></>,
    models: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
    workflow: <><rect x="3" y="3" width="6" height="6" rx="1.5"/><rect x="15" y="15" width="6" height="6" rx="1.5"/><path d="M9 6h6a3 3 0 0 1 3 3v6M6 9v6a3 3 0 0 0 3 3h6"/></>,
    language: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c5 5 5 13 0 18-5-5-5-13 0-18Z"/></>,
    manual: <><path d="M12 5C9 3 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-6-1-9 1Zm0 0v15"/></>,
    environment: <><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2" fill="white"/><circle cx="16" cy="12" r="2" fill="white"/><circle cx="9" cy="18" r="2" fill="white"/></>,
    panel: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></>,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" className="app-sidebar-icon">{paths[name]}</svg>;
}

export function AppSidebar({ collapsed, toggle, mode, historyRef, onNavigate }: {
  collapsed: boolean; toggle: () => void; mode: "unknown" | "local" | "online";
  historyRef: (node: HTMLDivElement | null) => void; onNavigate: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === "zh";
  const label = (en: string, cn: string) => zh ? cn : en;
  const location = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRoot = useRef<HTMLElement>(null);
  const settingsButton = useRef<HTMLButtonElement>(null);
  const settingsPanel = useRef<HTMLDivElement>(null);
  const route = location.pathname.replace(/^\/(en|zh)(?=\/|$)/, "");
  useEffect(() => { setSettingsOpen(false); }, [route, location.search, collapsed]);
  useEffect(() => {
    if (!settingsOpen) return;
    settingsPanel.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
    const outside = (event: PointerEvent) => {
      if (!settingsRoot.current?.contains(event.target as Node)) setSettingsOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation(); setSettingsOpen(false); settingsButton.current?.focus();
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape, true);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape, true); };
  }, [settingsOpen]);
  function navigateFromSettings() { setSettingsOpen(false); onNavigate(); }
  return <aside className={`vf2-sidebar app-sidebar${collapsed ? " collapsed" : ""}`} aria-label={label("Sidebar", "侧栏")}>
    <div className="app-sidebar-top">
      <LangNavLink to="/" className="app-sidebar-brand" title="VenusFactory2" aria-label="VenusFactory2">
        <img src={LOGO_VENUS_URL} width={24} height={24} alt=""/><span>VenusFactory2</span>
      </LangNavLink>
      <button type="button" className="app-sidebar-toggle" onClick={toggle} aria-expanded={!collapsed}
        title={label(collapsed ? "Expand sidebar" : "Collapse sidebar", collapsed ? "展开侧栏" : "收起侧栏")}
        aria-label={label(collapsed ? "Expand sidebar" : "Collapse sidebar", collapsed ? "展开侧栏" : "收起侧栏")}><Icon name="panel"/></button>
    </div>
    <nav className="app-sidebar-primary" aria-label={label("Main navigation", "主导航")}>
      <NewChatButton lang={lang} onNavigate={onNavigate}/>
      <LangNavLink to="/models" onPointerEnter={() => warmPage("ModelPlazaPage")} onFocus={() => warmPage("ModelPlazaPage")} onClick={onNavigate} title={label("Model Plaza", "模型广场")}
        className={({ isActive }) => `app-sidebar-link${isActive ? " active" : ""}`}>
        <Icon name="models"/><span>{label("Model Plaza", "模型广场")}</span>
      </LangNavLink>
      <LangNavLink to="/workflows" onPointerEnter={() => warmPage("ModelWorkflowPage")} onFocus={() => warmPage("ModelWorkflowPage")} onClick={onNavigate} title={label("Workflow", "工作流")}
        className={({ isActive }) => `app-sidebar-link${isActive ? " active" : ""}`}>
        <Icon name="workflow"/><span>{label("Workflow", "工作流")}</span>
      </LangNavLink>
      <LangNavLink to="/library" onClick={onNavigate} title={mode === "online" ? label("Library · Unavailable online", "文件库 · Online 模式不可用") : label("Library", "文件库")}
        className={({ isActive }) => `app-sidebar-link${isActive ? " active" : ""}`}>
        <Icon name="library"/><span>{label("Library", "文件库")}</span>
      </LangNavLink>
    </nav>
    <div className="app-sidebar-history" ref={historyRef}/>
    <footer className="app-sidebar-footer" ref={settingsRoot}
      onBlur={event => { if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) setSettingsOpen(false); }}>
      {settingsOpen && <div className="sidebar-settings-menu" id="sidebar-settings-menu" ref={settingsPanel}
        role="dialog" aria-label={label("Settings", "设置")}>
        <div className="sidebar-settings-language">
          <Icon name="language"/><span>{label("Language", "语言")}</span><LanguageSwitch variant="compact"/>
        </div>
        <div className="sidebar-settings-divider"/>
        <LangNavLink to="/manual/index" className="sidebar-settings-item" onClick={navigateFromSettings}>
          <Icon name="manual"/><span>{label("Manual", "使用手册")}</span>
        </LangNavLink>
        {mode === "online" && <button type="button" className="sidebar-settings-item" onClick={() => { navigateFromSettings(); openQuota(); }}><Icon name="chat"/><span>{label("Extra allowance", "额外额度")}</span></button>}
        {mode === "local" && <>
          <LangNavLink to="/admin" className="sidebar-settings-item" onClick={navigateFromSettings}><Icon name="settings"/><span>{label("Administration", "管理后台")}</span></LangNavLink>
        </>}
      </div>}
      <button ref={settingsButton} type="button" className={`app-sidebar-link${settingsOpen ? " active" : ""}`}
        onClick={() => setSettingsOpen(open => !open)} title={label("Settings", "设置")} aria-label={label("Settings", "设置")}
        aria-haspopup="dialog" aria-expanded={settingsOpen} aria-controls={settingsOpen ? "sidebar-settings-menu" : undefined}>
        <Icon name="settings"/><span>{label("Settings", "设置")}</span>
        <small>{mode === "local" ? label("Local", "本地") : mode === "online" ? label("Online", "在线") : ""}</small>
      </button>
    </footer>
  </aside>;
}
