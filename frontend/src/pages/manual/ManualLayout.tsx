import { NavLink, Outlet } from "react-router-dom";
import { useMemo, useState, type Dispatch, type SetStateAction } from "react";
import { MANUAL_SECTIONS, type ManualLanguage } from "../../lib/manualContent";
import { PageFooter } from "../../components/PageFooter";

export type ManualLayoutContext = {
  language: ManualLanguage;
  setLanguage: Dispatch<SetStateAction<ManualLanguage>>;
};

const MANUAL_INDEX_TAB = { path: "index", label: "Index" };

export function ManualLayout() {
  const [language, setLanguage] = useState<ManualLanguage>("English");

  const tabs = useMemo(
    () => [MANUAL_INDEX_TAB, ...MANUAL_SECTIONS.map((item) => ({ path: item.key, label: item.label }))],
    []
  );

  return (
    <div className="manual-v2-page">
      <header className="chat-header manual-v2-header">
        <div>
          <h2>Manual</h2>
          <p>Browse product documentation and usage guidance.</p>
        </div>
        <div className="manual-v2-language">
          <label htmlFor="manual-language-select">Language</label>
          <select
            id="manual-language-select"
            value={language}
            onChange={(event) => setLanguage(event.target.value as ManualLanguage)}
          >
            <option value="English">English</option>
            <option value="Chinese">Chinese</option>
          </select>
        </div>
      </header>

      <nav className="manual-v2-switcher">
        {tabs.map((tab) => (
          <NavLink
            key={tab.path}
            to={tab.path}
            className={({ isActive }) => `manual-v2-switch-item ${isActive ? "active" : ""}`}
          >
            {tab.label}
          </NavLink>
        ))}
      </nav>

      <Outlet context={{ language, setLanguage } satisfies ManualLayoutContext} />
      <PageFooter />
    </div>
  );
}
