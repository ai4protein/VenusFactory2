import { NavLink, Outlet } from "react-router-dom";
import { useMemo } from "react";
import "./manual-guide.css";
import { MANUAL_SECTIONS, langToManual, type ManualLanguage } from "../../lib/manualContent";
import { PageFooter } from "../../components/PageFooter";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";

export type ManualLayoutContext = {
  language: ManualLanguage;
};

const STRINGS = {
  en: {
    title: "User guide",
    subtitle: "From your first question to a reusable research result.",
    indexLabel: "Guide overview",
  },
  zh: {
    title: "手册",
    subtitle: "从第一次提问，到可复用的研究结果。",
    indexLabel: "手册目录",
  }
};

export function ManualLayout() {
  const { lang, t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const language = langToManual(lang);

  const tabs = useMemo(
    () => [
      { path: "index", label: t.indexLabel },
      ...MANUAL_SECTIONS.map((item) => ({
        path: item.key,
        label: item.title[lang]
      }))
    ],
    [t, lang]
  );

  return (
    <div className="manual-v2-page">
      <header className="chat-header manual-v2-header">
        <div>
          <h2>{t.title}</h2>
          <p>{t.subtitle}</p>
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

      <Outlet context={{ language } satisfies ManualLayoutContext} />
      <PageFooter showNotice={false} />
    </div>
  );
}
