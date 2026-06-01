import { NavLink, Outlet } from "react-router-dom";
import { useMemo } from "react";
import { MANUAL_SECTIONS, langToManual, type ManualLanguage } from "../../lib/manualContent";
import { PageFooter } from "../../components/PageFooter";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";

export type ManualLayoutContext = {
  language: ManualLanguage;
};

const STRINGS = {
  en: {
    title: "Manual",
    subtitle: "Browse product documentation and usage guidance.",
    indexLabel: "Index",
    sectionLabels: {
      report: "Report",
      agent: "Agent",
      training: "Training",
      prediction: "Prediction",
      evaluation: "Evaluation",
      "quick-tools": "Quick Tools",
      "advanced-tools": "Advanced Tools",
      download: "Download",
      faq: "FAQ"
    } as Record<string, string>
  },
  zh: {
    title: "手册",
    subtitle: "浏览产品文档与使用指南。",
    indexLabel: "目录",
    sectionLabels: {
      report: "报告",
      agent: "智能体",
      training: "训练",
      prediction: "预测",
      evaluation: "评估",
      "quick-tools": "快速工具",
      "advanced-tools": "高级工具",
      download: "下载",
      faq: "常见问题"
    } as Record<string, string>
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
        label: t.sectionLabels[item.key] ?? item.label
      }))
    ],
    [t]
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
      <PageFooter />
    </div>
  );
}
