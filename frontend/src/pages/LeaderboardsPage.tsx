import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

type Board = {
  id: string;
  name: string;
  venue?: string;
  href: string;
  descEn: string;
  descZh: string;
};

const BOARDS: Board[] = [
  {
    id: "venusx",
    name: "VenusX",
    venue: "ICLR 2026",
    href: "https://ai4protein.github.io/venusx/",
    descEn:
      "Benchmark for functional-residue identification across protein language models, structural models, and conservation baselines.",
    descZh: "面向蛋白语言模型、结构模型与保守性基线的功能残基识别评测基准。"
  }
];

const STRINGS = {
  en: {
    docTitle: "Leaderboards — VenusFactory2",
    docDescription:
      "Public leaderboards and benchmarks tracked by the VenusFactory2 project, including VenusX (ICLR 2026).",
    title: "Leaderboards",
    subtitle: "Public benchmarks tracked by the VenusFactory2 project.",
    visit: "Open dashboard",
    external: "External link"
  },
  zh: {
    docTitle: "排行榜 — VenusFactory2",
    docDescription:
      "VenusFactory2 项目跟踪的公开排行榜与基准，包括 VenusX (ICLR 2026)。",
    title: "排行榜",
    subtitle: "VenusFactory2 项目跟踪的公开基准。",
    visit: "打开 Dashboard",
    external: "外部链接"
  }
};

export function LeaderboardsPage() {
  const { lang, t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });

  return (
    <div className="vf2-boards">
      <header className="vf2-boards-header">
        <div>
          <h2>{t.title}</h2>
          <p>{t.subtitle}</p>
        </div>
      </header>

      <div className="vf2-boards-grid">
        {BOARDS.map((b) => (
          <a
            key={b.id}
            className="vf2-boards-card"
            href={b.href}
            target="_blank"
            rel="noreferrer"
            title={t.external}
          >
            <div className="vf2-boards-card-head">
              <h3>{b.name}</h3>
              {b.venue && <span className="vf2-boards-venue">{b.venue}</span>}
            </div>
            <p className="vf2-boards-desc">{lang === "zh" ? b.descZh : b.descEn}</p>
            <span className="vf2-boards-cta">
              {t.visit}
              <span className="vf2-boards-arrow" aria-hidden> →</span>
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}
