import { useMemo } from "react";
import { useLang } from "../lib/i18n";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { LangLink } from "../components/LangLink";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
  en: {
    docTitle: "VenusFactory2 — Agent-Driven Protein Engineering Platform",
    docDescription: "One platform for prediction, training, design and discovery — built on 40+ AI models and 11 biological databases, accessible from chat, web or API.",
    headline1: "Agent-Driven Protein",
    headline2: "Engineering, Unified.",
    lede: "One platform for prediction, training, design and discovery — built on 40+ models and 11 biological databases, accessible from chat, web, or API.",
    stat1: "models",
    stat2: "databases",
    stat3: "tool categories",
    primaryCta: "Try the Agent",
    secondaryCta: "Read the paper",
    tertiaryCta: "See Leaderboards",
    trusted: "Featured at",
    capPredict: "Predict",
    capPredictDesc: "Function, fitness, stability — zero-shot or fine-tuned.",
    capTrain: "Train",
    capTrainDesc: "LoRA, QLoRA and 7 PEFT methods in minutes.",
    capDesign: "Design",
    capDesignDesc: "Directed evolution and de novo sequence design.",
    capDiscover: "Discover",
    capDiscoverDesc: "Mine UniProt, AlphaFold, RCSB and more.",
    footer: "Open source · AI4Protein",
    license: "Non-Commercial License",
    licenseHint: "Commercial use requires a license"
  },
  zh: {
    docTitle: "VenusFactory2 — 智能体驱动的蛋白质工程平台",
    docDescription: "一站式蛋白质工程平台：预测、训练、设计与发现，集成 40+ AI 模型与 11 个生物数据库，支持对话、网页和 API 多种访问方式。",
    headline1: "智能体驱动的",
    headline2: "蛋白质工程平台",
    lede: "一个平台覆盖预测、训练、设计与发现——集成 40+ 模型与 11 个生物数据库，可通过对话、网页或 API 使用。",
    stat1: "模型",
    stat2: "数据库",
    stat3: "工具门类",
    primaryCta: "开始使用 Agent",
    secondaryCta: "阅读论文",
    tertiaryCta: "查看排行榜",
    trusted: "已发表于",
    capPredict: "预测",
    capPredictDesc: "功能、适应度、稳定性——零样本或微调。",
    capTrain: "训练",
    capTrainDesc: "LoRA、QLoRA 等 7 种 PEFT 方法，分钟级。",
    capDesign: "设计",
    capDesignDesc: "定向进化与 de novo 序列设计。",
    capDiscover: "发现",
    capDiscoverDesc: "覆盖 UniProt、AlphaFold、RCSB 等。",
    footer: "开源项目 · AI4Protein",
    license: "非商用许可",
    licenseHint: "商业使用需申请授权"
  }
};

const PAPER_URL = "http://arxiv.org/abs/2603.27303";
const GITHUB_URL = "https://github.com/ai4protein/VenusFactory2";
const HF_URL = "https://huggingface.co/AI4Protein";

const CAPABILITY_LINKS: Array<{
  key: "capPredict" | "capTrain" | "capDesign" | "capDiscover";
  to: string;
}> = [
  { key: "capPredict", to: "/quick-tools/protein-function" },
  { key: "capTrain", to: "/custom-model/training" },
  { key: "capDesign", to: "/quick-tools/sequence-design" },
  { key: "capDiscover", to: "/quick-tools/protein-discovery" }
];

export function HomePage() {
  const { t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription });

  const stats = useMemo(
    () => [
      { value: "40+", label: t.stat1 },
      { value: "11", label: t.stat2 },
      { value: "9", label: t.stat3 }
    ],
    [t]
  );

  return (
    <div className="vf2-home">
      <div className="vf2-home-aurora" aria-hidden />
      <header className="vf2-home-topbar">
        <div className="vf2-home-brand">
          <span className="vf2-home-wordmark">VenusFactory<sup>2</sup></span>
        </div>
        <nav className="vf2-home-topnav">
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="vf2-home-topnav-link">
            GitHub
          </a>
          <a href={HF_URL} target="_blank" rel="noreferrer" className="vf2-home-topnav-link">
            Hugging Face
          </a>
          <a href={PAPER_URL} target="_blank" rel="noreferrer" className="vf2-home-topnav-link">
            arXiv
          </a>
          <LanguageSwitch variant="pill" />
        </nav>
      </header>

      <main className="vf2-home-hero">
        <div className="vf2-home-hero-copy">
          <div className="vf2-home-eyebrow">
            <span className="vf2-home-dot" aria-hidden />
            Agent · Web · API
          </div>
          <h1 className="vf2-home-headline">
            <span>{t.headline1}</span>
            <span className="vf2-home-headline-accent">{t.headline2}</span>
          </h1>
          <p className="vf2-home-lede">{t.lede}</p>

          <div className="vf2-home-stats">
            {stats.map((s) => (
              <div key={s.label} className="vf2-home-stat">
                <div className="vf2-home-stat-value">{s.value}</div>
                <div className="vf2-home-stat-label">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="vf2-home-cta-row">
            <LangLink to="/agent/chat" className="vf2-home-cta vf2-home-cta-primary">
              <span>{t.primaryCta}</span>
              <span className="vf2-home-cta-arrow" aria-hidden>
                →
              </span>
            </LangLink>
            <a
              href={PAPER_URL}
              target="_blank"
              rel="noreferrer"
              className="vf2-home-cta vf2-home-cta-ghost"
            >
              {t.secondaryCta}
            </a>
            <LangLink to="/leaderboards" className="vf2-home-cta vf2-home-cta-ghost">
              {t.tertiaryCta}
            </LangLink>
          </div>

          <div className="vf2-home-trust">
            <span className="vf2-home-trust-label">{t.trusted}</span>
            <span className="vf2-home-trust-sep" aria-hidden />
            <span className="vf2-home-trust-item">ACL 2025</span>
            <span className="vf2-home-trust-item">ICLR 2026</span>
            <span className="vf2-home-trust-item">ISMB / ECCB</span>
            <span className="vf2-home-trust-item">SJTU &amp; SII AI4Protein</span>
          </div>
        </div>

        <div className="vf2-home-hero-visual" aria-hidden>
          <HeroVisual />
        </div>
      </main>

      <section className="vf2-home-caps" aria-label="capabilities">
        {CAPABILITY_LINKS.map(({ key, to }) => (
          <LangLink key={key} to={to} className="vf2-home-cap">
            <span className="vf2-home-cap-arrow">▸</span>
            <span className="vf2-home-cap-text">
              <span className="vf2-home-cap-title">{t[key]}</span>
              <span className="vf2-home-cap-desc">{t[`${key}Desc` as keyof typeof t]}</span>
            </span>
          </LangLink>
        ))}
      </section>

      <footer className="vf2-home-footer">
        <span>{t.footer}</span>
        <span className="vf2-home-footer-dot" aria-hidden>
          ·
        </span>
        <a
          className="vf2-home-footer-link"
          href="https://github.com/ai4protein/VenusFactory2/blob/main/LICENSE"
          target="_blank"
          rel="noreferrer"
          title={t.licenseHint}
        >
          {t.license}
        </a>
      </footer>
    </div>
  );
}

function HeroVisual() {
  return (
    <svg viewBox="0 0 420 460" className="vf2-home-visual-svg" role="img" aria-hidden>
      <defs>
        <radialGradient id="vfHaloA" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stopColor="#e8c98a" stopOpacity="0.55" />
          <stop offset="55%" stopColor="#e8c98a" stopOpacity="0.08" />
          <stop offset="100%" stopColor="#e8c98a" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="vfStrand" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#bfa37a" />
          <stop offset="100%" stopColor="#7a6a55" />
        </linearGradient>
        <linearGradient id="vfStrandAlt" x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#d6b482" />
          <stop offset="100%" stopColor="#8f7f6a" />
        </linearGradient>
      </defs>

      <circle cx="210" cy="200" r="190" fill="url(#vfHaloA)" />

      <g
        fill="none"
        strokeLinecap="round"
        strokeWidth="2.2"
        opacity="0.95"
        stroke="url(#vfStrand)"
      >
        <path d="M70 60 C 260 90, 260 170, 70 200 C 260 230, 260 310, 70 340 C 260 370, 260 430, 70 440" />
      </g>
      <g
        fill="none"
        strokeLinecap="round"
        strokeWidth="2.2"
        opacity="0.95"
        stroke="url(#vfStrandAlt)"
      >
        <path d="M350 60 C 160 90, 160 170, 350 200 C 160 230, 160 310, 350 340 C 160 370, 160 430, 350 440" />
      </g>

      <g stroke="#a4906c" strokeWidth="1" opacity="0.55">
        {Array.from({ length: 16 }).map((_, i) => {
          const y = 70 + i * 24;
          return <line key={i} x1="92" x2="328" y1={y} y2={y} />;
        })}
      </g>

      <g fill="#d6b482">
        {Array.from({ length: 16 }).map((_, i) => {
          const y = 70 + i * 24;
          const x = i % 2 === 0 ? 92 : 328;
          return <circle key={`a-${i}`} cx={x} cy={y} r="3.4" />;
        })}
        {Array.from({ length: 16 }).map((_, i) => {
          const y = 70 + i * 24;
          const x = i % 2 === 0 ? 328 : 92;
          return <circle key={`b-${i}`} cx={x} cy={y} r="3.4" />;
        })}
      </g>
    </svg>
  );
}
