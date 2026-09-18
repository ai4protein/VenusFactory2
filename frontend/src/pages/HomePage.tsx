import { useMemo, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useLang } from "../lib/i18n";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { LangLink } from "../components/LangLink";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const PAPER_URL = "http://arxiv.org/abs/2603.27303";
const ACL_URL = "https://aclanthology.org/2025.acl-demo.23/";
const VENUSX_URL = "https://ai4protein.github.io/venusx/";
const VENUSREM_URL = "https://doi.org/10.1093/bioinformatics/btaf189";
const GITHUB_URL = "https://github.com/ai4protein/VenusFactory2";
const HF_URL = "https://huggingface.co/AI4Protein";
const SJTU_URL = "https://www.sjtu.edu.cn/";
const SII_URL = "https://www.sii.edu.cn/";
const HOMEPAGE_URL = "https://tyang816.github.io/";
const LICENSE_URL = "https://github.com/ai4protein/VenusFactory2/blob/main/LICENSE";

const STRINGS = {
  en: {
    docTitle: "VenusFactory2 — Self-Evolving AI Agents for Protein Discovery and Directed Evolution",
    docDescription:
      "Self-evolving AI agents for protein discovery and directed evolution. VenusFactory2 plans an experiment, runs 40+ protein models against 11 biological databases, critiques the evidence, and iterates — from chat, web, API or CLI.",
    docKeywords:
      "self-evolving AI agents, self-evolving agents, protein discovery, directed evolution, protein engineering agent, VenusFactory2, VenusREM, Science Expert, Science Agent, AI4Protein, protein language model, LoRA, AlphaFold, UniProt",
    navAgent: "Agent",
    navTools: "Tools",
    navBoards: "Leaderboards",
    navManual: "Manual",
    navPapers: "Papers",
    banner: "Maintained by Shanghai Jiao Tong University & Shanghai Innovation Institute — free for academic use.",
    headline: "Self-evolving agents for protein discovery",
    lede: "VenusFactory2 plans a campaign, runs models and databases, critiques the evidence, then iterates — so you can spend time on the science.",
    composer: "Plan a directed-evolution campaign, then iterate on the top variants…",
    composerGo: "Start the agent",
    secondaryCta: "Read the paper",
    primaryCta: "Start a session",
    venueAcl: "ACL 2025",
    venueIsmb: "ISMB/ECCB 2025",
    venueIclr: "ICLR 2026",
    venueArxiv: "arXiv 2603.27303",
    stat1: "protein models",
    stat2: "biological databases",
    stat3: "steps in the loop",
    stat3Value: "4",
    stat3Note: "Plan · Run · Critique · Iterate",
    hlEyebrow: "The loop",
    hlTitle: "How the agents evolve",
    hlLede: "Plan an experiment, run the models, critique the evidence, then iterate.",
    hl1Title: "Plan",
    hl1Lead: "A hypothesis, then a plan you can inspect before anything runs.",
    hl1a: "Science Expert: PI → computational biologist → ML scientist → critic",
    hl1b: "Reorder, rewrite or drop steps before execution",
    hl1c: "Every later result traces back to this plan",
    hl2Title: "Run",
    hl2Lead: "Models, databases and design tools, in one session.",
    hl2a: "40+ protein language and structure models",
    hl2b: "UniProt, AlphaFold, RCSB and InterPro",
    hl2c: "Prediction, training, design and discovery as callable steps",
    hl3Title: "Critique",
    hl3Lead: "Results are checked, not just accepted.",
    hl3a: "A scientific critic reads the evidence before the next move",
    hl3b: "Scores stay next to the model that produced them",
    hl3c: "Paper-level reports you can defend, not a one-shot answer",
    hl4Title: "Iterate",
    hl4Lead: "The next round starts from what failed.",
    hl4a: "Directed evolution loops that refine the library",
    hl4b: "Fine-tune a specialist on the data you just produced",
    hl4c: "The agent updates its plan and tries again",
    mockBar: "Session",
    mockProv: "Zero-shot · 6 models · reproducible",
    mockUser: "Plan a directed-evolution campaign for TEM-1, then iterate on the top substitutions.",
    mockColRank: "Rank",
    mockColMut: "Mutation",
    mockColDdg: "ΔΔG",
    mockColAgree: "Models",
    pubEyebrow: "Publications",
    pubTitle: "Peer-reviewed, and open.",
    pub1Venue: "ACL 2025 · System Demonstration",
    pub1Title: "VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
    pub2Venue: "arXiv 2603.27303",
    pub2Title: "Self-evolving AI agents for protein discovery and directed evolution",
    pub3Venue: "ICLR 2026",
    pub3Title: "VenusX — functional-residue identification benchmark and public leaderboard",
    pub4Venue: "ISMB/ECCB 2025 · Bioinformatics",
    pub4Title: "VenusREM — retrieval-enhanced mutation effect prediction, ProteinGym substitution #1",
    pubOpen: "Open",
    closeTitle: "Start a self-evolving session.",
    closeLede: "Give the agents a protein question. They will plan, run, critique and iterate.",
    footerSjtu: "Shanghai Jiao Tong University",
    footerSii: "Shanghai Innovation Institute",
    footerHome: "tyang816.github.io",
    license: "Non-Commercial License",
    licenseHint: "Academic use is free; commercial use requires written approval"
  },
  zh: {
    docTitle: "VenusFactory2 — 面向蛋白质发现与定向进化的自进化智能体",
    docDescription:
      "面向蛋白质发现与定向进化的自进化智能体。VenusFactory2 会规划实验、调用 40+ 蛋白质模型与 11 个生物数据库、审查证据，再迭代推进——可通过对话、网页、API 或命令行使用。",
    docKeywords:
      "自进化智能体, self-evolving AI agents, 蛋白质发现, 定向进化, 蛋白质工程, VenusFactory2, VenusREM, Science Expert, Science Agent, AI4Protein, 蛋白语言模型",
    navAgent: "智能体",
    navTools: "工具",
    navBoards: "排行榜",
    navManual: "手册",
    navPapers: "论文",
    banner: "由上海交通大学 & 上海创智学院维护 — 学术使用免费。",
    headline: "面向蛋白质发现的自进化智能体",
    lede: "VenusFactory2 会规划实验、运行模型与数据库、审查证据，再迭代推进——把时间留给科学本身。",
    composer: "规划一轮定向进化，并对最优变体继续迭代…",
    composerGo: "开始对话",
    secondaryCta: "阅读论文",
    primaryCta: "开始会话",
    venueAcl: "ACL 2025",
    venueIsmb: "ISMB/ECCB 2025",
    venueIclr: "ICLR 2026",
    venueArxiv: "arXiv 2603.27303",
    stat1: "蛋白质模型",
    stat2: "生物数据库",
    stat3: "循环中的步骤",
    stat3Value: "4",
    stat3Note: "规划 · 执行 · 批评 · 迭代",
    hlEyebrow: "循环",
    hlTitle: "智能体如何进化",
    hlLede: "先规划实验，再跑模型，审查证据，然后迭代。",
    hl1Title: "规划",
    hl1Lead: "先有假说，再有你可以检查的计划，然后才执行。",
    hl1a: "Science Expert：PI → 计算生物学家 → 机器学习科学家 → 审稿人",
    hl1b: "执行前可重排、改写或删除步骤",
    hl1c: "后续结果都能回溯到这份计划",
    hl2Title: "执行",
    hl2Lead: "模型、数据库与设计工具，同一次会话里完成。",
    hl2a: "40+ 蛋白质语言与结构模型",
    hl2b: "UniProt、AlphaFold、RCSB、InterPro",
    hl2c: "预测、训练、设计与发现都是可调用的步骤",
    hl3Title: "批评",
    hl3Lead: "结果会被核对，而不是直接接受。",
    hl3a: "科学审稿人先读证据，再决定下一步",
    hl3b: "每项分数都和产生它的模型放在一起",
    hl3c: "产出可答辩的 paper 级报告，而不是一次性回答",
    hl4Title: "迭代",
    hl4Lead: "下一轮从失败的地方开始。",
    hl4a: "定向进化循环，持续收紧文库",
    hl4b: "用刚刚产生的数据微调专属模型",
    hl4c: "智能体更新计划，再试一次",
    mockBar: "会话",
    mockProv: "零样本 · 6 个模型 · 可复现",
    mockUser: "为 TEM-1 规划一轮定向进化，并对最优置换继续迭代。",
    mockColRank: "排序",
    mockColMut: "突变",
    mockColDdg: "ΔΔG",
    mockColAgree: "模型",
    pubEyebrow: "发表",
    pubTitle: "经过同行评审，公开可查。",
    pub1Venue: "ACL 2025 · System Demonstration",
    pub1Title: "VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
    pub2Venue: "arXiv 2603.27303",
    pub2Title: "Self-evolving AI agents for protein discovery and directed evolution",
    pub3Venue: "ICLR 2026",
    pub3Title: "VenusX — 功能残基识别基准与公开排行榜",
    pub4Venue: "ISMB/ECCB 2025 · Bioinformatics",
    pub4Title: "VenusREM — 检索增强的突变效应预测，ProteinGym 替换突变榜第一",
    pubOpen: "打开",
    closeTitle: "开始一轮自进化会话。",
    closeLede: "给智能体一个蛋白质问题。它们会规划、执行、批评，再迭代。",
    footerSjtu: "上海交通大学",
    footerSii: "上海创智学院",
    footerHome: "tyang816.github.io",
    license: "非商用许可",
    licenseHint: "学术使用免费；商业使用需书面授权"
  }
};

const HIGHLIGHTS = [
  {
    no: "01",
    titleKey: "hl1Title",
    leadKey: "hl1Lead",
    bullets: ["hl1a", "hl1b", "hl1c"],
    to: "/agent/chat"
  },
  {
    no: "02",
    titleKey: "hl2Title",
    leadKey: "hl2Lead",
    bullets: ["hl2a", "hl2b", "hl2c"],
    to: "/quick-tools/protein-function"
  },
  {
    no: "03",
    titleKey: "hl3Title",
    leadKey: "hl3Lead",
    bullets: ["hl3a", "hl3b", "hl3c"],
    to: "/report"
  },
  {
    no: "04",
    titleKey: "hl4Title",
    leadKey: "hl4Lead",
    bullets: ["hl4a", "hl4b", "hl4c"],
    to: "/quick-tools/directed-evolution"
  }
] as const;

const PUBLICATIONS = [
  { venueKey: "pub1Venue", titleKey: "pub1Title", href: ACL_URL },
  { venueKey: "pub4Venue", titleKey: "pub4Title", href: VENUSREM_URL },
  { venueKey: "pub2Venue", titleKey: "pub2Title", href: PAPER_URL },
  { venueKey: "pub3Venue", titleKey: "pub3Title", href: VENUSX_URL }
] as const;

export function HomePage() {
  const { t: translate } = useLang();
  const t = translate(STRINGS);
  useDocumentMeta({ title: t.docTitle, description: t.docDescription, keywords: t.docKeywords });

  const stats = useMemo(
    () => [
      { value: "40+", label: t.stat1 },
      { value: "11", label: t.stat2 },
      { value: t.stat3Value, label: t.stat3, note: t.stat3Note }
    ],
    [t]
  );

  return (
    <div className="vf2-home">
      <p className="vf2-home-banner">{t.banner}</p>

      <header className="vf2-home-nav">
        <LangLink to="/" className="vf2-home-brand" aria-label="VenusFactory2">
          <Mark />
          <span className="vf2-home-wordmark">
            VenusFactory<sup>2</sup>
          </span>
        </LangLink>

        <nav className="vf2-home-nav-mid" aria-label="primary">
          <LangLink to="/agent/chat" className="vf2-home-nav-link">
            {t.navAgent}
          </LangLink>
          <LangLink to="/quick-tools/protein-function" className="vf2-home-nav-link">
            {t.navTools}
          </LangLink>
          <LangLink to="/leaderboards" className="vf2-home-nav-link">
            {t.navBoards}
          </LangLink>
          <a href="#publications" className="vf2-home-nav-link">
            {t.navPapers}
          </a>
          <LangLink to="/manual/index" className="vf2-home-nav-link">
            {t.navManual}
          </LangLink>
        </nav>

        <div className="vf2-home-nav-end">
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="vf2-home-nav-ext">
            GitHub
          </a>
          <LanguageSwitch variant="pill" />
          <LangLink to="/agent/chat" className="vf2-home-nav-cta">
            {t.primaryCta}
          </LangLink>
        </div>
      </header>

      <main>
        <section className="vf2-home-hero">
          <h1 className="vf2-home-headline">{t.headline}</h1>
          <p className="vf2-home-lede">{t.lede}</p>
          <HomeComposer placeholder={t.composer} goLabel={t.composerGo} />
          <div className="vf2-home-actions">
            <LangLink to="/agent/chat" className="vf2-home-btn vf2-home-btn-primary">
              {t.primaryCta}
            </LangLink>
            <a href={PAPER_URL} target="_blank" rel="noreferrer" className="vf2-home-btn vf2-home-btn-secondary">
              {t.secondaryCta}
            </a>
          </div>
        </section>

        <section className="vf2-home-stats" aria-label="stats">
          {stats.map((s) => (
            <div key={s.label} className="vf2-home-stat">
              <div className="vf2-home-stat-value">{s.value}</div>
              <div className="vf2-home-stat-label">{s.label}</div>
              {"note" in s && s.note ? <div className="vf2-home-stat-note">{s.note}</div> : null}
            </div>
          ))}
        </section>

        <section className="vf2-home-feature" aria-labelledby="vf2-hl-title">
          <div className="vf2-home-feature-head">
            <h2 id="vf2-hl-title" className="vf2-home-section-title">
              {t.hlTitle}
            </h2>
            <p className="vf2-home-feature-lede">{t.hlLede}</p>
          </div>
          <div className="vf2-home-acc">
            {HIGHLIGHTS.map((item) => (
              <LangLink key={item.no} to={item.to} className="vf2-home-acc-item">
                <span className="vf2-home-acc-title">{t[item.titleKey]}</span>
                <span className="vf2-home-acc-lead">{t[item.leadKey]}</span>
              </LangLink>
            ))}
          </div>
        </section>

        <section className="vf2-home-pubs" id="publications" aria-labelledby="vf2-pub-title">
          <div className="vf2-home-feature-head">
            <h2 id="vf2-pub-title" className="vf2-home-section-title">
              {t.pubTitle}
            </h2>
            <p className="vf2-home-feature-lede">{t.pubEyebrow}</p>
          </div>
          <div className="vf2-home-pub-list">
            {PUBLICATIONS.map((pub) => (
              <a
                key={pub.href}
                href={pub.href}
                target="_blank"
                rel="noreferrer"
                className="vf2-home-pub"
              >
                <span className="vf2-home-pub-venue">{t[pub.venueKey]}</span>
                <span className="vf2-home-pub-title">{t[pub.titleKey]}</span>
                <span className="vf2-home-pub-open">{t.pubOpen} →</span>
              </a>
            ))}
          </div>
        </section>

        <section className="vf2-home-close">
          <h2 className="vf2-home-close-title">{t.closeTitle}</h2>
          <p className="vf2-home-close-lede">{t.closeLede}</p>
          <LangLink to="/agent/chat" className="vf2-home-cta vf2-home-cta-primary">
            <span>{t.primaryCta}</span>
            <span aria-hidden>→</span>
          </LangLink>
        </section>
      </main>

      <footer className="vf2-home-footer">
        <div className="vf2-home-footer-org">
          <a href={SJTU_URL} target="_blank" rel="noreferrer">
            {t.footerSjtu}
          </a>
          <span aria-hidden>&</span>
          <a href={SII_URL} target="_blank" rel="noreferrer">
            {t.footerSii}
          </a>
        </div>
        <div className="vf2-home-footer-links">
          <a href={HOMEPAGE_URL} target="_blank" rel="noreferrer">
            {t.footerHome}
          </a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a href={HF_URL} target="_blank" rel="noreferrer">
            Hugging Face
          </a>
          <a href={LICENSE_URL} target="_blank" rel="noreferrer" title={t.licenseHint}>
            {t.license}
          </a>
        </div>
      </footer>
    </div>
  );
}

function HomeComposer({ placeholder, goLabel }: { placeholder: string; goLabel: string }) {
  const navigate = useNavigate();
  const { lang } = useLang();
  const [value, setValue] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();
    navigate(`/${lang}/agent/chat`);
  };

  return (
    <form className="vf2-home-composer" onSubmit={submit}>
      <span className="vf2-home-composer-mark" aria-hidden>
        <Mark />
      </span>
      <input
        className="vf2-home-composer-input"
        placeholder={placeholder}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        aria-label={placeholder}
      />
      <button type="submit" className="vf2-home-composer-go" aria-label={goLabel}>
        <span aria-hidden>→</span>
      </button>
    </form>
  );
}

function Mark() {
  return (
    <img
      className="vf2-home-mark"
      src="/logo-venus.png"
      alt=""
      width={32}
      height={32}
      decoding="async"
    />
  );
}

