import { useEffect, useRef, useState } from "react";
import { useLang } from "../lib/i18n";
import { LanguageSwitch } from "../components/LanguageSwitch";
import { LangLink } from "../components/LangLink";
import { QuotaDialog } from "../components/QuotaDialog";
import { openQuotaRequest } from "../lib/onlineQuota";
import { useDocumentMeta } from "../lib/useDocumentMeta";
import { LOGO_VENUS_URL } from "../lib/brandAssets";
import proteinFigure from "../assets/home-crambin-ribbon.png";
import "./HomePage.css";

const GITHUB = "https://github.com/ai4protein/VenusFactory2";
const PAPER = "https://arxiv.org/abs/2603.27303";
const MODELS = ["ProSST", "VenusREM", "VenusVaccine", "ProtSolM", "VenusX"];

const COPY = {
  en: {
    title: "Recursive Self-improvement AI Agents for Protein Engineering",
    headlineA: "Recursive Self-improvement",
    headlineB: "AI Agents for Protein Engineering",
    lede: "Explore a protein. Develop a model. Advance your research with agents that plan, execute, evaluate and iterate.",
    academic: "Free for academic use", credit: "Request extra credits", maintained: "Shanghai Jiao Tong University · Shanghai Innovation Institute",
    chat: "Chat", models: "Model Plaza", papers: "Publications", manual: "Manual", start: "Start exploring", paper: "Read the paper",
    figure: "From sequence to scientific insight", structure: "Crambin", structureDetail: "46 residues · Experimental structure", helix: "α-helix", sheet: "β-sheet", loop: "Loop", sequence: "SEQUENCE", cycle: "Plan · Execute · Evaluate · Improve",
    modeTitle: "One workspace. Three ways to explore.", modeLabel: "YOUR RESEARCH, YOUR WAY",
    agentTitle: "Ask & explore", agentLead: "Ask protein questions, find evidence and run the right tools in a conversation.",
    scientistTitle: "Create & improve", scientistLead: "Bring your data and a goal. Develop, evaluate and refine a model you can use again.",
    researcherTitle: "Investigate in depth", researcherLead: "Bring together literature, biological databases and predictions in a detailed protein report.",
    enter: "Open", modelLabel: "BUILT FOR PROTEINS", modelTitle: "Specialist models, ready to use.",
    modelLead: "From mutation effects and solubility to immune properties and functional residues.", browse: "Explore all models",
    loopLabel: "RECURSIVE SELF-IMPROVEMENT", loopTitle: "Each result informs the next step.",
    loopLead: "Scientist turns a research goal into an evolving plan. Review the work as it runs, refine the direction and keep useful models for future questions.",
    steps: ["Define the goal", "Build & train", "Evaluate & refine", "Register & reuse"],
    stepNotes: ["Your question and data", "Tools and specialist models", "Evidence guides the next round", "Available in Chat and Model Plaza"],
    pubLabel: "RESEARCH", pubTitle: "Research behind VenusFactory.",
    pubLead: "Explore the papers and open-source projects behind VenusFactory.",
    closeTitle: "Your next discovery starts with a question.", closeLead: "Bring your protein, your data, or an idea.",
    licence: "Non-commercial license", back: "Back to top", structureAlt: "Ribbon representation of Crambin from PDB 1CRN; teal helices, gold beta sheets and slate loops",
  },
  zh: {
    title: "面向蛋白质工程的递归自改进 AI 智能体",
    headlineA: "递归自改进 AI 智能体",
    headlineB: "面向蛋白质工程",
    lede: "从理解一个蛋白质，到开发一个新模型。让智能体规划、执行、评估与迭代，推进你的研究。",
    academic: "学术免费使用", credit: "申请额外额度", maintained: "上海交通大学 · 上海创智学院",
    chat: "对话", models: "模型广场", papers: "研究论文", manual: "使用手册", start: "开始探索", paper: "阅读论文",
    figure: "从蛋白质序列，到科学发现", structure: "Crambin", structureDetail: "46 个残基 · 实验结构", helix: "α 螺旋", sheet: "β 折叠", loop: "环区", sequence: "蛋白质序列", cycle: "规划 · 执行 · 评估 · 改进",
    modeTitle: "一个工作空间，三种研究方式。", modeLabel: "从你的问题出发",
    agentTitle: "问答与探索", agentLead: "围绕蛋白质问题展开对话，查找证据，调用合适的工具与模型。",
    scientistTitle: "创造与改进", scientistLead: "提供数据和研究目标，开发、评估并迭代模型，积累可复用的能力。",
    researcherTitle: "深入专题研究", researcherLead: "汇集文献、数据库与模型预测，为一个蛋白质形成全面的研究报告。",
    enter: "进入", modelLabel: "为蛋白质研究而构建", modelTitle: "专用模型，随时调用。",
    modelLead: "覆盖突变效应、溶解度、免疫相关性质与功能残基等研究问题。", browse: "浏览全部模型",
    loopLabel: "递归自改进 · RSI", loopTitle: "让每一轮结果，推动下一步研究。",
    loopLead: "Scientist 将研究目标转化为可迭代的计划。过程中查看结果、调整方向，将有用的模型注册下来，用于后续研究。",
    steps: ["明确研究目标", "构建与训练", "评估与改进", "注册与复用"],
    stepNotes: ["从问题和数据出发", "调用工具，开发专用模型", "根据证据推进下一轮", "在对话和模型广场继续使用"],
    pubLabel: "研究基础", pubTitle: "VenusFactory 的研究基础",
    pubLead: "了解 VenusFactory 背后的论文与开源项目。",
    closeTitle: "下一次发现，从一个问题开始。", closeLead: "带上你的蛋白质、数据，或一个想法。",
    licence: "非商用许可", back: "返回顶部", structureAlt: "根据 PDB 1CRN 坐标渲染的 Crambin 带状结构：青绿色螺旋、金色折叠与灰蓝色环区",
  },
};

const PUBLICATIONS = [
  {
    "name": "VenusFactory2",
    "venue": "arXiv · 2026 · Preprint",
    "title": "Self-evolving AI agents for protein discovery and directed evolution",
    "href": "https://arxiv.org/abs/2603.27303"
  },
  {
    "name": "VenusRAR",
    "venue": "arXiv · 2026 · Preprint",
    "title": "Rank-and-Reason: Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction",
    "href": "https://arxiv.org/abs/2602.00197"
  },
  {
    "name": "VenusX",
    "venue": "ICLR · 2026",
    "title": "VenusX: Unlocking Fine-Grained Functional Understanding of Proteins",
    "href": "https://openreview.net/forum?id=zcmL592XRG"
  },
  {
    "name": "VenusFactory",
    "venue": "ACL Demo · 2025",
    "title": "VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
    "href": "https://aclanthology.org/2025.acl-demo.23/"
  },
  {
    "name": "VenusREM",
    "venue": "Bioinformatics · 2025",
    "title": "From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model",
    "href": "https://doi.org/10.1093/bioinformatics/btaf189"
  },
  {
    "name": "VenusVaccine",
    "venue": "ICLR · 2025",
    "title": "Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection",
    "href": "https://openreview.net/forum?id=hWmwL9gizZ"
  },
  {
    "name": "ProtSSN",
    "venue": "eLife · 2025",
    "title": "Semantical and geometrical protein encoding toward enhanced bioactivity and thermostability",
    "href": "https://elifesciences.org/articles/98033"
  },
  {
    "name": "ProSST",
    "venue": "NeurIPS · 2024",
    "title": "ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention",
    "href": "https://proceedings.neurips.cc/paper_files/paper/2024/file/3ed57b293db0aab7cc30c44f45262348-Paper-Conference.pdf"
  },
  {
    "name": "ProtLOCA",
    "venue": "IEEE BIBM · 2024",
    "title": "Protein Representation Learning with Sequence Information Embedding: Does it Always Lead to a Better Performance?",
    "href": "https://doi.org/10.1109/BIBM62325.2024.10822035"
  },
  {
    "name": "ProtSolM",
    "venue": "IEEE BIBM · 2024",
    "title": "PROTSOLM: Protein Solubility Prediction with Multi-Modal Features",
    "href": "https://ieeexplore.ieee.org/document/10822310"
  },
  {
    "name": "CPDiffusion-SS",
    "venue": "IEEE BIBM · 2024",
    "title": "Secondary Structure-Guided Novel Protein Sequence Generation with Latent Graph Diffusion",
    "href": "https://ieeexplore.ieee.org/document/10822117"
  },
  {
    "name": "PETA",
    "venue": "Journal of Cheminformatics · 2024",
    "title": "PETA: evaluating the impact of protein transfer learning with sub-word tokenization on downstream applications",
    "href": "https://link.springer.com/article/10.1186/s13321-024-00884-3"
  }
];

export function HomePage() {
  const { lang, t: translate } = useLang();
  const t = translate(COPY);
  useDocumentMeta({ title: `VenusFactory2 — ${t.title}`, description: t.lede, keywords: "RSI, recursive self-improvement, protein engineering, AI agents, VenusFactory2, ProSST, VenusREM, VenusVaccine, ProtSolM, VenusX" });
  const modes = [
    { name: "Agent", title: t.agentTitle, lead: t.agentLead, mode: "agent", icon: "chat" },
    { name: "Scientist", title: t.scientistTitle, lead: t.scientistLead, mode: "scientist", icon: "flask" },
    { name: "Researcher", title: t.researcherTitle, lead: t.researcherLead, mode: "researcher", icon: "book" },
  ];
  return <div className="landing" lang={lang} id="top">
    <QuotaDialog />
    <div className="landing-announcement">
      <span className="landing-academic"><span aria-hidden="true" className="landing-dot" />{t.academic}</span>
      <span className="landing-announcement-divider" aria-hidden="true" />
      <button type="button" onClick={openQuotaRequest}>{t.credit}<Arrow diagonal /></button>
    </div>
    <header className="landing-nav landing-wrap">
      <LangLink to="/" className="landing-brand" aria-label="VenusFactory2">
        <img src={LOGO_VENUS_URL} alt="" width={38} height={38} /><span>VenusFactory<sup>2</sup></span>
      </LangLink>
      <nav aria-label={lang === 'zh' ? '主导航' : 'Main navigation'}>
        <LangLink to="/agent/chat">{t.chat}</LangLink><LangLink to="/models">{t.models}</LangLink><a href="#publications">{t.papers}</a><LangLink to="/manual/index">{t.manual}</LangLink>
      </nav>
      <div className="landing-nav-end"><a className="landing-github" href={GITHUB} target="_blank" rel="noreferrer">GitHub<Arrow diagonal /></a><LanguageSwitch variant="pill" /></div>
    </header>
    <main>
      <section className="landing-hero landing-wrap" aria-labelledby="landing-title">
        <div className="landing-hero-copy">
          <p className="landing-eyebrow"><span aria-hidden="true" />AI4PROTEIN <span className="landing-eyebrow-separator">/</span> VENUSFACTORY 2</p>
          <h1 id="landing-title" aria-label={t.title}><span>{lang === "en" ? <>Recursive<br />Self-improvement</> : <><em className="landing-title-phrase">递归自改进</em>{" "}<em className="landing-title-phrase">AI 智能体</em></>}</span><span>{t.headlineB}</span></h1>
          <p className="landing-lede">{t.lede}</p>
          <div className="landing-actions"><LangLink to="/agent/chat" className="landing-button">{t.start}<Arrow /></LangLink><a href={PAPER} target="_blank" rel="noreferrer" className="landing-text-link">{t.paper}<Arrow diagonal /></a></div>
          <p className="landing-affiliation">{t.maintained}</p>
        </div>
        <figure className="landing-science">
          <div className="landing-figure-heading"><div><span>{t.figure}</span><strong>{t.structure}</strong></div><a href="https://www.rcsb.org/structure/1CRN" target="_blank" rel="noreferrer" className="landing-structure-source">PDB 1CRN<Arrow diagonal /></a></div>
          <div className="landing-molecule"><img src={proteinFigure} width={868} height={697} alt={t.structureAlt} fetchPriority="high" /></div>
          <div className="landing-structure-detail"><span>{t.structureDetail}</span><div className="landing-structure-legend"><span><i className="is-helix" />{t.helix}</span><span><i className="is-sheet" />{t.sheet}</span><span><i className="is-loop" />{t.loop}</span></div></div>
          <figcaption><span className="landing-cycle-icon" aria-hidden="true">↻</span><span>{t.cycle}</span><span className="landing-rsi">RSI</span></figcaption>
        </figure>
      </section>
      <section className="landing-modes landing-wrap" aria-labelledby="landing-modes-title">
        <div className="landing-section-heading"><div><p className="landing-eyebrow">{t.modeLabel}</p><h2 id="landing-modes-title">{t.modeTitle}</h2></div><span className="landing-section-index" aria-hidden="true">01 — 03</span></div>
        <div className="landing-mode-grid">{modes.map((mode, i) => <LangLink key={mode.name} to={`/agent/chat?mode=${mode.mode}`} className={`landing-mode${mode.mode === "scientist" ? " is-scientist" : ""}`}>
          <div className="landing-mode-top"><ModeIcon type={mode.icon} /><span>{mode.name}</span><span className="landing-mode-number">{mode.mode === "scientist" ? "RSI" : `0${i + 1}`}</span></div><h3>{mode.title}</h3><p>{mode.lead}</p><span className="landing-mode-open">{t.enter} {mode.name}<Arrow /></span>
        </LangLink>)}</div>
      </section>
      <section className="landing-models landing-wrap" aria-labelledby="landing-models-title">
        <div><p className="landing-eyebrow">{t.modelLabel}</p><h2 id="landing-models-title">{t.modelTitle}</h2><p>{t.modelLead}</p><LangLink to="/models" className="landing-text-link">{t.browse}<Arrow /></LangLink></div>
        <div className="landing-model-list" aria-label={t.models}>{MODELS.map((name, i) => <span key={name}><span className="landing-model-symbol" aria-hidden="true">{['⌁', '↗', '✳', '≈', '⨯'][i]}</span>{name}</span>)}</div>
      </section>
      <section className="landing-loop" aria-labelledby="landing-loop-title"><div className="landing-wrap">
        <div className="landing-loop-head"><div><p className="landing-eyebrow">{t.loopLabel}</p><h2 id="landing-loop-title">{t.loopTitle}</h2></div><p>{t.loopLead}</p></div>
        <ol className="landing-steps">{t.steps.map((step, i) => <li key={step}><span className="landing-step-number">0{i + 1}</span><h3>{step}</h3><p>{t.stepNotes[i]}</p></li>)}</ol>
        <LangLink to="/agent/chat?mode=scientist" className="landing-text-link">{t.enter} Scientist<Arrow /></LangLink>
      </div></section>
      <PublicationCarousel lang={lang} title={t.pubTitle} label={t.pubLabel} />
      <section className="landing-close landing-wrap"><div><h2>{t.closeTitle}</h2><p>{t.closeLead}</p></div><LangLink to="/agent/chat" className="landing-button">{t.start}<Arrow /></LangLink></section>
    </main>
    <footer className="landing-footer landing-wrap"><div><span className="landing-footer-brand">VenusFactory<sup>2</sup></span><p>{t.maintained}</p></div><div className="landing-footer-links"><a href={GITHUB} target="_blank" rel="noreferrer">GitHub</a><a href="https://huggingface.co/AI4Protein" target="_blank" rel="noreferrer">Hugging Face</a><a href={`${GITHUB}/blob/main/LICENSE`} target="_blank" rel="noreferrer">{t.licence}</a><a href="#top" aria-label={t.back}>↑</a></div></footer>
  </div>;
}

function PublicationCarousel({ lang, title, label }: { lang: string; title: string; label: string }) {
  const rail = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ previous: false, next: true });
  const zh = lang === "zh";
  function updateEdges() {
    const el = rail.current;
    if (el) setEdges({ previous: el.scrollLeft > 2, next: el.scrollLeft + el.clientWidth < el.scrollWidth - 2 });
  }
  useEffect(() => {
    const el = rail.current;
    if (!el) return;
    updateEdges();
    const observer = new ResizeObserver(updateEdges);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  function move(direction: number) {
    const el = rail.current, card = el?.firstElementChild;
    if (!el || !card) return;
    const stride = card.getBoundingClientRect().width + parseFloat(getComputedStyle(el).columnGap || '0');
    const count = Math.max(1, Math.floor(el.clientWidth / stride));
    el.scrollBy({ left: direction * stride * count, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  }
  return <section className="landing-publications landing-wrap" id="publications" aria-labelledby="landing-pubs-title">
    <div className="landing-section-heading"><div><p className="landing-eyebrow">{label}</p><h2 id="landing-pubs-title">{title}</h2></div>
      <div className="landing-pub-controls" role="group" aria-label={zh ? '浏览研究论文' : 'Browse publications'}>
        <button type="button" aria-label={zh ? '上一组论文' : 'Previous publications'} aria-controls="landing-publication-rail" disabled={!edges.previous} onClick={() => move(-1)}><span className="landing-pub-previous"><Arrow /></span></button>
        <button type="button" aria-label={zh ? '下一组论文' : 'Next publications'} aria-controls="landing-publication-rail" disabled={!edges.next} onClick={() => move(1)}><Arrow /></button>
      </div>
    </div>
    <div ref={rail} id="landing-publication-rail" className="landing-publication-list" role="region" aria-label={zh ? '论文卡片，可左右滚动' : 'Publication cards, scroll horizontally'} tabIndex={0} onScroll={updateEdges}
      onKeyDown={event => { if (event.target !== event.currentTarget) return; if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); move(event.key === 'ArrowRight' ? 1 : -1); } }}>
      {PUBLICATIONS.map(pub => <a key={pub.href} href={pub.href} target="_blank" rel="noreferrer"><span className="landing-pub-meta"><strong>{pub.name}</strong><span className="landing-pub-venue">{pub.venue}</span></span><h3>{pub.title}</h3><Arrow diagonal /></a>)}
    </div>
  </section>;
}

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">{diagonal ? <path d="M6 18 18 6M6 6h12v12" /> : <path d="M4 12h16m-6-6 6 6-6 6" />}</svg>;
}

function ModeIcon({ type }: { type: string }) {
  return <svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">{type === 'chat' ? <><path d="M20 11a8 8 0 0 1-8 8H4l1.5-4A8 8 0 1 1 20 11Z" /><path d="M8 9h8m-8 4h5" /></> : type === 'flask' ? <><path d="M9 3h6m-5 0v7L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3" /><path d="M7 15h10m-6 3h2" /></> : <><path d="M12 5C8 3 5 3 3 4v15c3-1 6 0 9 2 3-2 6-3 9-2V4c-2-1-5-1-9 1Zm0 0v16" /><path d="m6 8 3 1m6 0 3-1m-12 4 3 1m6 0 3-1" /></>}</svg>;
}
