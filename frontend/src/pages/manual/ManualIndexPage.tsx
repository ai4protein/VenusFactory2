import { useEffect, useMemo } from "react";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    nav: {
      welcome: "Welcome to VenusFactory2",
      howToUse: "How to Use VenusFactory2",
      questionnaires: "Research Questionnaires",
      partners: "Partner Institutions",
      developer: "Developer Information",
      citation: "Citation",
      additional: "Additional Information",
      models: "Supported Models",
      datasets: "Supported Datasets"
    },
    welcomeBody:
      "VenusFactory2 is a unified open-source platform for protein engineering, designed to simplify data acquisition, model fine-tuning and functional analysis for biologists and AI researchers.",
    feature: {
      agent: "🤖 Agent-0.1:",
      agentDesc: "Intelligent assistant for platform and protein AI Q&A.",
      quick: "🧰 Quick Tools:",
      quickDesc: "One-click mutation and function related analyses.",
      advanced: "🛠️ Advanced Tools:",
      advancedDesc: "Zero-shot prediction and expert workflows.",
      download: "📥 Download:",
      downloadDesc: "Access AlphaFold, RCSB, UniProt, InterPro data."
    },
    howToUseBody:
      "Choose the module according to your goal: use Agent for guidance, Quick Tools for fast tasks, Advanced Tools for model-level control, and Download for data retrieval.",
    googleSurvey: "Google Survey",
    wenjuanxing: "Wenjuanxing",
    sjtu: "Shanghai Jiao Tong University",
    ecust: "East China University of Science and Technology",
    shailab: "Shanghai AI Laboratory",
    cooperationTitle: "Cooperation Platform & Developer Information",
    coopPlatform: "🤝 Cooperation Platform:",
    fewShot: "🧬 Few-shot mutation prediction tool:",
    zeroShot: "⚡ Zero-shot protein prediction tool:",
    devHomepage: "🏠 Developer homepage:",
    contact: "✉️ Contact:"
  },
  zh: {
    nav: {
      welcome: "欢迎使用 VenusFactory2",
      howToUse: "如何使用 VenusFactory2",
      questionnaires: "研究问卷",
      partners: "合作机构",
      developer: "开发者信息",
      citation: "引用",
      additional: "更多信息",
      models: "支持的模型",
      datasets: "支持的数据集"
    },
    welcomeBody:
      "VenusFactory2 是一个面向蛋白质工程的统一开源平台，旨在为生物学家和 AI 研究者简化数据获取、模型微调与功能分析流程。",
    feature: {
      agent: "🤖 Agent-0.1：",
      agentDesc: "面向平台与蛋白 AI 问答的智能助手。",
      quick: "🧰 快速工具：",
      quickDesc: "一键完成突变与功能相关分析。",
      advanced: "🛠️ 高级工具：",
      advancedDesc: "零样本预测与专家级工作流。",
      download: "📥 下载：",
      downloadDesc: "获取 AlphaFold、RCSB、UniProt、InterPro 等数据。"
    },
    howToUseBody:
      "根据目标选择模块：Agent 提供引导，快速工具用于轻量任务，高级工具提供模型级控制，下载用于数据获取。",
    googleSurvey: "Google 问卷",
    wenjuanxing: "问卷星",
    sjtu: "上海交通大学",
    ecust: "华东理工大学",
    shailab: "上海人工智能实验室",
    cooperationTitle: "合作平台与开发者信息",
    coopPlatform: "🤝 合作平台：",
    fewShot: "🧬 少样本突变预测工具：",
    zeroShot: "⚡ 零样本蛋白预测工具：",
    devHomepage: "🏠 开发者主页：",
    contact: "✉️ 联系方式："
  }
};

export function ManualIndexPage() {
  const t = useLang().t(STRINGS);

  const indexNav = useMemo(
    () => [
      { id: "welcome", label: t.nav.welcome, level: 2 },
      { id: "how-to-use", label: t.nav.howToUse, level: 2 },
      { id: "questionnaires", label: t.nav.questionnaires, level: 2 },
      { id: "partners", label: t.nav.partners, level: 2 },
      { id: "developer", label: t.nav.developer, level: 2 },
      { id: "citation", label: t.nav.citation, level: 2 },
      { id: "additional", label: t.nav.additional, level: 2 },
      { id: "models", label: t.nav.models, level: 3 },
      { id: "datasets", label: t.nav.datasets, level: 3 }
    ],
    [t]
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      fetch("/api/stats/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ module: "total_visits" })
      }).catch(() => {});
    }, 1000);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <section className="manual-v2-panel">
      <aside className="manual-v2-nav">
        <ul>
          {indexNav.map((item) => (
            <li key={item.id}>
              <a href={`#${item.id}`} className={`manual-v2-nav-link level-${item.level}`}>
                {item.label}
              </a>
            </li>
          ))}
        </ul>
      </aside>

      <div className="manual-v2-content-wrap">
        <article className="manual-v2-content manual-v2-index">
          <section id="welcome" className="manual-v2-section">
            <h1>{t.nav.welcome}</h1>
            <p>{t.welcomeBody}</p>
            <ul className="manual-v2-feature-list">
              <li><strong>{t.feature.agent}</strong> {t.feature.agentDesc}</li>
              <li><strong>{t.feature.quick}</strong> {t.feature.quickDesc}</li>
              <li><strong>{t.feature.advanced}</strong> {t.feature.advancedDesc}</li>
              <li><strong>{t.feature.download}</strong> {t.feature.downloadDesc}</li>
            </ul>
          </section>

          <section id="how-to-use" className="manual-v2-section">
            <h1>{t.nav.howToUse}</h1>
            <p>{t.howToUseBody}</p>
            <div className="manual-v2-demo-grid">
              <div className="manual-v2-demo-card">
                <h3>🤖 Agent-0.1</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/agent.gif"
                  alt="Agent demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>🧰 {t.feature.quick.replace(":", "").replace("：", "")}</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/quick_tool.gif"
                  alt="Quick tools demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>🛠️ {t.feature.advanced.replace(":", "").replace("：", "")}</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/advanced_tool.gif"
                  alt="Advanced tools demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>📥 {t.feature.download.replace(":", "").replace("：", "")}</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/download.gif"
                  alt="Download demo"
                />
              </div>
            </div>
          </section>

          <section id="questionnaires" className="manual-v2-section">
            <h2>{t.nav.questionnaires}</h2>
            <div className="manual-v2-two-col">
              <div className="manual-v2-card">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_googleform.png"
                  alt={t.googleSurvey}
                />
                <h4>{t.googleSurvey}</h4>
              </div>
              <div className="manual-v2-card">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_wenjuanxing.png"
                  alt={t.wenjuanxing}
                />
                <h4>{t.wenjuanxing}</h4>
              </div>
            </div>
          </section>

          <section id="partners" className="manual-v2-section">
            <h2>{t.nav.partners}</h2>
            <div className="manual-v2-three-col">
              <a href="https://www.sjtu.edu.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/sjtu_logo.jpg"
                  alt="SJTU"
                />
                <h4>{t.sjtu}</h4>
              </a>
              <a href="https://www.ecust.edu.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/ecust_logo.jpg"
                  alt="ECUST"
                />
                <h4>{t.ecust}</h4>
              </a>
              <a href="https://www.shlab.org.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/shailab_logo.jpg"
                  alt="SHAILab"
                />
                <h4>{t.shailab}</h4>
              </a>
            </div>
          </section>

          <section id="developer" className="manual-v2-section">
            <h2>{t.cooperationTitle}</h2>
            <div className="manual-v2-two-col">
              <div className="manual-v2-card">
                <p>
                  <strong>{t.coopPlatform}</strong>{" "}
                  <a href="https://hyper.ai/cn/tutorials/38568" target="_blank" rel="noreferrer">
                    HyperAI
                  </a>
                </p>
                <p>
                  <strong>{t.fewShot}</strong>{" "}
                  <a href="https://github.com/ai4protein/Pro-FSFP" target="_blank" rel="noreferrer">
                    Pro-FSFP
                  </a>
                </p>
                <p>
                  <strong>{t.zeroShot}</strong>{" "}
                  <a href="https://github.com/ai4protein/VenusREM" target="_blank" rel="noreferrer">
                    VenusREM
                  </a>
                </p>
              </div>
              <div className="manual-v2-card">
                <p>
                  <strong>{t.devHomepage}</strong>{" "}
                  <a href="https://tyang816.github.io/" target="_blank" rel="noreferrer">
                    https://tyang816.github.io/
                  </a>
                </p>
                <p>
                  <strong>{t.contact}</strong>{" "}
                  <a href="mailto:tanyang.august@sjtu.edu.cn">tanyang.august@sjtu.edu.cn</a>,{" "}
                  <a href="mailto:zlr_zmm@163.com">zlr_zmm@163.com</a>
                </p>
              </div>
            </div>
          </section>

          <section id="citation" className="manual-v2-section">
            <h2>{t.nav.citation}</h2>
            <pre className="manual-v2-citation">
{`@inproceedings{tan-etal-2025-venusfactory,
  title = {VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning},
  author = {Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin},
  booktitle = {Proceedings of ACL 2025 System Demonstrations},
  year = {2025},
  url = {https://aclanthology.org/2025.acl-demo.23/},
  doi = {10.18653/v1/2025.acl-demo.23}
}`}
            </pre>
          </section>

          <section id="additional" className="manual-v2-section">
            <h1>{t.nav.additional}</h1>
            <div className="manual-v2-two-col">
              <div id="models" className="manual-v2-card">
                <h3>{t.nav.models}</h3>
                <ul>
                  <li>ESM-1v / ESM-1b / ESM-650M</li>
                  <li>SaProt</li>
                  <li>MIF-ST</li>
                  <li>ProSST-2048</li>
                  <li>ProtSSN</li>
                  <li>Ankh-large</li>
                  <li>ProtBert-uniref50 / ProtT5-xl-uniref50</li>
                </ul>
              </div>
              <div id="datasets" className="manual-v2-card">
                <h3>{t.nav.datasets}</h3>
                <ul>
                  <li>DeepSol / DeepSoluE / ProtSolM</li>
                  <li>DeepLocBinary / DeepLocMulti</li>
                  <li>MetalIonBinding</li>
                  <li>Thermostability</li>
                  <li>SortingSignal</li>
                  <li>DeepET_Topt</li>
                </ul>
              </div>
            </div>
          </section>
        </article>
      </div>
    </section>
  );
}
