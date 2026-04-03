import { useEffect } from "react";

const INDEX_NAV = [
  { id: "welcome", label: "Welcome to VenusFactory2", level: 2 },
  { id: "how-to-use", label: "How to Use VenusFactory2", level: 2 },
  { id: "questionnaires", label: "Research Questionnaires", level: 2 },
  { id: "partners", label: "Partner Institutions", level: 2 },
  { id: "developer", label: "Developer Information", level: 2 },
  { id: "citation", label: "Citation", level: 2 },
  { id: "additional", label: "Additional Information", level: 2 },
  { id: "models", label: "Supported Models", level: 3 },
  { id: "datasets", label: "Supported Datasets", level: 3 }
];

export function ManualIndexPage() {
  useEffect(() => {
    const timer = window.setTimeout(() => {
      fetch("/api/stats/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ module: "total_visits" })
      }).catch(() => {
        // Keep parity with legacy behavior: silent fallback.
      });
    }, 1000);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <section className="manual-v2-panel">
      <aside className="manual-v2-nav">
        <ul>
          {INDEX_NAV.map((item) => (
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
            <h1>Welcome to VenusFactory2</h1>
            <p>
              VenusFactory2 is a unified open-source platform for protein engineering, designed to
              simplify data acquisition, model fine-tuning and functional analysis for biologists
              and AI researchers.
            </p>
            <ul className="manual-v2-feature-list">
              <li>
                <strong>🤖 Agent-0.1:</strong> Intelligent assistant for platform and protein AI
                Q&amp;A.
              </li>
              <li>
                <strong>🧰 Quick Tools:</strong> One-click mutation and function related analyses.
              </li>
              <li>
                <strong>🛠️ Advanced Tools:</strong> Zero-shot prediction and expert workflows.
              </li>
              <li>
                <strong>📥 Download:</strong> Access AlphaFold, RCSB, UniProt, InterPro data.
              </li>
            </ul>
          </section>

          <section id="how-to-use" className="manual-v2-section">
            <h1>How to Use VenusFactory2</h1>
            <p>
              Choose the module according to your goal: use Agent for guidance, Quick Tools for
              fast tasks, Advanced Tools for model-level control, and Download for data retrieval.
            </p>
            <div className="manual-v2-demo-grid">
              <div className="manual-v2-demo-card">
                <h3>🤖 Agent-0.1</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/agent.gif"
                  alt="Agent demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>🧰 Quick Tools</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/quick_tool.gif"
                  alt="Quick tools demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>🛠️ Advanced Tools</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/advanced_tool.gif"
                  alt="Advanced tools demo"
                />
              </div>
              <div className="manual-v2-demo-card">
                <h3>📥 Download</h3>
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/download.gif"
                  alt="Download demo"
                />
              </div>
            </div>
          </section>

          <section id="questionnaires" className="manual-v2-section">
            <h2>Research Questionnaires</h2>
            <div className="manual-v2-two-col">
              <div className="manual-v2-card">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_googleform.png"
                  alt="Google Survey"
                />
                <h4>Google Survey</h4>
              </div>
              <div className="manual-v2-card">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_wenjuanxing.png"
                  alt="Wenjuanxing Survey"
                />
                <h4>问卷星</h4>
              </div>
            </div>
          </section>

          <section id="partners" className="manual-v2-section">
            <h2>Partner Institutions</h2>
            <div className="manual-v2-three-col">
              <a href="https://www.sjtu.edu.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/sjtu_logo.jpg"
                  alt="SJTU"
                />
                <h4>Shanghai Jiao Tong University</h4>
              </a>
              <a href="https://www.ecust.edu.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/ecust_logo.jpg"
                  alt="ECUST"
                />
                <h4>East China University of Science and Technology</h4>
              </a>
              <a href="https://www.shlab.org.cn/" target="_blank" rel="noreferrer" className="manual-v2-card link">
                <img
                  src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/shailab_logo.jpg"
                  alt="SHAILab"
                />
                <h4>Shanghai AI Laboratory</h4>
              </a>
            </div>
          </section>

          <section id="developer" className="manual-v2-section">
            <h2>Cooperation Platform &amp; Developer Information</h2>
            <div className="manual-v2-two-col">
              <div className="manual-v2-card">
                <p>
                  <strong>🤝 Cooperation Platform:</strong>{" "}
                  <a href="https://hyper.ai/cn/tutorials/38568" target="_blank" rel="noreferrer">
                    HyperAI
                  </a>
                </p>
                <p>
                  <strong>🧬 Few-shot mutation prediction tool:</strong>{" "}
                  <a href="https://github.com/ai4protein/Pro-FSFP" target="_blank" rel="noreferrer">
                    Pro-FSFP
                  </a>
                </p>
                <p>
                  <strong>⚡ Zero-shot protein prediction tool:</strong>{" "}
                  <a href="https://github.com/ai4protein/VenusREM" target="_blank" rel="noreferrer">
                    VenusREM
                  </a>
                </p>
              </div>
              <div className="manual-v2-card">
                <p>
                  <strong>🏠 Developer homepage:</strong>{" "}
                  <a href="https://tyang816.github.io/" target="_blank" rel="noreferrer">
                    https://tyang816.github.io/
                  </a>
                </p>
                <p>
                  <strong>✉️ Contact:</strong>{" "}
                  <a href="mailto:tanyang.august@sjtu.edu.cn">tanyang.august@sjtu.edu.cn</a>,{" "}
                  <a href="mailto:zlr_zmm@163.com">zlr_zmm@163.com</a>
                </p>
              </div>
            </div>
          </section>

          <section id="citation" className="manual-v2-section">
            <h2>Citation</h2>
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
            <h1>Additional Information</h1>
            <div className="manual-v2-two-col">
              <div id="models" className="manual-v2-card">
                <h3>Supported Models</h3>
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
                <h3>Supported Datasets</h3>
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
