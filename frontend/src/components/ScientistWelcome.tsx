import { CompletedExamples } from "./CompletedExamples";
import "./ScientistWelcome.css";

const stages = [
  { name: "PI", detail: ["Set direction", "确定方向"], title: ["Principal Investigator: define the research goal", "研究负责人：明确研究目标"] },
  { name: "Plan", detail: ["Design experiments", "规划实验"], title: ["Biology and modelling experts plan data and experiments", "生物学与建模专家规划数据和实验"] },
  { name: "Implement", detail: ["Build, train & test", "开发、训练与测试"], title: ["Develop methods, run experiments and collect evidence", "开发方法、执行实验并积累证据"] },
  { name: "SC", detail: ["Scientific acceptance", "科学验收"], title: ["Assess usefulness from measured evidence; return unmet goals to PI", "依据实测证据验收成果价值，未达到目标时退回 PI"] },
  { name: "Review", detail: ["Validate & deliver", "验证与交付"], title: ["Review the evidence, code and model deliverables", "审核证据、代码与模型交付物"] },
];

const examples = [
  {
    title: "Improve enzyme activity",
    detail: "Build a predictor from wet-lab measurements.",
    prompt: "Use my enzyme activity measurements to develop and improve a variant activity predictor. Evaluate it and suggest variants to test.",
  },
  {
    title: "Learn from a small dataset",
    detail: "Predict solubility with limited measurements.",
    prompt: "Develop a few-shot protein solubility predictor from my small expression dataset, and compare it with simple baselines.",
  },
  {
    title: "Extend a published study",
    detail: "Develop and test a new idea from a paper.",
    prompt: "Use the paper and data I upload to identify a testable improvement, develop a complementary model and design controlled experiments.",
  },
  {
    title: "Develop an enzyme mining model",
    detail: "Turn public data into a candidate ranking method.",
    prompt: "For my target reaction, collect public enzyme data and develop a candidate ranking model. Deliver reproducible code, an evaluation report and a candidate shortlist.",
  },
];

export function ScientistWelcome({ lang, disabled, onExample }: {
  lang: string; disabled: boolean; onExample: (prompt: string) => void;
}) {
  const language = lang === "zh" ? 1 : 0;
  return <div className="scientist-welcome">
    <ol className="scientist-research-cycle" aria-label={language ? "科研协作与迭代流程" : "Research and improvement cycle"}>
      {stages.map((stage, index) => <li key={stage.name} title={stage.title[language]}>
        <strong>{stage.name}</strong><span>{stage.detail[language]}</span>
        {index < stages.length - 1 && <i aria-hidden="true">{index === 2 ? "↔" : "›"}</i>}
      </li>)}
    </ol>
    <h2 style={{ margin: "0 0 9px", color: "#79818b", fontSize: 12, fontWeight: 550 }}>{language ? "推荐研究目标" : "Suggested research goals"}</h2>
    <div className="scientist-start-examples" role="group" aria-label={language ? "研究示例" : "Research examples"}>
      {examples.map(example => <button key={example.title} type="button" disabled={disabled} onClick={() => onExample(example.prompt)}>
        <strong>{example.title}</strong><span>{example.detail}</span>
        <svg aria-hidden="true" viewBox="0 0 16 16"><path d="M4 12 12 4M5 4h7v7"/></svg>
      </button>)}
    </div>
    <CompletedExamples mode="scientist" lang={lang}/>
  </div>;
}
