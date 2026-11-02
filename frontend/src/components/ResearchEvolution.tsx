import type { ScienceGoal } from "../lib/scienceGoals";
import { openResearchFile } from "./SessionFilesPanel";

export function ResearchEvolution({ goal, lang }: { goal: ScienceGoal; lang: string }) {
  const zh = lang === "zh", say = (en: string, cn: string) => zh ? cn : en;
  const development = goal.research?.development;
  const generations = development?.generations || [];
  const deliveries = goal.research?.deliveries || [];
  if (!generations.length && !deliveries.length) return null;
  const variant = (value: string) => ({ baseline: say("Baseline", "基线"), candidate: say("Revision", "改进实验"), ablation: say("Ablation", "消融实验") }[value] || value);
  const score = (value: number) => Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : "—";
  return <section className="research-evolution" aria-label={say("Model iterations", "模型迭代")}>
    <header><strong>{say("Model iterations", "模型迭代")}</strong><span>{development?.count || 0} {say("evaluated experiments", "个已评估实验")}</span></header>
    {!!generations.length && <div className="research-evolution-table"><table>
      <thead><tr><th>{say("Experiment", "实验")}</th><th>{say("Validation", "验证结果")}</th><th>{say("Change vs parent", "相对父代变化")}</th></tr></thead>
      <tbody>{generations.map(row => <tr key={row.id}>
        <td><details><summary><span className="research-variant">{variant(row.variant)}</span> {row.name}</summary>
          <p>{row.hypothesis}</p><small>{row.strategy.replace(/_/g, " ")}</small>
          {row.parent && <p>{say("Parent", "父代")} · {generations.find(parent => parent.id === row.parent)?.name || row.parent.slice(0, 8)}</p>}
          <button type="button" onClick={() => openResearchFile(goal.id, `experiments/${row.id}.json`)}>{say("Experiment record", "实验记录")} ↗</button>
        </details></td>
        <td><strong>{score(row.validation.score)}</strong><small>{row.validation.metric} {row.validation.direction === "minimize" ? "↓" : "↑"}</small></td>
        <td>{row.comparison ? <><strong>{row.comparison.delta_in_preferred_direction > 0 ? "+" : ""}{score(row.comparison.delta_in_preferred_direction)}</strong><small>{row.comparison.improved_on_validation ? say("Improved on validation", "验证结果改善") : say("No validation gain", "验证结果未改善")}</small></> : "—"}</td>
      </tr>)}</tbody>
    </table></div>}
    {!!deliveries.length && <div className="research-model-deliveries">{deliveries.map(delivery => <div key={delivery.candidate_id}>
      <strong>{say("Model delivery", "模型交付")}</strong>
      {[[delivery.model_card, say("Model card", "模型说明")], [delivery.test_report, say("Test report", "测试报告")], [delivery.model_package, say("Model package", "模型包")]].map(([path, label]) => <button key={path} type="button" onClick={() => openResearchFile(goal.id, path)}>{label} ↗</button>)}
    </div>)}</div>}
  </section>;
}
