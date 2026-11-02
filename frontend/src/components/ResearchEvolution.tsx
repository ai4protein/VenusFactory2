import type { ScienceGoal, ResearchDelivery } from "../lib/scienceGoals";
import { openResearchFile } from "./SessionFilesPanel";

export function ResearchEvolution({ goal, lang }: { goal: ScienceGoal; lang: string }) {
  const zh = lang === "zh", say = (en: string, cn: string) => zh ? cn : en;
  const development = goal.research?.development;
  const generations = development?.generations || [];
  const history = goal.research?.deliveries || [];
  const deliveries = goal.research?.selected_deliveries ?? Array.from(new Map(history.map(row => [row.family_id || row.candidate_id, row])).values());
  const previous = history.filter(row => !deliveries.some(selected => selected.candidate_id === row.candidate_id));
  if (!generations.length && !deliveries.length) return null;
  const variant = (value: string) => ({ baseline: say("Baseline", "基线"), candidate: say("Revision", "改进实验"), ablation: say("Ablation", "消融实验") }[value] || value);
  const score = (value: number) => Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : "—";
  const deliveryCard = (delivery: ResearchDelivery) => <div key={delivery.candidate_id}>
    <strong>{delivery.name || (delivery.kind === "method" ? say("Method", "方法") : say("Model", "模型"))}{delivery.version ? ` · v${delivery.version}` : ""}</strong>
    {[
      ...(delivery.readme ? [[delivery.readme, "README"]] : []),
      [delivery.model_card, delivery.kind === "method" ? say("Method card", "方法说明") : say("Model card", "模型说明")],
      [delivery.test_report, say("Test report", "测试报告")],
      [delivery.package || delivery.model_package, say("Download", "下载")],
    ].map(([path, label]) => <button key={path} type="button" onClick={() => openResearchFile(goal.id, path)}>{label} ↗</button>)}
  </div>;
  return <section className="research-evolution" aria-label={say("Research outputs", "研究成果")}>
    <header><strong>{say("Research outputs", "研究成果")}</strong><span>{development?.count || 0} {say("evaluated experiments", "个已评估实验")}</span></header>
    {!!generations.length && <details open={!deliveries.length}><summary>{say("Experiments", "实验记录")} · {development?.count || generations.length}</summary><div className="research-evolution-table"><table>
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
    </table></div></details>}
    {!!deliveries.length && <div className="research-model-deliveries">{deliveries.map(deliveryCard)}</div>}
    {!!previous.length && <details><summary>{say("Previous versions", "历史版本")} · {previous.length}</summary><div className="research-model-deliveries">{previous.map(deliveryCard)}</div></details>}
  </section>;
}
