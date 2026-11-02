"""Plot independently audited complete author ALDE trajectories."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_runs/rsi/20261011_alde_agent_full420_audit_v1/trajectory_replay/report.json"
OUT = ROOT / "docs/paper/0overleaf/figures/article"
data = json.loads(SOURCE.read_text())
assert data["completed_and_verified"] == 420
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.30, wspace=0.30)
colors = {
    "GREEDY": "#72a6ca",
    "UCB": "#39745a",
    "TS": "#b58a36",
    "author_random": "#999999",
    "unique_random_control": "#4d4d4d",
}
labels = {
    "GREEDY": "Greedy",
    "UCB": "UCB",
    "TS": "Thompson sampling",
    "author_random": "Author random",
    "unique_random_control": "Random, unique additions",
}
for ax, protein, letter in zip(axes, ["GB1", "TrpB"], ["a", "b"]):
    for condition in colors:
        row = next(r for r in data["summaries"] if r["protein"] == protein and r["condition"] == condition)
        mean = np.array(row["mean"])
        sem = np.array(row["sem"])
        x = data["query_budgets"]
        ax.plot(
            x,
            mean,
            marker="o",
            ms=4,
            lw=1.7,
            color=colors[condition],
            label=labels[condition],
            ls="-" if condition in ["GREEDY", "UCB", "TS"] else (":" if condition == "author_random" else "--"),
        )
        if condition in ["GREEDY", "UCB", "TS"]:
            ax.fill_between(x, mean - sem, mean + sem, color=colors[condition], alpha=0.09, lw=0)
    ax.set(title=protein, xlabel="Cumulative query events", ylabel="Best normalized fitness", ylim=(0, 1.06), xticks=x)
    ax.text(-0.17, 1.10, letter, transform=ax.transAxes, fontsize=17, fontweight="bold")
    ax.text(0.50, 1.12, "70 paired author seeds", ha="center", transform=ax.transAxes, fontsize=9)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.52, 0.005), ncol=3, frameon=False, fontsize=9)
for extension in ["pdf", "svg", "png"]:
    fig.savefig(OUT / ("figure3_alde_reproduction." + extension), dpi=220, facecolor="white")
(OUT / "figure3_alde_reproduction.sources.json").write_text(
    json.dumps(
        {
            "source": str(SOURCE.relative_to(ROOT)),
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "uncertainty": "SEM over 70 paired seeds for each acquisition; shared random controls counted once per seed",
            "scope": data["scope"],
        },
        indent=2,
    )
)
