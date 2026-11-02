"""Descriptive paper panels derived from archived reports."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/paper/0overleaf/figures/article"
RUNS = ROOT / "research_runs/rsi"
paths = {
    "vhh": RUNS / "20261011_rfantibody_ha_8x4_deterministic_v2/quality_audit/audit.json",
    "rsv": RUNS / "20261011_rfantibody_rsv_8x4_v1/quality_audit/audit.json",
    "publication": RUNS / "20261011_turbopetase_publication_audit_v1/report.json",
    "pet": RUNS / "20261011_turbopetase_author_inference_v1/report.json",
    "generated": RUNS / "20261011_generated_alde_to_mcba_glm51_v1/report.json",
    "landscapes": RUNS / "20261011_generated_transfer_gb1_trpb_glm51_v2/report.json",
}
data = {k: json.loads(p.read_text()) for k, p in paths.items()}
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
fig, axes = plt.subplots(3, 2, figsize=(9, 10.1))
fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.08, hspace=0.66, wspace=0.38)
blue = "#72a6ca"
green = "#39745a"
gold = "#e8c46d"


def title(ax, letter, text):
    ax.set_title(text, loc="left", fontsize=12, pad=12)
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=18)


records = data["vhh"]["records"]
ax = axes[0, 0]
title(ax, "a", "HA VHH (32 designs)")
for passed, color, label in [(False, blue, "Does not pass"), (True, green, "Passes filter (2/32)")]:
    rr = [r for r in records if r["additional_interface_filter_pass"] == passed]
    ax.scatter(
        [r["scores"]["pae"] for r in rr],
        [r["scores"]["target_aligned_antibody_rmsd"] for r in rr],
        c=color,
        s=35,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
        label=label,
        zorder=3,
    )
ax.axvline(10, color="gray", ls="--", lw=1)
ax.axhline(2, color="gray", ls="--", lw=1)
ax.set(
    xlabel="Global predicted aligned error",
    ylabel="Target-aligned VHH RMSD (Å)",
    xlim=(1.5, 11),
    ylim=(0.65, 90),
    yscale="log",
)
ax.set_yticks([1, 2, 5, 10, 20, 50])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
ax = axes[0, 1]
title(ax, "b", "RSV site-3 VHH (32 designs)")
rsv = data["rsv"]["records"]
ax.scatter(
    [r["scores"]["pae"] for r in rsv],
    [r["scores"]["target_aligned_antibody_rmsd"] for r in rsv],
    color=blue,
    s=35,
    alpha=0.8,
    edgecolors="white",
    linewidths=0.4,
)
ax.axvline(10, color="gray", ls="--", lw=1)
ax.axhline(2, color="gray", ls="--", lw=1)
ax.set(
    xlabel="Global predicted aligned error",
    ylabel="Target-aligned VHH RMSD (Å)",
    xlim=(1.5, 11),
    ylim=(0.65, 90),
    yscale="log",
)
ax.set_yticks([1, 2, 5, 10, 20, 50])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.text(0.5, -0.29, "Passes filter: 0/32", transform=ax.transAxes, ha="center", fontsize=9)
ax = axes[1, 0]
title(ax, "c", "Retained designs by backbone")
counts = [
    sum(r["additional_interface_filter_pass"] and r["id"].startswith(f"samples_design_{i}_") for r in records)
    for i in range(8)
]
rsv_counts = [
    sum(r["additional_interface_filter_pass"] and r["id"].startswith(f"samples_design_{i}_") for r in rsv)
    for i in range(8)
]
x = np.arange(8)
ax.bar(x - 0.18, 4, width=0.32, color="#e2e7eb")
ax.bar(x + 0.18, 4, width=0.32, color="#f4ead4")
ax.bar(x - 0.18, counts, width=0.32, color=green, label="HA retained")
ax.bar(x + 0.18, rsv_counts, width=0.32, color=gold, label="RSV retained")
ax.set(
    xlabel="Backbone identity (within target)",
    ylabel="Number of designs",
    ylim=(0, 4.8),
    xticks=x,
    yticks=[0, 1, 2, 3, 4],
)
ax.legend(frameon=False, fontsize=9)
ax.text(0.02, -0.29, "Pale bars: 4 designed sequences per backbone", transform=ax.transAxes, fontsize=9)
ax = axes[1, 1]
title(ax, "d", "PET indexing sensitivity")
overlaps = []
for checkpoint in ["PF01083_19.pt", "PTM311_19.pt", "pretrained-model_19.pt"]:
    values = [r for r in data["pet"]["comparisons"] if r["checkpoint"] == checkpoint]
    original = next(r for r in values if r["indexing"] == "author_notebook")
    corrected = next(r for r in values if r["indexing"] == "residue_index_corrected")
    assert original["independent_author_function_max_abs_error"] == 0
    overlaps.append(len({x["position"] for x in original["top10"]} & {x["position"] for x in corrected["top10"]}))
ax.bar(np.arange(3), overlaps, color=gold, width=0.6)
for i, n in enumerate(overlaps):
    ax.text(i, n + 0.2, f"{n}/10", ha="center")
ax.set(
    ylabel="Top-10 positions shared", ylim=(0, 11), xticks=np.arange(3), xticklabels=["PF01083", "PTM311", "Pretrained"]
)
ax.text(0.02, -0.23, "Original notebook vs corrected CLS indexing", transform=ax.transAxes, fontsize=9)
ax = axes[2, 0]
title(ax, "e", "PET published-ranking agreement")
comparisons = data["publication"]["checkpoint_to_published_rankings"]
checkpoints = ["PF01083_19.pt", "PTM311_19.pt", "pretrained-model_19.pt"]
columns = ["BhrPETase_Uniclust30_BFD", "BhrPETase_UniProt"]
matrix = np.array(
    [
        [
            next(
                v["top10_overlap"]
                for v in comparisons
                if v["checkpoint"] == checkpoint and v["indexing"] == indexing and v["published_column"] == column
            )
            for checkpoint in checkpoints
            for indexing in ["author_notebook", "residue_index_corrected"]
        ]
        for column in columns
    ]
)
from matplotlib.colors import LinearSegmentedColormap

ax.imshow(
    matrix, vmin=0, vmax=10, cmap=LinearSegmentedColormap.from_list("rank", ["#f5f8fa", blue, green]), aspect="auto"
)
for i in range(2):
    for j in range(6):
        ax.text(j, i, f"{matrix[i, j]}/10", ha="center", va="center", fontsize=9)
ax.set(xticks=np.arange(6), xticklabels=["Orig", "CLS+1"] * 3, yticks=[0, 1], yticklabels=["Uniclust/\nBFD", "UniProt"])
ax.tick_params(axis="both", labelsize=8)
for x, label in [(0.5, "PF01083"), (2.5, "PTM311"), (4.5, "Pretrained")]:
    ax.text(x, -0.56, label, ha="center", fontsize=9, transform=ax.transData)
ax.text(0.0, -0.28, "Published BhrPETase columns; no exact ranking match", transform=ax.transAxes, fontsize=8.5)
ax = axes[2, 1]
title(ax, "f", "Frozen-source predictive transfer")
generated = data["generated"]
gb1 = next(r for r in data["landscapes"]["results"] if r["case"] == "GB1")
rows = [generated, generated["transfer"], gb1]
means = [r["reuse"]["future_metrics"]["training_mean_rmse"] for r in rows]
ratios = [r["reuse"]["future_metrics"]["rmse"] / m for r, m in zip(rows, means)]
fixed = [r["fixed_recipe_baseline"]["future"]["rmse"] / m for r, m in zip(rows, means)]
x = np.arange(3)
ax.bar(x - 0.17, ratios, width=0.32, color=blue, label="Generated source")
ax.bar(x + 0.17, fixed, width=0.32, color=gold, label="Fixed ridge recipe")
ax.axhline(1, color="gray", lw=1, ls="--", label="Training-mean predictor")
ax.set(
    ylabel="Later-set RMSE / mean baseline",
    ylim=(0, 1.5),
    xticks=x,
    xticklabels=["ParPgb\n(n=90)", "McbA\n(n=14)", "GB1\n(n=1,024)"],
)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(0.02, -0.28, "TrpB rejected at acceptance (skill = -0.714)", transform=ax.transAxes, fontsize=9)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(OUT / ("figure2_computational_audits." + ext), dpi=220, facecolor="white")
manifest = {
    "scope": "descriptive audits; no independent-lineage confidence intervals or experimental binding claim",
    "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
    "pet_top10_overlaps": overlaps,
    "vhh_passes_by_backbone": counts,
    "rsv_passes_by_backbone": rsv_counts,
    "published_rank_overlap_matrix": matrix.tolist(),
    "generated_rmse_over_mean": ratios,
    "fixed_rmse_over_mean": fixed,
}
(OUT / "figure2_computational_audits.sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
