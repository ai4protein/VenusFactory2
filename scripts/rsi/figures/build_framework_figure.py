"""Vector framework figure using the previous figure's blue/yellow/peach palette."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/paper/0overleaf/figures/article"
BLUE = "#e4f0f8"
YELLOW = "#fff2c9"
PEACH = "#fce6d4"
GRAY = "#f2f2f2"
INK = "#24384b"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(9, 6.3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.4)
ax.axis("off")
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.018, top=0.985)


def box(x, y, w, h, text, color=GRAY, size=12, weight="normal", edge="#b9c1c8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.07", facecolor=color, edgecolor=edge, linewidth=1
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, linespacing=1.35
    )


def arrow(x1, y1, x2, y2, color=INK, style="-", rad=0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color,
            linestyle=style,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def panel(letter, title, x, y):
    ax.text(x, y, letter, fontsize=20, weight="bold", va="center", color="#111111")
    ax.text(x + 0.38, y, title, fontsize=15, weight="bold", va="center", color=INK)


panel("a", "Two modes, shared scientific interfaces", 0.1, 8.05)
box(0.25, 6.25, 2.1, 1.15, "Goal / question\n+ data", PEACH, 12, weight="bold")
ax.text(2.9, 7.65, "Science mode · expert team + persistent goal", fontsize=12, weight="bold", color=INK)
ax.text(2.9, 6.15, "Agent mode · interactive questions and tasks", fontsize=12, weight="bold", color=INK)
xs = [2.9, 5.1, 7.3, 9.5]
w = 1.85
for x, label in zip(xs, ["PI\nResearch", "CB\nPlan", "MLS\nImplement", "SC\nReview"]):
    box(x, 6.68, w, 0.7, label, BLUE, 12)
for x, label in zip(
    xs, ["Read\nRequest", "Choose\nAnswer or act", "Answer / tools\nAs needed", "Respond\nCurrent turn"]
):
    box(x, 5.36, w, 0.7, label, BLUE, 11.5)
for y in [7.03, 5.71]:
    for x in xs[:-1]:
        arrow(x + w, y, x + 2.2, y)
arrow(2.35, 6.98, 2.9, 7.03)
arrow(2.35, 6.48, 2.9, 5.71)
arrow(10.4, 6.55, 3.8, 6.55, style="--", rad=0)
ax.text(
    7.1,
    6.55,
    "Unmet goal: checkpoint progress and continue",
    ha="center",
    va="center",
    fontsize=8.5,
    color=INK,
    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
)
box(
    0.25, 4.82, 11.1, 0.38, "Shared plugins: protein tools  |  model development  |  reproduction workflows", YELLOW, 11
)
arrow(7.9, 5.3, 7.9, 5.2)
box(
    0.25,
    4.3,
    11.1,
    0.38,
    "Replaceable harness adapter  |  Independent model provider  |  Common evidence gateway",
    GRAY,
    10.6,
)
arrow(6.0, 4.82, 6.0, 4.68)
ax.plot([11.35, 11.65, 11.65], [7.03, 7.03, 5.01], color=INK, linewidth=1.25)
arrow(11.65, 5.01, 11.35, 5.01)
ax.plot([0.1, 11.8], [4.2, 4.2], color="#d9dde1", linewidth=1)

panel("b", "Model construction, validation and reuse", 0.1, 3.82)
coords = [(0.3, 2.65), (2.85, 2.65), (5.4, 2.65), (5.4, 1.15), (2.85, 1.15), (0.3, 1.15)]
texts = [
    "Task-specific\nmodelling need",
    "Author model\nsource code",
    "Train + debug\nTraining data only",
    "Independent\nacceptance test",
    "Publish validated\nmodel + scope",
    "Reuse or refit\nNew protein task",
]
colors = [PEACH, BLUE, YELLOW, GRAY, YELLOW, BLUE]
for (x, y), t, c in zip(coords, texts, colors):
    box(x, y, 2.15, 0.82, t, c, 11.5)
arrow(2.45, 3.06, 2.85, 3.06)
arrow(5, 3.06, 5.4, 3.06)
arrow(6.48, 2.65, 6.48, 1.97)
arrow(5.4, 1.56, 5, 1.56, color="#3d7a58")
arrow(2.85, 1.56, 2.45, 1.56)
arrow(1.37, 1.97, 1.37, 2.65, style="--")
ax.text(1.55, 2.30, "next task", fontsize=9.5, color="#637180", va="center")
ax.text(7.1, 2.32, "controller", fontsize=9.5, color="#637180", va="center", ha="center")
arrow(6.48, 1.15, 6.48, 0.70, color="#a95151")
box(5.4, 0.25, 2.15, 0.43, "Fail: no publication", GRAY, 10.5, edge="#c38686")
ax.text(0.3, 0.64, "Separate source, weights and model identity", fontsize=10, color=INK)
ax.text(0.3, 0.32, "Isolated execution + independent assessment", fontsize=10, color=INK)

ax.plot([7.83, 7.83], [0.18, 4.00], color="#d9dde1", linewidth=1)
panel("c", "Protein-study cases", 8.05, 3.82)
box(8.15, 2.63, 3.45, 0.79, "Antibody design\nThree-stage pipeline", BLUE, 10.5)
box(8.15, 1.52, 3.45, 0.79, "PET-hydrolase study\nInference + residue audit", YELLOW, 10.5)
box(8.15, 0.41, 3.45, 0.79, "Fitness landscapes\nFour-round campaigns", PEACH, 10.5)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(OUT / ("figure1_dual_mode_pipeline." + ext), dpi=240, facecolor="white")
plt.close(fig)
print(OUT / "figure1_dual_mode_pipeline.pdf")
