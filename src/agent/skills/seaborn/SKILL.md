---
name: seaborn
description: >-
  Seaborn statistical plots for exploratory analysis via agent_generated_code. Use for quick relational/distribution/categorical charts. Do NOT use for submission-grade Nature figures (nature_figure) or low-level artists control (matplotlib).
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# Seaborn Statistical Visualization

## Overview

High-level statistical viz on top of matplotlib. In VenusFactory, plots run through **`agent_generated_code`** / `python_repl`. Publication figures must load **`nature_figure`** first.

## VenusFactory execution

- No `seaborn_*` `@tool`. Optional helpers: `src/tools/visualize/matplotlib/plot_template.py`, `style_configurator.py`.
- Deep API: `read_skill("seaborn", relative_path="references/function_reference.md")` or `references/legacy_guide.md`.

## When to use / NOT

| Use seaborn | Prefer instead |
|-------------|----------------|
| EDA bar/box/violin/heatmap | — |
| Publication multi-panel Nature style | `nature_figure` |
| Pixel-perfect artists / custom projections | `matplotlib` |

## Quick start

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=df, x="mutation", y="score")
fig = ax.get_figure()
fig.savefig(out_png, dpi=300, bbox_inches="tight")
```

## Plot selection (index)

- Relational: `scatterplot`, `lineplot`, `relplot`
- Distribution: `histplot`, `kdeplot`, `ecdfplot`
- Categorical: `boxplot`, `violinplot`, `barplot`, `stripplot`
- Matrix: `heatmap`, `clustermap`
- Objects API: `seaborn.objects` — see `references/objects_interface.md` if present / legacy guide

## Common mistakes

- Skipping `nature_figure` for manuscript figures
- Not saving PNG under session `output_dir` at dpi≥300
- Passing wide data without melting for categorical plots

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="seaborn", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.
