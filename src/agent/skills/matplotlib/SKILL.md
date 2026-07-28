---
name: matplotlib
description: >-
  Matplotlib OO/pyplot guidance for custom plots via agent_generated_code. Use for fine-grained control. Prefer nature_figure for manuscript figures and seaborn for quick statistical EDA.
license: https://github.com/matplotlib/matplotlib/tree/main/LICENSE
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# Matplotlib

## Overview

Foundational plotting. In VenusFactory execute via **`agent_generated_code`**. Helpers: `src/tools/visualize/matplotlib/plot_template.py`, `style_configurator.py`.

## When to use / NOT

| Use matplotlib | Prefer |
|----------------|--------|
| Custom artists, insets, unusual projections | — |
| Statistical EDA defaults | `seaborn` |
| Nature submission figures | `nature_figure` (load first) |

## Quick patterns

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(x, y)
ax.set_xlabel("..."); ax.set_ylabel("...")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
```

Prefer the **OO API** (`fig, ax = plt.subplots`) over pyplot state for multi-step agent code.

## Common mistakes

- Forgetting dpi≥300 / session `output_dir`
- Leaving interactive `plt.show()` in headless runs
- Skipping `nature_figure` for publication output

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="matplotlib", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.
