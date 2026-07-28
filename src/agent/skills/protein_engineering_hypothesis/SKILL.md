---
name: protein_engineering_hypothesis
description: >-
  Evidence-bounded hypothesis and experiment planning for protein engineering. Use when the user asks what to mutate next, how to prioritize variants, how to falsify a mechanism, or how to design a directed-evolution round. Do NOT invent wet-lab results; chain VenusFactory tools for computational evidence and label model scores as hypotheses.
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Protein Engineering Hypothesis

## Overview

Instruction-only skill: structure the scientific argument, then point MLS at concrete hub tools. Inspired by scientific-agent-skills hypothesis-generation, adapted to VenusFactory protein tools.

## VenusFactory execution

1. Load this skill.
2. Produce a short plan with: Observation → Hypothesis → Computational test → Wet-lab falsifier.
3. Execute tests via other skills/tools (do not reimplement).

## Recommended evidence chain

| Question | Skills / tools |
|----------|----------------|
| What is known about the protein? | `uniprot_database`, `interpro_domain_annotation`, `pubmed` |
| Where is it expressed? | `hpa_expression_context` |
| Structure confidence? | `protein_structure_pipeline` / `alphafold_database` |
| Which mutations look beneficial? | `zero_shot_mutation_workflow` |
| Redesign backbone-constrained seq? | `proteinmpnn_design_workflow` |
| Kinetics / EC? | `brenda_database` |
| Partners / pathways? | `string_database`, `kegg_database` |
| Train a custom head? | `venus_finetune_workflow` |

## Output template (for CB/MLS)

```markdown
### Observation
...
### Hypothesis (falsifiable)
...
### Computational tests (ordered)
1. tool/skill … success = …
### Predicted outcome
...
### Wet-lab falsifier (suggested, not executed here)
...
### Risks / confounders
- Model score ≠ experimental activity
```

## Rules

- Never present zero-shot / PLM ranks as measured ΔΔG or activity.
- Prefer protecting catalytic residues (`interpro` + literature) before design.
- End computational rounds with at least one figure when numeric tables exist.

## When NOT to use

- User only wants a single tool call with no scientific framing — call that tool's skill directly.
