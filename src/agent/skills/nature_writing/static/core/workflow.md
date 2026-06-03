# Writing workflow

Run these eight steps for any drafting or restructuring task. Steps 1-3 are planning, 4-6 are drafting, 7-8 are checking.

## 1. Build a one-sentence argument

> In [system/problem], we show [advance] using [approach], supported by [evidence], with [boundary].

Force every section to serve this sentence. If the sentence cannot be written, the paper does not yet have an argument — surface that to the user.

## 1b. Build the Terminology Ledger

On first contact with the material, extract the recurring terms, abbreviations, notation, and proper names into a Terminology Ledger before drafting any prose. Lock the canonical forms and reuse them across every section. See `../../../_shared/core/terminology-ledger.md`.

## 2. Choose section architecture

Pick the section structure from the relevant `section/*.md` fragment and, if needed, deeper patterns from `references/article-architecture.md`.

## 3. Map each paragraph to one job

Each paragraph must do exactly one job from: context, gap, approach, result, comparison, mechanism, implication, limitation.

If a paragraph carries two jobs, split it before drafting.

## 4. Draft from evidence outward

Keep claims near the data that support them. Do not stack claims at the top of a section then leave evidence at the bottom.

## 5. Calibrate verbs to evidence strength

`show` / `demonstrate` need strong direct evidence. `suggest` / `indicate` are for trend-level or indirect evidence. `may` / `could` are for plausible but unverified mechanisms.

## 6. Remove unsupported novelty and universal claims

Sweep for `first`, `unique`, `unprecedented`, `comprehensive`, `complete`, `always`, `never`. Replace with bounded claims or delete.

## 7. Run a paragraph-flow check

- One paragraph, one message.
- The first sentence is the topic / claim.
- Each subsequent sentence has an explicit relation to the previous one (cause, comparison, restriction, example).

For full reverse-outlining, open `references/paragraph-flow.md`.

## 8. Return prose plus notes

Output the draft together with explicit notes on assumptions, missing inputs, and where evidence is needed. See `output-format.md`.
