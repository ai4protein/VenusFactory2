# SKEMPI 2.0 binding-affinity data

SKEMPI adds a molecular-recognition task to Scientist research: predict the change
in binding free energy caused by mutations at a protein–protein interface.
Preparing this dataset does not train a model or register a public tool.

## Source and measurement meaning

Use the [official CSV and cleaned structure downloads](https://life.bsc.es/pid/skempi2/database/index),
retain the [database citation](https://doi.org/10.1093/bioinformatics/bty635),
and preserve its [CC BY 4.0 attribution terms](https://life.bsc.es/pid/skempi2/info/terms).

The [official field guide](https://life.bsc.es/pid/skempi2/info/faq_and_help)
defines Kd in molar units, original and cleaned residue numbering, insertion
codes, partner chains, original references and holdout relationships. Temperature
is Kelvin, as also labelled in the [official browser](https://life.bsc.es/pid/skempi2/database/browse).
For exact positive Kd values at an explicitly measured temperature:

```text
binding_ddg_kcal_mol = (8.314 / 4184) * T_kelvin * (ln(Kd_mut_M) - ln(Kd_wt_M))
```

Positive values therefore indicate weaker mutant binding. The baseline is the
reference form represented by the source structure; it need not be a complete
natural full-length protein. Association/dissociation rates and enthalpy/entropy
are retained with their own units; they are not merged into the ddG target.

Source fields remain intact. Inequalities, no detectable binding, unfolding,
approximate measurements, missing values and assumed temperatures are typed
explicitly. In particular, a numeric `*_parsed` value never overrides a censored
raw measurement. Only exact Kd with measured temperature is converted to a point
target; no room-temperature imputation is used.

## Staging workflow

All commands work offline from frozen, hash-bound snapshots and create new output
directories. Downloads and large derived datasets belong in ignored runtime/data
storage, outside versioned source. No upstream data or model weights are bundled.

```bash
python scripts/rsi/data_preparation/import_skempi_source.py \
  --manifest <source-cache>/manifest.json --sha256 <source-manifest-sha256> \
  --structure-manifest <structure-cache>/manifest.json \
  --structure-sha256 <structure-manifest-sha256> --output <mapped-cache>

python scripts/rsi/data_preparation/audit_skempi_groups.py \
  --manifest <mapped-cache>/manifest.json --sha256 <mapped-manifest-sha256> \
  --output <group-audit-cache>

python scripts/rsi/data_preparation/prepare_skempi_task.py \
  --audit <group-audit-cache>/manifest.json --sha256 <group-audit-sha256> \
  --output <prepared-task>
```

The CSV snapshot uses the existing `property_source_snapshot.v1` format.
The structure snapshot uses `skempi_structure_snapshot.v1` and binds three
official assets: `SKEMPI2_PDBs.tgz`, `terms.html` and `faq.html`, with their source
URLs, final URLs, byte sizes and SHA-256 hashes. Its license is `CC-BY-4.0`.
Archive members are read in memory and never extracted to arbitrary paths.

The mapper verifies original numbering and insertion codes against the supplied
`.mapping` files, cleaned coordinate residues, wild-type identities, both partner
chain sets and each mutation position. Solvent/ion HETATM numbers are separate
from polymer residue identities. Nonstandard partner residues, repeated mutation
sites, missing sites, ambiguous mappings and WT mismatches require review; they
are not repaired by guessing.

Observed-coordinate sequences are retained as observed, without filling unresolved
residues or claiming that they are complete UniProt sequences.

## Grouping and admission

Before filtering individual observations, form connected components using:

- the same source complex;
- the same original study;
- the official holdout-protein and holdout-type relations;
- identical observed partner sequences;
- similarity clusters of either partner, from local BLOSUM62 alignments with gap
  opening −10, extension −0.5, at least 30% identity over alignment columns and
  at least 80% aligned-residue coverage of each sequence.

This is an explicit operational similarity rule, not proof of remote-homology
independence. Short partner chains below 20 residues are excluded from this task
rather than claiming reliable homology separation for them. Shared domains below
the stated coverage threshold remain a limitation.

The task retains complete exclusion groups. It balances group sizes deterministically
without target values, preserves a maximum of 512 rows in each native holdout,
requires at least four groups per partition and fails if whole groups cannot meet
the minimum sample requirements. It never cuts a group to fill a quota.
Only identical complete source rows are deduplicated; distinct measurements at
different conditions remain available. Uncertainty should be resampled by exclusion
group, not by correlated individual rows.

The native table supplies observed WT/mutant chain sequences, partner membership,
temperature and assay method. Kd values, ddG targets and study identifiers are not
input features. `native_prepare` in the protocol uses the existing protected-data
store: only the training table reaches the Scientist workspace; validation and test
labels remain behind a host reference.

## Verified snapshot, 12 November 2026

The unchanged source contains 7,085 observations, 348 complex identifiers and 295
original references. The official structure archive is 30,482,090 bytes.
6,624 observations pass the residue/partner mapping checks; 461 require review.
For example, the supplied 1KBH mapping repeats original sites and is not silently
collapsed.

The all-pairs audit covers 669 unique observed sequences, computes 45,030
length-compatible alignments and records all qualifying edges. The full source
has 78 exclusion components; the largest contains 3,551 observations.
After exact-measurement, mapping, length and duplicate exclusions, the prepared
task contains:

| Partition | Observations | Exclusion groups | Complexes | Original studies |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,345 | 40 | 171 | 120 |
| Validation | 503 | 7 | 24 | 22 |
| Test | 503 | 6 | 11 | 9 |

The largest selected component has 1,491 observations. The small number of
independent holdout groups limits precision of generalization estimates.
Exclusions are 2,445 nonexact/temperature-ineligible rows, 461 mapping failures,
826 short-partner rows and two exact duplicate rows in the otherwise eligible
cohort. Exclusion counts are sequential and should not be added to overlapping
raw quality flags.

State: **prepared_not_trained**. One model-development task is prepared.
No fitted model, accepted model, public registration or external model request
is implied. Planned conventional comparators in the protocol remain proposals
until executed and measured. Independent scientific review is still required.

## Verification

```bash
python -m pytest -q tests/rsi/test_skempi_source.py tests/rsi/test_skempi_task.py
```

Tests cover censored values, units and ddG sign, source checksums, archive paths,
residue/insertion correspondence, WT mismatches, duplicate mappings, whole-group
selection, label independence and the real protected-data preparation path.
Synthetic test fixtures are software checks and are not counted as biological results.
