# Versioned residue references in generated models

Generated-model tasks expose `residue_reference_catalog` through `inspect_model_code_task`. The catalog is provider-independent: it is available through the same tool contract across harnesses and language models. The frozen implementation includes `protein_references.py`, so a model plugin fingerprint binds the exact reference data and checker.

The initial catalog, `protein_residue_references.v1`, contains two pinned tables from Biopython 1.87. Their source bytes were compared with the official `biopython-187` release:

- `kyte_doolittle.hydropathy.v1`: [ProtParamData.py](https://raw.githubusercontent.com/biopython/biopython/biopython-187/Bio/SeqUtils/ProtParamData.py), symbol `kd`; dimensionless hydropathy scale.
- `free_amino_acid.average_mass.v1`: [IUPACData.py](https://raw.githubusercontent.com/biopython/biopython/biopython-187/Bio/Data/IUPACData.py), symbol `protein_weights`; free amino-acid average masses in Da. These are not polymer residue masses or the molecular mass of a peptide.

The catalog digest is `d29135c8d338f850b4cdab6292e4e16608ebfeab17eb999a5279da395905daa3`. Versioned reference IDs must not be silently repointed to changed data. A new table or revision needs an explicit ID, source provenance, an admission record and contract tests.

## Declaration contract

A generated source file can declare a module-level literal array or mapping:

```python
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = np.array([
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
], dtype=float)
PROTEIN_FEATURE_REFERENCES = {
    "HYDRO": {
        "reference_id": "kyte_doolittle.hydropathy.v1",
        "residue_order": "ACDEFGHIKLMNPQRSTVWY",
        "unit": "dimensionless hydropathy scale",
    }
}
```

The declaration itself must be a literal mapping: use `"residue_order": "ACDEFGHIKLMNPQRSTVWY"` directly, not a variable reference. Supported constant initializers are literal lists, tuples, dictionaries, or a syntactic `array`/`asarray` wrapper with a literal argument and optional `dtype`. The checker does not execute the source. Declared vectors must contain twenty finite numeric entries, and the declared order must contain every canonical amino acid exactly once. Literal values are compared with absolute tolerance `1e-6`; this is an exact-reference contract, not an experimental uncertainty threshold.

An unknown reference, incorrect unit, mismatched value or unsupported declared initializer is rejected before fitting. Unannotated generated source remains admissible, and the returned audit explicitly reports `not_declared`. Heuristics list some unclaimed property-like variable names; they are not a comprehensive detector of biological claims.

## Evidence lifecycle and limits

`submit_model_code` records `source_reference_audit`. The host independently recomputes this audit during model assessment, and registered-model discovery carries the assessment into a fresh reuse session. Tool-contract validation rejects tampered catalog or source-audit results. Reference parity does not replace independent validation and reuse checks. Previously exported models retain their original implementation binding and require that frozen bundle for replay. They are not automatically reclassified as validated by the new checker; implementation fingerprints change when this dependency is added.

`matched_declared_literals` means only that the declared literals agree with the specified table, residue order and units. It does not prove that the model actually uses those constants correctly, that its computed features have the claimed meaning, or that its predictions improve. `biological_interpretation_verified` is always false. A correct unused table can pass this limited check. A sequence mean of either reference table is a linear combination of amino-acid composition, not independent sequence information or evidence of a new model architecture.

The lifecycle tests use synthetic sequence targets to verify rejection, repaired-source training, independent registration and fresh-session reuse. They are engineering validation, not biological or autonomous-improvement evidence. Historical model diagnostics are performed on copies; existing campaigns keep their frozen source and are not refitted by this feature.
