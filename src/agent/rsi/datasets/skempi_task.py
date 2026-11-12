"""Freeze one exact binding-ddG task after mapped, grouped SKEMPI2 source audit."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from .skempi_protocol import ALIGNMENT
from .skempi_protocol import SCHEMA as AUDIT_SCHEMA
from .skempi_source import LICENSE, PAPER, canonical, read_regular, require, sha256

SCHEMA = "skempi_native_protocol.v1"
SALT = "skempi2-structure-ddg-v1:"
HOLDOUT_LIMIT = 512
PROVENANCE = "SKEMPI 2.0; DOI 10.1093/bioinformatics/bty635; exact Kd ratios at reported measured Kelvin temperatures."
GROUPING = ("Connected components of source complex, original study, official holdout proteins/types, identical "
            "observed partner sequences and local BLOSUM62 partner alignments at >=30% identity and >=80% coverage each.")
INPUT_COLUMNS = ["wild_type_chains", "mutant_chains", "partner_chains", "temperature_kelvin", "assay_method"]


def eligible_rows(rows):
    selected, excluded, seen = [], Counter(), set()
    for row in sorted(rows, key=lambda r: r["source_line"]):
        if row["sequence_mapping_status"] != "verified_observed_residues":
            excluded["unverified_mapping"] += 1
        elif row["derived_ddg"]["value"] is None:
            excluded["nonexact_or_unmeasured_temperature"] += 1
        elif any(len(chain["observed_sequence"]) < ALIGNMENT["minimum_chain_length"]
                 for chain in row["partner_sequences"].values()):
            excluded["short_partner_chain"] += 1
        elif row["duplicate_fingerprint"] in seen:
            excluded["exact_source_duplicate"] += 1
        else:
            selected.append(row)
            seen.add(row["duplicate_fingerprint"])
    return selected, dict(excluded)


def grouped_splits(rows):
    """Whole groups, deterministic size balancing; no labels or measured scores."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["exclusion_group"]].append(row)
    total = len(rows)
    targets = {"train": total * 0.70, "validation": min(total * 0.15, HOLDOUT_LIMIT),
               "test": min(total * 0.15, HOLDOUT_LIMIT)}
    result = {name: [] for name in targets}
    counts = dict.fromkeys(targets, 0)
    for _group, values in sorted(groups.items(), key=lambda item: (-len(item[1]), sha256((SALT + item[0]).encode()))):
        choices = [name for name in targets if name == "train" or counts[name] + len(values) <= HOLDOUT_LIMIT]
        chosen = min(choices, key=lambda name: counts[name] / max(1, targets[name]))
        result[chosen].extend(sorted(values, key=lambda row: row["id"]))
        counts[chosen] += len(values)
    require(counts["train"] >= 200 and all(counts[n] >= 32 for n in ("validation", "test")),
            "Whole biological groups do not yield enough observations; never split groups to fit quotas")
    require(all(len({r["exclusion_group"] for r in values}) >= 4 for values in result.values()),
            "At least four independent exclusion groups per split are required")
    return result


def model_inputs(row):
    wild = {name: chain["observed_sequence"] for name, chain in row["partner_sequences"].items()}
    mutated = {name: list(sequence) for name, sequence in wild.items()}
    require(len(row["mutation_mapping"]) == len(row["mutations_pdb"]), "Incomplete mutation mapping")
    for original, link in zip(row["mutations_pdb"], row["mutation_mapping"]):
        chain, index = original["chain"], link["sequence_index"]
        require(mutated[chain][index] == original["wild_type"], "Mutation does not match the verified WT sequence")
        mutated[chain][index] = original["mutant"]
    return {"wild_type_chains": canonical(wild),
            "mutant_chains": canonical({chain: "".join(sequence) for chain, sequence in mutated.items()}),
            "partner_chains": canonical(row["partner_chains"]),
            "temperature_kelvin": row["temperature"]["kelvin"], "assay_method": row["raw"]["Method"]}


def export_protocol(audit_path, expected_sha256, output):
    from agent.harness.research_evaluation import evaluation_bytes, validate_data
    audit_path = Path(audit_path).absolute()
    raw = read_regular(audit_path, 16 * 1024**2)
    require(sha256(raw) == expected_sha256, "Grouping audit manifest checksum changed")
    audit = json.loads(raw)
    require(audit.get("schema") == AUDIT_SCHEMA and audit.get("complete") is True and audit.get("method") == ALIGNMENT,
            "Complete reviewed SKEMPI grouping audit required")
    content = read_regular(audit_path.parent / "observations.jsonl", 64 * 1024**2)
    require(sha256(content) == audit["observations_sha256"], "Grouped observations checksum changed")
    rows = [json.loads(line) for line in content.splitlines()]
    selected, excluded = eligible_rows(rows)
    splits = grouped_splits(selected)
    native = {"task": "regression", "provenance": PROVENANCE, "grouping": GROUPING,
              "splits": {name: [{"id": row["id"], "group": row["exclusion_group"], "inputs": model_inputs(row),
                                "target": row["derived_ddg"]["value"]} for row in values]
                         for name, values in splits.items()}}
    validate_data(native, "supervised")
    native_size = len(evaluation_bytes(native))
    # Verify input identity cannot cross groups after conversion to the exact API
    # input representation (this does not rely solely on source row IDs).
    owners = {}
    for name, values in native["splits"].items():
        for row in values:
            identity = sha256(canonical(row["inputs"]).encode())
            require(identity not in owners or owners[identity] == name, "Identical model inputs cross partitions")
            owners[identity] = name
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["id", "group", "split", *INPUT_COLUMNS, "binding_ddg_kcal_mol"], lineterminator="\n")
    writer.writeheader()
    membership = []
    for name, values in splits.items():
        for row, example in zip(values, native["splits"][name]):
            writer.writerow({"id": row["id"], "group": row["exclusion_group"], "split": name,
                             **example["inputs"], "binding_ddg_kcal_mol": example["target"]})
            membership.append({"id": row["id"], "group": row["exclusion_group"], "split": name,
                               "complex_id": row["complex_id"], "original_study": row["raw"]["Reference"],
                               "source_fingerprint": row["duplicate_fingerprint"]})
    dataset = buffer.getvalue().encode()
    require(len(dataset) <= 20 * 1024**2, "Native protected table size exceeded")
    protocol = {"schema": SCHEMA, "state": "prepared_not_trained", "source_dataset_id": "skempi2",
                "source_audit_sha256": expected_sha256, "property_family": "molecular_recognition",
                "target": {"name": "binding_ddg", "unit": "kcal mol^-1", "definition": "G_mut - G_wt"},
                "scope": "Exact affinity changes for structure-mapped mutations with measured temperatures and standard observed partner sequences. Not full-length sequence reconstruction or a complete published SKEMPI benchmark.",
                "model_task_count": 1, "selection": {"uses_targets": False, "method": "whole-group descending size, salted hash ties, relative split-capacity balancing",
                                                  "salt": SALT, "holdout_limit": HOLDOUT_LIMIT, "deduplication": "Only identical complete source rows"},
                "cohorts": {name: {"rows": len(values), "groups": len({r["exclusion_group"] for r in values}),
                                  "complexes": len({r["complex_id"] for r in values}),
                                  "studies": len({r["raw"]["Reference"] for r in values})} for name, values in splits.items()},
                "excluded_rows": excluded, "alignment_rule": ALIGNMENT,
                "limitations": ["Observed coordinate sequences can be incomplete; only verified residue sites are mutated.",
                                "Remote homology/shared domains below the declared alignment rule are not ruled out.",
                                "Measurements from different assay methods remain distinct; exact duplicates alone are removed.",
                                "Non-binding, unfolded, bounded, approximate and assumed-temperature observations are excluded, not assigned point targets.",
                                "This one task does not establish a trained, independently accepted or public model."],
                "reference_methods": ["training_mean", "mutation_physicochemical_ridge", "sequence_composition_ridge"],
                "metrics": {"primary": "mae", "secondary": ["rmse", "spearman"], "grouped_uncertainty": "bootstrap exclusion groups, never individual correlated rows"},
                "native_prepare": {"data_file": "inputs/binding_ddg.csv", "target_column": "binding_ddg_kcal_mol",
                                   "input_columns": INPUT_COLUMNS, "group_column": "group", "id_column": "id",
                                   "split_column": "split", "task": "regression", "provenance": PROVENANCE, "grouping": GROUPING},
                "input_encoding": "wild_type_chains and mutant_chains are JSON maps from chain IDs to observed amino-acid sequences; partner_chains is a JSON list of the two chain groups. No affinities or reference-study IDs are model features.",
                "native_evaluation_bytes_before_preparation_metadata": native_size,
                "data_terms": {"license": "CC-BY-4.0", "source_license": LICENSE, "citation": PAPER,
                               "public_redistribution": False},
                "trained_models": 0, "external_model_requests": 0}
    output = Path(output).absolute()
    require(not output.exists() and not any(p.is_symlink() for p in output.parents), "Output must be a new non-symlink directory")
    output.mkdir(parents=True, mode=0o700)
    artifacts = {"dataset.csv": dataset,
                 "membership.jsonl": ("\n".join(canonical(r) for r in membership) + "\n").encode()}
    for name, body in artifacts.items():
        path = output / name
        path.write_bytes(body)
        path.chmod(0o600)
    protocol["artifacts"] = {name: {"bytes": len(body), "sha256": sha256(body)} for name, body in artifacts.items()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "protocol.json").chmod(0o600)
    return protocol
