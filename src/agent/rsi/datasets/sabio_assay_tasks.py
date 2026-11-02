"""Prepare source-bound property prediction tasks from public plant enzyme assays."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from agent.rsi import generated_assay, model_tasks
from agent.rsi import sabio_assay_encoder as encoder
from agent.rsi.engine.protein_model import digest
from .research_project_catalog import inspect_candidate
from agent.rsi.engine.storage import atomic_json, sha256_file

PROPERTIES = ("kcat", "Km", "kcat/Km")
UNITS = {"kcat": "s^(-1)", "Km": "M", "kcat/Km": "M^(-1)*s^(-1)"}
TASK_IDS = {"kcat": "plant_kcat", "Km": "plant_km", "kcat/Km": "plant_kcat_over_km"}
ALLOWED_ORGANISMS = {
    "Arabidopsis thaliana",
    "Zea mays",
    "Oryza sativa",
    "Solanum lycopersicum",
    "Triticum aestivum",
    "Hordeum vulgare",
    "Spinacia oleracea",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]), "Regular source file required")
    return {"path": str(path), "sha256": sha256_file(path)}


def bind_sequences(rows, entry_documents, taxonomy_documents=()):
    """Replay sequence and taxon checks from frozen official API responses."""
    entries = defaultdict(dict)
    for document in entry_documents:
        for entry in document["results"]:
            primary = entry["primaryAccession"]
            for accession in [primary, *entry.get("secondaryAccessions", [])]:
                if primary in entries[accession]:
                    require(entries[accession][primary] == entry, "Inconsistent duplicate reference entry")
                entries[accession][primary] = entry
    taxonomies = {record["taxonId"]: record for record in taxonomy_documents}
    references, exclusions = {}, []
    for accession in sorted({r["protein"] for r in rows}):
        matches = entries[accession]
        reason = None
        if len(matches) != 1:
            exclusions.append({"protein": accession, "reason": "missing_or_ambiguous_reference"})
            continue
        entry = next(iter(matches.values()))
        origins = {(r["organism"], r["taxonomy_id"]) for r in rows if r["protein"] == accession}
        require(len(origins) == 1, "Conflicting source organism identity")
        species, source_taxon = next(iter(origins))
        require(species in ALLOWED_ORGANISMS, "Outside declared plant source scope")
        organism = entry["organism"]
        taxonomy = taxonomies.get(organism["taxonId"], {})
        lineage = {x["taxonId"] for x in taxonomy.get("lineage", [])}
        if "parent" in taxonomy:
            lineage.add(taxonomy["parent"]["taxonId"])
        exact = organism["taxonId"] == source_taxon
        ancestor = source_taxon in lineage and taxonomy.get("scientificName") == organism["scientificName"]
        sequence = entry.get("sequence", {}).get("value", "")
        if not exact and not ancestor:
            reason = "source_reference_taxonomy_mismatch"
        elif entry.get("proteinDescription", {}).get("flag") in {"Fragment", "Fragments"}:
            reason = "fragment_reference"
        elif not isinstance(sequence, str) or not 1 <= len(sequence) <= 4096 or set(sequence) - set(encoder.AA):
            reason = "unsupported_reference_sequence"
        elif entry["sequence"].get("length") != len(sequence):
            reason = "sequence_length_mismatch"
        elif entry["sequence"].get("md5", "").lower() != hashlib.md5(sequence.encode()).hexdigest():
            reason = "sequence_checksum_mismatch"
        if reason:
            exclusions.append({"protein": accession, "reason": reason})
            continue
        references[accession] = {
            "source_accession": accession,
            "primary_accession": entry["primaryAccession"],
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "sequence_version": entry["entryAudit"]["sequenceVersion"],
            "sequence_updated": entry["entryAudit"]["lastSequenceUpdateDate"],
            "source_organism": species,
            "source_taxonomy": source_taxon,
            "reference_taxonomy": organism["taxonId"],
            "taxonomy_binding": "exact_taxonomy" if exact else "verified_taxonomic_ancestor",
            "experimental_construct_identity": "Reference sequence; exact experimental construct not established",
        }
    return references, exclusions


def similarity_evidence(references):
    import Bio
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10.0, -0.5
    accessions = sorted(references)
    pairs, edges = [], []
    for index, left in enumerate(accessions):
        a = references[left]["sequence"]
        for right in accessions[index + 1 :]:
            b = references[right]["sequence"]
            alignment = aligner.align(a, b)[0]
            aligned = sum(int(end - start) for start, end in alignment.aligned[0])
            identities = int(alignment.counts().identities)
            identity = identities / aligned if aligned else 0.0
            coverage = min(aligned / len(a), aligned / len(b))
            item = {
                "left": left,
                "right": right,
                "identities": identities,
                "aligned_residues": aligned,
                "identity": identity,
                "minimum_coverage": coverage,
            }
            pairs.append(item)
            if identity >= 0.4 and coverage >= 0.8:
                edges.append([left, right])
    return {
        "schema": "plant_reference_similarity.v1",
        "biopython": Bio.__version__,
        "alignment": "first optimal global BLOSUM62 alignment; gap open -10, extension -0.5",
        "identity_definition": "identical residues / aligned residue pairs",
        "edge_policy": {"minimum_identity": 0.4, "minimum_coverage_both_sequences": 0.8},
        "pairs": pairs,
        "edges": edges,
        "interpretation": "A declared sequence-similarity separation, not proof of absence of remote homology.",
    }


def components(rows, references, edges):
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for row in rows:
        protein = "protein:" + row["protein"]
        union(protein, "publication:" + row["publication_group"])
        union(protein, "sequence:" + references[row["protein"]]["sequence_sha256"])
    present = {r["protein"] for r in rows}
    for left, right in edges:
        if left in present and right in present:
            union("protein:" + left, "protein:" + right)
    members = defaultdict(list)
    for node in sorted(parent):
        members[find(node)].append(node)
    names = {key: "pc_" + digest(value)[:20] for key, value in members.items()}
    return {row["id"]: names[find("protein:" + row["protein"])] for row in rows}


def split_components(rows, groups):
    """Select by endpoint counts only; never inspect target magnitudes or fitted errors."""
    counts = defaultdict(Counter)
    for row in rows:
        counts[groups[row["id"]]][row["property"]] += 1
    keys = sorted(counts)
    require(len(keys) >= 11, "Insufficient independent components")
    n_validation = max(3, round(len(keys) * 0.2))
    n_future = max(3, round(len(keys) * 0.2))
    n_training = len(keys) - n_validation - n_future
    require(n_training >= 5, "Insufficient training components")
    totals = Counter(row["property"] for row in rows)
    rng = random.Random(20261027)
    best = None
    for _ in range(4096):
        order = keys[:]
        rng.shuffle(order)
        assignment = {
            key: "training" if i < n_training else "validation" if i < n_training + n_validation else "future"
            for i, key in enumerate(order)
        }
        summary = {p: {phase: {"rows": 0, "groups": 0} for phase in model_tasks.PHASES} for p in PROPERTIES}
        for key, phase in assignment.items():
            for prop, count in counts[key].items():
                summary[prop][phase]["rows"] += count
                summary[prop][phase]["groups"] += 1
        if any(
            summary[prop][phase][kind] < minima[phase]
            for prop in PROPERTIES
            for kind, minima in [
                ("rows", {"training": 40, "validation": 15, "future": 15}),
                ("groups", {"training": 5, "validation": 3, "future": 3}),
            ]
            for phase in model_tasks.PHASES
        ):
            continue
        score = sum(
            (summary[prop][phase]["rows"] / totals[prop] - fraction) ** 2
            for prop in PROPERTIES
            for phase, fraction in [("training", 0.6), ("validation", 0.2), ("future", 0.2)]
        )
        key = (score, tuple(assignment[x] for x in keys))
        if best is None or key < best[0]:
            best = (key, assignment, summary)
    require(best is not None, "No label-independent split meets scientific sample minima")
    return best[1], {
        "schema": "assay_component_split.v1",
        "seed": 20261027,
        "candidates": 4096,
        "selection": "minimum endpoint row-fraction squared deviation from 60/20/20; labels unused",
        "components": len(keys),
        "counts": best[2],
        "assignment": best[1],
    }


def prepare(observations, sequence_snapshots, taxonomy_snapshots, output):
    output = Path(output).resolve()
    require(not output.exists(), "Task output must be fresh")
    inputs = [reference(observations), *map(reference, sequence_snapshots), *map(reference, taxonomy_snapshots)]
    rows = json.loads(Path(observations).read_text())
    require(
        isinstance(rows, list) and rows and len({r["id"] for r in rows}) == len(rows),
        "Unique source observations required",
    )
    references, excluded_proteins = bind_sequences(
        rows,
        [json.loads(Path(p).read_text()) for p in sequence_snapshots],
        [json.loads(Path(p).read_text()) for p in taxonomy_snapshots],
    )
    retained, features, excluded_rows = [], {}, []
    for row in rows:
        if row["protein"] not in references:
            excluded_rows.append({"id": row["id"], "reason": "unavailable_complete_reference"})
            continue
        require(
            row["property"] in PROPERTIES and row["unit"] == UNITS[row["property"]], "Source endpoint units changed"
        )
        require(
            encoder.finite(row["value"])
            and row["value"] > 0
            and math.isclose(math.log10(row["value"]), row["log10_value"], rel_tol=1e-12, abs_tol=1e-12),
            "Source log target differs from observed physical value",
        )
        try:
            vector = encoder.encode(row, references[row["protein"]]["sequence"])
        except ValueError as exc:
            excluded_rows.append({"id": row["id"], "reason": str(exc)})
            continue
        features[row["id"]] = vector
        retained.append(row)
    require(retained, "No eligible assay observations")
    similarity = similarity_evidence({p: references[p] for p in {r["protein"] for r in retained}})
    groups = components(retained, references, similarity["edges"])
    assignment, split = split_components(retained, groups)
    output.mkdir(parents=True)
    atomic_json(output / "reference_sequences.json", references)
    atomic_json(output / "reference_exclusions.json", excluded_proteins)
    atomic_json(output / "observation_exclusions.json", excluded_rows)
    atomic_json(output / "similarity.json", similarity)
    atomic_json(output / "split.json", split)
    feature_rows = [
        {
            "id": row["id"],
            "protein": row["protein"],
            "sequence": references[row["protein"]]["sequence"],
            "substrate": encoder.substrate_identity(row),
            "study": row["publication_group"],
            "group": groups[row["id"]],
            "values": features[row["id"]],
            "property": row["property"],
        }
        for row in retained
    ]
    atomic_json(output / "encoded_inputs.json", feature_rows)
    encoder_path = Path(encoder.__file__).resolve()
    references_bound = (
        inputs
        + [
            reference(output / name)
            for name in ["reference_sequences.json", "similarity.json", "split.json", "encoded_inputs.json"]
        ]
        + [reference(encoder_path)]
    )
    paper = {"title": "SABIO-RK—database for biochemical reaction kinetics", "doi": "10.1093/nar/gkr1046"}
    context = {
        "source": "SABIO-RK public in-vitro wildtype plant measurements",
        "representation": encoder.NAME,
        "partition": "protein, publication and declared 40-percent sequence-similarity components",
        "conditions": "pH/temperature intervals plus source buffer/other-condition categorical tokens",
    }
    tasks, catalog = [], []
    indexed = {r["id"]: r for r in feature_rows}
    for prop in PROPERTIES:
        folder = output / TASK_IDS[prop]
        folder.mkdir()
        data = {phase: [] for phase in model_tasks.PHASES}
        for raw in retained:
            if raw["property"] != prop:
                continue
            row = {key: value for key, value in indexed[raw["id"]].items() if key != "property"}
            row["target"] = raw["log10_value"]
            data[assignment[row["group"]]].append(row)
        for phase in data:
            data[phase].sort(key=lambda row: row["id"])
        evidence = {
            "schema": "assay_groups.v1",
            "level": "protein_publication_component",
            "study_doi": paper["doi"],
            "sources": references_bound[:-1],
            "observations": {
                r["id"]: {"group": r["group"], "phase": phase, "input_sha256": generated_assay.input_digest(r)}
                for phase, part in data.items()
                for r in part
            },
        }
        atomic_json(folder / "groups.json", evidence)
        scope = {
            "protein": "source_resolved_plant_enzyme_panel",
            "assay": "in-vitro wildtype enzyme kinetics",
            "target": f"log10_{prop}",
            "unit": f"log10({UNITS[prop]})",
            "context": context,
            "interface_version": generated_assay.PLUGIN_ID,
            "covariates": encoder.covariates(),
            "representation": {
                "name": encoder.NAME,
                "dimensions": encoder.DIMENSIONS,
                "encoder_sha256": sha256_file(encoder_path),
            },
            "grouping": {
                "level": "protein_publication_component",
                "evidence_sha256": sha256_file(folder / "groups.json"),
            },
        }
        for phase, part in data.items():
            atomic_json(folder / f"{phase}.json", part)
        task = {
            "schema": 1,
            "id": TASK_IDS[prop],
            "stage": "kinetics",
            "plugin_id": generated_assay.PLUGIN_ID,
            "scope": scope,
            "paper": paper,
            "references": references_bound + [reference(folder / "groups.json")],
            "data": {phase: reference(folder / f"{phase}.json") for phase in model_tasks.PHASES},
            "minimum_skill": 0.0,
        }
        atomic_json(folder / "task.json", task)
        model_tasks.load_task(task)
        assignments = {r["id"]: {"group": r["group"], "phase": phase} for phase, part in data.items() for r in part}
        atomic_json(folder / "catalog_groups.json", assignments)
        candidate = {
            "id": task["id"],
            "project_id": "sabio_plant_kinetics",
            "task": reference(folder / "task.json"),
            "protein_scope": {"kind": "project", "identifier": scope["protein"]},
            "property": scope["target"],
            "scenario": context,
            "question": f"Predict source-observed wildtype plant enzyme {prop} on withheld protein/publication components",
            "groups": reference(folder / "catalog_groups.json"),
            "group_basis": scope["grouping"]["level"],
            "scenario_evidence": inputs[0],
        }
        record, _ = inspect_candidate(candidate, {"study_doi": paper["doi"]})
        require(record["status"] == "admitted", "Prepared task fails scientific catalog admission")
        catalog.append({"candidate": candidate, "admission": record})
        tasks.append(reference(folder / "task.json"))
    manifest = {
        "schema": "sabio_plant_assay_tasks.v1",
        "source_inputs": inputs,
        "tasks": tasks,
        "catalog": catalog,
        "source_observations": len(rows),
        "retained_observations": len(retained),
        "proteins": len({r["protein"] for r in retained}),
        "publications": len({r["publication_group"] for r in retained}),
        "independent_components": split["components"],
        "descriptor_dimensions": encoder.DIMENSIONS,
        "new_admitted_tasks": len(tasks),
        "new_autonomous_models": 0,
        "external_model_calls": 0,
        "baseline_comparisons_run": False,
        "final_labels_scored": False,
        "limitations": [
            "Categorical identity/text hashes are not chemical structure fingerprints.",
            "Reference protein sequences do not establish exact assay constructs.",
            "Declared alignment separation does not establish absence of remote homology.",
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest
