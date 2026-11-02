#!/usr/bin/env python3
"""Prepare source-verified retrospective tasks; no model service or training."""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.rsi.published_sequence_tasks import (  # noqa: E402
    DESIGN_CLUSTERS,
    ESOL_ATTRIBUTION,
    ESOL_ENDPOINTS,
    PARTITION_VERSION,
    _write,
    audit_catalog,
    bound,
    designed_records,
    esol_records,
    group_assignments,
    write_task,
)
from agent.rsi.storage import sha256_file  # noqa: E402

EXPECTED = {
    "esol.csv": "dfb04bdd0b7de1c0b0c51940c09390e93216adaef88fd297b1707aec8aea7068",
    "w3110_AP009048.1.gb": "5f120a5decf657b7f76af6d25ee8250352daac91b5b5363f996ba7a030c530fb",
    "w3110_annotations.json": "ff5ac045324f704b8a40b14ad2a866ffb91cd91e34048cfe8bac0f524303c03f",
    "esol_README.html": "5a8a353831e59444f0fb14e960c3c34bde5f650f14b21d903b30f1661839b59b",
    "Processed_K50_dG_datasets.zip": "e1c52e14d27b18b846cd87c719eb3f3adae6f736e700962bb7da82fd031488e9",
    "tsuboyama_article.xml": "3e759e2b4dc8140b6805d021c0c50ebbb543012efba5f0c701a54c072ab934ee",
}


def fasta(path, records):
    with path.open("x") as handle:
        for row in sorted(records, key=lambda row: row["id"]):
            handle.write(f">{row['id']}\n{row['sequence']}\n")


def fasta_ids(path):
    return {line[1:].strip() for line in path.read_text().splitlines() if line.startswith(">")}


def run_command(command, log):
    with log.open("x") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Sequence clustering failed; see {log}")
    return command


def enzyme_partitions(records, folder):
    """Cluster once, then screen all earlier-phase sequences without labels.

    CD-HIT is heuristic. Passing this screen is not proof of complete homology
    independence, remote homology absence or identical biological function.
    """
    folder.mkdir()
    source = folder / "all.fasta"
    fasta(source, records)
    cluster = folder / "clusters.fasta"
    common = ["-c", "0.4", "-n", "2", "-aL", "0.8", "-aS", "0.8", "-T", "2", "-M", "2000", "-d", "0"]
    commands = [
        run_command(["/usr/bin/cd-hit", "-i", str(source), "-o", str(cluster), *common], folder / "cluster.log")
    ]
    members, groups = [], {}
    for line in (folder / "clusters.fasta.clstr").read_text().splitlines():
        if line.startswith(">"):
            members.append([])
        else:
            match = re.search(r">([^\s]+)\.\.\.", line)
            if not match or not members:
                raise ValueError("Unrecognized CD-HIT cluster record")
            members[-1].append(match[1])
    for cluster_members in members:
        for identity in cluster_members:
            if identity in groups:
                raise ValueError("Repeated sequence in clustering result")
            groups[identity] = min(cluster_members)
    assignments = group_assignments(records, groups)
    kept, dropped = [], []
    for phase in ("training", "validation", "future"):
        candidates = [row for row in records if assignments[row["id"]]["phase"] == phase]
        if not candidates:
            raise ValueError("Empty global partition")
        if phase != "training":
            references, queries = folder / f"{phase}_references.fasta", folder / f"{phase}_queries.fasta"
            fasta(references, kept)
            fasta(queries, candidates)
            output = folder / f"{phase}_unmatched.fasta"
            # Relax length ordering: longer queries must also be screened.
            command = [
                "/usr/bin/cd-hit-2d",
                "-i",
                str(references),
                "-i2",
                str(queries),
                "-o",
                str(output),
                *common,
                "-s2",
                "0",
                "-S2",
                "999999",
            ]
            commands.append(run_command(command, folder / f"{phase}_screen.log"))
            retained = fasta_ids(output)
            if not retained <= {row["id"] for row in candidates}:
                raise ValueError("Unknown sequence in similarity screen")
            dropped.extend(
                {"id": row["id"], "phase": phase, "reason": "similar_to_earlier_partition"}
                for row in candidates
                if row["id"] not in retained
            )
            candidates = [row for row in candidates if row["id"] in retained]
        kept.extend(candidates)
    audit = {
        "commands": commands,
        "identity_threshold": 0.4,
        "both_sequence_coverage": 0.8,
        "length_order_relaxed": True,
        "dropped": dropped,
        "clusters": len(members),
        "input_rows": len(records),
        "retained_rows": len(kept),
        "homology_independence_established": False,
        "interpretation": "Later partitions exclude hits detected by CD-HIT-2D against all earlier-phase sequences; heuristic screen only.",
    }
    _write(folder / "screen.json", audit)
    keep_ids = {row["id"] for row in kept}
    return kept, {key: value for key, value in assignments.items() if key in keep_ids}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.sources.absolute(), args.output.absolute()
    for name, expected in EXPECTED.items():
        path = source / name
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"Source digest mismatch: {name}")
    if output.exists():
        raise ValueError("Use a new output directory; previous evidence is immutable")
    output.mkdir(parents=True)
    _write(
        output / "protocol.json",
        {
            "schema": 1,
            "version": PARTITION_VERSION,
            "sources": [bound(source / name) for name in EXPECTED],
            "implementation": [bound(Path(__file__)), bound(ROOT / "src/agent/rsi/published_sequence_tasks.py")],
            "task_type": "retrospective_computational_prediction",
            "partition_rule": "SHA256(version:group) modulo 100: training<60, validation<80, otherwise future",
            "future_means": "Held-out retrospective labels, not newly acquired experimental observations",
            "caps": {"training": 600, "validation": 250, "future": 250},
            "cap_selection": "SHA256(version:selection:sample_id), independent of labels",
            "minimum_skill": 0.0,
            "acceptance": "MSE skill over training-mean baseline; no threshold tuning",
            "model_calls": 0,
            "published_models": 0,
        },
    )
    _write(
        output / "licenses.json",
        {
            "esol": {
                "license": "CC BY-SA 2.1 Japan",
                "attribution": ESOL_ATTRIBUTION,
                "url": "https://dbarchive.biosciencedbc.jp/data/esol/LATEST/README_e.html",
                "derived_data_license": "CC BY-SA 2.1 Japan",
            },
            "tsuboyama": {
                "license": "CC BY 4.0",
                "attribution": "Tsuboyama et al. (2023), doi:10.1038/s41586-023-06328-6",
                "url": "https://zenodo.org/records/7992926",
                "derived_data_license": "CC BY 4.0",
            },
            "scope": "Licenses apply to the respective derived data, separately from repository software.",
        },
    )
    records, exclusions = esol_records(source / "esol.csv", source / "w3110_annotations.json")
    _write(
        output / "esol_mapping.json",
        {
            "strain": "E. coli K-12 W3110",
            "accession": "AP009048.1",
            "taxonomy": 316407,
            "selection": "Type of gene product exactly e (annotated enzyme)",
            "excluded": exclusions,
            "admitted_before_similarity_screen": records,
        },
    )
    records, assignments, screen = enzyme_partitions(records, output / "esol_similarity")
    _write(output / "esol_assignments.json", assignments)
    refs = [
        bound(output / name)
        for name in ("protocol.json", "licenses.json", "esol_mapping.json", "esol_assignments.json")
    ]
    refs.append(bound(output / "esol_similarity/screen.json"))
    # The protocol binds raw sources; selected source records are also directly bound.
    refs.extend(bound(source / name) for name in ("esol.csv", "w3110_AP009048.1.gb", "w3110_annotations.json"))
    entries = []
    for identity, paper, stage, endpoint, unit in ESOL_ENDPOINTS:
        entries.append(
            write_task(
                output,
                identity=identity,
                stage=stage,
                paper=paper,
                protein="Annotated enzymes of E. coli K-12 W3110; source-mapped AP009048.1",
                endpoint=endpoint,
                unit=unit,
                records=records,
                assignments=assignments,
                references=refs,
            )
        )
    records, exclusions, parents = designed_records(source / "Processed_K50_dG_datasets.zip")
    _write(
        output / "designed_curation.json",
        {
            "selection": "Author G0 and explicit designed-scaffold topology labels; finite dG_ML only",
            "quality_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10412457/",
            "parent_scaffolds": parents,
            "excluded": exclusions,
            "interpretation": "Partial computational benchmark using published stability measurements; no sequence design or wet-lab reproduction",
        },
    )
    assignments = group_assignments(records)
    _write(output / "designed_assignments.json", assignments)
    # Preserve selected exact published observations without extracting the full ZIP.
    _write(output / "designed_observations.json", records)
    refs = [
        bound(output / name)
        for name in (
            "protocol.json",
            "licenses.json",
            "designed_curation.json",
            "designed_assignments.json",
            "designed_observations.json",
        )
    ]
    refs.append(bound(source / "tsuboyama_article.xml"))
    for topology in sorted(DESIGN_CLUSTERS):
        subset = [row for row in records if row["topology"] == topology]
        selected_ids = {row["id"] for row in subset}
        subset_assignments = {key: value for key, value in assignments.items() if key in selected_ids}
        entries.append(
            write_task(
                output,
                identity=f"tsuboyama2023_designed_{topology.lower()}",
                stage="stability",
                paper="tsuboyama2023",
                protein=f"Published G0 de novo designed scaffolds: {topology}",
                endpoint="dG_ML",
                unit="kcal/mol",
                records=subset,
                assignments=subset_assignments,
                references=refs,
            )
        )
    audit = audit_catalog(entries)
    audit.update(
        esol_screen=screen,
        designed_parent_groups=len({row["group"] for row in records}),
        designed_group_phase_counts=dict(
            collections.Counter(
                value["phase"] for value in {row["group"]: assignments[row["id"]] for row in records}.values()
            )
        ),
    )
    _write(output / "catalog.json", {"schema": 1, "entries": entries, "audit": audit})
    print(json.dumps({key: value for key, value in audit.items() if key != "esol_screen"}, indent=2))


if __name__ == "__main__":
    main()
