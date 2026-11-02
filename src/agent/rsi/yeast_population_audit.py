"""Independent pandas/NumPy replay of the public yeast quantitative source.

This implementation does not call the production quantifier, QC filter, target
aggregation or task exporter. It audits source reconstruction, not model skill.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .model_tasks import bound_file, ordinary
from .storage import atomic_json, sha256_file

Q_FIELDS = (
    "Q.Value",
    "PG.Q.Value",
    "Protein.Q.Value",
    "Global.Q.Value",
    "Global.PG.Q.Value",
    "Lib.PG.Q.Value",
    "Lib.Q.Value",
)
MEMBER = "211020_SC_media_Diann18_MBR_report.tsv"


def _keys(values):
    return values.str.replace(r"^.*[\\/]", "", regex=True).str.split(".rawIdx", regex=False).str[0]


def _aliases(annotation, accessions):
    genes = pd.read_csv(annotation, dtype=str, keep_default_na=False)
    candidates = {}
    for _, row in genes.iterrows():
        gene = row["systematic_name"].strip()
        if gene:
            for alias in (gene, row["Gene"].strip()):
                if alias:
                    candidates.setdefault(alias, set()).add(gene)
    aliases = {name: next(iter(values)) for name, values in candidates.items() if len(values) == 1}
    refs = pd.read_csv(accessions, sep="\t", dtype=str, keep_default_na=False)
    if set(refs["Organism (ID)"]) != {"559292"}:
        raise ValueError("Identifier reference has a non-yeast organism")
    accession_genes = {}
    for _, row in refs.iterrows():
        loci = row["Gene Names (ordered locus)"].split()
        if len(loci) == 1 and aliases.get(loci[0]) == loci[0]:
            accession_genes.setdefault(row["Entry"], set()).add(loci[0])
    for name, values in accession_genes.items():
        if len(values) == 1:
            gene = next(iter(values))
            if name in aliases and aliases[name] != gene:
                raise ValueError("Ambiguous accession alias")
            aliases[name] = gene
    return aliases


def _metadata(path):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame["run"] = _keys(frame["File.Name.Short"])
    if frame["run"].duplicated().any():
        raise ValueError("Repeated run metadata")
    frame["strain"] = frame["Strain"].str.replace(r"^SACE\.", "SACE_", regex=True)
    frame["qc"] = frame["Strain"].str.startswith("QC-")
    frame["batch"] = frame["Batches"]
    if set(frame["batch"]) != {"1", "2", "3"}:
        raise ValueError("Unexpected acquisition batches")
    return frame.set_index("run")[["strain", "qc", "batch"]]


def _scan(archive, metadata, aliases, selected, *, qc, counters):
    columns = [
        "File.Name",
        "Genes",
        "Protein.Ids",
        "Proteotypic",
        "Precursor.Id",
        "Stripped.Sequence",
        "Precursor.Quantity",
        *Q_FIELDS,
    ]
    pieces = []
    process = subprocess.Popen(
        ["/usr/bin/unzip", "-p", str(archive), MEMBER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        reader = pd.read_csv(
            process.stdout,
            sep="\t",
            usecols=columns,
            dtype=str,
            keep_default_na=False,
            chunksize=100000,
        )
        for frame in reader:
            counters["rows_read"] = counters.get("rows_read", 0) + len(frame)
            frame["run"] = _keys(frame["File.Name"])
            frame = frame.join(metadata, on="run")
            frame = frame.loc[frame["qc"] == qc].copy()
            if not qc:
                frame = frame.loc[frame["strain"].isin(selected)].copy()
            if frame.empty:
                continue
            counters["metadata_rows"] = counters.get("metadata_rows", 0) + len(frame)
            gene_text = frame["Genes"].str.strip()
            accession_text = frame["Protein.Ids"].str.strip()
            gene_match, accession_match = gene_text.map(aliases), accession_text.map(aliases)
            frame["gene"] = gene_match.where(gene_text != "", accession_match)
            identity = (
                frame["gene"].notna()
                & ~gene_text.str.contains(";", regex=False)
                & ~accession_text.str.contains(";", regex=False)
                & ((gene_text == "") | (gene_match == frame["gene"]))
                & ((accession_text == "") | (accession_match == frame["gene"]))
            )
            numbers = frame[["Precursor.Quantity", "Proteotypic", *Q_FIELDS]].apply(pd.to_numeric, errors="coerce")
            quality = (
                np.isfinite(numbers).all(axis=1)
                & (numbers["Precursor.Quantity"] > 0)
                & (numbers["Proteotypic"] == 1)
                & numbers[list(Q_FIELDS)].ge(0).all(axis=1)
                & numbers[list(Q_FIELDS)].le(0.01).all(axis=1)
                & (frame["Precursor.Id"] != "")
                & (frame["Stripped.Sequence"] != "")
            )
            frame["quantity"] = numbers["Precursor.Quantity"]
            frame = frame.loc[identity & quality].copy()
            for source, target in (("Precursor.Id", "pc"), ("Stripped.Sequence", "peptide")):
                mapping = {value: hashlib.sha256(value.encode()).hexdigest() for value in frame[source].unique()}
                frame[target] = frame[source].map(mapping)
            pieces.append(frame[["run", "batch", "strain", "gene", "pc", "peptide", "quantity"]])
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise ValueError("Independent decompression failed: " + error[-500:])
    finally:
        if process.poll() is None:
            process.terminate()
        process.stdout.close()
        process.stderr.close()
        process.wait(timeout=10)
    if not pieces:
        raise ValueError("No independently qualified observations")
    return pd.concat(pieces, ignore_index=True)


def _panel(qc, metadata):
    if (qc.groupby(["pc", "run"])["quantity"].nunique() > 1).any():
        raise ValueError("Conflicting QC observations")
    identities = qc.groupby("pc")[["gene", "peptide"]].nunique()
    good = identities.index[(identities == 1).all(axis=1)]
    qc = qc.loc[qc["pc"].isin(good)].drop_duplicates(["pc", "run"])
    reference = qc.groupby(["pc", "batch"])["quantity"].agg(["count", "median", "mean", "std"])
    reference["cv"] = reference["std"] / reference["mean"]
    totals = metadata.loc[metadata["qc"]].groupby("batch").size()
    if len(totals) != 3 or totals.min() < 2:
        raise ValueError("Repeated QCs required for every batch")
    threshold = reference.index.get_level_values("batch").map((0.8 * totals).apply(np.ceil).clip(lower=2))
    reference = reference.loc[(reference["count"] >= threshold) & (reference["cv"] <= 0.3)].copy()
    complete = reference.groupby(level="pc").size()
    reference = reference.loc[reference.index.get_level_values("pc").isin(complete.index[complete == 3])]
    labels = qc.drop_duplicates("pc").set_index("pc")[["gene", "peptide"]]
    return reference.join(labels, on="pc")


def _relative_samples(samples, reference):
    values = samples.merge(
        reference.reset_index()[["pc", "batch", "median", "gene", "peptide"]],
        on=["pc", "batch", "gene", "peptide"],
        validate="many_to_one",
    )
    if (values.groupby(["pc", "run"])["quantity"].nunique() > 1).any():
        raise ValueError("Conflicting sample observations")
    values = values.drop_duplicates(["pc", "run"])
    values["target"] = np.log2(values["quantity"]) - np.log2(values["median"])
    runs = values.groupby(["gene", "strain", "run", "batch"]).agg(
        target=("target", "median"),
        distinct_peptides=("peptide", "nunique"),
        precursors=("pc", "nunique"),
    )
    runs = runs.loc[runs["distinct_peptides"] >= 2]
    observations = runs.groupby(["gene", "strain"])["target"].median()
    return runs, observations


def audit_numeric(archive, metadata_path, annotation, accessions, panel, gene_records, selected):
    metadata, aliases = _metadata(metadata_path), _aliases(annotation, accessions)
    qc_counts, sample_counts = {}, {}
    qc = _scan(archive, metadata, aliases, selected, qc=True, counters=qc_counts)
    reference = _panel(qc, metadata)
    del qc
    expected_keys = {(pc, batch) for pc, item in panel.items() for batch in item["medians"]}
    if set(reference.index) != expected_keys:
        raise ValueError("QC panel membership differs from independent reconstruction")
    differences = []
    for (pc, batch), row in reference.iterrows():
        expected = panel[pc]
        if row["gene"] != expected["gene"] or row["peptide"] != expected["peptide_sha256"]:
            raise ValueError("QC precursor identity differs")
        if int(row["count"]) != expected["qc_observations"][batch]:
            raise ValueError("QC detection counts differ")
        for actual, saved in ((row["median"], expected["medians"][batch]), (row["cv"], expected["qc_cv"][batch])):
            if not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("QC arithmetic differs")
            differences.append(abs(actual - saved))
    samples = _scan(archive, metadata, aliases, selected, qc=False, counters=sample_counts)
    runs, observations = _relative_samples(samples, reference)
    del samples
    recorded = {(gene, sample) for gene, record in gene_records.items() for sample in record["observations"]}
    if set(observations.index) != recorded:
        raise ValueError("Quantified observation membership differs")
    target_difference = 0.0
    for (gene, sample), target in observations.items():
        stored = gene_records[gene]["observations"][sample]
        if not math.isclose(target, stored["target"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Quantified target differs from raw-data replay")
        target_difference = max(target_difference, abs(target - stored["target"]))
        actual_runs = runs.loc[(gene, sample)]
        if {(row["run"], row["batch"]) for row in stored["runs"]} != set(actual_runs.index):
            raise ValueError("Quantified injection identity differs")
        for row in stored["runs"]:
            actual = actual_runs.loc[(row["run"], row["batch"])]
            if (
                row["distinct_peptides"] != int(actual["distinct_peptides"])
                or row["precursors"] != int(actual["precursors"])
                or not math.isclose(row["target"], actual["target"], rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("Run-level aggregation differs")
    return {
        "qc_rows": qc_counts,
        "sample_rows": sample_counts,
        "precursors": len(panel),
        "genes": len(gene_records),
        "observations": len(recorded),
        "max_qc_absolute_difference": max(differences, default=0.0),
        "max_target_absolute_difference": target_difference,
    }


def audit_source(quantification, tasks, output, progress=None):
    quantification, tasks, output = (Path(p).absolute() for p in (quantification, tasks, output))
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": "source_hashes", "external_model_calls": 0, "future_scored": False}
    atomic_json(output / "report.json", status)
    try:
        source = json.loads(ordinary(quantification / "source_evidence.json").read_text())
        if source["schema"] != "yeast_food_population_quantification.v1":
            raise ValueError("Unexpected source evidence")
        for ref in source["sources"].values():
            if sha256_file(ordinary(Path(ref["path"]))) != ref["sha256"]:
                raise ValueError("Public source bytes changed")
        plan = json.loads(bound_file(source["plan"]).read_text())
        protocol = json.loads(bound_file(source["protocol"]).read_text())
        if set(protocol["quality_fields"]) != set(Q_FIELDS) or protocol["maximum_q_value"] != 0.01:
            raise ValueError("Unsupported frozen quality protocol")
        panel = json.loads(bound_file(source["qc_panel"]).read_text())
        index = json.loads(ordinary(quantification / "gene_index.json").read_text())
        records = {gene: json.loads(bound_file(ref).read_text()) for gene, ref in index.items()}
        paths = {name: Path(ref["path"]) for name, ref in source["sources"].items()}
        selected = {row["id"] for row in plan["samples"]}
        status["phase"] = "independent_raw_replay"
        atomic_json(output / "report.json", status)
        if progress:
            progress(dict(status))
        numeric = audit_numeric(
            paths["report"],
            paths["runs"],
            paths["annotation"],
            paths["accessions"],
            panel,
            records,
            selected,
        )
        rna = pd.read_csv(
            paths["rna"], usecols=["ORF", *sorted(selected)], keep_default_na=False, float_precision="round_trip"
        )
        if rna["ORF"].duplicated().any():
            raise ValueError("Repeated RNA source locus")
        rna = rna.set_index("ORF")
        for gene, record in records.items():
            values = pd.to_numeric(rna.loc[gene], errors="coerce")
            values = values.loc[np.isfinite(values) & (values >= -30) & (values <= 40)]
            if set(values.index) != set(record["rna"]):
                raise ValueError("RNA coverage differs")
            if any(
                not math.isclose(value, record["rna"][sample], rel_tol=1e-12, abs_tol=1e-12)
                for sample, value in values.items()
            ):
                raise ValueError("RNA covariate differs from author data")
        manifest = json.loads(ordinary(tasks / "manifest.json").read_text())
        checked_rows = 0
        for candidate in manifest["candidates"]:
            task = json.loads(bound_file(candidate["task"]).read_text())
            gene = task["scope"]["protein"]
            if task["scope"]["grouping"]["level"] != "genetic_population":
                raise ValueError("Genetic population relabelled")
            for phase, ref in task["data"].items():
                actual = json.loads(bound_file(ref).read_text())
                samples = [
                    row
                    for row in plan["samples"]
                    if row["phase"] == phase
                    and row["id"] in records[gene]["observations"]
                    and row["id"] in records[gene]["rna"]
                ]
                if len(actual) != len(samples):
                    raise ValueError("Exported task coverage differs")
                for row, sample in zip(actual, samples, strict=True):
                    target = records[gene]["observations"][sample["id"]]["target"]
                    features = [
                        records[gene]["rna"][sample["id"]],
                        sample["od_rna"],
                        sample["od_protein"],
                        sample["ploidy"],
                    ]
                    if (
                        row["id"] != gene + "_" + sample["id"]
                        or row["protein"] != gene
                        or row["group"] != plan["study_doi"] + ":" + sample["population"]
                        or row["values"] != features
                        or row["target"] != target
                    ):
                        raise ValueError("Exported task row differs from independent source join")
                    checked_rows += 1
        status.update(
            status="passed",
            phase="completed",
            numeric=numeric,
            exported_tasks=len(manifest["candidates"]),
            exported_rows=checked_rows,
            rna_genes_verified=len(records),
            new_autonomous_models=0,
            implementation_sha256=sha256_file(Path(__file__)),
            pandas_version=pd.__version__,
            numpy_version=np.__version__,
        )
        atomic_json(output / "report.json", status)
        if progress:
            progress(dict(status))
        return status
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output / "report.json", status)
        raise
