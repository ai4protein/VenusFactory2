"""Stream raw yeast DIA-NN quantities through a frozen, QC-relative endpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from agent.rsi.datasets.model_tasks import ordinary
from .storage import atomic_json, sha256_file
from agent.rsi.datasets.yeast_population import canonical_sample, target_protocol, validate_report_header

RAW_SHA256 = "5f4e0cc1d5d09e9d7bdf7510d088d7daa5dc18e01d5cc3b58c982940f50dbd19"
MEMBER = "211020_SC_media_Diann18_MBR_report.tsv"
QUALITY_FIELDS = tuple(target_protocol()["quality_fields"])


def run_key(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1].split(".rawIdx", 1)[0]


def run_metadata(path):
    runs = {}
    with ordinary(Path(path).absolute()).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = run_key(row["File.Name.Short"])
            if not key or key in runs:
                raise ValueError("Empty or duplicate acquisition identity")
            runs[key] = {
                "sample": canonical_sample(row["Strain"]),
                "batch": row["Batches"],
                "qc": row["Strain"].startswith("QC-"),
            }
    if {r["batch"] for r in runs.values()} != {"1", "2", "3"}:
        raise ValueError("Expected the three documented acquisition batches")
    return runs


def gene_aliases(path):
    targets = defaultdict(set)
    with ordinary(Path(path).absolute()).open() as handle:
        for row in csv.DictReader(handle):
            gene = row["systematic_name"].strip()
            if not gene:
                continue
            for alias in (gene, row["Gene"].strip()):
                if alias:
                    targets[alias].add(gene)
    return {alias: next(iter(genes)) for alias, genes in targets.items() if len(genes) == 1}


def with_accessions(path, aliases):
    """Bind current reference accessions to exactly one author-annotated ORF."""
    targets = defaultdict(set)
    with ordinary(Path(path).absolute()).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["Organism (ID)"] != "559292":
                raise ValueError("Reference accession belongs to another organism")
            loci = row["Gene Names (ordered locus)"].split()
            if len(loci) == 1 and aliases.get(loci[0]) == loci[0]:
                targets[row["Entry"]].add(loci[0])
    result = dict(aliases)
    for accession, genes in targets.items():
        if len(genes) == 1:
            gene = next(iter(genes))
            if accession in result and result[accession] != gene:
                raise ValueError("Accession collides with another gene alias")
            result[accession] = gene
    return result


def resolve_gene(row, aliases):
    identifiers = [row.get(field, "").strip() for field in ("Genes", "Protein.Ids")]
    identifiers = [value for value in identifiers if value]
    if not identifiers or any(";" in value for value in identifiers):
        return None
    genes = {aliases.get(value) for value in identifiers}
    return next(iter(genes)) if len(genes) == 1 and None not in genes else None


def stream_report(archive, runs, *, qc, selected, counters):
    """Select by metadata before parsing quantities; verify decompressor exit."""
    proc = subprocess.Popen(
        ["/usr/bin/unzip", "-p", str(ordinary(Path(archive).absolute())), MEMBER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8-sig",
    )
    try:
        reader = csv.reader(proc.stdout, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError("Empty or invalid raw report archive")
        required = validate_report_header(header) + ["Protein.Ids"]
        if "Protein.Ids" not in header:
            raise ValueError("Raw accession field is required")
        indices = {name: header.index(name) for name in required}
        for values in reader:
            counters["source_rows"] += 1
            if len(values) != len(header):
                raise ValueError("Ragged raw DIA-NN report row")
            key = run_key(values[indices["File.Name"]])
            metadata = runs.get(key)
            if metadata is None:
                counters["unmapped_run_rows"] += 1
                continue
            if metadata["qc"] != qc or (not qc and metadata["sample"] not in selected):
                continue
            counters["selected_metadata_rows"] += 1
            yield key, {name: values[index] for name, index in indices.items()}
        error = proc.stderr.read()
        if proc.wait() != 0:
            raise ValueError("Raw archive decompression failed: " + error[-500:])
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.stdout.close()
        proc.stderr.close()
        proc.wait(timeout=10)


def qualified(row, aliases):
    """Raw positive signal, explicit Q thresholds and one unambiguous gene."""
    gene = resolve_gene(row, aliases)
    if gene is None:
        return None
    try:
        quantity = float(row["Precursor.Quantity"])
        q_values = [float(row[field]) for field in QUALITY_FIELDS]
        proteotypic = float(row["Proteotypic"])
    except (ValueError, TypeError):
        return None
    if (
        proteotypic != 1
        or not math.isfinite(quantity)
        or quantity <= 0
        or any(not math.isfinite(q) or not 0 <= q <= 0.01 for q in q_values)
        or not row["Precursor.Id"]
        or not row["Stripped.Sequence"]
    ):
        return None
    precursor = hashlib.sha256(row["Precursor.Id"].encode()).hexdigest()
    peptide = hashlib.sha256(row["Stripped.Sequence"].encode()).hexdigest()
    return precursor, gene, peptide, quantity


def qc_panel(records, runs, aliases, counters):
    totals = Counter(r["batch"] for r in runs.values() if r["qc"])
    if len(totals) != 3 or min(totals.values()) < 2:
        raise ValueError("Repeated QC injections required in every acquisition batch")
    signals, identities, ambiguous = defaultdict(dict), {}, set()
    for run, row in records:
        if not runs[run]["qc"]:
            raise ValueError("Only QC measurements may define the precursor panel")
        item = qualified(row, aliases)
        if item is None:
            counters["unresolved_gene_rows" if resolve_gene(row, aliases) is None else "quality_rejected"] += 1
            continue
        precursor, gene, peptide, quantity = item
        identity = (gene, peptide)
        if precursor in identities and identities[precursor] != identity:
            ambiguous.add(precursor)
        identities[precursor] = identity
        if run in signals[precursor]:
            if signals[precursor][run] != quantity:
                raise ValueError("Conflicting quantities for one QC precursor/injection")
            counters["identical_duplicate_rows"] += 1
        signals[precursor][run] = quantity
    panel = {}
    for precursor, values in signals.items():
        if precursor in ambiguous:
            counters["ambiguous_precursors"] += 1
            continue
        medians, coverage, cvs = {}, {}, {}
        for batch, total in totals.items():
            numbers = [q for run, q in values.items() if runs[run]["batch"] == batch]
            if len(numbers) < max(2, math.ceil(0.8 * total)):
                break
            cv = statistics.stdev(numbers) / statistics.mean(numbers)
            if cv > 0.3:
                break
            medians[batch] = statistics.median(numbers)
            coverage[batch] = len(numbers)
            cvs[batch] = cv
        if len(medians) == len(totals):
            gene, peptide = identities[precursor]
            panel[precursor] = {
                "gene": gene,
                "peptide_sha256": peptide,
                "medians": medians,
                "qc_observations": coverage,
                "qc_cv": cvs,
            }
    counters["retained_precursors"] = len(panel)
    counters["qc_injections"] = sum(totals.values())
    return panel


def quantify_samples(records, runs, aliases, panel, selected, counters):
    """Use the fixed panel; one quantified observation per strain and gene."""
    values = defaultdict(dict)
    for run, row in records:
        metadata = runs[run]
        if metadata["qc"] or metadata["sample"] not in selected:
            raise ValueError("Nonselected sample reached quantification")
        item = qualified(row, aliases)
        if item is None:
            counters["unresolved_gene_rows" if resolve_gene(row, aliases) is None else "quality_rejected"] += 1
            continue
        precursor, gene, peptide, quantity = item
        reference = panel.get(precursor)
        if reference is None:
            counters["outside_qc_panel"] += 1
            continue
        if (gene, peptide) != (reference["gene"], reference["peptide_sha256"]):
            counters["inconsistent_precursor_identity"] += 1
            continue
        target = math.log2(quantity) - math.log2(reference["medians"][metadata["batch"]])
        entry = values[(run, gene)]
        if precursor in entry and entry[precursor] != (peptide, target):
            raise ValueError("Conflicting quantities for one sample precursor/injection")
        entry[precursor] = (peptide, target)
    observations = defaultdict(list)
    for (run, gene), precursors in sorted(values.items()):
        distinct = len({peptide for peptide, _ in precursors.values()})
        if distinct < 2:
            counters["protein_runs_below_two_peptides"] += 1
            continue
        observations[(gene, runs[run]["sample"])].append(
            {
                "run": run,
                "batch": runs[run]["batch"],
                "target": statistics.median(target for _, target in precursors.values()),
                "distinct_peptides": distinct,
                "precursors": len(precursors),
            }
        )
    by_gene = defaultdict(dict)
    for (gene, sample), technical in sorted(observations.items()):
        by_gene[gene][sample] = {
            "target": statistics.median(v["target"] for v in technical),
            "runs": technical,
        }
    counters["quantified_genes"] = len(by_gene)
    counters["quantified_observations"] = len(observations)
    return dict(by_gene)


def rna_covariates(path, genes, selected):
    result, seen = {}, set()
    with ordinary(Path(path).absolute()).open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != len(set(header)) or header[0] != "ORF":
            raise ValueError("Unique author RNA sample columns and ORF index required")
        indices = {sample: header.index(sample) for sample in selected}
        for fields in reader:
            if len(fields) != len(header):
                raise ValueError("Ragged author RNA row")
            gene = fields[0]
            if gene not in genes:
                continue
            if gene in seen:
                raise ValueError("Repeated gene in author RNA table")
            seen.add(gene)
            values = {}
            for sample, index in indices.items():
                try:
                    value = float(fields[index])
                except ValueError:
                    continue
                if math.isfinite(value) and -30 <= value <= 40:
                    values[sample] = value
            result[gene] = values
    return result


def quantify_source(raw_root, frozen_root, output, progress=None):
    """Reconstruct all eligible numeric observations; do not fit or score models."""
    raw_root, frozen_root, output = (Path(p).absolute() for p in (raw_root, frozen_root, output))
    plan_path, protocol_path = frozen_root / "population_plan.json", frozen_root / "target_protocol.json"
    plan = json.loads(ordinary(plan_path).read_text())
    protocol = json.loads(ordinary(protocol_path).read_text())
    if (
        sha256_file(plan_path) != "9570ddab508384cff780dc6dcca1c37e4db1e08f049ee6acc798a04751009ed8"
        or sha256_file(protocol_path) != "b1380debb21da5a09f8f05b3f523cb7ba3bbfa1955c2aa567db6eea73c810b97"
        or protocol != target_protocol()
    ):
        raise ValueError("Frozen metadata or target protocol differs")
    inputs = {
        "report": (raw_root / MEMBER.replace(".tsv", ".zip"), RAW_SHA256),
        "accessions": (
            raw_root / "uniprot_reference_identifiers_20261025.tsv",
            "c9315c99e2a077042ef998bd1a86b4634cab14b2989522ede305c6adad9f6718",
        ),
        "runs": (
            raw_root / "1kCollection_SC_Media_metadata.txt",
            "cda3c019882bd388d80dde2454b214ddb3f84b6b55324bc1e43dc883e970f7de",
        ),
        "annotation": (
            raw_root / "author_data/data_upload/CompleteGeneAnnot_strand.csv",
            "93ba8ffe486d50081688291a88289235090203357b628b2342654b650f85f5e9",
        ),
        "rna": (
            raw_root / "author_data/data_upload/RNAseq_969_log2tpm_matrix.tab",
            "83fe0fea45a69682560bb718b13c2fcd485af474d46168f467f66b8f4debaef6",
        ),
    }
    for name, (path, expected) in inputs.items():
        if sha256_file(ordinary(path)) != expected:
            raise ValueError("Public source changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": "qc_panel", "new_trained_models": 0, "external_model_calls": 0}

    def checkpoint(phase):
        status["phase"] = phase
        atomic_json(output / "status.json", status)
        if progress:
            progress(dict(status))

    checkpoint("qc_panel")
    try:
        runs = run_metadata(inputs["runs"][0])
        aliases = with_accessions(inputs["accessions"][0], gene_aliases(inputs["annotation"][0]))
        selected = {s["id"] for s in plan["samples"]}
        qc_counts, sample_counts = Counter(), Counter()
        with closing(stream_report(inputs["report"][0], runs, qc=True, selected=selected, counters=qc_counts)) as rows:
            panel = qc_panel(rows, runs, aliases, qc_counts)
        status["qc_counts"] = dict(qc_counts)
        if not panel:
            raise ValueError("No precursors pass the frozen QC rules")
        atomic_json(output / "qc_panel.json", panel)
        status["qc_counts"] = dict(qc_counts)
        checkpoint("sample_quantification")
        with closing(
            stream_report(inputs["report"][0], runs, qc=False, selected=selected, counters=sample_counts)
        ) as rows:
            quantities = quantify_samples(rows, runs, aliases, panel, selected, sample_counts)
        status["sample_counts"] = dict(sample_counts)
        checkpoint("rna_covariates")
        rna = rna_covariates(inputs["rna"][0], set(quantities), selected)
        genes = {}
        for gene, observations in sorted(quantities.items()):
            if not gene or any(c in gene for c in "/\\") or gene in {".", ".."}:
                raise ValueError("Unsafe gene identifier")
            target = output / "genes" / (gene + ".json")
            atomic_json(target, {"gene": gene, "observations": observations, "rna": rna.get(gene, {})})
            genes[gene] = {"path": str(target), "sha256": sha256_file(target)}
        evidence = {
            "schema": "yeast_food_population_quantification.v1",
            "study_doi": plan["study_doi"],
            "sources": {name: {"path": str(p), "sha256": expected} for name, (p, expected) in inputs.items()},
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "qc_panel": {"path": str(output / "qc_panel.json"), "sha256": sha256_file(output / "qc_panel.json")},
            "implementation": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "qc_counts": dict(qc_counts),
            "sample_counts": dict(sample_counts),
            "future_scored": False,
            "model_calls": 0,
        }
        atomic_json(output / "source_evidence.json", evidence)
        atomic_json(output / "gene_index.json", genes)
        status.update(status="completed", genes=len(genes), genes_with_rna=len(set(rna) & set(genes)))
        checkpoint("completed")
        return status
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output / "status.json", status)
        raise
