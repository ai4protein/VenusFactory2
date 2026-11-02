"""Source-bound, retrospective GFP and TrpB prediction tasks.

Author code is parsed as literal data, never imported or executed. All sequence
reconstruction refers to published observations, not proposed new variants.
"""

from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from .model_tasks import load_task
from .published_sequence_tasks import _write, bound

VERSION = "published_mutation_landscapes.v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
CAPS = {"training": 600, "validation": 250, "future": 250}
PAPERS = {
    "gfp2022": {
        "title": "Heterogeneity of the GFP fitness landscape and data-driven protein design",
        "doi": "10.7554/eLife.75842",
    },
    "gfp2016": {"title": "Local fitness landscape of the green fluorescent protein", "doi": "10.1038/nature17995"},
    "trpb2024": {
        "title": "A combinatorially complete epistatic fitness landscape in an enzyme active site",
        "doi": "10.1073/pnas.2400439121",
    },
}


def canonical_parent(value):
    if not isinstance(value, str) or not value.endswith("*") or not set(value[:-1]) <= AA:
        raise ValueError("Expected a canonical author parent with one terminal stop")
    if not 20 <= len(value) - 1 <= 2048:
        raise ValueError("Parent length is outside the admitted model interface")
    return value[:-1]


def gfp_parents(notebook):
    result = {}
    for cell in json.loads(Path(notebook).read_text())["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "wt_seqs[" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (
                not isinstance(target, ast.Subscript)
                or not isinstance(target.value, ast.Name)
                or target.value.id != "wt_seqs"
            ):
                continue
            gene = ast.literal_eval(target.slice)
            parent = canonical_parent(ast.literal_eval(node.value))
            if gene in result:
                raise ValueError("Repeated author parent declaration")
            result[gene] = parent
    if set(result) != {"avGFP", "amacGFP", "cgreGFP", "ppluGFP"}:
        raise ValueError("Incomplete GFP author parent declarations")
    return result


def reconstruct(parent, genotype, *, av_format=False):
    """Validate published wild-type residues before applying recorded changes."""
    if genotype == "wt":
        return parent, 0
    if not isinstance(genotype, str) or not genotype:
        raise ValueError("Missing published genotype")
    sequence, seen = list(parent), set()
    pattern = r"S([A-Z])(\d+)([A-Z])" if av_format else r"([A-Z])(\d+)([A-Z])"
    for token in genotype.split(":"):
        match = re.fullmatch(pattern, token)
        if match is None:
            raise ValueError("Unsupported published substitution notation")
        before, position, after = match.groups()
        # The archived GFP loader maps the original avGFP indices by +1 to M=0.
        position = int(position) + int(av_format)
        if position in seen or not 0 <= position < len(parent) or parent[position] != before:
            raise ValueError("Repeated position or inconsistent wild-type residue")
        if before not in AA or after not in AA or before == after:
            raise ValueError("Noncanonical or unchanged substitution")
        seen.add(position)
        sequence[position] = after
    sequence = "".join(sequence)
    return sequence, sum(a != b for a, b in zip(parent, sequence, strict=True))


def make_record(identity, sequence, mutations, target, family):
    if not sequence or not set(sequence) <= AA or not math.isfinite(target):
        raise ValueError("Invalid measured sequence or target")
    return {
        "id": identity,
        "sequence": sequence,
        "mutation_count": mutations,
        "target": target,
        "family": family,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }


def gfp_records(native_table, av_table, parents):
    records, excluded = [], collections.Counter()
    with Path(native_table).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header and header[0] == "":
            header = header[1:]
        required = {"gene", "aa_genotype_native", "replicates_mean_brightness"}
        if not required <= set(header) or len(header) != len(set(header)):
            raise ValueError("Unrecognized GFP observation table")
        for values in reader:
            if len(values) != len(header) + 1:
                raise ValueError("Author GFP row/index layout changed")
            index, row = values[0], dict(zip(header, values[1:], strict=True))
            gene, genotype = row["gene"], row["aa_genotype_native"]
            if gene not in {"amacGFP", "cgreGFP", "ppluGFP"}:
                raise ValueError("Unknown GFP family")
            if "*" in genotype:
                excluded[gene + ":stop_genotype"] += 1
                continue
            if "." in genotype:
                excluded[gene + ":indel_outside_substitution_scope"] += 1
                continue
            value = float(row["replicates_mean_brightness"])
            if not math.isfinite(value) or value <= 1:
                excluded[gene + ":author_nonpositive_log_filter"] += 1
                continue
            sequence, count = reconstruct(parents[gene], genotype)
            records.append(make_record(f"{gene}:{index}", sequence, count, math.log10(value), gene))
    with Path(av_table).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"aaMutations", "medianBrightness"} <= set(reader.fieldnames or []):
            raise ValueError("Unrecognized avGFP observation table")
        for index, row in enumerate(reader):
            genotype = row["aaMutations"]
            if "*" in genotype:
                excluded["avGFP:stop_genotype"] += 1
                continue
            # This column already stores log10 brightness. The archived loader
            # only tests log10(column)>0 in this branch; it retains the column.
            value = float(row["medianBrightness"])
            if not math.isfinite(value) or value <= 1:
                excluded["avGFP:author_filter"] += 1
                continue
            sequence, count = reconstruct(parents["avGFP"], genotype, av_format=True)
            records.append(make_record(f"avGFP:{index}", sequence, count, value, "avGFP"))
    validate_records(records)
    return records, dict(excluded)


def trpb_parents(source):
    values = {}
    for node in ast.parse(Path(source).read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in {"TM9D8S_AA", "LIBRARY_INFO_DICT"}:
            if name in values:
                raise ValueError("Repeated TrpB literal declaration")
            values[name] = ast.literal_eval(node.value)
    parent = canonical_parent(values["TM9D8S_AA"])
    libraries = values["LIBRARY_INFO_DICT"]
    if set(libraries) != set("ABCDEFGHI") | {"4-site"}:
        raise ValueError("TrpB library inventory changed")
    for name, row in libraries.items():
        sites = row["positions"]
        if set(sites) != set(range(1, (4 if name == "4-site" else 3) + 1)):
            raise ValueError("Invalid TrpB library position keys")
        if len(set(sites.values())) != len(sites):
            raise ValueError("Repeated TrpB position")
        for key, position in sites.items():
            if type(position) is not int or not 1 <= position <= len(parent) or parent[position - 1] != row["AAs"][key]:
                raise ValueError("TrpB position/parent amino-acid mismatch")
    return parent, libraries


def validate_records(records):
    identities, sequences = set(), set()
    for row in records:
        identity = row["id"]
        key = (row["family"], row["sequence"])
        if identity in identities or key in sequences:
            raise ValueError("Repeated source identity or sequence within one assay")
        identities.add(identity)
        sequences.add(key)
        if type(row["mutation_count"]) is not int or row["mutation_count"] < 0:
            raise ValueError("Invalid mutation distance")


def trpb_records(simplified, summary, parent, libraries, library):
    """Cross-check the author's reported fitness against measured replicate rows."""
    sites = libraries[library]["positions"]
    keys = sorted(sites)
    with Path(summary).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        replicas = [str(i) for i in (1, 2) if "mu_" + str(i) in fields]
        if not replicas or "AAs" not in fields:
            raise ValueError("Missing measured TrpB replicate fields")
        endpoint = "avg_mu-bg/max" if len(replicas) == 2 else "mu_1-bg/max"
        if endpoint not in fields:
            raise ValueError("Unknown author TrpB normalization")
        input_columns = sorted(field for field in fields if re.fullmatch(r"InputCount_[12]", field))
        if not input_columns or input_columns[0] != "InputCount_1":
            raise ValueError("Missing published input-count support")
        observed = {}
        for row in reader:
            if row["AAs"] in observed:
                raise ValueError("Repeated measured TrpB genotype")
            observed[row["AAs"]] = row
    records, excluded, seen = [], collections.Counter(), set()
    with Path(simplified).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"AAs", "fitness", "# Stop", *("AA" + str(i) for i in keys)}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Unrecognized TrpB observation table")
        for row in reader:
            genotype = row["AAs"]
            if genotype in seen or genotype not in observed:
                raise ValueError("Repeated or unmeasured TrpB genotype")
            seen.add(genotype)
            if len(genotype) != len(keys) or not set(genotype) <= AA | {"*"}:
                raise ValueError("Malformed TrpB amino-acid tuple")
            if any(row["AA" + str(i)] != genotype[i - 1] for i in keys):
                raise ValueError("TrpB tuple columns disagree")
            if float(row["# Stop"]) != genotype.count("*"):
                raise ValueError("TrpB stop annotation disagrees")
            if "*" in genotype:
                excluded["stop_genotype"] += 1
                continue
            original = observed[genotype]
            numbers = [row["fitness"], original[endpoint]]
            numbers += [original["mu_" + replica] for replica in replicas]
            numbers += [original["mu_" + replica + "-bg/max"] for replica in replicas]
            counts = [original[column] for column in input_columns]
            if any(value == "" for value in numbers + counts):
                excluded["missing_measured_support"] += 1
                continue
            values, read_counts = [float(x) for x in numbers], [float(x) for x in counts]
            if not all(math.isfinite(x) for x in values + read_counts) or min(read_counts) <= 0:
                excluded["missing_measured_support"] += 1
                continue
            if not math.isclose(values[0], values[1], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Reported fitness differs from measured normalized growth")
            normalized = values[2 + len(replicas) :]
            if not math.isclose(values[1], sum(normalized) / len(normalized), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Reported mean differs from normalized measured replicates")
            sequence = list(parent)
            for key in keys:
                sequence[sites[key] - 1] = genotype[key - 1]
            sequence = "".join(sequence)
            distance = sum(a != b for a, b in zip(parent, sequence, strict=True))
            family = "trpb_" + library.lower().replace("-", "")
            records.append(make_record(f"{family}:{genotype}", sequence, distance, values[0], family))
    if seen != set(observed):
        raise ValueError("Simplified and measured TrpB table inventories differ")
    validate_records(records)
    return records, {
        "excluded": dict(excluded),
        "observed_replicates": len(replicas),
        "input_count_columns": input_columns,
        "author_endpoint": endpoint,
    }


def partition_records(records):
    """Fit/accept on <=2 mutations; keep >=3 mutations for later evaluation."""
    validate_records(records)
    partitions = {phase: [] for phase in CAPS}
    counts = collections.Counter()
    for row in records:
        key = row["sequence_sha256"]
        if key != hashlib.sha256(row["sequence"].encode()).hexdigest():
            raise ValueError("Changed sequence identity")
        bucket = int(hashlib.sha256(f"{VERSION}:partition:{key}".encode()).hexdigest(), 16) % 100
        phase = "future" if row["mutation_count"] >= 3 else "training" if bucket < 75 else "validation"
        counts[phase] += 1
        rank = hashlib.sha256(f"{VERSION}:selection:{key}".encode()).hexdigest()
        partitions[phase].append((rank, row))
    data, selection = {}, {}
    for phase, rows in partitions.items():
        chosen = [row for _, row in sorted(rows, key=lambda item: item[0])[: CAPS[phase]]]
        if len(chosen) < (10 if phase == "training" else 1):
            raise ValueError("Insufficient observations for the declared extrapolation task")
        data[phase] = [{key: row[key] for key in ("id", "sequence", "target")} for row in chosen]
        selection[phase] = [{key: row[key] for key in ("id", "sequence_sha256", "mutation_count")} for row in chosen]
    return data, selection, dict(counts)


def write_mutation_task(output, *, identity, family, paper, protein, target, unit, records, references):
    subset = [row for row in records if row["family"] == family]
    data, selection, eligible = partition_records(subset)
    lengths = {len(row["sequence"]) for row in subset}
    if len(lengths) != 1:
        raise ValueError("Mutation landscape must retain its complete fixed-length construct")
    length = lengths.pop()
    folder = Path(output) / identity
    folder.mkdir()
    _write(folder / "selection.json", {"scheme": VERSION, "eligible": eligible, "selected": selection})
    assets = {}
    for phase, rows in data.items():
        _write(folder / (phase + ".json"), rows)
        assets[phase] = bound(folder / (phase + ".json"))
    definition = {
        "schema": 1,
        "id": identity,
        "stage": "fitness",
        "plugin_id": "generated_sequence.v1",
        "scope": {
            "protein": protein,
            "assay": identity,
            "target": target,
            "unit": unit,
            "minimum_sequence_length": length,
            "maximum_sequence_length": length,
        },
        "paper": PAPERS[paper],
        "references": [*references, bound(folder / "selection.json")],
        "data": assets,
        "minimum_skill": 0.0,
    }
    load_task(definition)
    _write(folder / "task.json", definition)
    return {
        "id": identity,
        "task": bound(folder / "task.json"),
        "stage": "fitness",
        "paper": PAPERS[paper]["doi"],
        "rows": {k: len(v) for k, v in data.items()},
        "eligible_rows": eligible,
    }
