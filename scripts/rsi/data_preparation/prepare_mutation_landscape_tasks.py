#!/usr/bin/env python3
"""Prepare immutable computational tasks from pinned GFP and TrpB observations."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import collections
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agent.rsi.datasets.mutation_landscape_tasks import (  # noqa: E402
    CAPS,
    VERSION,
    gfp_parents,
    gfp_records,
    trpb_parents,
    trpb_records,
    write_mutation_task,
)
from agent.rsi.datasets.published_sequence_tasks import _write, audit_catalog, bound  # noqa: E402
from agent.rsi.engine.storage import sha256_file  # noqa: E402

EXPECTED = {
    "code.zip": "235e797c156406dccb90c6731eb2c0837bc0cf2c9d4c8c37e998a67c9e63b09b",
    "data.zip": "857ce7d69e47af105297872250f69c0e912c1e82c9f422e5bbd096f0ab66181b",
    "gfp_source3.xlsx": "7e1d04176867d9d2a478d0cf9f8a5298cda36bf5869cf1053f2133946e0d309d",
    "gfp_api_av.tsv": "f21bde7892e150a8f537981c35afd0335f463a923dc8682cd3b44fddc5301364",
    "gfp_api_load.ipynb": "4a0afe765703a0a0253784d19020a68f49d96fc9763ffa529e325644daeb5ca5",
    "gfp_api_license.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}


def export_gfp(workbook_path, output):
    import openpyxl

    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        rows = workbook["aminoacid_genotypes"].iter_rows(values_only=True)
        header = next(rows)
        required = {"gene", "aa_genotype_native", "replicates_mean_brightness"}
        if len(header) != len(set(header)) or not required <= set(header):
            raise ValueError("Publisher workbook columns changed")
        count, seen = 0, set()
        with output.open("x", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["", *header])
            for index, values in enumerate(rows, 2):
                row = dict(zip(header, values, strict=True))
                key = (row["gene"], row["aa_genotype_native"])
                if key in seen:
                    raise ValueError("Duplicated publisher genotype")
                seen.add(key)
                writer.writerow([index, *values])
                count += 1
        return {"sheet": "aminoacid_genotypes", "rows": count, "exporter": "openpyxl", "version": openpyxl.__version__}
    finally:
        workbook.close()


def extract_member(archive, member, target):
    """Extract one explicitly named table; never execute or unpack author code."""
    info = archive.getinfo(member)
    if info.is_dir() or not 0 < info.file_size <= 150_000_000:
        raise ValueError("Unexpected source member size")
    with archive.open(info) as source, target.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    return {"member": member, **bound(target)}


def prepare(source, output):
    source, output = Path(source).absolute(), Path(output).absolute()
    verified = []
    for name, expected in EXPECTED.items():
        path = source / name
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"Source digest mismatch: {name}")
        verified.append({"path": str(path), "sha256": expected})
    if output.exists():
        raise ValueError("Use a fresh output directory; previous evidence is immutable")
    output.mkdir(parents=True)
    _write(
        output / "protocol.json",
        {
            "schema": 1,
            "version": VERSION,
            "sources": verified,
            "implementation": [bound(Path(__file__)), bound(ROOT / "src/agent/rsi/datasets/mutation_landscape_tasks.py")],
            "task_type": "retrospective_computational_prediction",
            "partition_rule": "Mutation distance >=3: future; otherwise SHA256(version:partition:sequence_sha256) modulo 100 <75: training; remaining: validation",
            "cap_selection": "SHA256(version:selection:sequence_sha256), independent of targets",
            "caps": CAPS,
            "minimum_skill": 0.0,
            "acceptance": "Validation MSE skill relative to the training-target mean; no threshold search",
            "future_means": "Unscored published higher-order mutants, not prospective experiments",
            "gfp2022_source": "Final eLife source data 3 v3; archive table is an earlier, discordant version",
            "gfp_transform": "2022 raw fluorescence: log10 once; 2016 medianBrightness already logarithmic",
            "trpb_transform": "Retain the author's measured, normalized growth fitness, including negative values; omit imputed table",
            "gfp_scope": "Canonical, fixed-length substitution-only genotypes; explicitly exclude stop and indel records",
            "trpb_scope": "Published T. maritima TrpB Tm9D8* constructs with recorded affinity tag retained",
            "homology_independence_established": False,
            "new_autonomous_models": 0,
            "physical_experiments_reproduced": False,
        },
    )
    _write(
        output / "licenses.json",
        {
            "gfp2022": {
                "license": "CC BY 4.0",
                "url": "https://elifesciences.org/articles/75842",
                "attribution": "Somermeyer et al. (2022), doi:10.7554/eLife.75842",
            },
            "gfp_archive": {
                "license": "GPL-3.0 repository license; original 2016 data attributed separately",
                "url": "https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks",
                "commit": "4bb2791013567442a8ea4d7735ca79311502bdee",
                "attribution": "Sarkisyan et al. (2016), doi:10.1038/nature17995; preserved in Somermeyer et al. archive",
            },
            "trpb2024": {
                "license": "CC0-1.0",
                "url": "https://data.caltech.edu/records/h5rah-5z170",
                "attribution": "Johnston et al. (2024), doi:10.1073/pnas.2400439121; dataset doi:10.22002/h5rah-5z170",
            },
            "scope": "Source-specific terms and attribution retained separately from project software.",
        },
    )
    tables = output / "sources"
    tables.mkdir()
    for name in ("gfp_api_av.tsv", "gfp_api_load.ipynb", "gfp_api_license.txt"):
        shutil.copyfile(source / name, tables / name)
        if sha256_file(tables / name) != EXPECTED[name]:
            raise ValueError("Source changed during preparation")
    export = export_gfp(source / "gfp_source3.xlsx", tables / "gfp_final_source3.tsv")
    parents = gfp_parents(tables / "gfp_api_load.ipynb")
    records, exclusions = gfp_records(tables / "gfp_final_source3.tsv", tables / "gfp_api_av.tsv", parents)
    _write(
        output / "gfp_curation.json",
        {
            "export": export,
            "parent_lengths": {key: len(value) for key, value in parents.items()},
            "admitted": dict(collections.Counter(row["family"] for row in records)),
            "excluded": exclusions,
            "scope": "Reconstructed published observations; partial computational prediction task, not full author training reproduction",
        },
    )
    common = [bound(output / name) for name in ("protocol.json", "licenses.json")]
    gfp_refs = (
        common
        + [bound(output / "gfp_curation.json")]
        + [
            bound(tables / name)
            for name in ("gfp_final_source3.tsv", "gfp_api_av.tsv", "gfp_api_load.ipynb", "gfp_api_license.txt")
        ]
    )
    entries = []
    for family in ("avGFP", "amacGFP", "cgreGFP", "ppluGFP"):
        entries.append(
            write_mutation_task(
                output,
                identity=f"gfp_{family.lower()}_mutation_extrapolation",
                family=family,
                paper="gfp2016" if family == "avGFP" else "gfp2022",
                protein="Published " + ("ppluGFP2" if family == "ppluGFP" else family),
                target="log10 fluorescence brightness",
                unit="log10 fluorescence units",
                records=records,
                references=gfp_refs,
            )
        )
    members = []
    with zipfile.ZipFile(source / "code.zip") as archive:
        members.append(
            extract_member(
                archive, "code/scripts/growth_analysis/growth_analysis/globals.py", tables / "trpb_globals.py"
            )
        )
    parent, libraries = trpb_parents(tables / "trpb_globals.py")
    reviews = {}
    # These ten paths are the only observations opened from the larger archive.
    with zipfile.ZipFile(source / "data.zip") as archive:
        for library in [*"ABCDEFGHI", "4-site"]:
            prefix = (
                "data/figure_data/4-site_merged_replicates/20230827/four-site"
                if library == "4-site"
                else f"data/figure_data/3-site_merged_replicates/Lib{library}/20230926/Lib{library}"
            )
            refs = []
            for kind in ("simplified", "summary"):
                member = extract_member(
                    archive, prefix + "_" + kind + "_AA_data.csv", tables / f"trpb_{library}_{kind}.csv"
                )
                members.append(member)
                refs.append({key: member[key] for key in ("path", "sha256")})
            rows, review = trpb_records(
                tables / f"trpb_{library}_simplified.csv",
                tables / f"trpb_{library}_summary.csv",
                parent,
                libraries,
                library,
            )
            reviews[library] = {"admitted": len(rows), **review}
            review_path = output / f"trpb_{library}_curation.json"
            _write(
                review_path,
                {
                    "library": library,
                    "parent_length_with_recorded_tag": len(parent),
                    "source_members": members[-2:],
                    **reviews[library],
                },
            )
            family = "trpb_" + library.lower().replace("-", "")
            entries.append(
                write_mutation_task(
                    output,
                    identity=family + "_mutation_extrapolation",
                    family=family,
                    paper="trpb2024",
                    protein="Thermotoga maritima TrpB Tm9D8*; author library " + library,
                    target="author normalized growth fitness",
                    unit="dimensionless growth proxy",
                    records=rows,
                    references=common + [bound(review_path), bound(tables / "trpb_globals.py"), *refs],
                )
            )
    _write(output / "archive_members.json", members)
    audit = audit_catalog(entries)
    audit.update(trpb_libraries=reviews, future_metrics_evaluated=False, external_model_calls=0)
    _write(output / "catalog.json", {"schema": 1, "entries": entries, "audit": audit})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output), indent=2))


if __name__ == "__main__":
    main()
