"""Biological grouping audit for SKEMPI2; never random-split measurements."""
from __future__ import annotations

import time
from collections import Counter

from .skempi_source import canonical, exclusion_groups, require, sha256

ALIGNMENT = {"matrix": "BLOSUM62", "mode": "local", "open_gap_score": -10.0,
             "extend_gap_score": -0.5, "minimum_identity": 0.30,
             "minimum_coverage_each": 0.80, "minimum_chain_length": 20}
SCHEMA = "skempi_group_audit.v1"


def homology_groups(observations, *, wall_seconds=600, progress=None):
    from Bio import __version__ as biopython_version
    from Bio.Align import PairwiseAligner, substitution_matrices

    seqs = sorted({chain["observed_sequence"] for row in observations
                   if row["sequence_mapping_status"] == "verified_observed_residues"
                   for chain in row["partner_sequences"].values()})
    require(len(seqs) <= 1200 and all(len(s) <= 5000 for s in seqs), "Sequence audit exceeds reviewed limits")
    identities = {s: sha256(s.encode()) for s in seqs}
    parents = {identities[s]: identities[s] for s in seqs}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    aligner = PairwiseAligner(mode=ALIGNMENT["mode"],
                             substitution_matrix=substitution_matrices.load(ALIGNMENT["matrix"]),
                             open_gap_score=ALIGNMENT["open_gap_score"], extend_gap_score=ALIGNMENT["extend_gap_score"])
    started, compared, edges, skipped = time.monotonic(), 0, [], Counter()
    for i, first in enumerate(seqs):
        for second in seqs[i+1:]:
            require(time.monotonic() - started <= wall_seconds, "Incomplete homology audit: time budget exhausted")
            if min(len(first), len(second)) < ALIGNMENT["minimum_chain_length"]:
                skipped["short_chain"] += 1
                continue
            # With >=80% aligned residue coverage on each sequence, this pair
            # cannot qualify if the shorter sequence is <80% of the longer.
            if min(len(first), len(second)) / max(len(first), len(second)) < ALIGNMENT["minimum_coverage_each"]:
                skipped["length_ratio"] += 1
                continue
            alignments = aligner.align(first, second)
            compared += 1
            if alignments.score <= 0:
                continue
            alignment = alignments[0]
            coverage = [sum(end-start for start, end in spans) / length
                        for spans, length in zip(alignment.aligned, (len(first), len(second)))]
            identity = alignment.counts().identities / alignment.length
            if identity >= ALIGNMENT["minimum_identity"] and min(coverage) >= ALIGNMENT["minimum_coverage_each"]:
                left, right = identities[first], identities[second]
                a, b = find(left), find(right)
                parents[max(a,b)] = min(a,b)
                edges.append({"left": left, "right": right, "identity": float(identity),
                              "coverage": list(map(float, coverage))})
        if progress and (i+1) % 100 == 0:
            progress({"sequence": i+1, "total": len(seqs), "compared": compared, "edges": len(edges)})
    clusters = {identity: find(identity) for identity in parents}
    for row in observations:
        if row["sequence_mapping_status"] == "verified_observed_residues":
            row["sequence_clusters"] = [clusters[identities[c["observed_sequence"]]]
                                       for c in row["partner_sequences"].values()]
    groups = exclusion_groups(observations)
    eligible = [r for r in observations if r["sequence_mapping_status"] == "verified_observed_residues"
                and r["derived_ddg"]["value"] is not None
                and all(len(c["observed_sequence"]) >= ALIGNMENT["minimum_chain_length"]
                        for c in r["partner_sequences"].values())]
    group_counts = Counter(r["exclusion_group"] for r in eligible)
    return {"schema": SCHEMA, "complete": True, "method": ALIGNMENT, "biopython_version": biopython_version,
            "sequence_set_sha256": sha256(canonical(seqs).encode()), "sequences": len(seqs),
            "comparisons": compared, "safe_skips": dict(skipped), "edges": edges,
            "sequence_clusters": len(set(clusters.values())), "exclusion_groups": groups,
            "strict_exact_mapped_rows_with_long_enough_chains": len(eligible),
            "candidate_group_sizes": dict(sorted(group_counts.items())),
            "admitted_tasks": 0, "training_started": False,
            "interpretation": "Observed-structure sequence similarity under the declared alignment rule; not proof of absence of shared domains or remote homology. Short chains are excluded from candidate training rows."}
