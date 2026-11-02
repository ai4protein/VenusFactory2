"""
AlphaFold local analysis: pLDDT confidence summary and PAE domain decomposition.

Both functions read JSON files previously downloaded via
`download_alphafold_metadata_by_uniprot_id` (pLDDT global fractions) or via the
AlphaFold predicted-aligned-error endpoint (PAE matrix). No network I/O.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/alphafold_database_fetch_and_analyze/scripts/analyze_plddt.py
- skills/alphafold_database_fetch_and_analyze/scripts/analyze_pae.py
"""

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_PREVIEW_LEN = 500
_SOURCE = "AlphaFold DB (local analysis)"

CONFIDENT_THRESHOLD = 0.7
MODERATE_THRESHOLD = 0.4
NOTABLE_DISORDER_THRESHOLD = 0.15
MIXED_DISORDER_THRESHOLD = 0.3
MOSTLY_DISORDERED_THRESHOLD = 0.5


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _success_response(
    content: str,
    biological_metadata: Dict[str, Any],
    elapsed_ms: int,
) -> str:
    out: Dict[str, Any] = {
        "status": "success",
        "content": content,
        "content_preview": content[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"analysis_time_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


def _plddt_conclusion(frac_conf: float, frac_vhigh: float, frac_vlow: float) -> str:
    conf_total = frac_conf + frac_vhigh
    if conf_total >= CONFIDENT_THRESHOLD:
        if frac_vlow > NOTABLE_DISORDER_THRESHOLD:
            return "Protein is mostly confidently predicted, but contains notable disordered regions."
        return "Protein is confidently predicted and likely fully ordered/structured."
    if conf_total >= MODERATE_THRESHOLD:
        if frac_vlow >= MIXED_DISORDER_THRESHOLD:
            return (
                "Protein has a mixture of confidently predicted structured domains "
                "and significant intrinsically disordered regions."
            )
        return "Protein has moderate prediction confidence. Certain regions might be flexible or poorly predicted."
    if frac_vlow >= MOSTLY_DISORDERED_THRESHOLD:
        return "Protein is mostly poorly predicted, likely being highly intrinsically disordered."
    return "Protein prediction is of low confidence overall."


def analyze_alphafold_plddt_by_metadata_file(metadata_path: str) -> str:
    """Analyze pLDDT confidence fractions from a saved AlphaFold metadata JSON.

    The metadata JSON is expected to be either a single entry dict or a list
    whose first item is that dict (matches the AlphaFold prediction API shape).
    Returns the standard rich JSON envelope.
    """
    t0 = time.perf_counter()
    path = str(metadata_path or "").strip()
    if not path:
        return _error_response("ValidationError", "empty metadata_path", suggestion="Pass the path to an AlphaFold metadata JSON file.")
    if not os.path.exists(path):
        return _error_response("NotFound", f"metadata file not found: {path}", suggestion="Run download_alphafold_metadata_by_uniprot_id first.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return _error_response("ParseError", f"failed to read metadata JSON: {e}", suggestion="Check that the file is valid AlphaFold metadata JSON.")

    entry = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(entry, dict):
        return _error_response("ParseError", "metadata is not a JSON object", suggestion="Expected AlphaFold metadata JSON.")

    accession = entry.get("uniprotAccession", "Unknown")
    global_plddt = float(entry.get("globalMetricValue", 0.0) or 0.0)
    frac_vlow = float(entry.get("fractionPlddtVeryLow", 0.0) or 0.0)
    frac_low = float(entry.get("fractionPlddtLow", 0.0) or 0.0)
    frac_conf = float(entry.get("fractionPlddtConfident", 0.0) or 0.0)
    frac_vhigh = float(entry.get("fractionPlddtVeryHigh", 0.0) or 0.0)
    conclusion = _plddt_conclusion(frac_conf, frac_vhigh, frac_vlow)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    content = (
        f"AlphaFold pLDDT report for {accession}\n"
        f"  global pLDDT      : {global_plddt:.2f}\n"
        f"  fraction very low : {frac_vlow:.3f} ({frac_vlow*100:.1f}%)\n"
        f"  fraction low      : {frac_low:.3f} ({frac_low*100:.1f}%)\n"
        f"  fraction confident: {frac_conf:.3f} ({frac_conf*100:.1f}%)\n"
        f"  fraction very high: {frac_vhigh:.3f} ({frac_vhigh*100:.1f}%)\n"
        f"  conclusion        : {conclusion}"
    )
    meta = {
        "uniprot_id": accession,
        "global_plddt": round(global_plddt, 4),
        "fractions": {
            "very_low": round(frac_vlow, 4),
            "low": round(frac_low, 4),
            "confident": round(frac_conf, 4),
            "very_high": round(frac_vhigh, 4),
        },
        "conclusion": conclusion,
        "source_file": os.path.basename(path),
    }
    return _success_response(content, meta, elapsed_ms)


def _find_sub_domains(pae: List[List[float]], distance_cutoff: float, min_domain_size: int) -> List[List[int]]:
    n_res = len(pae)
    domains: List[List[int]] = []
    current: List[int] = []
    for i in range(n_res):
        if not current:
            current.append(i)
            continue
        window_size = min(20, len(current))
        recent = current[-window_size:]
        pae_sum = sum(pae[r][i] + pae[i][r] for r in recent)
        avg = pae_sum / (2.0 * window_size)
        if avg < distance_cutoff:
            current.append(i)
        else:
            if len(current) >= min_domain_size:
                domains.append(current)
            current = [i]
    if len(current) >= min_domain_size:
        domains.append(current)
    return [[d[0] + 1, d[-1] + 1] for d in domains]


def _merge_global_domains(boundaries: List[List[int]], pae: List[List[float]], merge_cutoff: float = 15.0) -> List[List[int]]:
    if not boundaries:
        return []
    if len(boundaries) == 1:
        merged = [list(b) for b in boundaries]
    else:
        merged = [list(boundaries[0])]
        for i in range(1, len(boundaries)):
            prev_end = merged[-1][1] - 1
            curr_start = boundaries[i][0] - 1
            lookback = max(merged[-1][0] - 1, prev_end - 30)
            lookfwd = min(boundaries[i][1] - 1, curr_start + 30)
            pae_sum = 0.0
            n_pairs = 0
            for r1 in range(lookback, prev_end + 1):
                for r2 in range(curr_start, lookfwd + 1):
                    pae_sum += pae[r1][r2] + pae[r2][r1]
                    n_pairs += 2
            if n_pairs > 0 and (pae_sum / n_pairs) < merge_cutoff:
                merged[-1][1] = boundaries[i][1]
            else:
                merged.append(list(boundaries[i]))
    return [d for d in merged if (d[1] - d[0] + 1) > 50]


def analyze_alphafold_pae_by_pae_file(
    pae_path: str,
    distance_cutoff: float = 7.0,
    min_domain_size: int = 40,
) -> str:
    """Identify structural domains from an AlphaFold PAE JSON file.

    Accepts either AFDB classical format (`distance` key) or the newer
    `predicted_aligned_error` key. Returns the standard rich JSON envelope; the
    `biological_metadata` field contains per-domain boundaries and aggregate
    PAE statistics.
    """
    t0 = time.perf_counter()
    path = str(pae_path or "").strip()
    if not path:
        return _error_response("ValidationError", "empty pae_path", suggestion="Pass the path to an AlphaFold PAE JSON file.")
    if not os.path.exists(path):
        return _error_response("NotFound", f"PAE file not found: {path}", suggestion="Download the PAE JSON from the AlphaFold file endpoint first.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return _error_response("ParseError", f"failed to read PAE JSON: {e}", suggestion="Check that the file is a valid AlphaFold PAE JSON.")

    data = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(data, dict):
        return _error_response("ParseError", "PAE JSON is not an object", suggestion="Expected AlphaFold PAE JSON.")

    if "predicted_aligned_error" in data:
        pae = data["predicted_aligned_error"]
    elif "distance" in data:
        pae = data["distance"]
    else:
        return _error_response("ParseError", f"PAE matrix key not found; available keys: {list(data.keys())}", suggestion="Expected `predicted_aligned_error` or `distance` key.")

    if not pae or not isinstance(pae, list) or not isinstance(pae[0], list):
        return _error_response("ParseError", "PAE matrix is empty or malformed", suggestion="Check the source JSON.")

    flat = list(itertools.chain.from_iterable(pae))
    if not flat:
        return _error_response("ParseError", "PAE matrix is empty", suggestion="Check the source JSON.")

    mean_pae = sum(flat) / len(flat)
    max_pae = max(flat)
    min_pae = min(flat)
    confident_pct = sum(1 for p in flat if p < 5.0) / len(flat) * 100.0

    sub_domains = _find_sub_domains(pae, distance_cutoff=distance_cutoff, min_domain_size=min_domain_size)
    global_domains = _merge_global_domains(sub_domains, pae, merge_cutoff=15.0)

    if len(global_domains) == 1:
        conclusion = "The protein consists of a single well-folded, rigid composite domain."
    elif len(global_domains) > 1:
        conclusion = f"The protein has {len(global_domains)} independently positioned global domains separated by truly flexible joints."
    else:
        conclusion = "The protein is likely entirely disordered or lacks rigid tertiary structure."

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    shape = f"{len(pae)}x{len(pae[0])}"
    domain_lines = "\n".join(
        f"    domain {i}: residues {s}-{e} (length {e - s + 1})"
        for i, (s, e) in enumerate(global_domains, 1)
    ) or "    (no rigid domains > 50 AAs detected)"
    content = (
        f"AlphaFold PAE report for {os.path.basename(path)}\n"
        f"  matrix shape           : {shape}\n"
        f"  mean PAE               : {mean_pae:.2f} Å\n"
        f"  max PAE                : {max_pae:.2f} Å\n"
        f"  min PAE                : {min_pae:.2f} Å\n"
        f"  confident pairs (<5 Å) : {confident_pct:.1f}%\n"
        f"  global domains         : {len(global_domains)}\n"
        f"{domain_lines}\n"
        f"  conclusion             : {conclusion}"
    )
    meta = {
        "source_file": os.path.basename(path),
        "matrix_shape": shape,
        "mean_pae": round(mean_pae, 2),
        "max_pae": round(max_pae, 2),
        "min_pae": round(min_pae, 2),
        "confident_pairs_pct": round(confident_pct, 1),
        "domains": [{"start": s, "end": e, "length": e - s + 1} for s, e in global_domains],
        "conclusion": conclusion,
        "params": {"distance_cutoff": distance_cutoff, "min_domain_size": min_domain_size},
    }
    return _success_response(content, meta, elapsed_ms)


__all__ = [
    "analyze_alphafold_plddt_by_metadata_file",
    "analyze_alphafold_pae_by_pae_file",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaFold pLDDT/PAE local analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plddt = sub.add_parser("plddt", help="Analyze pLDDT fractions from metadata JSON")
    p_plddt.add_argument("metadata_path")
    p_pae = sub.add_parser("pae", help="Analyze PAE matrix and detect domains")
    p_pae.add_argument("pae_path")
    p_pae.add_argument("--distance-cutoff", type=float, default=7.0)
    p_pae.add_argument("--min-domain-size", type=int, default=40)
    args = parser.parse_args()

    if args.cmd == "plddt":
        print(analyze_alphafold_plddt_by_metadata_file(args.metadata_path))
    else:
        print(analyze_alphafold_pae_by_pae_file(args.pae_path, distance_cutoff=args.distance_cutoff, min_domain_size=args.min_domain_size))
