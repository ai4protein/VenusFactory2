"""
Features operations: calculate_xxx style; write results to file and return rich JSON with file_info.

Success: status, file_info, content_preview, biological_metadata, execution_context.
Error: status "error", error { type, message, suggestion }, file_info null.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root on path when script is run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .calculate_physchem import calculate_physchem_from_fasta as _do_physchem
    from .calculate_rsa import calculate_rsa_from_pdb as _do_rsa
    from .calculate_sasa import calculate_sasa_from_pdb as _do_sasa
    from .calculate_secondary_structure import calculate_ss_from_pdb as _do_ss
    from .calculate_all_property import calculate_all_properties as _do_all_properties
except ImportError:
    from src.tools.predict.features.calculate_physchem import calculate_physchem_from_fasta as _do_physchem
    from src.tools.predict.features.calculate_rsa import calculate_rsa_from_pdb as _do_rsa
    from src.tools.predict.features.calculate_sasa import calculate_sasa_from_pdb as _do_sasa
    from src.tools.predict.features.calculate_secondary_structure import calculate_ss_from_pdb as _do_ss
    from src.tools.predict.features.calculate_all_property import calculate_all_properties as _do_all_properties
from src.tools.path_sanitizer import to_client_file_path


_PREVIEW_LEN = 500
_SOURCE = "Predict_Features"


def _default_agent_out_dir() -> str:
    """Default output directory for agent/tool calls when out_dir is omitted."""
    base = Path(os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs")).resolve()
    target = base / "agent" / "predict_features"
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _download_success_response(
    file_path: str,
    content_preview: Optional[str] = None,
    biological_metadata: Optional[Dict[str, Any]] = None,
    compute_time_ms: int = 0,
    source: str = _SOURCE,
) -> str:
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0
    fmt = path.suffix.lstrip(".").lower() or "json"
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": file_size,
            "format": fmt,
        },
        "content_preview": (content_preview or "")[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata or {},
        "execution_context": {"compute_time_ms": compute_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _default_output_path(input_path: str, suffix: str, out_dir: Optional[str] = None) -> str:
    stem = Path(input_path).stem
    name = f"{stem}{suffix}"
    target_dir = out_dir or _default_agent_out_dir()
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, name)


def calculate_physchem_from_fasta(
    fasta_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate physicochemical properties from FASTA; write JSON to output file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    if not fasta_file or not Path(fasta_file).exists():
        return _error_response("ValidationError", f"File not found: {fasta_file}", suggestion="Check fasta_file path.")
    try:
        result = _do_physchem(fasta_file)
        if not result:
            return _error_response("ComputeError", "Empty result; FASTA may be empty or invalid.", suggestion="Check FASTA format.")
        content = json.dumps(result, ensure_ascii=False, indent=2)
        out_path = output_file or _default_output_path(fasta_file, "_physchem.json", out_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        meta = {"sequence_id": result.get("sequence_id"), "sequence_length": result.get("sequence_length")}
        return _download_success_response(
            out_path,
            content_preview=content[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response("ComputeError", str(e), suggestion="Check fasta_file and install biopython.")


def calculate_rsa_from_pdb(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate RSA per residue from PDB; write JSON to output file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    if not pdb_file or not Path(pdb_file).exists():
        return _error_response("ValidationError", f"File not found: {pdb_file}", suggestion="Check pdb_file path.")
    try:
        result = _do_rsa(pdb_file, chain_id=chain_id)
        if not result:
            return _error_response("ComputeError", "Empty result; PDB or chain may be invalid.", suggestion="Install DSSP and check chain_id.")
        content = json.dumps(result, ensure_ascii=False, indent=2)
        out_path = output_file or _default_output_path(pdb_file, f"_rsa_chain{chain_id}.json", out_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        meta = {"chain_id": chain_id, "residue_count": len(result)}
        return _download_success_response(
            out_path,
            content_preview=content[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response("ComputeError", str(e), suggestion="Check pdb_file, chain_id, and DSSP installation.")


def calculate_sasa_from_pdb(
    pdb_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate SASA per residue from PDB; write JSON to output file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    if not pdb_file or not Path(pdb_file).exists():
        return _error_response("ValidationError", f"File not found: {pdb_file}", suggestion="Check pdb_file path.")
    try:
        result = _do_sasa(pdb_file)
        if not result:
            return _error_response("ComputeError", "Empty result; PDB may be invalid.", suggestion="Check PDB file.")
        content = json.dumps(result, ensure_ascii=False, indent=2)
        out_path = output_file or _default_output_path(pdb_file, "_sasa.json", out_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        meta = {"chains": list(result.keys()), "chain_count": len(result)}
        return _download_success_response(
            out_path,
            content_preview=content[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response("ComputeError", str(e), suggestion="Check pdb_file and biopython.")


def calculate_ss_from_pdb(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate secondary structure per residue from PDB; write JSON to output file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    if not pdb_file or not Path(pdb_file).exists():
        return _error_response("ValidationError", f"File not found: {pdb_file}", suggestion="Check pdb_file path.")
    try:
        result = _do_ss(pdb_file, chain_id=chain_id)
        if not result:
            return _error_response("ComputeError", "Empty result; PDB or chain may be invalid.", suggestion="Install DSSP and check chain_id.")
        content = json.dumps(result, ensure_ascii=False, indent=2)
        out_path = output_file or _default_output_path(pdb_file, f"_ss_chain{chain_id}.json", out_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        meta = {"chain_id": chain_id, "residue_count": len(result)}
        return _download_success_response(
            out_path,
            content_preview=content[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response("ComputeError", str(e), suggestion="Check pdb_file, chain_id, and DSSP.")


def calculate_all_properties(
    input_file: str,
    file_type: str = "auto",
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate all properties (physchem, SASA, RSA, SS) from FASTA or PDB; write JSON to output file. Returns rich JSON with file_info."""
    t0 = time.perf_counter()
    if not input_file or not Path(input_file).exists():
        return _error_response("ValidationError", f"File not found: {input_file}", suggestion="Check input_file path.")
    try:
        result = _do_all_properties(input_file, file_type=file_type, chain_id=chain_id)
        content = json.dumps(result, ensure_ascii=False, indent=2)
        out_path = output_file or _default_output_path(input_file, "_all_properties.json", out_dir)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        meta = {
            "input_file": result.get("input_file"),
            "file_type": result.get("file_type"),
            "chain_id": result.get("chain_id"),
            "analysis_timestamp": result.get("analysis_timestamp"),
        }
        return _download_success_response(
            out_path,
            content_preview=content[:_PREVIEW_LEN],
            biological_metadata=meta,
            compute_time_ms=int((time.perf_counter() - t0) * 1000),
            source=_SOURCE,
        )
    except Exception as e:
        return _error_response("ComputeError", str(e), suggestion="Check input_file, file_type ('fasta'/'pdb'/'auto'), chain_id.")


def _print_result(name: str, res: str) -> None:
    obj = json.loads(res)
    print(f"  {name}: status={obj.get('status')}")
    if obj.get("status") == "success":
        if obj.get("file_info"):
            print(f"    file_info: {obj['file_info']}")
        if obj.get("biological_metadata"):
            print(f"    biological_metadata: {obj['biological_metadata']}")
    if obj.get("status") == "error" and obj.get("error"):
        print(f"    error: {obj['error']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Features operations: calculate_xxx (physchem, rsa, sasa, ss, all_properties). --test runs all with sample FASTA/PDB.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all feature calculation operations; output under out_dir.",
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="example/database/P60002.fasta",
        help="FASTA file for physchem and all_properties (fasta). Default: example/database/P60002.fasta.",
    )
    parser.add_argument(
        "--pdb_file",
        type=str,
        default="example/database/alphafold/A0A1B0GTW7.pdb",
        help="PDB file for rsa, sasa, ss and all_properties (pdb). Default: example/database/alphafold/A0A1B0GTW7.pdb.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="example/predict/features",
        help="Output directory for result JSONs. Default: example/predict/features.",
    )
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID for PDB-based operations. Default: A.")
    args = parser.parse_args()

    if not args.test:
        print("Use --test to run all feature operations (physchem, rsa, sasa, ss, all_properties).")
        exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir
    fasta_file = args.fasta_file
    pdb_file = args.pdb_file
    chain_id = args.chain_id

    print("=== features_operations (calculate_xxx) ===")
    print(f"  fasta_file: {fasta_file}")
    print(f"  pdb_file: {pdb_file}")
    print(f"  out_dir: {out_dir}")
    print("")

    # 1. calculate_physchem_from_fasta (FASTA only)
    print("--- calculate_physchem_from_fasta ---")
    res = calculate_physchem_from_fasta(fasta_file, out_dir=out_dir)
    _print_result("calculate_physchem_from_fasta", res)
    with open(os.path.join(out_dir, "sample_physchem.json"), "w", encoding="utf-8") as f:
        f.write(res)
    print("")

    # 2. calculate_rsa_from_pdb (PDB only)
    print("--- calculate_rsa_from_pdb ---")
    res = calculate_rsa_from_pdb(pdb_file, chain_id=chain_id, out_dir=out_dir)
    _print_result("calculate_rsa_from_pdb", res)
    with open(os.path.join(out_dir, "sample_rsa.json"), "w", encoding="utf-8") as f:
        f.write(res)
    print("")

    # 3. calculate_sasa_from_pdb (PDB only)
    print("--- calculate_sasa_from_pdb ---")
    res = calculate_sasa_from_pdb(pdb_file, out_dir=out_dir)
    _print_result("calculate_sasa_from_pdb", res)
    with open(os.path.join(out_dir, "sample_sasa.json"), "w", encoding="utf-8") as f:
        f.write(res)
    print("")

    # 4. calculate_ss_from_pdb (PDB only)
    print("--- calculate_ss_from_pdb ---")
    res = calculate_ss_from_pdb(pdb_file, chain_id=chain_id, out_dir=out_dir)
    _print_result("calculate_ss_from_pdb", res)
    with open(os.path.join(out_dir, "sample_ss.json"), "w", encoding="utf-8") as f:
        f.write(res)
    print("")

    # 5. calculate_all_properties from FASTA
    print("--- calculate_all_properties (file_type=fasta) ---")
    res = calculate_all_properties(fasta_file, file_type="fasta", out_dir=out_dir)
    _print_result("calculate_all_properties(fasta)", res)
    with open(os.path.join(out_dir, "sample_all_properties_fasta.json"), "w", encoding="utf-8") as f:
        f.write(res)
    print("")

    # 6. calculate_all_properties from PDB
    print("--- calculate_all_properties (file_type=pdb) ---")
    res = calculate_all_properties(pdb_file, file_type="pdb", chain_id=chain_id, out_dir=out_dir)
    _print_result("calculate_all_properties(pdb)", res)
    with open(os.path.join(out_dir, "sample_all_properties_pdb.json"), "w", encoding="utf-8") as f:
        f.write(res)

    print("")
    print(f"Done. Output under {out_dir}")
