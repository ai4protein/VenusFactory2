"""Sequence similarity search operations: MMseqs2 (ColabFold) and BLAST (EBI).

Both return the standard VenusFactory rich JSON envelope. Heavy payload (hits
list) goes to a JSON file on disk; the response carries a `content_preview`
with the top 10 hits as a markdown table.

Adapted from google-deepmind/science-skills (Apache-2.0).
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.path_sanitizer import to_client_file_path

try:
    from .mmseqs2_colabfold import search_mmseqs2_colabfold
    from .ebi_blast import search_ebi_blast
except ImportError:
    from src.tools.database.seq_search.mmseqs2_colabfold import search_mmseqs2_colabfold
    from src.tools.database.seq_search.ebi_blast import search_ebi_blast


_PREVIEW_LEN = 1500


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _hits_to_markdown_mmseqs(hits: List[Dict[str, Any]], limit: int = 10) -> str:
    if not hits:
        return "_(no homologues found)_"
    lines = [
        f"### Top {min(len(hits), limit)} Sequence Homologues (MMseqs2)",
        "| Target ID | Q-Cov | E-value | Seq Identity | Aln Length |",
        "|---|---|---|---|---|",
    ]
    for h in hits[:limit]:
        lines.append(
            f"| {h.get('target_id', '?')} | {h.get('q_cov', 0):.1f}% | "
            f"{h.get('e_value', float('nan')):.2e} | "
            f"{h.get('identity', 0) * 100:.1f}% | {h.get('aln_len', 0)} |"
        )
    return "\n".join(lines)


def _hits_to_markdown_blast(hits: List[Dict[str, Any]], limit: int = 10) -> str:
    if not hits:
        return "_(no homologues found)_"
    lines = [
        f"### Top {min(len(hits), limit)} Sequence Homologues (EBI BLAST)",
        "| Target ID | Q-Cov | E-value | Seq Identity | Aln Length |",
        "|---|---|---|---|---|",
    ]
    for h in hits[:limit]:
        desc = (h.get("description") or "")[:60]
        target = f"{h.get('target_id', '?')} {desc}".strip()
        try:
            e = float(h.get("e_value")) if h.get("e_value") is not None else None
            e_str = f"{e:.2e}" if e is not None else "N/A"
        except (TypeError, ValueError):
            e_str = str(h.get("e_value"))
        ident = h.get("identity_pct")
        ident_str = f"{ident}%" if ident is not None else "N/A"
        lines.append(
            f"| {target} | {h.get('q_cov', 0):.1f}% | {e_str} | "
            f"{ident_str} | {h.get('aln_len', 'N/A')} |"
        )
    return "\n".join(lines)


def _save_and_envelope(
    hits: List[Dict[str, Any]],
    meta: Dict[str, Any],
    out_dir: str,
    file_basename: str,
    preview_md: str,
    elapsed_ms: int,
    source: str,
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, file_basename)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"hits": hits, "metadata": meta}, f, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write hits JSON: {e}")
    size = os.path.getsize(out_path)
    out = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(out_path),
            "file_name": os.path.basename(out_path),
            "file_size": size,
            "format": "json",
        },
        "content_preview": preview_md[:_PREVIEW_LEN],
        "biological_metadata": meta,
        "execution_context": {"elapsed_ms": elapsed_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def download_mmseqs2_homologs_by_sequence(
    sequence_or_fasta_path: str,
    out_dir: str,
    include_mgnify: bool = False,
    poll_interval: float = 10.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Run MMseqs2 (ColabFold) homologue search. Returns rich JSON envelope.

    `sequence_or_fasta_path` may be a raw amino-acid sequence or a path to a
    FASTA file (first record is used).
    """
    t0 = time.perf_counter()
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    try:
        hits, meta = search_mmseqs2_colabfold(
            sequence_or_fasta_path,
            include_mgnify=include_mgnify,
            poll_interval=poll_interval,
            timeout_secs=timeout_secs,
        )
    except ValueError as e:
        return _error_response("ValidationError", str(e))
    except TimeoutError as e:
        return _error_response("Timeout", str(e), suggestion="Increase timeout_secs or rerun later.")
    except RuntimeError as e:
        return _error_response("EngineError", str(e), suggestion="Try EBI BLAST (download_blast_homologs_by_sequence) as a fallback.")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    basename = f"mmseqs2_{meta.get('ticket_id', 'job')}.json"
    preview = _hits_to_markdown_mmseqs(hits)
    return _save_and_envelope(hits, meta, out_dir, basename, preview, elapsed_ms, "ColabFold MMseqs2")


def download_blast_homologs_by_sequence(
    sequence_or_fasta_path: str,
    out_dir: str,
    database: str = "uniprotkb_swissprot",
    email: Optional[str] = None,
    poll_interval: float = 30.0,
    timeout_secs: int = 15 * 60,
) -> str:
    """Run EBI NCBI BLAST homologue search. Returns rich JSON envelope.

    `database` may be a single name or a comma-separated list of supported
    UniProt/UniRef/PDB databases.
    """
    t0 = time.perf_counter()
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    try:
        hits, meta = search_ebi_blast(
            sequence_or_fasta_path,
            database=database,
            email=email,
            poll_interval=poll_interval,
            timeout_secs=timeout_secs,
        )
    except ValueError as e:
        return _error_response("ValidationError", str(e))
    except TimeoutError as e:
        return _error_response("Timeout", str(e), suggestion="Increase timeout_secs or rerun later.")
    except RuntimeError as e:
        return _error_response("EngineError", str(e))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    basename = f"blast_{meta.get('job_id', 'job')}.json"
    preview = _hits_to_markdown_blast(hits)
    return _save_and_envelope(hits, meta, out_dir, basename, preview, elapsed_ms, "EBI BLAST")


__all__ = [
    "download_mmseqs2_homologs_by_sequence",
    "download_blast_homologs_by_sequence",
]
