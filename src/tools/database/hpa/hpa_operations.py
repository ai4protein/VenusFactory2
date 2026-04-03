"""
Human Protein Atlas (HPA) operations: download functions returning rich JSON.

All functions return the project-standard rich JSON envelope:
  Success: {status, file_info, content_preview, biological_metadata, execution_context}
  Error:   {status, error: {type, message, suggestion}, file_info: null}

API endpoint: https://www.proteinatlas.org/search/{gene}?format=json
No authentication required.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from src.tools.path_sanitizer import to_client_file_path

try:
    from .hpa_api import hpa_get_exact_entry, hpa_get_tissue_expression
except ImportError:
    _dir = Path(__file__).resolve().parent
    if str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.database.hpa.hpa_api import hpa_get_exact_entry, hpa_get_tissue_expression

_PREVIEW_LEN = 500
_SOURCE_HPA = "Human Protein Atlas"


# ---------- Helper: standard JSON envelopes ----------

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
    download_time_ms: int = 0,
    source: str = _SOURCE_HPA,
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
        "execution_context": {"download_time_ms": download_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)


def _read_preview(path: str, max_chars: int = _PREVIEW_LEN) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# ---------- download_* functions ----------

def download_hpa_protein_by_gene(gene_name: str, out_path: str) -> str:
    """
    Download Human Protein Atlas full protein entry for a gene symbol to a JSON file.

    Includes: protein name, Ensembl ID, UniProt ID, tissue expression summary,
    subcellular location, pathology (cancer) data, and RNA expression overview.

    Args:
        gene_name: Gene symbol (e.g. 'TP53', 'BRCA1').
        out_path:  Output JSON file path (e.g. 'output/TP53_hpa.json').

    Returns:
        Rich JSON string: status, file_info, content_preview, biological_metadata,
        execution_context. On error: status 'error' with details.
    """
    t0 = time.perf_counter()
    try:
        entry = hpa_get_exact_entry(gene_name)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "gene": entry.get("Gene", gene_name),
            "ensembl_id": entry.get("Ensembl"),
            "uniprot_id": entry.get("Uniprot"),
            "gene_description": entry.get("Gene description"),
            "source": _SOURCE_HPA,
        }
        return _download_success_response(
            out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except LookupError as e:
        return _error_response("NotFound", str(e), suggestion="Check gene symbol at https://www.proteinatlas.org")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check gene symbol and network connection.")


def download_hpa_subcellular_location_by_gene(gene_name: str, out_path: str) -> str:
    """
    Download Human Protein Atlas subcellular location, secretome, and protein class data.

    Extracts fields needed to classify a protein as:
    - Secreted protein  → Secretome location is not None
    - Membrane protein  → Subcellular location contains 'Plasma membrane'
    - Intracellular     → nucleus, cytoplasm, ER, etc. with no secretome entry

    Returned fields:
    - Subcellular location (list of HPA-annotated compartments)
    - Secretome location & function (e.g. 'Secreted to blood', 'Transport')
    - Protein class (e.g. 'Membrane proteins', 'Secreted proteins', 'CD markers')
    - Reliability (IH)

    Args:
        gene_name: Gene symbol (e.g. 'TP53', 'IL6', 'MS4A1').
        out_path:  Output JSON file path.

    Returns:
        Rich JSON string with location + secretome + protein class data, or error.
    """
    t0 = time.perf_counter()
    try:
        entry = hpa_get_exact_entry(gene_name)

        subcell = {
            "Gene": entry.get("Gene"),
            "Ensembl": entry.get("Ensembl"),
            "Uniprot": entry.get("Uniprot"),
            "Protein class": entry.get("Protein class"),
            "Subcellular location": entry.get("Subcellular location"),
            "Subcellular main location": entry.get("Subcellular main location"),
            "Subcellular additional location": entry.get("Subcellular additional location"),
            "Secretome location": entry.get("Secretome location"),
            "Secretome function": entry.get("Secretome function"),
            "Reliability (IH)": entry.get("Reliability (IH)"),
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subcell, f, ensure_ascii=False, indent=2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        loc = entry.get("Subcellular location")
        sec_loc = entry.get("Secretome location")
        protein_class = entry.get("Protein class") or []
        reliability = entry.get("Reliability (IH)")

        reliab_str = str(reliability or "").lower()
        if "uncertain" in reliab_str:
            loc = []  # Exclude uncertain locations from parsed output

        # Classify protein localization type with priority weights
        loc_str = str(loc or "").lower()
        sec_loc_str = str(sec_loc or "").lower()
        protein_class_str = " ".join([str(c) for c in protein_class]).lower()
        
        if sec_loc and "secreted" in sec_loc_str:
            localization_type = "Secreted"
        elif "predicted secreted proteins" in protein_class_str:
            localization_type = "Secreted"
        elif ("cytosol" in loc_str or "nucleoplasm" in loc_str) and not sec_loc:
            localization_type = "Intracellular"
        elif ("plasma membrane" in loc_str and "enhanced" in reliab_str) or (sec_loc and "membrane" in sec_loc_str):
            localization_type = "Membrane"
        elif "predicted membrane proteins" in protein_class_str or "cd markers" in protein_class_str:
            localization_type = "Membrane"
        elif loc or "predicted intracellular proteins" in protein_class_str:
            localization_type = "Intracellular"
        else:
            localization_type = "Unknown"

        meta = {
            "gene": entry.get("Gene", gene_name),
            "ensembl_id": entry.get("Ensembl"),
            "subcellular_location": loc,
            "secretome_location": sec_loc,
            "secretome_function": entry.get("Secretome function"),
            "protein_class": protein_class,
            "reliability_ih": reliability,
            "localization_type": localization_type,
            "source": _SOURCE_HPA,
        }
        return _download_success_response(
            out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except LookupError as e:
        return _error_response("NotFound", str(e), suggestion="Check gene symbol at https://www.proteinatlas.org")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check gene symbol and network connection.")

def download_hpa_tissue_expression_by_gene(gene_name: str, out_path: str) -> str:
    """
    Download Human Protein Atlas RNA tissue expression data for a gene to a JSON file.

    Uses the HPA /search endpoint (hpa_get_exact_entry) which provides the complete
    tissue expression profile including:
    - RNA tissue specificity category (e.g. 'Tissue enriched', 'Low tissue specificity')
    - RNA tissue distribution (e.g. 'Detected in all', 'Detected in single') -- key for
      ubiquitous genes like TP53 where nTPM is null but the gene is expressed everywhere
    - RNA tissue specificity score
    - Per-tissue nTPM expression values (only present for tissue-enriched genes)
    - RNA tissue cell type enrichment (dominant cell types within tissues)
    - Tissue expression cluster (functional annotation)

    Args:
        gene_name: Gene symbol (e.g. 'GFAP', 'INS', 'GAPDH', 'TP53').
        out_path:  Output JSON file path.

    Returns:
        Rich JSON string with tissue expression data, or error details.
    """
    t0 = time.perf_counter()
    try:
        entry = hpa_get_exact_entry(gene_name)

        tissue_data = {
            "Gene": entry.get("Gene"),
            "Ensembl": entry.get("Ensembl"),
            "Uniprot": entry.get("Uniprot"),
            "RNA tissue specificity": entry.get("RNA tissue specificity"),
            "RNA tissue distribution": entry.get("RNA tissue distribution"),
            "RNA tissue specificity score": entry.get("RNA tissue specificity score"),
            "RNA tissue specific nTPM": entry.get("RNA tissue specific nTPM"),
            "RNA tissue cell type enrichment": entry.get("RNA tissue cell type enrichment"),
            "Tissue expression cluster": entry.get("Tissue expression cluster"),
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tissue_data, f, ensure_ascii=False, indent=2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Per-tissue nTPM dict — may be None for ubiquitous genes (Low tissue specificity)
        tissue_ntpm = entry.get("RNA tissue specific nTPM") or {}
        # Top expressed tissue(s) by nTPM value
        top_tissues = sorted(
            tissue_ntpm.items(),
            key=lambda kv: float(kv[1]) if kv[1] else 0,
            reverse=True
        )[:5]  # top 5

        meta = {
            "gene": entry.get("Gene", gene_name),
            "ensembl_id": entry.get("Ensembl"),
            "rna_tissue_specificity": entry.get("RNA tissue specificity"),
            "rna_tissue_distribution": entry.get("RNA tissue distribution"),
            "rna_specificity_score": entry.get("RNA tissue specificity score"),
            "top_expressed_tissues": {t: v for t, v in top_tissues},
            "all_tissue_ntpm": tissue_ntpm,
            "tissue_cell_type_enrichment": entry.get("RNA tissue cell type enrichment"),
            "tissue_expression_cluster": entry.get("Tissue expression cluster"),
            "source": _SOURCE_HPA,
        }
        return _download_success_response(
            out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except LookupError as e:
        return _error_response("NotFound", str(e), suggestion="Check gene symbol at https://www.proteinatlas.org")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check gene symbol and network connection.")


def download_hpa_single_cell_type_by_gene(gene_name: str, out_path: str) -> str:
    """
    Download Human Protein Atlas RNA single cell type specificity data for a gene.

    Uses the /search endpoint (hpa_get_exact_entry) which returns single cell type
    fields alongside all other gene metadata. Extracts:
    - RNA single cell type specificity category (e.g. 'Cell type enriched')
    - Per-cell-type nCPM values (e.g. {'Astrocytes': '781.6'})
    - RNA single nuclei brain cell type data (finer brain cell resolution)
    - Single cell expression cluster annotation

    Args:
        gene_name: Gene symbol (e.g. 'GFAP', 'INS').
        out_path:  Output JSON file path.

    Returns:
        Rich JSON string with single cell type data, or error details.
    """
    t0 = time.perf_counter()
    try:
        entry = hpa_get_exact_entry(gene_name)

        sc_data = {
            "Gene": entry.get("Gene"),
            "Ensembl": entry.get("Ensembl"),
            "Uniprot": entry.get("Uniprot"),
            "RNA single cell type specificity": entry.get("RNA single cell type specificity"),
            "RNA single cell type distribution": entry.get("RNA single cell type distribution"),
            "RNA single cell type specificity score": entry.get("RNA single cell type specificity score"),
            "RNA single cell type specific nCPM": entry.get("RNA single cell type specific nCPM"),
            "RNA single nuclei brain specificity": entry.get("RNA single nuclei brain specificity"),
            "RNA single nuclei brain specific nCPM": entry.get("RNA single nuclei brain specific nCPM"),
            "Single cell expression cluster": entry.get("Single cell expression cluster"),
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sc_data, f, ensure_ascii=False, indent=2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Top cell types by nCPM
        cell_ntpm = entry.get("RNA single cell type specific nCPM") or {}
        top_cells = sorted(
            cell_ntpm.items(),
            key=lambda kv: float(kv[1]) if kv[1] else 0,
            reverse=True
        )[:5]

        meta = {
            "gene": entry.get("Gene", gene_name),
            "ensembl_id": entry.get("Ensembl"),
            "sc_specificity": entry.get("RNA single cell type specificity"),
            "sc_specificity_score": entry.get("RNA single cell type specificity score"),
            "top_cell_types": {c: v for c, v in top_cells},
            "all_cell_type_ncpm": cell_ntpm,
            "sc_cluster": entry.get("Single cell expression cluster"),
            "source": _SOURCE_HPA,
        }
        return _download_success_response(
            out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except LookupError as e:
        return _error_response("NotFound", str(e), suggestion="Check gene symbol at https://www.proteinatlas.org")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check gene symbol and network connection.")




def download_hpa_blood_expression_by_gene(gene_name: str, out_path: str) -> str:
    """
    Download Human Protein Atlas blood cell expression and serum concentration data for a gene.

    Works for ANY gene (not just blood-specific ones). This is useful for:
    - Antibody design (off-target expression in immune cells)
    - Biomarker discovery (serum concentration)
    - Immune cell targeting (which blood cell type expresses it most)

    Extracts:
    - RNA blood cell specificity category (e.g. 'Group enriched', 'Low immune cell specificity')
    - Per-blood-cell-type nTPM (T cells, B cells, NK cells, monocytes, neutrophils...)
    - Blood lineage specificity and enrichment
    - Blood serum/plasma concentration (immunoassay and mass spectrometry measurements)
    - Blood expression cluster annotation

    Args:
        gene_name: Gene symbol (e.g. 'IL6', 'MS4A1', 'TP53', 'ALB', 'GAPDH').
        out_path:  Output JSON file path.

    Returns:
        Rich JSON string with blood expression data, or error details.
    """
    t0 = time.perf_counter()
    try:
        entry = hpa_get_exact_entry(gene_name)

        blood_data = {
            "Gene": entry.get("Gene"),
            "Ensembl": entry.get("Ensembl"),
            "Uniprot": entry.get("Uniprot"),
            "RNA blood cell specificity": entry.get("RNA blood cell specificity"),
            "RNA blood cell distribution": entry.get("RNA blood cell distribution"),
            "RNA blood cell specificity score": entry.get("RNA blood cell specificity score"),
            "RNA blood cell specific nTPM": entry.get("RNA blood cell specific nTPM"),
            "RNA blood lineage specificity": entry.get("RNA blood lineage specificity"),
            "RNA blood lineage distribution": entry.get("RNA blood lineage distribution"),
            "RNA blood lineage specific nTPM": entry.get("RNA blood lineage specific nTPM"),
            "Blood concentration - Conc. blood IM [pg/L]": entry.get("Blood concentration - Conc. blood IM [pg/L]"),
            "Blood concentration - Conc. blood MS [pg/L]": entry.get("Blood concentration - Conc. blood MS [pg/L]"),
            "Blood expression cluster": entry.get("Blood expression cluster"),
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blood_data, f, ensure_ascii=False, indent=2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Top blood cell types by nTPM
        cell_ntpm = entry.get("RNA blood cell specific nTPM") or {}
        if cell_ntpm:
            top_blood_cells = sorted(
                cell_ntpm.items(),
                key=lambda kv: float(kv[1]) if kv[1] else 0,
                reverse=True
            )[:5]
            top_blood_dict = {c: v for c, v in top_blood_cells}
        else:
            # For ubiquitous genes (like GAPDH) where specific nTPM is null,
            # return a Non-specific indicator to avoid empty dictionaries
            # Note: HPA's /search endpoint does not provide raw non-specific nTPMs.
            dist = entry.get("RNA blood cell distribution") or "unknown distribution"
            top_blood_dict = {"Non-specific": f"Expression data omitted for {dist}"}

        # Blood concentration: prefer IM, fallback to MS
        conc_im = entry.get("Blood concentration - Conc. blood IM [pg/L]")
        conc_ms = entry.get("Blood concentration - Conc. blood MS [pg/L]")

        meta = {
            "gene": entry.get("Gene", gene_name),
            "ensembl_id": entry.get("Ensembl"),
            "blood_cell_specificity": entry.get("RNA blood cell specificity"),
            "top_blood_cells": top_blood_dict,
            "all_blood_cell_ntpm": cell_ntpm,
            "blood_lineage_specificity": entry.get("RNA blood lineage specificity"),
            "blood_lineage_ntpm": entry.get("RNA blood lineage specific nTPM") or {},
            "blood_concentration_im_pgL": conc_im,
            "blood_concentration_ms_pgL": conc_ms,
            "blood_expression_cluster": entry.get("Blood expression cluster"),
            "source": _SOURCE_HPA,
        }
        return _download_success_response(
            out_path,
            content_preview=_read_preview(out_path),
            biological_metadata=meta,
            download_time_ms=elapsed_ms,
        )
    except LookupError as e:
        return _error_response("NotFound", str(e), suggestion="Check gene symbol at https://www.proteinatlas.org")
    except Exception as e:
        return _error_response("DownloadError", str(e), suggestion="Check gene symbol and network connection.")


__all__ = [
    "download_hpa_protein_by_gene",
    "download_hpa_subcellular_location_by_gene",
    "download_hpa_tissue_expression_by_gene",
    "download_hpa_single_cell_type_by_gene",
    "download_hpa_blood_expression_by_gene",
]
