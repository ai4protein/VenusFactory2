"""
Human Protein Atlas (HPA) REST API client.

Public API, no authentication required.
Docs: https://www.proteinatlas.org/about/help/api

The main search endpoint returns a JSON list of matching gene entries.
Each entry has fields like Gene, Ensembl, Uniprot, Gene description,
Tissue specificity, Subcellular location, etc.
"""

import requests
from typing import Optional, Dict, Any, List

BASE_URL = "https://www.proteinatlas.org"
_DEFAULT_TIMEOUT = 30

# Comprehensive column set for /search endpoint
_DEFAULT_COLUMNS = (
    "g,gs,eg,gd,up,pe,sl,xt,rna_cancer_category,"
    "rna_tissue_category,rna_tissue_specificity"
)


def hpa_search(gene_name: str, timeout: int = _DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Search HPA for a gene name. Returns a list of matching gene entry dicts.

    Uses the /search/{gene}?format=json endpoint which is the correct
    public API endpoint (the /{gene}.json URL format does not work).

    Args:
        gene_name: Gene symbol (e.g. 'TP53', 'BRCA1') or Ensembl ID.
        timeout: HTTP timeout in seconds. Default 30.

    Returns:
        List of entry dicts, or raises LookupError / RuntimeError.
    """
    url = f"{BASE_URL}/search/{gene_name}"
    params = {"format": "json"}

    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code == 404:
        raise LookupError(f"Gene '{gene_name}' not found in Human Protein Atlas.")
    if resp.status_code != 200:
        raise RuntimeError(f"HPA API error {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse HPA JSON response: {e}")

    if not data:
        raise LookupError(f"Gene '{gene_name}' not found in Human Protein Atlas.")

    return data


def hpa_get_exact_entry(gene_name: str, timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Search HPA and return the single entry whose Gene field exactly matches
    gene_name (case-insensitive). Falls back to the first result if no exact match.

    Args:
        gene_name: Gene symbol (e.g. 'TP53').
        timeout: HTTP timeout in seconds.

    Returns:
        Single entry dict.
    """
    entries = hpa_search(gene_name, timeout=timeout)
    gene_upper = gene_name.upper()
    for entry in entries:
        if entry.get("Gene", "").upper() == gene_upper:
            return entry
    # fallback: first result
    return entries[0]


def hpa_get_tissue_expression(gene_name: str, timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Fetch RNA tissue specificity data for a gene using the HPA search_download endpoint.

    Uses columns: g (gene), eg (Ensembl), up (UniProt), rnats (RNA tissue specificity
    category), rnatss (specificity score), rnatsm (per-tissue nTPM values).

    Args:
        gene_name: Gene symbol (e.g. 'GFAP', 'INS').
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with tissue expression fields, or raises LookupError / RuntimeError.
    """
    url = f"{BASE_URL}/api/search_download.php"
    params = {
        "search": gene_name,
        "format": "json",
        "columns": "g,eg,up,rnats,rnatss,rnatsm",
        "compress": "no",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HPA search_download error {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse HPA search_download JSON: {e}")

    if not data:
        raise LookupError(f"Gene '{gene_name}' not found in Human Protein Atlas.")

    gene_upper = gene_name.upper()
    for entry in data:
        if entry.get("Gene", "").upper() == gene_upper:
            return entry
    return data[0]
