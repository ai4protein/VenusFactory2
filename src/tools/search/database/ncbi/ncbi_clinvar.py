"""
NCBI ClinVar: atomic E-utilities (esearch, esummary, efetch), term builder, and FTP helpers.

Skill and references (for Agent): src/agent/skills/ncbi_clinvar/
  - SKILL.md          : When to use, project tools table (this module's functions)
  - references/api_reference.md     : E-utilities params, rate limits, workflow
  - references/clinical_significance.md : ACMG/AMP terms, review status, conflicts
  - references/data_formats.md       : FTP layout, XML/VCF/tab formats
"""
import json
import os
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

import requests

CLINVAR_DB = "clinvar"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FTP_BASE = "ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar"
# Rate limit: 3 req/s without API key, 10 with key
_DELAY_NO_KEY = 0.34
_DELAY_WITH_KEY = 0.11


def _delay(api_key: Optional[str] = None) -> None:
    time.sleep(_DELAY_WITH_KEY if api_key else _DELAY_NO_KEY)


def _ids_arg(id_list: List[str]) -> str:
    """Comma-separated IDs for E-utilities."""
    if isinstance(id_list, str):
        return id_list.strip()
    return ",".join(str(x).strip() for x in id_list if x is not None)


# -----------------------------------------------------------------------------
# Atomic E-utilities
# -----------------------------------------------------------------------------


def clinvar_esearch(
    term: str,
    retmax: int = 20,
    retstart: int = 0,
    retmode: str = "json",
    api_key: Optional[str] = None,
    sort: Optional[str] = None,
) -> str:
    """
    ClinVar esearch: search for variation IDs matching term. Returns response text (JSON or XML).
    term: e.g. "BRCA1[gene]", "pathogenic[CLNSIG]", "BRCA1[gene] AND pathogenic[CLNSIG]".
    """
    params = {
        "db": CLINVAR_DB,
        "term": term,
        "retmax": retmax,
        "retstart": retstart,
        "retmode": retmode,
    }
    if api_key:
        params["api_key"] = api_key
    if sort:
        params["sort"] = sort
    url = f"{EUTILS_BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
    try:
        resp = requests.get(url, timeout=30)
        _delay(api_key)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        return json.dumps({"success": False, "error": str(e), "tool": "esearch"})


def clinvar_esummary(
    id_list: List[str],
    retmode: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """
    ClinVar esummary: get summaries for variation IDs. Returns response text (JSON or XML).
    id_list: list of Variation IDs (VCV) or single comma-separated string.
    """
    ids = _ids_arg(id_list)
    if not ids:
        return json.dumps({"success": False, "error": "id_list is empty", "tool": "esummary"})
    params = {"db": CLINVAR_DB, "id": ids, "retmode": retmode}
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS_BASE}/esummary.fcgi?" + urllib.parse.urlencode(params)
    try:
        resp = requests.get(url, timeout=60)
        _delay(api_key)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        return json.dumps({"success": False, "error": str(e), "tool": "esummary"})


def clinvar_efetch(
    id_list: List[str],
    rettype: str = "vcv",
    retmode: str = "xml",
    api_key: Optional[str] = None,
) -> str:
    """
    ClinVar efetch: fetch full records for variation IDs. Returns XML or other format as text.
    id_list: list of Variation IDs (VCV).
    rettype: e.g. vcv (variation-centric), default vcv.
    retmode: xml, text, etc.
    """
    ids = _ids_arg(id_list)
    if not ids:
        return json.dumps({"success": False, "error": "id_list is empty", "tool": "efetch"})
    params = {"db": CLINVAR_DB, "id": ids, "rettype": rettype, "retmode": retmode}
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS_BASE}/efetch.fcgi?" + urllib.parse.urlencode(params)
    try:
        resp = requests.get(url, timeout=120)
        _delay(api_key)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        return json.dumps({"success": False, "error": str(e), "tool": "efetch"})


# -----------------------------------------------------------------------------
# Term builder (from ncbi_clinvar.md search patterns)
# -----------------------------------------------------------------------------


def build_clinvar_term(
    gene: Optional[str] = None,
    clinical_significance: Optional[str] = None,
    condition: Optional[str] = None,
    chr: Optional[str] = None,
    variant_name: Optional[str] = None,
    exclude_conflicting: bool = False,
    raw_term: Optional[str] = None,
) -> str:
    """
    Build an E-utilities query term for ClinVar. All non-None args are ANDed.
    gene: e.g. "BRCA1" -> BRCA1[gene]
    clinical_significance: e.g. "pathogenic" -> pathogenic[CLNSIG]
    condition: e.g. "breast cancer" -> "breast cancer"[disorder]
    chr: e.g. "13" -> 13[chr]
    variant_name: e.g. "NM_000059.3:c.1310_1313del" -> "NM_000059.3:c.1310_1313del"[variant name]
    exclude_conflicting: add NOT conflicting[RVSTAT]
    raw_term: if set, returned as-is (other args ignored).
    """
    if raw_term:
        return raw_term.strip()
    parts = []
    if gene:
        parts.append(f"{gene.strip()}[gene]")
    if clinical_significance:
        parts.append(f"{clinical_significance.strip()}[CLNSIG]")
    if condition:
        parts.append(f'"{condition.strip()}"[disorder]')
    if chr:
        parts.append(f"{str(chr).strip()}[chr]")
    if variant_name:
        parts.append(f'"{variant_name.strip()}"[variant name]')
    if exclude_conflicting:
        parts.append("NOT conflicting[RVSTAT]")
    return " AND ".join(parts) if parts else ""


# -----------------------------------------------------------------------------
# High-level query (compose atoms)
# -----------------------------------------------------------------------------


def query_clinvar(
    term: str,
    retmax: int = 20,
    retstart: int = 0,
    retmode: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """
    Search ClinVar with term; returns esearch response text (JSON with idlist, count, etc.).
    term can be built with build_clinvar_term() or passed as raw string.
    """
    return clinvar_esearch(term=term, retmax=retmax, retstart=retstart, retmode=retmode, api_key=api_key)


def get_clinvar_summary(
    variation_ids: List[str],
    retmode: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """Get variant summaries for given Variation IDs. Returns esummary response text."""
    return clinvar_esummary(id_list=variation_ids, retmode=retmode, api_key=api_key)


def fetch_clinvar_records(
    variation_ids: List[str],
    rettype: str = "vcv",
    retmode: str = "xml",
    api_key: Optional[str] = None,
) -> str:
    """Fetch full ClinVar records for given Variation IDs. Returns XML (or other) text."""
    return clinvar_efetch(id_list=variation_ids, rettype=rettype, retmode=retmode, api_key=api_key)


# -----------------------------------------------------------------------------
# Parse esearch JSON (minimal: idlist and count)
# -----------------------------------------------------------------------------


def parse_esearch_ids(esearch_response_text: str) -> List[str]:
    """
    Parse esearch JSON response; return list of IDs (idlist). Returns [] on error.
    """
    try:
        data = json.loads(esearch_response_text)
        if "error" in data or data.get("esearchresult") is None:
            return []
        return list(data.get("esearchresult", {}).get("idlist", []))
    except (json.JSONDecodeError, TypeError):
        return []


def parse_esearch_count(esearch_response_text: str) -> int:
    """Parse esearch JSON; return total count. Returns 0 on error."""
    try:
        data = json.loads(esearch_response_text)
        if "error" in data or data.get("esearchresult") is None:
            return 0
        return int(data.get("esearchresult", {}).get("count", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0


# -----------------------------------------------------------------------------
# FTP URLs and download (atomic: one URL -> one file)
# -----------------------------------------------------------------------------

# Keys for get_clinvar_ftp_url
FTP_VARIANT_SUMMARY = "variant_summary"
FTP_VCF_GRCH37 = "vcf_grch37"
FTP_VCF_GRCH38 = "vcf_grch38"
FTP_VCF_GRCH37_TBI = "vcf_grch37_tbi"
FTP_VCF_GRCH38_TBI = "vcf_grch38_tbi"
FTP_XML_LATEST = "xml_latest"
FTP_VAR_CITATIONS = "var_citations"
FTP_CROSS_REFS = "cross_references"


def get_clinvar_ftp_url(key: str) -> str:
    """
    Return FTP URL for a known ClinVar file. Keys:
    variant_summary, vcf_grch37, vcf_grch38, vcf_grch37_tbi, vcf_grch38_tbi,
    xml_latest, var_citations, cross_references.
    """
    urls = {
        FTP_VARIANT_SUMMARY: f"{FTP_BASE}/tab_delimited/variant_summary.txt.gz",
        FTP_VCF_GRCH37: f"{FTP_BASE}/vcf_GRCh37/clinvar.vcf.gz",
        FTP_VCF_GRCH38: f"{FTP_BASE}/vcf_GRCh38/clinvar.vcf.gz",
        FTP_VCF_GRCH37_TBI: f"{FTP_BASE}/vcf_GRCh37/clinvar.vcf.gz.tbi",
        FTP_VCF_GRCH38_TBI: f"{FTP_BASE}/vcf_GRCh38/clinvar.vcf.gz.tbi",
        FTP_XML_LATEST: f"{FTP_BASE}/xml/clinvar_variation/ClinVarVariationRelease_00-latest.xml.gz",
        FTP_VAR_CITATIONS: f"{FTP_BASE}/tab_delimited/var_citations.txt.gz",
        FTP_CROSS_REFS: f"{FTP_BASE}/tab_delimited/cross_references.txt.gz",
    }
    return urls.get(key.lower(), "")


def download_clinvar_ftp(
    url_or_key: str,
    out_dir: str,
    filename: Optional[str] = None,
) -> str:
    """
    Download a file from ClinVar FTP (or from URL if url_or_key is a full URL).
    url_or_key: FTP key (e.g. "variant_summary", "vcf_grch38") or full URL.
    out_dir: directory to save file.
    filename: optional output filename; otherwise derived from URL.
    Returns message string (success or error). Uses urllib for ftp://, requests for http(s).
    """
    os.makedirs(out_dir, exist_ok=True)
    if url_or_key.startswith(("http://", "https://", "ftp://")):
        url = url_or_key
    else:
        url = get_clinvar_ftp_url(url_or_key)
        if not url:
            return f"Unknown FTP key: {url_or_key}"
    name = filename or os.path.basename(urllib.parse.urlparse(url).path or "clinvar_download")
    out_path = os.path.join(out_dir, name)
    try:
        if url.startswith("ftp://"):
            urllib.request.urlretrieve(url, out_path)
        else:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return f"Downloaded to {out_path}"
    except (urllib.error.URLError, requests.RequestException, OSError) as e:
        return f"Download failed: {e}"


# -----------------------------------------------------------------------------
# Convenience: query (return JSON text) and download (save to file)
# -----------------------------------------------------------------------------


def query_clinvar_variants(
    term: str,
    retmax: int = 20,
    retstart: int = 0,
    api_key: Optional[str] = None,
) -> str:
    """
    Search ClinVar and return esearch result as JSON string. For idlist + count use parse_esearch_ids/count.
    """
    return query_clinvar(term=term, retmax=retmax, retstart=retstart, retmode="json", api_key=api_key)


def download_clinvar_esearch(
    term: str,
    out_path: str,
    retmax: int = 100,
    retmode: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """Run esearch and save response to file. Returns message string."""
    text = query_clinvar(term=term, retmax=retmax, retmode=retmode, api_key=api_key)
    if text.strip().startswith("{"):
        try:
            json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            return f"Saved to {out_path}"
    if '"success": false' in text or '"error"' in text.lower():
        return f"Search failed: {text[:200]}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


def download_clinvar_summary(
    variation_ids: List[str],
    out_path: str,
    retmode: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """Fetch esummary for IDs and save to file. Returns message string."""
    text = get_clinvar_summary(variation_ids, retmode=retmode, api_key=api_key)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Saved to {out_path}"
    except OSError as e:
        return f"Write failed: {e}"


def download_clinvar_fetch(
    variation_ids: List[str],
    out_path: str,
    rettype: str = "vcv",
    retmode: str = "xml",
    api_key: Optional[str] = None,
) -> str:
    """Fetch full records and save to file. Returns message string."""
    text = fetch_clinvar_records(variation_ids, rettype=rettype, retmode=retmode, api_key=api_key)
    if text.strip().startswith("{") and "error" in text:
        return f"Fetch failed: {text[:300]}"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Saved to {out_path}"
    except OSError as e:
        return f"Write failed: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCBI ClinVar: search (esearch) and optional download.")
    parser.add_argument("--test", action="store_true", help="Run tests for build_clinvar_term, clinvar_esearch, get_clinvar_ftp_url, etc.; output under example/database/ncbi")
    parser.add_argument("--term", type=str, help="E-utilities search term (e.g. BRCA1[gene] AND pathogenic[CLNSIG])")
    parser.add_argument("--gene", type=str, help="Gene symbol (builds term with [gene])")
    parser.add_argument("--significance", type=str, help="Clinical significance (e.g. pathogenic -> [CLNSIG])")
    parser.add_argument("--no-conflict", action="store_true", help="Exclude conflicting interpretations")
    parser.add_argument("--retmax", type=int, default=20)
    parser.add_argument("--retstart", type=int, default=0)
    parser.add_argument("--out", type=str, help="Save esearch JSON to this file")
    parser.add_argument("--api_key", type=str, default=None, help="NCBI API key for higher rate limit")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "ncbi", "clinvar")
        os.makedirs(out_base, exist_ok=True)
        clinvar_dir = os.path.join(out_base, "clinvar")
        os.makedirs(clinvar_dir, exist_ok=True)
        print("Testing build_clinvar_term(...)")
        term = build_clinvar_term(gene="BRCA1", clinical_significance="pathogenic")
        print(f"  term: {term}")
        print("Testing clinvar_esearch(...)")
        esearch_text = clinvar_esearch(term, retmax=5, retmode="json", api_key=args.api_key)
        esearch_path = os.path.join(clinvar_dir, "esearch_sample.json")
        with open(esearch_path, "w", encoding="utf-8") as f:
            f.write(esearch_text[:10000] if len(esearch_text) > 10000 else esearch_text)
        print(f"  saved to {esearch_path}")
        ids = parse_esearch_ids(esearch_text)
        if ids:
            print("Testing get_clinvar_summary(...)")
            summary_text = get_clinvar_summary(ids[:3], retmode="json", api_key=args.api_key)
            with open(os.path.join(clinvar_dir, "esummary_sample.json"), "w", encoding="utf-8") as f:
                f.write(summary_text[:15000] if len(summary_text) > 15000 else summary_text)
            print("  saved esummary_sample.json")
        print("Testing get_clinvar_ftp_url(...)")
        url = get_clinvar_ftp_url("variant_summary")
        print(f"  variant_summary URL: {url[:80]}...")
        print(f"Done. Output under {out_base}")
        exit(0)

    term = args.term or build_clinvar_term(
        gene=args.gene,
        clinical_significance=args.significance,
        exclude_conflicting=args.no_conflict,
    )
    if not term:
        print("Error: provide --term or (e.g.) --gene and/or --significance")
        exit(1)

    result = query_clinvar(term=term, retmax=args.retmax, retstart=args.retstart, api_key=args.api_key)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Saved to {args.out}")
    else:
        print(result[:2000] + ("..." if len(result) > 2000 else ""))
