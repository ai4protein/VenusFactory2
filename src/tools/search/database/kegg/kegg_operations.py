"""
KEGG REST API: query (return text) and download (save to file) for all operations.
Academic use only. Base client: kegg_rest.

Operations: info, list, find, get, conv, link, ddi.
See kegg.md and src/agent/skills/kegg/ for references.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Union

try:
    from .kegg_rest import kegg_request, _join_ids, BASE_URL
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "kegg" and str(_dir.parents[4]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[4]))
    from src.tools.search.database.kegg.kegg_rest import kegg_request, _join_ids, BASE_URL


# ---------- info ----------
def query_kegg_info(database: str) -> str:
    """Query KEGG database metadata/statistics. Returns response text. No file save."""
    return kegg_request("info", database)


def download_kegg_info(database: str, out_dir: str, filename: Optional[str] = None) -> str:
    """Download KEGG info to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or f"{database.replace(':', '_')}_info.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_info(database)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- list ----------
def query_kegg_list(database: str, org_or_ids: Optional[Union[str, list]] = None) -> str:
    """Query KEGG list entries. Returns response text."""
    if org_or_ids is None:
        return kegg_request("list", database)
    sid = _join_ids(org_or_ids) if isinstance(org_or_ids, list) else str(org_or_ids).strip()
    return kegg_request("list", database, sid)


def download_kegg_list(database: str, out_dir: str, org_or_ids: Optional[Union[str, list]] = None, filename: Optional[str] = None) -> str:
    """Download KEGG list result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or f"{database.replace(':', '_')}_list.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_list(database, org_or_ids)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- find ----------
def query_kegg_find(database: str, query: str, option: Optional[str] = None) -> str:
    """Search KEGG database. Returns response text."""
    if option:
        return kegg_request("find", database, query, option)
    return kegg_request("find", database, query)


def download_kegg_find(database: str, query: str, out_dir: str, option: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Download KEGG find result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or f"find_{database}_{query.replace(' ', '_')[:20]}.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_find(database, query, option)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- get ----------
def query_kegg_get(entry_id: Union[str, List[str]], format: Optional[str] = None) -> str:
    """Get KEGG entry/entries. Returns response text."""
    eid = _join_ids(entry_id)
    if format:
        return kegg_request("get", eid, format)
    return kegg_request("get", eid)


def download_kegg_get(entry_id: Union[str, List[str]], out_dir: str, format: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Download KEGG get result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    eid = _join_ids(entry_id) if isinstance(entry_id, list) else str(entry_id).strip()
    safe = eid.replace("+", "_").replace(":", "_")[:50]
    ext = "json" if format == "json" else "txt"
    name = filename or f"get_{safe}.{ext}"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_get(entry_id, format)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- conv ----------
def query_kegg_conv(target_db: str, source_id: Union[str, List[str]]) -> str:
    """Convert KEGG IDs to/from external DBs. Returns response text."""
    sid = _join_ids(source_id)
    return kegg_request("conv", target_db, sid)


def download_kegg_conv(target_db: str, source_id: Union[str, List[str]], out_dir: str, filename: Optional[str] = None) -> str:
    """Download KEGG conv result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or f"conv_{target_db}.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_conv(target_db, source_id)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- link ----------
def query_kegg_link(target_db: str, source_id: Union[str, List[str]]) -> str:
    """Query KEGG cross-references. Returns response text."""
    sid = _join_ids(source_id)
    return kegg_request("link", target_db, sid)


def download_kegg_link(target_db: str, source_id: Union[str, List[str]], out_dir: str, filename: Optional[str] = None) -> str:
    """Download KEGG link result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or f"link_{target_db}.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_link(target_db, source_id)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# ---------- ddi ----------
def query_kegg_ddi(drug_id: Union[str, List[str]]) -> str:
    """Query KEGG drug-drug interactions. Returns response text."""
    did = _join_ids(drug_id)
    return kegg_request("ddi", did)


def download_kegg_ddi(drug_id: Union[str, List[str]], out_dir: str, filename: Optional[str] = None) -> str:
    """Download KEGG ddi result to file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    name = filename or "ddi.txt"
    out_path = os.path.join(out_dir, name)
    text = query_kegg_ddi(drug_id)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved to {out_path}"


# Backward-compat aliases (tools_agent and others use kegg_info, kegg_find, kegg_get)
kegg_info = query_kegg_info
kegg_list = query_kegg_list
kegg_find = query_kegg_find


def kegg_get(entry_id: Union[str, List[str]], format: Optional[str] = None, option: Optional[str] = None) -> str:
    """Get KEGG entry/entries. option is alias for format (backward compat)."""
    return query_kegg_get(entry_id, format=format or option)


kegg_conv = query_kegg_conv
kegg_link = query_kegg_link
kegg_ddi = query_kegg_ddi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEGG REST: query and download (info, list, find, get, conv, link, ddi). Academic use only.")
    parser.add_argument("--test", action="store_true", help="Run tests for all query_* and download_*; output under example/database/kegg")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "kegg")
        os.makedirs(out_base, exist_ok=True)
        print("Testing KEGG operations (query + download); output under example/database/kegg")

        print("  1. query_kegg_info, download_kegg_info(...)")
        t = query_kegg_info("pathway")
        with open(os.path.join(out_base, "query_info_pathway_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:2000] if len(t) > 2000 else t)
        download_kegg_info("pathway", out_base)

        print("  2. query_kegg_list, download_kegg_list(...)")
        t = query_kegg_list("pathway", "hsa")
        with open(os.path.join(out_base, "query_list_pathway_hsa_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        download_kegg_list("pathway", out_base, "hsa")

        print("  3. query_kegg_find, download_kegg_find(...)")
        t = query_kegg_find("genes", "p53")
        with open(os.path.join(out_base, "query_find_genes_p53_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        download_kegg_find("genes", "p53", out_base)

        print("  4. query_kegg_get, download_kegg_get(...)")
        t = query_kegg_get("hsa:7535")
        with open(os.path.join(out_base, "query_get_hsa7535_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:3000] if len(t) > 3000 else t)
        download_kegg_get("hsa:7535", out_base)

        print("  5. query_kegg_conv, download_kegg_conv(...)")
        t = query_kegg_conv("ncbi-geneid", "hsa:7535")
        with open(os.path.join(out_base, "query_conv_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:2000] if len(t) > 2000 else t)
        download_kegg_conv("ncbi-geneid", "hsa:7535", out_base)

        print("  6. query_kegg_link, download_kegg_link(...)")
        t = query_kegg_link("pathway", "hsa:7535")
        with open(os.path.join(out_base, "query_link_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:2000] if len(t) > 2000 else t)
        download_kegg_link("pathway", "hsa:7535", out_base)

        print("  7. query_kegg_ddi, download_kegg_ddi(...)")
        t = query_kegg_ddi("D00001")
        with open(os.path.join(out_base, "query_ddi_sample.txt"), "w", encoding="utf-8") as f:
            f.write(t[:2000] if len(t) > 2000 else t)
        download_kegg_ddi("D00001", out_base)

        print(f"Done. Output under {out_base}")
        exit(0)

    print("Use --test to run all tests, or call query_*/download_* from Python.")
