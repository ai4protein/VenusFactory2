"""
UniProt via BioServices: search, retrieve (FASTA/XML/tsv), mapping (UniProtKB↔KEGG, etc.).
Search frmt: xlsx, fasta, json, gff, tsv (UniProt no longer accepts "tab"; use "tsv").
See bioservices.md §1 Protein Analysis, §5 Identifier Mapping.
"""
import json
from typing import List, Optional

try:
    from bioservices import UniProt
    _HAS_BIOSERVICES = True
except ImportError:
    _HAS_BIOSERVICES = False

_ERR_NO_DEPS = json.dumps({"success": False, "error": "bioservices required: pip install bioservices"})


# UniProt search accepts: xlsx, fasta, json, gff, tsv (not "tab"; use "tsv" for tab-separated)
# database: uniprotkb (default), uniparc, uniref — see UniProt REST API
# Column names must be current API names (June 2022+); legacy "organism"/"genes" cause 400 Bad Request.
_UNIPROT_LEGACY_COLUMNS = {"organism": "organism_name", "genes": "gene_names", "entry name": "id"}
# Optional filters (taxonomy_id, keyword, subcellular_location, disease, proteome_id, literature, reviewed)
# are AND-ed to the query using UniProt query field syntax.
def uniprot_search(
    query: str,
    frmt: str = "tsv",
    columns: Optional[str] = None,
    limit: Optional[int] = 100,
    verbose: bool = False,
    database: str = "uniprotkb",
    taxonomy_id: Optional[int] = None,
    keyword: Optional[str] = None,
    subcellular_location: Optional[str] = None,
    disease: Optional[str] = None,
    proteome_id: Optional[str] = None,
    literature: Optional[str] = None,
    reviewed: Optional[bool] = None,
) -> str:
    """
    Search UniProt. Returns response text (tsv/XML/json depending on frmt).
    limit: max rows to return (default 100); without it bioservices fetches all pages and can appear to hang.
    frmt: one of xlsx, fasta, json, gff, tsv. columns: use current API names; legacy names organism/genes are auto-mapped.
    database: uniprotkb (UniProt Knowledgebase), uniparc (UniProt Archive), or uniref (UniRef clusters).
    Optional filters (taxonomy_id, keyword, subcellular_location, disease, proteome_id, literature, reviewed)
    are combined with the query using AND and UniProt field:value syntax.
    """
    if not _HAS_BIOSERVICES:
        return _ERR_NO_DEPS
    try:
        if frmt and frmt.lower() == "tab":
            frmt = "tsv"
        db = (database or "uniprotkb").strip().lower()
        if db not in ("uniprotkb", "uniparc", "uniref"):
            db = "uniprotkb"
        # Build final query: base query + optional filters (AND-ed)
        parts: List[str] = [query.strip()]
        if taxonomy_id is not None:
            parts.append(f"taxonomy_id:{taxonomy_id}")
        if keyword:
            # Allow quoted value if it contains spaces
            val = keyword.strip()
            if " " in val and not (val.startswith('"') and val.endswith('"')):
                val = f'"{val}"'
            parts.append(f"keyword:{val}")
        if subcellular_location:
            val = subcellular_location.strip()
            if " " in val and not (val.startswith('"') and val.endswith('"')):
                val = f'"{val}"'
            parts.append(f"cc_scl_term:{val}")
        if disease:
            val = disease.strip()
            if " " in val and not (val.startswith('"') and val.endswith('"')):
                val = f'"{val}"'
            parts.append(f"disease:{val}")
        if proteome_id:
            parts.append(f"proteome:{proteome_id.strip()}")
        if literature:
            lit = literature.strip()
            if lit.isdigit():
                parts.append(f"lit_pubmed_id:{lit}")
            else:
                parts.append(f"citation:{lit}")
        if reviewed is True:
            parts.append("reviewed:true")
        elif reviewed is False:
            parts.append("reviewed:false")
        final_query = " AND ".join(parts)
        u = UniProt(verbose=verbose)
        cols = columns or "accession,gene_names,organism_name"
        # Normalize legacy column names to current API (organism->organism_name, genes->gene_names)
        col_list = [c.strip() for c in cols.split(",") if c.strip()]
        col_list = [_UNIPROT_LEGACY_COLUMNS.get(c.lower(), c) for c in col_list]
        cols = ",".join(col_list)
        raw = u.search(final_query, frmt=frmt, columns=cols, database=db, limit=limit, progress=False)
        # Bioservices can return None for empty/no results; never return None so tool output has non-null results
        return raw if raw is not None else ""
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "tool": "uniprot_search"})


def uniprot_retrieve(
    uniprot_id: str,
    frmt: str = "fasta",
    verbose: bool = False,
) -> str:
    """
    Retrieve one UniProt entry. frmt: fasta, xml, tab, etc. Returns entry text.
    """
    if not _HAS_BIOSERVICES:
        return _ERR_NO_DEPS
    try:
        u = UniProt(verbose=verbose)
        return u.retrieve(uniprot_id, frmt=frmt)
    except Exception as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e), "tool": "uniprot_retrieve"})


def uniprot_mapping(
    fr: str,
    to: str,
    query: str,
    verbose: bool = False,
) -> str:
    """
    Map identifiers between databases. fr/to: e.g. UniProtKB_AC-ID, KEGG, PDB, RefSeq.
    query: one or more IDs (space or comma separated). Returns mapping response (format depends on service).
    """
    if not _HAS_BIOSERVICES:
        return _ERR_NO_DEPS
    try:
        u = UniProt(verbose=verbose)
        result = u.mapping(fr=fr, to=to, query=query)
        if result is None:
            return json.dumps({"success": False, "error": "No mapping result (API returned empty)", "tool": "uniprot_mapping"})
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "tool": "uniprot_mapping"})


def uniprot_search_and_retrieve(
    query: str,
    retrieve_frmt: str = "fasta",
    search_columns: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Search UniProt then retrieve first hit as FASTA (or retrieve_frmt). Returns retrieved entry or JSON error.
    """
    if not _HAS_BIOSERVICES:
        return _ERR_NO_DEPS
    try:
        u = UniProt(verbose=verbose)
        tsv = u.search(query, frmt="tsv", columns=search_columns or "id")
        if not tsv or not tsv.strip():
            return json.dumps({"success": False, "error": "No results", "query": query})
        lines = [l for l in tsv.strip().split("\n") if l]
        if len(lines) < 2:
            return json.dumps({"success": False, "error": "No hit IDs", "query": query})
        first_id = lines[1].split("\t")[0].strip()
        return u.retrieve(first_id, frmt=retrieve_frmt)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query})


if __name__ == "__main__":
    import argparse
    import os
    import sys
    p = argparse.ArgumentParser(description="UniProt tools: uniprot_search, retrieve, mapping, search_and_retrieve.")
    p.add_argument("--test", action="store_true", help="Run all non-helper functions; output under example/database/uniprot")
    p.add_argument("-q", "--query", default="urate", help="UniProt search query")
    p.add_argument("-o", "--output", help="Save TSV to this path (default: print only)")
    p.add_argument("-l", "--limit", type=int, default=25, help="Max rows (default 25)")
    p.add_argument("--columns", default="accession,gene_names,organism_name,protein_name,length", help="TSV columns")
    args = p.parse_args()

    if args.test:
        out_dir = os.path.join("example", "database", "uniprot", "search")
        os.makedirs(out_dir, exist_ok=True)
        test_id = "P43403"
        print("Testing uniprot_search(...)", flush=True)
        out = uniprot_search("urate", frmt="tsv", columns="accession,gene_names,organism_name", limit=5)
        print(f"  {len(out)} chars")
        with open(os.path.join(out_dir, f"{test_id}_search_sample.tsv"), "w", encoding="utf-8") as f:
            f.write(out)
        print("Testing uniprot_retrieve(...)", flush=True)
        out2 = uniprot_retrieve(test_id, frmt="fasta")
        print(f"  {len(out2)} chars")
        with open(os.path.join(out_dir, f"{test_id}_retrieve_sample.fasta"), "w", encoding="utf-8") as f:
            f.write(out2)
        print("Testing uniprot_mapping(...)", flush=True)
        # UniProt idmapping: valid 'to' from https://idmapping.uniprot.org (e.g. PDB, KEGG, RefSeq_Protein)
        out3 = uniprot_mapping("UniProtKB_AC-ID", "KEGG", test_id)
        if out3 is None:
            out3 = ""
        out3_str = out3 if isinstance(out3, str) else json.dumps(out3)
        print(f"  {len(out3_str)} chars")
        with open(os.path.join(out_dir, f"{test_id}_mapping_sample.txt"), "w", encoding="utf-8") as f:
            f.write(out3_str or "(no mapping or API error)")
        print("Testing uniprot_search_and_retrieve(...)", flush=True)
        # Use a narrow query to avoid slow pagination (no limit in search_and_retrieve)
        out4 = uniprot_search_and_retrieve("accession:P43403", retrieve_frmt="fasta")
        print(f"  {len(out4)} chars")
        with open(os.path.join(out_dir, f"{test_id}_search_and_retrieve_sample.fasta"), "w", encoding="utf-8") as f:
            f.write(out4)
        print(f"Done. Output under {out_dir}")
        sys.exit(0)

    print("Testing registered tool: uniprot_search", flush=True)
    out = uniprot_search(args.query, frmt="tsv", columns=args.columns, limit=args.limit)
    if out.strip().startswith("{") and '"success": false' in out.lower():
        print("  uniprot_search failed:", out[:300], file=sys.stderr)
        sys.exit(1)
    print(f"  uniprot_search: {len(out)} chars")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Done. Output: {args.output}")
    else:
        print(out[:4000] + ("..." if len(out) > 4000 else ""))
        if len(out) > 4000:
            print(f"... ({len(out)} chars total)")
