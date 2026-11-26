"""Literature, database and web retrieval for principal-investigator reports."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from config import get_config
from logger import get_logger
from mcp_tools.common.citations import (
    _format_dataset_citations,
    _format_literature_citations,
    _format_literature_for_reading,
    _format_web_citations,
    _format_web_for_reading,
)

_logger = get_logger("agent.utils")
SEARCH_MAX_RESULTS = get_config().agent.search_max_results
PI_SEARCH_TOOL_NAMES = [
    "query_pubmed", "query_semantic_scholar", "query_arxiv",
    "query_web", "query_serpapi", "query_tavily", "query_duckduckgo",
    "query_github", "query_hugging_face"
]


def _fetch_literature_for_pi(user_text: str, max_results: int = None) -> tuple:
    """Run literature_search tool. Returns (formatted_str_for_prompt, tool_input_dict, raw_output_str) for logging."""
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    empty = ("", {}, "")
    try:
        from mcp_tools.tools.tools_agent_hub import get_tools
        tools = get_tools()
        lit_tool = next((t for t in tools if getattr(t, "name", "") == "query_literature_by_keywords"), None)
        if not lit_tool:
            return empty

        query = (user_text or "").strip()[:120]

        if not query:
            return empty
        tool_input = {"query": query, "max_results": max_results, "source": "pubmed"}
        out = lit_tool.invoke(tool_input)
        raw_out = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

        if isinstance(out, str):
            data = json.loads(out) if out.strip().startswith("{") else {}
        else:
            data = out if isinstance(out, dict) else {}

        if not data.get("success"):
            return ("", tool_input, raw_out)
        refs_raw = data.get("references", [])

        if isinstance(refs_raw, str):
            try:
                refs = json.loads(refs_raw)
            except json.JSONDecodeError:
                refs = []
        else:
            refs = refs_raw if isinstance(refs_raw, list) else []

        lines = []
        for i, r in enumerate(refs[:5], 1):
            if not isinstance(r, dict):
                continue

            title = r.get("title") or r.get("citation") or "No title"
            authors = r.get("authors") or r.get("author") or ""
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors[:5])
            year = r.get("year") or r.get("published") or ""
            url = r.get("url") or r.get("link") or ""
            lines.append(f"[{i}] {title}. {authors}. {year}. {url}")
        return ("\n".join(lines) if lines else "", tool_input, raw_out)

    except Exception as e:
        _logger.warning("PI literature_search failed: %s", e)
        return empty


def _refine_query_for_search(user_text: str, max_len: int = 80) -> str:
    """Extract short, search-engine-friendly keywords from the user query.

    The previous version naively took the first 3–4 words, which for a
    request like "Search PubMed for recent PETase engineering studies,
    download PDB 5XJH, ..." kept the meta-instruction ("Search PubMed for
    recent") and dropped the real content keywords (PETase, 5XJH,
    ProteinMPNN). Now:

    1. Strip a small set of meta-verbs at the start ("search", "find",
       "look up", "query", "download", "fetch", "get") and the
       sources they reference ("PubMed for", "in PubMed", "from PDB",
       etc.).
    2. Keep biological identifiers (UniProt accessions, PDB IDs,
       all-caps gene-like tokens, EC numbers) verbatim.
    3. Drop stop-words and prepositions.
    4. Cap at max_len chars.
    """
    import re
    t = (user_text or "").strip()
    if not t:
        return ""

    # 1. Strip meta-instruction prefixes
    META_PREFIX = re.compile(
        r"^(please\s+)?(search|find|look\s*(up|for)|query|download|fetch|get|retrieve|use|"
        r"check|analyze|run|predict|examine|investigate)\s+(in\s+|for\s+|from\s+|the\s+)?"
        r"(pubmed|arxiv|semantic\s*scholar|biorxiv|uniprot|pdb|rcsb|alphafold|interpro|string|hpa|brenda|chembl|kegg|ncbi)"
        r"(\s+for\s+|\s+in\s+|\s+by\s+|\s+from\s+|\s+|\s*,\s*)?",
        re.IGNORECASE,
    )
    cleaned = META_PREFIX.sub("", t)
    # also strip ", download PDB ..." trailing meta clauses
    cleaned = re.sub(r",\s*(and\s+)?(use|download|fetch|run|then|use)\s+\S+\s+\S+", " ", cleaned, flags=re.IGNORECASE)

    # 2. Extract biological identifiers verbatim from the ORIGINAL text
    # so PDB IDs / UniProt accessions / EC numbers can never be lost to
    # the meta-prefix stripping above.
    bio_tokens: list[str] = []
    bio_tokens.extend(re.findall(r"\b([1-9][A-Za-z0-9]{3})\b", t))  # PDB IDs
    bio_tokens.extend(re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9])\b", t))  # UniProt
    bio_tokens.extend(re.findall(r"\b(\d+\.\d+\.\d+\.\d+)\b", t))  # EC numbers
    # 3. Word-level cleanup
    STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
        "is", "are", "was", "were", "be", "this", "that", "these", "those",
        "studies", "study", "paper", "papers", "literature", "article", "articles",
        "recent", "latest", "new",
    }
    word_re = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b")
    words = [w for w in word_re.findall(cleaned) if w.lower() not in STOP]
    # Dedup preserving order; prepend bio tokens so they're never dropped
    seen = set()
    out: list[str] = []
    for tok in bio_tokens + words:
        k = tok.upper()
        if k in seen:
            continue
        seen.add(k)
        out.append(tok)
        if sum(len(x) + 1 for x in out) > max_len:
            break
    return " ".join(out)[:max_len]


def _extract_deepsearch_data(data: dict) -> dict:
    """Helper to extract inner data from deepsearch tool wrapped response."""
    if isinstance(data, dict):
        if data.get("status") == "error":
            return {"success": False, "error": data.get("error", {})}
        if data.get("status") == "success" and "content" in data:
            content_val = data["content"]
            if isinstance(content_val, str) and content_val.strip().startswith("{"):
                try:
                    return json.loads(content_val)
                except Exception:
                    pass
            elif isinstance(content_val, dict):
                return content_val
    return data


def _is_search_result_empty(tool_name: str, data: dict) -> bool:
    """True if the search returned success but no usable results (empty refs/results/datasets)."""
    if not data.get("success"):
        return True
    if tool_name == "query_literature_by_keywords":
        refs = data.get("references") or []
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        return not (isinstance(refs, list) and len(refs) > 0)
    if tool_name == "query_web_by_keywords":
        res = data.get("results") or []
        return not (isinstance(res, list) and len(res) > 0) and not (isinstance(data.get("results"), str) and data.get("results"))
    if tool_name == "query_dataset_by_keywords":
        ds = data.get("datasets") or []
        if isinstance(ds, str):
            try:
                ds = json.loads(ds)
            except json.JSONDecodeError:
                return True
        return not (isinstance(ds, list) and len(ds) > 0)
    return True


def _translate_user_query_to_english(user_text: str, llm) -> str:
    """When user asks in non-English, use LLM to translate intent to 2-5 English search keywords."""
    if not llm or not (user_text or "").strip():
        return ""
    try:
        prompt = (
            "Translate this user question into 2-5 short English search keywords for PubMed/scientific literature. "
            "Output ONLY the keywords, nothing else. No quotes, no explanation."
            "\n\nUser question: " + (user_text or "").strip()[:500]
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        out = (resp.content if hasattr(resp, "content") else str(resp)).strip()[:120]
        # Ensure output is ASCII-safe (English); if LLM returned non-ASCII, fall through to empty
        if out and not any(ord(c) > 127 for c in out):
            return out
    except Exception:
        pass
    return ""


def _run_section_search(query: str, max_results: int = None, mode: str = "full") -> tuple:
    """Per-section deep-research: parallel-invoke literature + web sources,
    merge/dedupe, then fall back to structural-DB metadata if empty.

    ``mode``:
      - ``full``: PubMed + S2 + arXiv + bioRxiv + configured web routes
      - ``lite``: PubMed + Semantic Scholar + configured web routes (Expert timeline-friendly)
    """
    import concurrent.futures as _cf

    if max_results is None:
        max_results = SEARCH_MAX_RESULTS
    query = (query or "").strip()[:80]
    if not query:
        return ([], [])
    try:
        from mcp_tools.tools.tools_agent_hub import get_tools
        tools = get_tools()
        tools_dict = {getattr(t, "name", ""): t for t in tools}
    except Exception:
        return ([], [])

    sections: list = []
    logged: list = []

    if (mode or "full").strip().lower() == "lite":
        LITERATURE = ["query_pubmed", "query_semantic_scholar"]
        WEB = ["query_web"]
    else:
        LITERATURE = ["query_pubmed", "query_semantic_scholar", "query_arxiv", "query_biorxiv"]
        WEB = ["query_web"]
    plan = [(tn, {"query": query, "max_results": max_results})
            for tn in LITERATURE + WEB]

    def _invoke(tn, ti):
        tool = tools_dict.get(tn)
        if not tool:
            return (tn, ti, json.dumps({"success": False, "error": "tool not available"}))
        try:
            _logger.info("PI section invoking (parallel): %s with %s", tn, ti)
            out = tool.invoke(ti)
            raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            return (tn, ti, raw)
        except Exception as e:
            return (tn, ti, json.dumps({"success": False, "error": str(e)}))

    results_by_tool: dict = {}
    with _cf.ThreadPoolExecutor(max_workers=min(6, len(plan))) as ex:
        futures = {ex.submit(_invoke, tn, ti): (tn, ti) for tn, ti in plan}
        for fut in _cf.as_completed(futures, timeout=120):
            try:
                tn, ti, raw = fut.result()
                results_by_tool[tn] = raw
                logged.append((tn, ti, raw))
            except Exception:
                pass

    def _parse_refs(raw_str: str) -> list:
        try:
            data = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.strip().startswith("{") else {}
        except Exception:
            return []
        data = _extract_deepsearch_data(data) if isinstance(data, dict) else {}
        if not isinstance(data, dict) or data.get("success") is False:
            return []
        refs = (data.get("references") or data.get("results")
                or data.get("papers") or data.get("datasets") or [])
        if isinstance(refs, str):
            try:
                refs = json.loads(refs) if refs.strip().startswith("[") else []
            except Exception:
                refs = []
        return refs if isinstance(refs, list) else []

    # Merge literature across all 4 sources, dedupe by URL/DOI/PMID/title
    lit_refs: list = []
    for tn in LITERATURE:
        lit_refs.extend(_parse_refs(results_by_tool.get(tn, "")))
    seen = set()
    deduped_lit = []
    for r in lit_refs:
        if isinstance(r, dict):
            k = (r.get("url") or r.get("doi") or r.get("pmid") or r.get("title") or str(id(r)))[:120]
        else:
            k = str(r)[:120]
        if k in seen:
            continue
        seen.add(k)
        deduped_lit.append(r)
    if deduped_lit:
        lines = _format_literature_for_reading(deduped_lit, max_n=max_results * 2, abstract_max=400)
        if lines:
            sections.extend(lines)

    # Merge web across both sources
    web_refs: list = []
    for tn in WEB:
        web_refs.extend(_parse_refs(results_by_tool.get(tn, "")))
    seen = set()
    deduped_web = []
    for r in web_refs:
        if isinstance(r, dict):
            k = (r.get("url") or r.get("link") or r.get("title") or str(id(r)))[:120]
        else:
            k = str(r)[:120]
        if k in seen:
            continue
        seen.add(k)
        deduped_web.append(r)
    if deduped_web:
        lines = _format_web_for_reading(deduped_web, max_n=max_results * 2, snippet_max=300)
        if lines:
            sections.extend(lines)

    # Structural / sequence-database fallback when literature + web returned
    # nothing. Detect identifier patterns in the query (4-char PDB IDs,
    # UniProt accessions, common gene names) and call the corresponding
    # metadata endpoint directly. This is what a senior researcher would do:
    # if no paper exists for "5XJH", at least look up the RCSB metadata.
    if not sections:
        try:
            import re as _re
            db_sections = _pi_database_fallback(query, tools_dict)
            if db_sections:
                sections.extend(db_sections)
                logged.append(("__db_fallback__",
                              {"query": query, "via": "_pi_database_fallback"},
                              json.dumps({"sections_added": len(db_sections)})))
        except Exception as e:
            logged.append(("__db_fallback__", {"query": query},
                          json.dumps({"success": False, "error": str(e)})))

    if not sections:
        return ([], logged)
    return (sections, logged)


def _pi_database_fallback(query: str, tools_dict: dict) -> list:
    """Last-resort fallback: when literature + web search return nothing,
    parse the query for identifier patterns (PDB IDs, UniProt accessions,
    gene names) and call the corresponding structural / functional database
    endpoints directly. Returns a list of formatted text sections.

    Patterns recognized:
    - 4-character PDB ID: e.g. 5XJH, 1ABC → download_rcsb_entry_metadata_by_pdb_id
    - UniProt accession (1 letter + 5-10 alphanumeric): e.g. P04637, A0A0A0A → download_uniprot_meta_by_id
    - All-caps gene-like token (3-12 chars, ≥1 letter): TP53, EGFR, PETase →
      download_uniprot_search_by_query + download_string_map_ids
    """
    import re as _re

    sections: list = []
    seen_ids = set()

    def _try_tool(tool_name: str, tool_input: dict, label_prefix: str) -> str | None:
        tool = tools_dict.get(tool_name)
        if not tool:
            return None
        try:
            out = tool.invoke(tool_input)
            raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            data = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else {}
            if data.get("status") == "error" or data.get("success") is False:
                return None
            # Pull preview content / metadata so SC sees real data
            preview = (data.get("content_preview")
                       or data.get("biological_metadata")
                       or data.get("data")
                       or data)
            preview_str = (json.dumps(preview, ensure_ascii=False)[:2000]
                           if not isinstance(preview, str) else preview[:2000])
            return f"**{label_prefix}**\n{preview_str}"
        except Exception:
            return None

    # PDB IDs
    for pdb_id in _re.findall(r"\b([1-9][A-Za-z0-9]{3})\b", query):
        if pdb_id.upper() in seen_ids:
            continue
        seen_ids.add(pdb_id.upper())
        s = _try_tool(
            "download_rcsb_entry_metadata_by_pdb_id",
            {"pdb_id": pdb_id.upper(), "out_path": f"/tmp/pi_rcsb_{pdb_id.upper()}.json"},
            f"RCSB metadata for PDB {pdb_id.upper()}",
        )
        if s:
            sections.append(s)

    # UniProt accessions
    for up_id in _re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b", query):
        # Above regex returns tuples for the second alternative; collapse
        if isinstance(up_id, tuple):
            up_id = up_id[0]
        if up_id in seen_ids:
            continue
        seen_ids.add(up_id)
        s = _try_tool(
            "download_uniprot_meta_by_id",
            {"uniprot_id": up_id, "out_path": f"/tmp/pi_uniprot_{up_id}.json"},
            f"UniProt metadata for {up_id}",
        )
        if s:
            sections.append(s)

    # Gene-like all-caps tokens (only try if we have NO sections yet — keeps
    # this cheap; gene-symbol lookups can be slow). Try HPA (per-gene record)
    # first because it's a fast REST endpoint that returns rich
    # functional+expression context in one call. STRING mapping is a cheap
    # backup if HPA doesn't have an entry.
    if not sections:
        STOP_TOKENS = {"PDB", "FASTA", "JSON", "API", "HTTP", "DNA", "RNA",
                       "PETase", "EGFR", "STRING", "UNIPROT", "INTERPRO",
                       "HPA", "KEGG", "MMSEQS", "BLAST", "NCBI", "REST"}
        # Note: PETase / EGFR ARE valid gene names — STOP_TOKENS is just to
        # skip obvious non-gene acronyms. We still try them because the
        # gene-lookup just returns empty for non-genes.
        STOP_TOKENS = {"PDB", "FASTA", "JSON", "API", "HTTP", "DNA", "RNA",
                       "STRING", "UNIPROT", "INTERPRO", "HPA", "KEGG",
                       "MMSEQS", "BLAST", "NCBI", "REST"}
        for tok in _re.findall(r"\b([A-Z][A-Z0-9]{2,11})\b", query):
            if tok in seen_ids or tok in STOP_TOKENS:
                continue
            seen_ids.add(tok)
            s = _try_tool(
                "download_hpa_protein_by_gene",
                {"gene_name": tok,
                 "out_path": f"/tmp/pi_hpa_{tok}.json"},
                f"Human Protein Atlas record for gene {tok}",
            )
            if s:
                sections.append(s)
                break  # one gene is enough for context
            # HPA empty → try STRING id mapping (still gives a hit list with
            # functional annotations)
            s2 = _try_tool(
                "download_string_map_ids",
                {"identifiers": tok,
                 "out_dir": "/tmp",
                 "species": 9606,
                 "limit": 3,
                 "echo_query": True,
                 "filename": f"pi_string_{tok}.tsv"},
                f"STRING ID mapping for gene {tok}",
            )
            if s2:
                sections.append(s2)
                break
    return sections


def _fetch_search_for_pi_report(user_text: str, max_results: int = None, llm=None) -> tuple:
    """Deep-research mode: query ALL configured sources in parallel for each
    of three search groups (literature, web, datasets), merge results across
    sources within each group (deduping by URL/DOI/PMID), then fall back to
    structural-database metadata when everything else is empty.

    Returns (combined_str_with_citations, list of (tool_name, inputs, raw_output)).

    Source coverage (was fallback-chain, now merge):
    - Literature: PubMed + Semantic Scholar + arXiv + bioRxiv (4)
    - Web:        configured web routes (2)
    - Datasets:   GitHub + HuggingFace (2)
    - FDA:        query_fda when query mentions a drug-related keyword (1)
    - DB metadata fallback: RCSB/UniProt/HPA/STRING when bio identifiers
      are present in the prompt
    """
    import concurrent.futures as _cf

    if max_results is None:
        max_results = SEARCH_MAX_RESULTS

    # Use English keywords: if user_text has non-ASCII, translate via LLM
    has_non_ascii = any(ord(c) > 127 for c in (user_text or ""))
    query = ""
    if has_non_ascii and llm:
        query = _translate_user_query_to_english(user_text, llm)
    if not query:
        query = _refine_query_for_search(user_text, 80) or (user_text or "").strip()[:80]
    if not query:
        return ("", [])

    try:
        from mcp_tools.tools.tools_agent_hub import get_tools
        tools = get_tools()
        tools_dict = {getattr(t, "name", ""): t for t in tools}

        sections: list = []
        logged: list = []

        def _invoke_one(tool_name: str, tool_input: dict) -> tuple:
            """Invoke a single search tool; return (tool_name, input, raw_str)."""
            tool = tools_dict.get(tool_name)
            if not tool:
                return (tool_name, tool_input, json.dumps({"success": False, "error": "tool not available"}))
            try:
                _logger.info("PI deep-research invoking: %s with %s", tool_name, tool_input)
                out = tool.invoke(tool_input)
                raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                return (tool_name, tool_input, raw)
            except Exception as e:
                _logger.warning("PI %s failed: %s", tool_name, e)
                return (tool_name, tool_input, json.dumps({"success": False, "error": str(e)}))

        def _parse_results(raw_str: str) -> list:
            try:
                data = json.loads(raw_str) if isinstance(raw_str, str) and raw_str.strip().startswith("{") else {}
            except Exception:
                return []
            data = _extract_deepsearch_data(data) if isinstance(data, dict) else {}
            if not isinstance(data, dict) or data.get("success") is False:
                return []
            refs = (data.get("references") or data.get("results")
                    or data.get("papers") or data.get("datasets") or [])
            if isinstance(refs, str):
                try:
                    refs = json.loads(refs) if refs.strip().startswith("[") else []
                except Exception:
                    refs = []
            return refs if isinstance(refs, list) else []

        def _dedupe(items: list, key_fn) -> list:
            seen = set()
            out = []
            for it in items:
                try:
                    k = key_fn(it)
                except Exception:
                    k = id(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        # Build the parallel invocation plan
        LITERATURE = ["query_pubmed", "query_semantic_scholar", "query_arxiv", "query_biorxiv"]
        WEB = ["query_web"]
        DATASETS = ["query_github", "query_hugging_face"]

        # FDA only triggered when query looks drug/clinical-related
        FDA_KEYWORDS = ("drug", "compound", "inhibitor", "small molecule", "clinical",
                        "FDA", "approved", "trial", "adverse", "antibody", "biologic")
        plan: list[tuple[str, dict]] = []
        for tn in LITERATURE + WEB + DATASETS:
            plan.append((tn, {"query": query, "max_results": max_results}))
        if any(kw.lower() in user_text.lower() for kw in FDA_KEYWORDS) and "query_fda" in tools_dict:
            plan.append(("query_fda", {"query": query, "max_results": max_results}))

        # Parallel execution — bounded concurrency to be polite to upstream
        # rate limits but not so slow that everything is serialized.
        results_by_tool: dict[str, str] = {}
        with _cf.ThreadPoolExecutor(max_workers=min(6, len(plan))) as ex:
            futures = {ex.submit(_invoke_one, tn, ti): (tn, ti) for tn, ti in plan}
            for fut in _cf.as_completed(futures, timeout=120):
                try:
                    tn, ti, raw = fut.result()
                    results_by_tool[tn] = raw
                    logged.append((tn, ti, raw))
                except _cf.TimeoutError:
                    pass
                except Exception:
                    pass

        # Merge literature across PubMed + S2 + arXiv + bioRxiv
        lit_refs: list = []
        for tn in LITERATURE:
            lit_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        lit_refs = _dedupe(lit_refs, lambda r: (
            (r.get("url") or r.get("doi") or r.get("pmid") or r.get("title") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if lit_refs:
            lines = _format_literature_citations(lit_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive = [t for t in LITERATURE if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive) or "literature"
                sections.append(f"**Literature ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # Merge web across configured web routes
        web_refs: list = []
        for tn in WEB:
            web_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        web_refs = _dedupe(web_refs, lambda r: (
            (r.get("url") or r.get("link") or r.get("title") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if web_refs:
            lines = _format_web_citations(web_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive_web = [t for t in WEB if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive_web) or "web"
                sections.append(f"**Web ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # Merge datasets across GitHub + HuggingFace
        ds_refs: list = []
        for tn in DATASETS:
            ds_refs.extend(_parse_results(results_by_tool.get(tn, "")))
        ds_refs = _dedupe(ds_refs, lambda r: (
            (r.get("url") or r.get("html_url") or r.get("id") or r.get("name") or "")[:120]
            if isinstance(r, dict) else str(r)[:120]
        ))
        if ds_refs:
            lines = _format_dataset_citations(ds_refs[: max_results * 2], max_n=max_results * 2)
            if lines:
                productive_ds = [t for t in DATASETS if _parse_results(results_by_tool.get(t, ""))]
                source_tag = ", ".join(t.replace("query_", "") for t in productive_ds) or "datasets"
                sections.append(f"**Datasets ({source_tag}; cite as [1], [2], ...)**\n" + "\n".join(lines))

        # FDA (when triggered)
        if "query_fda" in results_by_tool:
            fda_refs = _parse_results(results_by_tool["query_fda"])
            if fda_refs:
                fda_lines = []
                for r in fda_refs[:max_results]:
                    if isinstance(r, dict):
                        nm = r.get("title") or r.get("name") or r.get("brand_name") or "FDA record"
                        url = r.get("url") or r.get("link") or ""
                        fda_lines.append(f"- [{nm}]({url})" if url else f"- {nm}")
                if fda_lines:
                    sections.append("**FDA (drug/clinical context)**\n" + "\n".join(fda_lines))

        # Structural / functional database fallback: when literature + web
        # + dataset search all returned nothing useful, parse the ORIGINAL
        # user query (not the truncated keyword form) for identifier
        # patterns and call the corresponding metadata endpoint. This is
        # what gets the report substantive content for prompts like
        # "Search for PDB 5XJH" where no paper indexes the PDB ID directly.
        if not sections:
            try:
                db_sections = _pi_database_fallback(user_text, tools_dict)
                if db_sections:
                    sections.append(
                        "**Database metadata (cite as [1], [2], ...)**\n"
                        + "\n\n".join(db_sections)
                    )
                    logged.append((
                        "__db_fallback__",
                        {"query": user_text},
                        json.dumps({"sections_added": len(db_sections)}),
                    ))
            except Exception as e:
                _logger.warning("PI DB fallback failed: %s", e)
                logged.append((
                    "__db_fallback__",
                    {"query": user_text},
                    json.dumps({"success": False, "error": str(e)}),
                ))

        if not sections:
            return ("No search results.", logged)
        intro = "References from search (use [1], [2], etc. in your report):\n\n"
        return (intro + "\n\n".join(sections), logged)
    except Exception as e:
        _logger.warning("PI report search failed: %s", e)
        return ("", [])
