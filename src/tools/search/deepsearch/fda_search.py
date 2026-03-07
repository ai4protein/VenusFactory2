from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, List

try:
    from .fda import FDAQuery
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "deepsearch" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.search.deepsearch.fda import FDAQuery

try:
    from .response_utils import error_response, query_success_response
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "deepsearch" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.search.deepsearch.response_utils import error_response, query_success_response


def _events_to_items(
    events: List[dict],
    query: str,
    max_content_length: int,
) -> List[dict[str, Any]]:
    """Map drug/event API results to same shape as label results (source, title, abstract, etc.)."""
    out: List[dict[str, Any]] = []
    for entry in events:
        patient = entry.get("patient") or {}
        drugs = patient.get("drug") or []
        reactions = patient.get("reaction") or []
        drug_name = ""
        indications: List[str] = []
        for d in drugs:
            names = d.get("medicinalproduct") or []
            if names:
                drug_name = drug_name or (names[0] if isinstance(names[0], str) else str(names[0]))
            ind = d.get("drugindication") or []
            if ind:
                indications.extend([x if isinstance(x, str) else str(x) for x in ind])
        reaction_strs: List[str] = []
        for r in reactions:
            pt = r.get("reactionmeddrapt")
            if isinstance(pt, list):
                reaction_strs.extend(str(x) for x in pt if x is not None)
            elif pt is not None:
                reaction_strs.append(str(pt))
        title = drug_name or f"Adverse event: {query}"
        abstract_parts = []
        if indications:
            abstract_parts.append("Indication: " + "; ".join(indications[:3]))
        if reaction_strs:
            abstract_parts.append("Reactions: " + "; ".join(reaction_strs[:5]))
        if not abstract_parts:
            abstract_parts.append("FAERS adverse event report.")
        abstract = " ".join(abstract_parts).replace("\n", " ")
        if max_content_length and len(abstract) > max_content_length:
            abstract = abstract[:max_content_length] + "..."
        receivedate = entry.get("receivedate", "")
        if len(receivedate) == 8 and receivedate.isdigit():
            receivedate = f"{receivedate[:4]}-{receivedate[4:6]}-{receivedate[6:]}"
        out.append({
            "source": "fda_event",
            "title": title,
            "url": "https://open.fda.gov/apis/drug/event/",
            "authors": [],
            "published": receivedate,
            "published_date": receivedate,
            "abstract": abstract,
        })
    return out


def _fda_search(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> List[dict[str, Any]]:
    """
    Search FDA drug data: try drug/label first, then drug/event (adverse events) as fallback.
    Uses openFDA API; see src/agent/skills/fda/references/drugs.md for search syntax.

    Args:
        query: Search query string (e.g. drug name).
        max_results: Maximum number of results to return.
        max_content_length: Max length for text fields (truncate if longer).

    Returns:
        List of dicts with keys: source, title, url, authors, published, abstract.
    """
    out: List[dict[str, Any]] = []
    fda = FDAQuery()
    query_clean = query.strip()
    query_upper = query_clean.upper()

    # 1) Drug label (drug/label): per drugs.md use openfda.brand_name / openfda.generic_name
    label_searches = [
        f"openfda.generic_name:{query_clean}+OR+openfda.brand_name:{query_clean}",
        f"openfda.generic_name:{query_upper}+OR+openfda.brand_name:{query_upper}",
        f"openfda.generic_name:*{query_clean}*+OR+openfda.brand_name:*{query_clean}*",
        f"purpose:*{query_clean}*",
        f"indications_and_usage:*{query_clean}*",
    ]
    resp: dict = {"results": []}
    for search_str in label_searches:
        resp = fda.query("drug", "label", search=search_str, limit=max_results)
        if "error" in resp:
            continue
        if resp.get("results"):
            break
    else:
        resp = resp if "results" in resp else {"results": []}

    if "results" in resp:
        for entry in resp["results"]:
            openfda = entry.get("openfda", {})
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            title = ""
            if brand_names:
                title = brand_names[0]
            elif generic_names:
                title = generic_names[0]
            if not title:
                continue
            abstract_parts = []
            if "description" in entry and entry["description"]:
                abstract_parts.append(entry["description"][0])
            elif "indications_and_usage" in entry and entry["indications_and_usage"]:
                abstract_parts.append(entry["indications_and_usage"][0])
            abstract = " ".join(abstract_parts).strip().replace("\n", " ")
            if max_content_length and len(abstract) > max_content_length:
                abstract = abstract[:max_content_length] + "..."
            published = entry.get("effective_time", "")
            if len(published) == 8 and published.isdigit():
                published = f"{published[:4]}-{published[4:6]}-{published[6:]}"
            authors = openfda.get("manufacturer_name", [])
            url = ""
            app_nums = openfda.get("application_number", [])
            if app_nums:
                url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_nums[0].replace('NDA', '').replace('ANDA', '').replace('BLA', '')}"
            out.append({
                "source": "fda",
                "title": title.title(),
                "url": url,
                "authors": authors,
                "published": published,
                "published_date": published,
                "abstract": abstract,
            })

    # 2) Fallback: drug adverse events (drug/event), per drugs.md patient.drug.medicinalproduct
    if not out:
        event_resp = fda.query_drug_events(query_clean, limit=max_results)
        if event_resp.get("results"):
            out = _events_to_items(
                event_resp["results"], query_clean, max_content_length
            )[:max_results]

    return out


def query_fda(query: str, max_results: int = 5, max_content_length: int = 10000) -> str:
    """
    Search FDA drug labeling database.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        results = _fda_search(query, max_results, max_content_length)
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="fda")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or FDA API key.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FDA Drug Label Search")
    parser.add_argument("--query", type=str, default="aspirin", help="Drug name query")
    parser.add_argument("--max_results", type=int, default=5, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_fda(query='{args.query}', max_results={args.max_results}) ===")
    res = query_fda(args.query, max_results=args.max_results)
    print(res)

