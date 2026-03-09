"""
FDA search: openFDA drug events, drug recalls, device events.
CLI: python src/tools/search/source/fda_search.py --query "..." [--max_results 10] [--source drug_events,drug_recalls]
"""
import json
import argparse

try:
    from src.tools.search.deepsearch.fda import FDAQuery
except ImportError:
    from ..deepsearch.fda import FDAQuery


def _normalize_drug_event(record: dict, index: int) -> dict:
    """Turn openFDA drug event record into unified item with title, url, abstract, source."""
    title = ""
    if isinstance(record.get("patient"), dict) and record["patient"].get("drug"):
        drugs = record["patient"]["drug"]
        if isinstance(drugs, list) and drugs:
            names = [d.get("medicinalproduct") or d.get("openfda", {}).get("brand_name", [""])[0] for d in drugs[:3]]
            title = "; ".join(str(n) for n in names if n)
    if not title:
        title = record.get("safetyreportid") or f"Event {index + 1}"
    received = record.get("receivedate") or ""
    report_id = record.get("safetyreportid") or ""
    url = f"https://api.fda.gov/drug/event.json?search=safetyreportid:{report_id}" if report_id else ""
    abstract = f"Received: {received}. Report ID: {report_id}."
    return {"source": "drug_events", "title": title, "url": url, "abstract": abstract, "raw": record}


def _normalize_recall(record: dict, index: int) -> dict:
    """Turn openFDA recall/enforcement record into unified item."""
    title = record.get("product_description") or record.get("recall_number") or f"Recall {index + 1}"
    url = "https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
    reason = record.get("reason_for_recall") or ""
    date = record.get("report_date") or record.get("recall_initiation_date") or ""
    abstract = f"Date: {date}. {reason[:300]}" if reason else f"Date: {date}"
    return {"source": "drug_recalls", "title": title, "url": url, "abstract": abstract, "raw": record}


def _normalize_device_event(record: dict, index: int) -> dict:
    """Turn openFDA device event into unified item."""
    dev = record.get("device") or []
    brand = (dev[0].get("brand_name") if isinstance(dev, list) and dev else "") or ""
    mfr = (dev[0].get("manufacturer_d_name") if isinstance(dev, list) and dev else "") or ""
    title = brand or mfr or record.get("report_number") or f"Device event {index + 1}"
    report_id = record.get("report_number") or ""
    url = f"https://api.fda.gov/device/event.json?search=report_number:{report_id}" if report_id else ""
    date = record.get("date_received") or ""
    abstract = f"Date: {date}. Report: {report_id}."
    return {"source": "device_events", "title": title, "url": url, "abstract": abstract, "raw": record}


def fda_search(
    query: str,
    max_results: int = 10,
    source: str = "drug_events",
) -> str:
    """
    Search openFDA: drug adverse events, drug recalls, or device events.
    source: comma-separated, e.g. drug_events, drug_recalls, device_events.
    Returns JSON string of unified results (source, title, url, abstract).
    """
    fda = FDAQuery()
    all_results = []
    sources = [s.strip().lower() for s in source.split(",") if s.strip()]

    if "drug_events" in sources:
        try:
            data = fda.query_drug_events(drug_name=query, limit=max_results)
            if isinstance(data, dict) and "results" in data:
                for i, rec in enumerate(data["results"][:max_results]):
                    all_results.append(_normalize_drug_event(rec, i))
        except Exception:
            pass

    if "drug_recalls" in sources:
        try:
            data = fda.query_drug_recalls(drug_name=query, limit=max_results)
            if isinstance(data, dict) and "results" in data:
                for i, rec in enumerate(data["results"][:max_results]):
                    all_results.append(_normalize_recall(rec, i))
        except Exception:
            pass

    if "device_events" in sources:
        try:
            data = fda.query_device_events(device_name=query, limit=max_results)
            if isinstance(data, dict) and "results" in data:
                for i, rec in enumerate(data["results"][:max_results]):
                    all_results.append(_normalize_device_event(rec, i))
        except Exception:
            pass

    # Deduplicate by title
    seen = set()
    final = []
    for r in all_results:
        key = (r.get("title", "").strip().lower(), r.get("source", ""))
        if key in seen:
            continue
        seen.add(key)
        final.append({k: v for k, v in r.items() if k != "raw"})
    return json.dumps(final[:max_results * 2], ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDA search (drug events, drug recalls, device events)")
    parser.add_argument("--query", type=str, default="aspirin", help="Drug or device name to search")
    parser.add_argument("--max_results", type=int, default=10, help="Max results per source")
    parser.add_argument("--source", type=str, default="drug_events,drug_recalls",
                        help="Comma-separated: drug_events, drug_recalls, device_events")
    args = parser.parse_args()
    result = fda_search(args.query, max_results=args.max_results, source=args.source)
    print(result)
