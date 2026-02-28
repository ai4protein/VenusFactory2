"""
InterPro family proteins: by UniProt (annotation), by InterPro ID (protein list, UniProt ID list).
Query returns JSON; download saves to file.
"""
import json
import os
import ssl
import sys
import time
import argparse
import requests
from time import sleep
from tqdm import tqdm
from urllib import request as url_request
from urllib.error import HTTPError

try:
    from fake_useragent import UserAgent
    ua = UserAgent()
except Exception:
    ua = None


# ----- By UniProt ID: annotation (InterPro entries + GO) -----

def query_interpro_by_uniprot(uniprot_id: str) -> str:
    """Query InterPro entries and GO annotations for a UniProt ID. Returns JSON string. No file save."""
    url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/{uniprot_id}/entry/?extra_fields=counters&page_size=100"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return json.dumps({
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Error during API call for {uniprot_id}: {str(e)}"
        }, indent=4)

    metadata = data.get("metadata", {})
    interpro_entries = data.get("entries", [])
    result = {
        "success": True,
        "uniprot_id": uniprot_id,
        "basic_info": {
            "uniprot_id": metadata.get("accession", ""),
            "protein_name": metadata.get("name", ""),
            "length": metadata.get("length", 0),
            "gene_name": metadata.get("gene", ""),
            "organism": metadata.get("source_organism", {}),
            "source_database": metadata.get("source_database", ""),
            "in_alphafold": metadata.get("in_alphafold", False),
        },
        "interpro_entries": interpro_entries,
        "go_annotations": {"molecular_function": [], "biological_process": [], "cellular_component": []},
        "counters": metadata.get("counters", {}),
        "num_entries": len(interpro_entries),
    }
    if "go_terms" in metadata:
        for go_term in metadata["go_terms"]:
            category_name = go_term.get("category", {}).get("name", "")
            go_annotation = {"go_id": go_term.get("identifier", ""), "name": go_term.get("name", "")}
            if category_name == "molecular_function":
                result["go_annotations"]["molecular_function"].append(go_annotation)
            elif category_name == "biological_process":
                result["go_annotations"]["biological_process"].append(go_annotation)
            elif category_name == "cellular_component":
                result["go_annotations"]["cellular_component"].append(go_annotation)
    return json.dumps(result, indent=4)


def download_interpro_by_uniprot(uniprot_id: str, out_dir: str) -> str:
    """Download InterPro annotation by UniProt ID. Saves JSON file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}_interpro.json")
    text = query_interpro_by_uniprot(uniprot_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return f"{uniprot_id} failed: {data.get('error_message', 'unknown')}"
    except json.JSONDecodeError:
        return f"{uniprot_id} failed: invalid response"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"{uniprot_id} interpro annotation downloaded"


# ----- By InterPro ID: protein list (reviewed) -----

def _fetch_proteins_page(url: str) -> list:
    data_list = []
    while url:
        response = requests.get(url, timeout=30)
        data = response.json()
        data_list.extend(data.get("results", []))
        url = data.get("next")
        if url:
            time.sleep(1)
    return data_list


def query_interpro_proteins(interpro_id: str, page_size: int = 200) -> str:
    """Query protein list for an InterPro ID. Returns JSON string. No file save."""
    url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/{interpro_id}/?extra_fields=counters&page_size={page_size}"
    try:
        data_list = _fetch_proteins_page(url)
        meta = {"accession": interpro_id, "num_proteins": len(data_list)}
        return json.dumps({"metadata": meta, "results": data_list}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "interpro_id": interpro_id, "error": str(e)})


def download_interpro_proteins(interpro_id: str, out_dir: str) -> str:
    """Download family proteins for an InterPro ID: detail.json, meta.json, uids.txt. Returns message string."""
    interpro_dir = os.path.join(out_dir, interpro_id)
    os.makedirs(interpro_dir, exist_ok=True)

    detail_path = os.path.join(interpro_dir, "detail.json")
    if os.path.exists(detail_path):
        return f"Skipping {interpro_id}, already exists"

    url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/{interpro_id}/?extra_fields=counters&page_size=20"
    try:
        info_data = _fetch_proteins_page(url)
    except Exception:
        return f"Error downloading {interpro_id}"

    if not info_data:
        return f"No data found for {interpro_id}"

    with open(detail_path, "w") as f:
        json.dump(info_data, f)

    meta_data = {"metadata": {"accession": interpro_id}, "num_proteins": len(info_data)}
    with open(os.path.join(interpro_dir, "meta.json"), "w") as f:
        json.dump(meta_data, f)

    uids = [d["metadata"]["accession"] for d in info_data]
    with open(os.path.join(interpro_dir, "uids.txt"), "w") as f:
        f.write("\n".join(uids))

    return f"Successfully downloaded {interpro_id}"


def download_single_interpro(interpro_id: str, out_dir: str) -> str:
    """Alias for download_interpro_proteins (backward compatibility)."""
    return download_interpro_proteins(interpro_id, out_dir)


def fetch_info_data(url: str) -> list:
    """Legacy alias for _fetch_proteins_page."""
    return _fetch_proteins_page(url)


# ----- By InterPro ID: UniProt ID list (paginated, optional filter) -----

def _fetch_uniprot_list_urllib(base_url: str, page_size: int = 200):
    """Fetch all UniProt accessions for an InterPro entry via pagination (urllib)."""
    context = ssl._create_unverified_context()
    next_url = base_url
    names = []
    attempts = 0
    headers = {"Accept": "application/json"}
    if ua:
        headers["user-agent"] = ua.random
    while next_url:
        try:
            req = url_request.Request(next_url, headers=headers)
            res = url_request.urlopen(req, context=context)
            if getattr(res, "status", res.getcode()) == 408:
                sleep(61)
                continue
            if getattr(res, "status", res.getcode()) == 204:
                break
            payload = json.loads(res.read().decode())
            res.close()
            next_url = payload.get("next")
            attempts = 0
            for item in payload.get("results", []):
                names.append(item["metadata"]["accession"])
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            if attempts < 3:
                attempts += 1
                sleep(61)
                continue
            raise e
    return list(set(names))


def query_interpro_uniprot_list(interpro_id: str, filter_name: str = None, page_size: int = 200) -> str:
    """Query UniProt ID list for an InterPro entry. Returns JSON string. No file save."""
    if filter_name:
        base_url = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{interpro_id}/{filter_name}/?page_size={page_size}"
    else:
        base_url = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{interpro_id}/?page_size={page_size}"
    try:
        names = _fetch_uniprot_list_urllib(base_url, page_size)
        return json.dumps({"interpro_id": interpro_id, "accessions": names, "count": len(names)})
    except Exception as e:
        return json.dumps({"success": False, "interpro_id": interpro_id, "error": str(e)})


def download_interpro_uniprot_list(
    interpro_id: str,
    out_dir: str,
    protein_name: str = "",
    chunk_size: int = 5000,
    filter_name: str = None,
    page_size: int = 200,
    re_collect: bool = False,
) -> str:
    """Download UniProt ID list for an InterPro entry: chunked txt files. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    if filter_name:
        base_url = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{interpro_id}/{filter_name}/?page_size={page_size}"
    else:
        base_url = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{interpro_id}/?page_size={page_size}"

    if re_collect:
        for f in os.listdir(out_dir):
            if f.startswith(f"af_raw_{protein_name or interpro_id}_") and f.endswith(".txt"):
                try:
                    os.remove(os.path.join(out_dir, f))
                except OSError:
                    pass

    names = _fetch_uniprot_list_urllib(base_url, page_size)
    length = len(names)
    max_i = (length // chunk_size) + 1 if chunk_size else 1
    prefix = protein_name or interpro_id
    for i in range(max_i):
        chunk = names[i * chunk_size : (i + 1) * chunk_size] if chunk_size else names
        out_file = os.path.join(out_dir, f"af_raw_{prefix}_{i}.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(chunk))
    return f"Successfully downloaded {length} accessions for {interpro_id}"


def output_list(args) -> None:
    """Legacy CLI entry: download_interpro_uniprot_list with argparse namespace."""
    download_interpro_uniprot_list(
        args.protein,
        args.output,
        protein_name=args.protein_name,
        chunk_size=args.chunk_size,
        filter_name=args.filter_name or None,
        page_size=args.page_size,
        re_collect=args.re_collect,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterPro family proteins: by-uniprot annotation, protein list, or uniprot list")
    parser.add_argument("--interpro_id", type=str, default=None)
    parser.add_argument("--interpro_json", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="download/interpro_domain")
    parser.add_argument("--error_file", type=str, default=None)
    parser.add_argument("--chunk_num", type=int, default=None)
    parser.add_argument("--chunk_id", type=int, default=None)
    sub = parser.add_subparsers(dest="cmd")

    p_anno = sub.add_parser("by_uniprot", help="Query/download annotation by UniProt ID")
    p_anno.add_argument("--uniprot_id", required=True)
    p_anno.add_argument("--output", help="If set, download to this path")

    p_prot = sub.add_parser("proteins", help="Query/download protein list by InterPro ID")
    p_prot.add_argument("--interpro_id")
    p_prot.add_argument("--interpro_json")
    p_prot.add_argument("--out_dir", default="download/interpro_domain")
    p_prot.add_argument("--error_file")
    p_prot.add_argument("--chunk_num", type=int)
    p_prot.add_argument("--chunk_id", type=int)

    p_list = sub.add_parser("uniprot_list", help="Download UniProt ID list for InterPro entry")
    p_list.add_argument("--protein", default="IPR001557")
    p_list.add_argument("--protein_name", default="MDH")
    p_list.add_argument("--chunk_size", type=int, default=5000)
    p_list.add_argument("--filter_name", default="")
    p_list.add_argument("--page_size", type=int, default=200)
    p_list.add_argument("--output", default="data/MDH")
    p_list.add_argument("--re_collect", action="store_true")

    args = parser.parse_args()

    if args.cmd is None and (getattr(args, "interpro_id", None) or getattr(args, "interpro_json", None)):
        args.cmd = "proteins"
    if args.cmd is None:
        parser.error("Missing subcommand: by_uniprot, proteins, or uniprot_list")

    if args.cmd == "by_uniprot":
        if getattr(args, "output", None):
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            text = query_interpro_by_uniprot(args.uniprot_id)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved to {args.output}")
        else:
            print(query_interpro_by_uniprot(args.uniprot_id))

    elif args.cmd == "proteins":
        if not getattr(args, "interpro_id", None) and not getattr(args, "interpro_json", None):
            print("Error: Must provide either interpro_id or interpro_json")
            sys.exit(1)
        errors, messages = [], []
        if args.interpro_id:
            msg = download_interpro_proteins(args.interpro_id, args.out_dir)
            print(msg)
            if "Error" in msg or "No data" in msg:
                errors.append(args.interpro_id)
                messages.append(msg)
        else:
            with open(args.interpro_json) as f:
                all_data = json.load(f)
            if args.chunk_num is not None and args.chunk_id is not None:
                start = args.chunk_id * len(all_data) // args.chunk_num
                end = (args.chunk_id + 1) * len(all_data) // args.chunk_num
                all_data = all_data[start:end]
            for data in tqdm(all_data):
                interpro_id = data["metadata"]["accession"]
                msg = download_interpro_proteins(interpro_id, args.out_dir)
                if "Error" in msg or "No data" in msg:
                    errors.append(interpro_id)
                    messages.append(msg)
        if errors and getattr(args, "error_file", None):
            d = os.path.dirname(args.error_file)
            os.makedirs(d, exist_ok=True)
            with open(args.error_file, "w") as f:
                for p, m in zip(errors, messages):
                    f.write(f"{p} - {m}\n")

    elif args.cmd == "uniprot_list":
        download_interpro_uniprot_list(
            args.protein,
            args.output,
            protein_name=args.protein_name,
            chunk_size=args.chunk_size,
            filter_name=args.filter_name or None,
            page_size=args.page_size,
            re_collect=args.re_collect,
        )
        print(f"UniProt list saved under {args.output}")
