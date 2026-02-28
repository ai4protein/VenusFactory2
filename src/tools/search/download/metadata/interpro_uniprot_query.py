"""
InterPro/GO query by UniProt ID (protein-centric). For InterPro-entry-centric download see download_interpro.py.
"""
import json
import requests


def query_interpro_by_uniprot(uniprot_id: str) -> str:
    """Fetch InterPro entries and GO annotations for a single UniProt ID. Returns JSON string."""
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
