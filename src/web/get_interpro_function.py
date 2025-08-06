import requests
import time
import json
import os
import argparse

def generate_interpro_ai_summary(interpro_data: dict, api_key: str = None) -> str:
    """Generate a concise AI summary for InterPro query results using a shorter prompt."""
    try:
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return "❌ No API key found. Please set the DEEPSEEK_API_KEY environment variable or provide the API key in the interface."
        
        # 1. Consolidate the essential data into a single dictionary
        summary_data = {
            "protein_info": interpro_data.get('basic_info', {}),
            "interpro_summary": interpro_data.get('interpro_entries', {}),
            "go_annotations": interpro_data.get('go_annotations', {})
        }
        
        # 2. Create the new, shorter prompt
        prompt = f"""
As a bioinformatics expert, provide a comprehensive functional analysis of the protein based on the following data.
Focus on its main function, key domains/families, biological roles, and cellular location.
Output a complete analysis, two to three paragraphs, do not output with formatting.

### Protein Data
{json.dumps(summary_data, indent=2)}
"""
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a protein bioinformatics expert who provides clear, structured analysis of protein data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        # Call AI API with a reasonable timeout
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=90  # Using a longer timeout for AI generation
        )
        
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
            
    except requests.exceptions.HTTPError as http_err:
        return f"❌ AI API call failed: {http_err.response.status_code} - {http_err.response.text}"
    except Exception as e:
        return f"❌ Error generating InterPro AI summary: {str(e)}"



def download_single_interpro(uniprot_id):
    """
    Fetches InterPro entries and GO annotations for a single UniProt ID.
    """
    url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/{uniprot_id}/entry/?extra_fields=counters&page_size=100"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
    except Exception as e:
        return {
            "success": False,
            "uniprot_id": uniprot_id,
            "error_message": f"Error during API call for {uniprot_id}: {str(e)}"
        }

    # --- Start of Corrected Logic ---
    
    # Directly get metadata and entries from the top-level of the response
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
            "in_alphafold": metadata.get("in_alphafold", False)
        },
        "interpro_entries": interpro_entries,
        "go_annotations": {
            "molecular_function": [],
            "biological_process": [],
            "cellular_component": []
        },
        "counters": metadata.get("counters", {}),
        "num_entries": len(interpro_entries)
    }

    # Process GO terms from the metadata
    if "go_terms" in metadata:
        for go_term in metadata["go_terms"]:
            category_name = go_term.get("category", {}).get("name", "")
            go_annotation = {
                "go_id": go_term.get("identifier", ""),
                "name": go_term.get("name", "")
            }
            
            if category_name == "molecular_function":
                result["go_annotations"]["molecular_function"].append(go_annotation)
            elif category_name == "biological_process":
                result["go_annotations"]["biological_process"].append(go_annotation)
            elif category_name == "cellular_component":
                result["go_annotations"]["cellular_component"].append(go_annotation)

    return json.dumps(result, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download InterPro and GO data for a UniProt ID.")
    parser.add_argument("--uniprot_id", type=str, required=True, help="UniProt ID (e.g., P00734)")
    args = parser.parse_args()
    
    result = download_single_interpro(args.uniprot_id)
    print(result)