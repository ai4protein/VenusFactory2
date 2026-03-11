"""
HPA (Human Protein Atlas) Standard Validation Script

Combined set of generic and representative proteins.
Pure assertions, no compensation/patching logic in the test script.

Proteins:
1. INS     (Secreted / Pancreas)
2. IL6     (Secreted / Immune)
3. ALB     (Secreted / Liver)
4. MS4A1   (Membrane / B-cell)
5. CD3E    (Membrane / T-cell)
6. PTPRC   (Membrane / Leukocyte)
7. TP53    (Intracellular / Nuclear)
8. LMNA    (Intracellular / Nuclear Lamina)
9. GFAP    (Intracellular / Astrocyte)
10. GAPDH  (Intracellular / Housekeeping)
11. EPCAM  (Membrane / Epithelial marker)
12. MKI67  (Intracellular / Proliferation marker)
13. ACTA2  (Intracellular / Smooth muscle marker)
14. COL1A1 (Secreted / Extracellular matrix)
15. PECAM1 (Membrane / Endothelial marker)

Usage:
    python src/tools/database/hpa/test_hpa.py
"""

import json
import os
import sys
import tempfile
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from src.tools.database.hpa.hpa_operations import (
    download_hpa_protein_by_gene,
    download_hpa_subcellular_location_by_gene,
    download_hpa_tissue_expression_by_gene,
    download_hpa_single_cell_type_by_gene,
    download_hpa_blood_expression_by_gene,
)

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg): print(f"  {GREEN}[OK]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL] {msg}{RESET}")
def info(msg): print(f"       {CYAN}{msg}{RESET}")

def call(fn, gene, tmp) -> dict:
    raw = fn(gene, tmp)
    return json.loads(raw)

def assert_success(r, gene, fn_name) -> bool:
    if r["status"] != "success":
        fail(f"{fn_name}({gene}) returned error: {r.get('error')}")
        return False
    ok(f"{fn_name}({gene})  ->  status=success  [{r['execution_context']['download_time_ms']}ms]")
    return True

def check_meta(meta, key, keyword, gene):
    if keyword is None: return True
    val = meta.get(key)
    if val is None:
        fail(f"{key} is None - expected '{keyword}'")
        return False
        
    val_str = str(val).lower()
    match = re.search(keyword.lower(), val_str)
    
    if match:
        ok(f"{key} contains '{keyword}'")
        return True
    else:
        fail(f"{key} does not contain '{keyword}' (got: {str(val)[:100]})")
        return False

TEST_CASES = [
    {"gene": "INS", "desc": "Insulin (Secreted)", "localization_type": "Secreted", "tissue_checks": {"top_expressed_tissues": "pancreas"}},
    {"gene": "IL6", "desc": "Interleukin-6 (Secreted)", "localization_type": "Secreted", "tissue_checks": {}},
    {"gene": "ALB", "desc": "Albumin (Secreted)", "localization_type": "Secreted", "tissue_checks": {"top_expressed_tissues": "liver"}},
    {"gene": "MS4A1", "desc": "CD20 (Membrane)", "localization_type": "Membrane", "sc_checks": {"top_cell_types": "b-cells"}},
    {"gene": "CD3E", "desc": "CD3E (Membrane)", "localization_type": "Membrane", "sc_checks": {"top_cell_types": "t-cells"}},
    {"gene": "PTPRC", "desc": "CD45 (Membrane)", "localization_type": "Membrane", "sc_checks": {}},
    {"gene": "TP53", "desc": "TP53 (Intracellular)", "localization_type": "Intracellular", "tissue_checks": {}},
    {"gene": "LMNA", "desc": "Lamin A/C (Intracellular)", "localization_type": "Intracellular", "tissue_checks": {}},
    {"gene": "GFAP", "desc": "GFAP (Intracellular)", "localization_type": "Intracellular", "tissue_checks": {"top_expressed_tissues": "brain"}},
    {"gene": "GAPDH", "desc": "GAPDH (Intracellular)", "localization_type": "Intracellular", "tissue_checks": {"rna_tissue_distribution": "detected in all"}},
    {"gene": "EPCAM", "desc": "EPCAM (Membrane/Epithelial)", "localization_type": "Membrane", "sc_checks": {"top_cell_types": "epithelial"}},
    {"gene": "MKI67", "desc": "Ki-67 (Intracellular/Cyc)", "localization_type": "Intracellular", "loc_checks": {"subcellular_location": "nucleoplasm"}},
    {"gene": "ACTA2", "desc": "alpha-SMA (Intracellular/Mus)", "localization_type": "Intracellular", "sc_checks": {"top_cell_types": "smooth muscle"}},
    {"gene": "COL1A1", "desc": "Collagen Type I (Secreted)", "localization_type": "Secreted", "tissue_checks": {}},
    {"gene": "PECAM1", "desc": "CD31 (Membrane/Endo)", "localization_type": "Membrane", "sc_checks": {"top_cell_types": "endothelial"}},
]

def run_all():
    tmp = tempfile.mktemp(suffix=".json")
    total, passed = 0, 0

    for tc in TEST_CASES:
        gene = tc["gene"]
        print(f"\n{BOLD}Gene: {gene}  --  {tc['desc']}{RESET}")

        r = call(download_hpa_protein_by_gene, gene, tmp)
        total += 1
        if assert_success(r, gene, "protein"): passed += 1

        r2 = call(download_hpa_tissue_expression_by_gene, gene, tmp)
        total += 1
        if assert_success(r2, gene, "tissue"):
            passed += 1
            meta = r2.get("biological_metadata", {})
            for k, kw in tc.get("tissue_checks", {}).items():
                total += 1
                if check_meta(meta, k, kw, gene): passed += 1

        r3 = call(download_hpa_subcellular_location_by_gene, gene, tmp)
        total += 1
        if assert_success(r3, gene, "subcell"):
            passed += 1
            meta = r3.get("biological_metadata", {})
            expect = tc.get("localization_type")
            if expect:
                total += 1
                if meta.get("localization_type") == expect:
                    ok(f"localization_type == '{expect}'")
                    passed += 1
                else:
                    fail(f"localization_type != '{expect}' (got: {meta.get('localization_type')})")
            for k, kw in tc.get("loc_checks", {}).items():
                total += 1
                if check_meta(meta, k, kw, gene): passed += 1

        r4 = call(download_hpa_single_cell_type_by_gene, gene, tmp)
        total += 1
        if assert_success(r4, gene, "sc"):
            passed += 1
            meta = r4.get("biological_metadata", {})
            for k, kw in tc.get("sc_checks", {}).items():
                total += 1
                if check_meta(meta, k, kw, gene): passed += 1

        r5 = call(download_hpa_blood_expression_by_gene, gene, tmp)
        total += 1
        if assert_success(r5, gene, "blood"): passed += 1

    if os.path.exists(tmp): os.remove(tmp)
    print(f"\n{BOLD}Results: {passed}/{total} passed ({int(100*passed/total)}%){RESET}")
    return passed == total

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
