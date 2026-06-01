"""
Tests for HPA tissue expression response-completeness check.

The download_hpa_tissue_expression_by_gene tool used to return success=true even
when HPA returned a metadata-only stub (Gene/Ensembl/Uniprot populated but every
tissue field None). Downstream agent_generated_code that parsed `tissues` /
nTPM data then silently failed. The tool now returns status='error' with
type='IncompleteData' when no tissue-expression field is populated, while
preserving the legitimate ubiquitous-gene case (empty nTPM but a populated
distribution / specificity label).

These tests mock hpa_get_exact_entry so no network call is made.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.database.hpa.hpa_operations import (  # noqa: E402
    download_hpa_tissue_expression_by_gene,
)


_FULL_ENTRY = {
    "Gene": "GFAP",
    "Ensembl": "ENSG00000131095",
    "Uniprot": "P14136",
    "RNA tissue specificity": "Tissue enriched (brain)",
    "RNA tissue distribution": "Detected in some",
    "RNA tissue specificity score": "20.3",
    "RNA tissue specific nTPM": {"brain": "1234.5", "liver": "0.4"},
    "RNA tissue cell type enrichment": {"brain": ["Astrocytes"]},
    "Tissue expression cluster": "Brain - Neuronal signaling",
}

_UBIQUITOUS_ENTRY = {
    "Gene": "GAPDH",
    "Ensembl": "ENSG00000111640",
    "Uniprot": "P04406",
    "RNA tissue specificity": "Low tissue specificity",
    "RNA tissue distribution": "Detected in all",
    "RNA tissue specificity score": None,
    "RNA tissue specific nTPM": None,
    "RNA tissue cell type enrichment": None,
    "Tissue expression cluster": "Non-specific - Mixed function",
}

_METADATA_ONLY_ENTRY = {
    "Gene": "OBSCURE1",
    "Ensembl": "ENSG00000999999",
    "Uniprot": "Q99999",
    "RNA tissue specificity": None,
    "RNA tissue distribution": None,
    "RNA tissue specificity score": None,
    "RNA tissue specific nTPM": None,
    "RNA tissue cell type enrichment": None,
    "Tissue expression cluster": None,
}


def _run(entry):
    """Invoke the tool with hpa_get_exact_entry mocked to return `entry`."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "out.json")
        with patch(
            "src.tools.database.hpa.hpa_operations.hpa_get_exact_entry",
            return_value=entry,
        ):
            raw = download_hpa_tissue_expression_by_gene(
                entry.get("Gene", "X"), out_path
            )
        result = json.loads(raw)
        on_disk = None
        if Path(out_path).exists():
            with open(out_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
        return result, on_disk


def test_full_tissue_data_returns_success():
    result, on_disk = _run(_FULL_ENTRY)
    assert result["status"] == "success", result
    meta = result["biological_metadata"]
    assert meta["top_expressed_tissues"], "expected non-empty top_expressed_tissues"
    assert "brain" in meta["top_expressed_tissues"]
    assert meta["all_tissue_ntpm"] == _FULL_ENTRY["RNA tissue specific nTPM"]
    # data_completeness flag should NOT be set when nTPM exists
    assert "data_completeness" not in meta
    assert on_disk is not None and on_disk["Gene"] == "GFAP"


def test_ubiquitous_gene_still_succeeds_with_marker():
    """Empty nTPM but distribution='Detected in all' is legitimate HPA output."""
    result, _ = _run(_UBIQUITOUS_ENTRY)
    assert result["status"] == "success", result
    meta = result["biological_metadata"]
    assert meta["rna_tissue_distribution"] == "Detected in all"
    # nTPM dict is empty so top_expressed_tissues is empty,
    # and the data_completeness marker tells consumers to branch on distribution.
    assert meta["top_expressed_tissues"] == {}
    assert meta.get("data_completeness") == "no_per_tissue_ntpm"


def test_metadata_only_entry_returns_error():
    """Entry with only identifiers and no tissue fields must NOT report success."""
    result, on_disk = _run(_METADATA_ONLY_ENTRY)
    assert result["status"] == "error", result
    assert result["error"]["type"] == "IncompleteData"
    assert "OBSCURE1" in result["error"]["message"]
    assert "suggestion" in result["error"]
    # Stub file is still written for debugging.
    assert on_disk is not None
    assert on_disk["Gene"] == "OBSCURE1"


def test_specificity_only_entry_is_complete_enough():
    """If at least one substantive field is populated, treat as success."""
    entry = dict(_METADATA_ONLY_ENTRY)
    entry["RNA tissue specificity"] = "Tissue enhanced (kidney)"
    result, _ = _run(entry)
    assert result["status"] == "success", result
    meta = result["biological_metadata"]
    assert meta["rna_tissue_specificity"] == "Tissue enhanced (kidney)"
    assert meta.get("data_completeness") == "no_per_tissue_ntpm"


if __name__ == "__main__":
    # Allow running directly without pytest for quick verification.
    test_full_tissue_data_returns_success()
    test_ubiquitous_gene_still_succeeds_with_marker()
    test_metadata_only_entry_returns_error()
    test_specificity_only_entry_is_complete_enough()
    print("All HPA completeness tests passed.")
