"""
Live integration tests for tools added in Phases 1 / 2 / 3.1.

For pure-local tools: real input file → real execution → assert real output.
For network tools: real HTTP request → real response → assert real data.

Slow tests (Clustal Omega, MMseqs2, EBI BLAST submit-poll-download) run only
when SLOW=1 is in the environment — each takes 1-5 min of wallclock.

Run all (skip slow):
    pytest tests/test_phase_tools_live.py -v

Run all including slow ones:
    SLOW=1 pytest tests/test_phase_tools_live.py -v

Run directly without pytest:
    python tests/test_phase_tools_live.py            # fast only
    SLOW=1 python tests/test_phase_tools_live.py     # everything
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Live test output is under tests/_phase_live_artifacts so re-runs can inspect outputs.
_ART = _REPO / "tests" / "_phase_live_artifacts"
_ART.mkdir(exist_ok=True)


SLOW = os.environ.get("SLOW") == "1"
slow = pytest.mark.skipif(not SLOW, reason="set SLOW=1 to run multi-minute submit-poll-download tests")


# ============================================================================
# Phase 1.1 — AlphaFold local analysis (pure local, no network)
# ============================================================================

def test_analyze_alphafold_plddt_real_metadata():
    from src.tools.database.alphafold import analyze_alphafold_plddt_by_metadata_file

    sample_path = _ART / "plddt_sample.json"
    # Realistic AlphaFold metadata fractions (P53 HUMAN — disordered N-term + structured DBD)
    sample = [{
        "uniprotAccession": "P04637",
        "globalMetricValue": 78.1,
        "fractionPlddtVeryLow": 0.18,
        "fractionPlddtLow": 0.12,
        "fractionPlddtConfident": 0.20,
        "fractionPlddtVeryHigh": 0.50,
    }]
    sample_path.write_text(json.dumps(sample))
    raw = analyze_alphafold_plddt_by_metadata_file(str(sample_path))
    r = json.loads(raw)
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["uniprot_id"] == "P04637"
    assert abs(bm["global_plddt"] - 78.1) < 0.001
    # conf_total = 0.70 → CONFIDENT_THRESHOLD; frac_vlow > NOTABLE_DISORDER_THRESHOLD (0.15)
    assert "notable disordered regions" in bm["conclusion"]
    assert "global pLDDT" in r["content"]


def test_analyze_alphafold_pae_real_matrix_two_domains():
    from src.tools.database.alphafold import analyze_alphafold_pae_by_pae_file

    # Synthetic 140-residue PAE matrix with two 70-residue domains (merge filter requires >50).
    n = 140
    mat = []
    for i in range(n):
        row = []
        for j in range(n):
            same_block = (i < 70 and j < 70) or (i >= 70 and j >= 70)
            row.append(1.5 if same_block else 25.0)
        mat.append(row)
    sample_path = _ART / "pae_sample.json"
    sample_path.write_text(json.dumps([{"predicted_aligned_error": mat}]))
    raw = analyze_alphafold_pae_by_pae_file(str(sample_path), min_domain_size=30)
    r = json.loads(raw)
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["matrix_shape"] == "140x140"
    assert bm["confident_pairs_pct"] > 40.0
    assert len(bm["domains"]) == 2, bm["domains"]
    # Verify the two domains are roughly correct sizes
    sizes = sorted(d["length"] for d in bm["domains"])
    assert all(60 <= s <= 80 for s in sizes), sizes


def test_analyze_alphafold_plddt_error_path():
    from src.tools.database.alphafold import analyze_alphafold_plddt_by_metadata_file
    r = json.loads(analyze_alphafold_plddt_by_metadata_file("/does/not/exist.json"))
    assert r["status"] == "error"
    assert r["error"]["type"] == "NotFound"


# ============================================================================
# Phase 1.5 — PyMOL (pure local, requires pymol on PATH)
# ============================================================================

def _pymol_available() -> bool:
    return shutil.which("pymol") is not None


@pytest.mark.skipif(not _pymol_available(), reason="pymol CLI not on PATH")
def test_pymol_render_real_structure_ss_coloring():
    from src.tools.visualize.pymol import render_protein_structure
    pdb_path = _ART / "trp.pdb"
    if not pdb_path.exists():
        _generate_trp_pdb(pdb_path)
    out_dir = _ART / "pymol_out"
    out_dir.mkdir(exist_ok=True)
    r = json.loads(render_protein_structure(str(pdb_path), str(out_dir), color_by="ss"))
    assert r["status"] == "success", r
    fi = r["file_info"]
    assert fi["format"] == "png"
    assert fi["file_size"] > 1000  # real PNG, not empty
    # Verify PSE session file also saved
    pse_path = out_dir / f"{pdb_path.stem}_ss.pse"
    assert pse_path.exists() and pse_path.stat().st_size > 0
    # Verify stdout content_preview was captured (flush bug regression)
    assert "atoms=" in r["content_preview"]


@pytest.mark.skipif(not _pymol_available(), reason="pymol CLI not on PATH")
def test_pymol_render_validation_errors():
    from src.tools.visualize.pymol import render_protein_structure
    # NotFound: bad pdb path
    r = json.loads(render_protein_structure("/nonexistent.pdb", str(_ART / "pymol_out")))
    assert r["status"] == "error" and r["error"]["type"] == "NotFound"
    # ValidationError: bad color_by — supply a REAL pdb so we hit the color validation
    pdb_path = _ART / "trp.pdb"
    if not pdb_path.exists():
        _generate_trp_pdb(pdb_path)
    r2 = json.loads(render_protein_structure(str(pdb_path), str(_ART / "pymol_out"), color_by="rainbow"))
    assert r2["status"] == "error" and r2["error"]["type"] == "ValidationError", r2


@pytest.mark.skipif(not _pymol_available(), reason="pymol CLI not on PATH")
def test_pymol_superpose_identical_structures_rmsd_zero():
    from src.tools.visualize.pymol import superpose_two_structures
    pdb_path = _ART / "trp.pdb"
    if not pdb_path.exists():
        _generate_trp_pdb(pdb_path)
    out_dir = _ART / "pymol_out"
    out_dir.mkdir(exist_ok=True)
    r = json.loads(superpose_two_structures(str(pdb_path), str(pdb_path), str(out_dir)))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    # Regression for the flush bug: rmsd must be parsed from stdout
    assert bm["rmsd_angstroms"] is not None
    assert bm["rmsd_angstroms"] == 0.0  # identical structures
    assert bm["aligned_residues"] is not None and bm["aligned_residues"] > 0


def _generate_trp_pdb(out_path: Path):
    """Use pymol headless to create a tryptophan PDB. Writes script to temp file then runs."""
    script = f'''import os
os.environ["PYOPENGL_PLATFORM"]="osmesa"
import pymol
pymol.pymol_argv=["pymol","-cq"]
pymol.finish_launching()
from pymol import cmd
cmd.fragment("trp")
cmd.save({str(out_path)!r})
cmd.quit()
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        sp = subprocess.run(["pymol", "-cq", script_path], capture_output=True, text=True, timeout=60)
        if sp.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"pymol fragment failed (rc={sp.returncode}): stdout={sp.stdout[-200:]!r} stderr={sp.stderr[-200:]!r}"
            )
    finally:
        os.unlink(script_path)


# ============================================================================
# Phase 1.3 — Clustal Omega (network, slow ~1-3 min)
# ============================================================================

_TEST_EMAIL = os.environ.get("USER_EMAIL") or "noreply@anthropic.com"


@slow
def test_clustalo_msa_real_alignment():
    from src.tools.database.clustalo import download_clustalo_msa_by_fasta
    in_path = _ART / "clustalo_input.fasta"
    in_path.write_text(
        ">human\nMVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK\n"
        ">mouse\nMVHLTDAEKSAVSCLWGKVNSDEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNPK\n"
        ">rat\nMVHLTDAEKAAVNALWGKVNPDDVGGEALGRLLVVYPWTQRYFDKFGDLSSASAIMGNAK\n"
        ">horse\nMVHLNGAEKSAVNGLWGKVKVDEVGAEALGRLLVVYPWTQRYFDSFGDLSNPGAVMGNPK\n"
    )
    out_dir = _ART / "clustalo_out"
    raw = download_clustalo_msa_by_fasta(str(in_path), str(out_dir), email=_TEST_EMAIL)
    r = json.loads(raw)
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["input_sequences"] == 4
    assert bm["aligned_sequences"] == 4
    # Verify the alignment file is real FASTA with all 4 records
    msa_file = next(out_dir.glob("*_msa.fasta"))
    text = msa_file.read_text()
    assert text.count(">") == 4
    # All aligned sequences in MSA share the same length (gaps padded as needed)
    seq_lines = [l for l in text.splitlines() if l and not l.startswith(">")]
    assert seq_lines, "no sequence lines in alignment"
    # In clustal omega FASTA output, each record is on a single line; with such
    # close homologs no gaps may be needed, but lengths must be equal.
    assert len({len(l) for l in seq_lines}) == 1, "all aligned sequences must have equal length"


# ============================================================================
# Phase 1.4 — MMseqs2 (ColabFold) and EBI BLAST (network, slow ~1-5 min each)
# ============================================================================

@slow
def test_mmseqs2_real_homolog_search():
    from src.tools.database.seq_search import download_mmseqs2_homologs_by_sequence
    # Hemoglobin beta — has thousands of homologs
    seq = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    out_dir = _ART / "mmseqs2_out"
    raw = download_mmseqs2_homologs_by_sequence(seq, str(out_dir))
    r = json.loads(raw)
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["engine"].startswith("MMseqs2")
    assert bm["query_length"] == len(seq)
    assert bm["hit_count"] > 5, f"expected many hemoglobin homologs, got {bm['hit_count']}"
    # Verify on-disk JSON contains hit records
    hits_file = next(out_dir.glob("mmseqs2_*.json"))
    payload = json.loads(hits_file.read_text())
    assert "hits" in payload
    assert len(payload["hits"]) == bm["hit_count"]
    assert all("e_value" in h for h in payload["hits"][:3])


@slow
def test_ebi_blast_real_search():
    from src.tools.database.seq_search import download_blast_homologs_by_sequence
    seq = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
    out_dir = _ART / "blast_out"
    raw = download_blast_homologs_by_sequence(seq, str(out_dir), database="uniprotkb_swissprot", email=_TEST_EMAIL)
    r = json.loads(raw)
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["engine"] == "EBI BLAST"
    assert bm["hit_count"] > 5


# ============================================================================
# Phase 2.1 — RCSB Search API v2 (network, fast)
# ============================================================================

def test_rcsb_search_real_text_query():
    from src.tools.database.rcsb import download_rcsb_search_by_query
    r = json.loads(download_rcsb_search_by_query(
        query={"type": "terminal", "service": "full_text", "parameters": {"value": "myoglobin"}},
        out_dir=str(_ART / "rcsb_out"),
        return_type="entry",
        rows=10,
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["total_count"] > 100  # myoglobin has many structures
    assert bm["result_count"] == 10
    out_file = _ART / "rcsb_out" / "rcsb_search_entry.json"
    body = json.loads(out_file.read_text())
    assert body.get("result_set"), "expected result_set in response"
    first = body["result_set"][0]
    assert first.get("identifier") and len(first["identifier"]) == 4  # PDB ID


def test_rcsb_search_count_only():
    from src.tools.database.rcsb import download_rcsb_search_by_query
    r = json.loads(download_rcsb_search_by_query(
        query={"type": "terminal", "service": "full_text", "parameters": {"value": "kinase"}},
        out_dir=str(_ART / "rcsb_out"), count_only=True,
    ))
    assert r["status"] == "success"
    assert r["biological_metadata"]["total_count"] > 1000


def test_rcsb_search_sequence_service():
    from src.tools.database.rcsb import download_rcsb_search_by_query
    r = json.loads(download_rcsb_search_by_query(
        query={
            "type": "terminal", "service": "sequence",
            "parameters": {
                "evalue_cutoff": 0.1, "identity_cutoff": 0.95,
                "sequence_type": "protein",
                "value": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK",
            },
        },
        out_dir=str(_ART / "rcsb_out"), return_type="polymer_entity", rows=5,
    ))
    assert r["status"] == "success", r
    assert r["biological_metadata"]["total_count"] >= 1


# ============================================================================
# Phase 2.2 — UniProt SPARQL (network, fast)
# ============================================================================

def test_uniprot_sparql_real_query():
    from src.tools.database.uniprot import download_uniprot_sparql_by_query
    q = """
    PREFIX up: <http://purl.uniprot.org/core/>
    SELECT ?p ?name
    WHERE {
      ?p a up:Protein .
      ?p up:recommendedName / up:fullName ?name .
    }
    LIMIT 5
    """
    r = json.loads(download_uniprot_sparql_by_query(q, str(_ART / "sparql_out")))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["row_count"] == 5
    assert "p" in bm["head_vars"] and "name" in bm["head_vars"]
    # The saved JSON file is valid SPARQL JSON
    fp = next((_ART / "sparql_out").glob("uniprot_sparql_*.json"))
    body = json.loads(fp.read_text())
    assert "head" in body and "results" in body
    assert len(body["results"]["bindings"]) == 5


def test_uniprot_sparql_validation():
    from src.tools.database.uniprot import download_uniprot_sparql_by_query
    r = json.loads(download_uniprot_sparql_by_query("", str(_ART / "sparql_out")))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"


# ============================================================================
# Phase 2.3 — NCBI extras (network, fast)
# ============================================================================

def test_ncbi_cds_translate_real_hbb_mrna():
    from src.tools.database.ncbi import translate_ncbi_cds_to_protein
    # NM_000518 = HBB mRNA, ~147 aa hemoglobin beta
    r = json.loads(translate_ncbi_cds_to_protein("NM_000518", str(_ART / "ncbi_out")))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["method"] == "fasta_cds_aa"
    assert 140 <= bm["selected_length"] <= 150
    # Verify the FASTA file actually exists with correct content
    fasta = _ART / "ncbi_out" / "NM_000518_protein.fasta"
    assert fasta.exists()
    text = fasta.read_text()
    assert text.startswith(">")
    aa_seq = "".join(text.splitlines()[1:])
    assert aa_seq.startswith("M")
    assert "*" not in aa_seq  # stop codon stripped


def test_ncbi_cds_translate_target_length():
    """Verify target_length picks the closest translation when many are returned."""
    from src.tools.database.ncbi import translate_ncbi_cds_to_protein
    r = json.loads(translate_ncbi_cds_to_protein("NM_000518", str(_ART / "ncbi_out"), target_length=147))
    bm = r["biological_metadata"]
    assert bm["selected_length"] == 147


def test_ncbi_search_protein_by_gene_organism_real():
    from src.tools.database.ncbi import search_ncbi_protein_by_gene_and_organism
    r = json.loads(search_ncbi_protein_by_gene_and_organism(
        "TP53", "Homo sapiens", str(_ART / "ncbi_out"), retmax=3,
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["total_count"] > 100  # many TP53 entries
    assert bm["returned_count"] >= 1
    # Verify multi-FASTA file
    fasta = _ART / "ncbi_out" / "TP53_Homo_sapiens_proteins.fasta"
    assert fasta.exists()
    text = fasta.read_text()
    assert text.count(">") >= 1
    # Verify the summary JSON
    summary = _ART / "ncbi_out" / "TP53_Homo_sapiens_proteins.json"
    assert summary.exists()
    sj = json.loads(summary.read_text())
    assert sj["gene"] == "TP53"
    assert "hits" in sj


def test_ncbi_search_validation_errors():
    from src.tools.database.ncbi import (
        translate_ncbi_cds_to_protein, search_ncbi_protein_by_gene_and_organism,
    )
    r = json.loads(translate_ncbi_cds_to_protein("", str(_ART / "ncbi_out")))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"
    r2 = json.loads(search_ncbi_protein_by_gene_and_organism("TP53", "", str(_ART / "ncbi_out")))
    assert r2["status"] == "error" and r2["error"]["type"] == "ValidationError"


# ============================================================================
# Phase 3.1 — OpenAlex (network, fast)
# ============================================================================

def test_openalex_search_real_works_query():
    from src.tools.database.openalex import download_openalex_entries_by_query
    r = json.loads(download_openalex_entries_by_query(
        "works", str(_ART / "openalex_out"),
        search="AlphaFold protein structure prediction",
        per_page=5, sort="cited_by_count:desc",
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["total_count"] > 1000  # AlphaFold has lots of mentions
    assert bm["result_count"] == 5
    # Glob picks only the search-pages (the "_p1_" segment isolates them from entry files)
    matches = list((_ART / "openalex_out").glob("openalex_works_*_p1_*.json"))
    assert matches, "expected at least one search-page file with `_p1_` token"
    body = json.loads(matches[-1].read_text())
    assert "results" in body and len(body["results"]) == 5
    # Top result should have huge citation count (Jumper 2021)
    top = body["results"][0]
    assert top.get("cited_by_count", 0) > 10000


def test_openalex_entry_by_id_real():
    from src.tools.database.openalex import download_openalex_entry_by_id
    # W3177828909 = "Highly accurate protein structure prediction with AlphaFold"
    r = json.loads(download_openalex_entry_by_id(
        "works", "W3177828909", str(_ART / "openalex_out"),
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert "AlphaFold" in bm["display_name"]
    assert bm["cited_by_count"] > 10000
    assert bm["publication_year"] == 2021


def test_openalex_validation_errors():
    from src.tools.database.openalex import download_openalex_entries_by_query
    r = json.loads(download_openalex_entries_by_query("bogus_entity", str(_ART / "openalex_out")))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"


# ============================================================================
# Phase 3.2 — arXiv paper download (network, fast)
# ============================================================================

def test_arxiv_download_real_pdf():
    from src.tools.database.arxiv import download_arxiv_paper_by_id
    # 2007.07235 = ColabFold paper, small (~360 KB PDF)
    r = json.loads(download_arxiv_paper_by_id(
        "2007.07235", str(_ART / "arxiv_out"), format="pdf",
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["format"] == "pdf"
    assert bm["size_bytes"] > 10000  # real PDF
    pdf_path = _ART / "arxiv_out" / "arxiv_2007.07235.pdf"
    assert pdf_path.exists()
    # PDF magic number
    assert pdf_path.read_bytes()[:4] == b"%PDF"


def test_arxiv_validation_and_not_found():
    from src.tools.database.arxiv import download_arxiv_paper_by_id
    r = json.loads(download_arxiv_paper_by_id("2007.07235", str(_ART / "arxiv_out"), format="bogus"))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"
    r2 = json.loads(download_arxiv_paper_by_id("9999.99999", str(_ART / "arxiv_out"), format="pdf"))
    assert r2["status"] == "error" and r2["error"]["type"] == "NotFound"


# ============================================================================
# Phase 3.3 — PubMed batch abstract fetch (network, fast)
# ============================================================================

def test_pubmed_abstracts_real_alphafold_papers():
    from src.tools.database.ncbi import download_pubmed_abstracts_by_pmids
    out_path = str(_ART / "pubmed_out" / "abstracts.json")
    # 3 real protein-science PMIDs: AlphaFold (Jumper 2021), AlphaFold DB (Varadi 2024),
    # ColabFold (Mirdita 2022)
    r = json.loads(download_pubmed_abstracts_by_pmids(
        ["34265844", "37962427", "35637307"], out_path,
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["requested_pmids"] == 3
    assert bm["articles_returned"] == 3
    body = json.loads(Path(out_path).read_text())
    titles = [a["title"] for a in body["articles"]]
    assert any("AlphaFold" in t or "ColabFold" in t for t in titles), titles
    # Verify abstract + DOI extraction
    first = body["articles"][0]
    assert first["abstract"] and len(first["abstract"]) > 100
    assert first["doi"] and first["doi"].startswith("10.")
    assert first["authors"]  # non-empty list


def test_pubmed_abstracts_validation():
    from src.tools.database.ncbi import download_pubmed_abstracts_by_pmids
    r = json.loads(download_pubmed_abstracts_by_pmids([], str(_ART / "pubmed_out" / "v.json")))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"
    # Too many
    r2 = json.loads(download_pubmed_abstracts_by_pmids([str(i) for i in range(300)], str(_ART / "pubmed_out" / "v.json")))
    assert r2["status"] == "error" and r2["error"]["type"] == "ValidationError"


# ============================================================================
# Phase 3.4 — bioRxiv per-DOI fetch (network, fast)
# ============================================================================

def test_biorxiv_by_doi_real():
    from src.tools.database.biorxiv import download_biorxiv_by_doi
    r = json.loads(download_biorxiv_by_doi(
        "10.1101/2023.05.16.541025", str(_ART / "biorxiv_out"),
    ))
    assert r["status"] == "success", r
    bm = r["biological_metadata"]
    assert bm["title"] and "Cysteine" in bm["title"]
    assert bm["versions_returned"] >= 1
    # Verify the JSON contains full metadata including abstract
    fp = next((_ART / "biorxiv_out").glob("biorxiv_10.1101_2023.05.16.541025*.json"))
    body = json.loads(fp.read_text())
    assert body["latest"].get("abstract")


def test_biorxiv_doi_url_normalization():
    """The tool should accept DOI URLs and strip the prefix."""
    from src.tools.database.biorxiv import download_biorxiv_by_doi
    r = json.loads(download_biorxiv_by_doi(
        "https://doi.org/10.1101/2023.05.16.541025",
        str(_ART / "biorxiv_out"),
    ))
    assert r["status"] == "success", r


def test_biorxiv_validation_and_not_found():
    from src.tools.database.biorxiv import download_biorxiv_by_doi
    r = json.loads(download_biorxiv_by_doi("", str(_ART / "biorxiv_out")))
    assert r["status"] == "error" and r["error"]["type"] == "ValidationError"
    r2 = json.loads(download_biorxiv_by_doi("any", str(_ART / "biorxiv_out"), server="not_a_server"))
    assert r2["status"] == "error" and r2["error"]["type"] == "ValidationError"
    r3 = json.loads(download_biorxiv_by_doi("10.1101/0000.00.00.000000", str(_ART / "biorxiv_out")))
    assert r3["status"] == "error" and r3["error"]["type"] == "NotFound"


# ============================================================================
# Runnable as a script — print PASS/FAIL summary
# ============================================================================

if __name__ == "__main__":
    import traceback
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    passed = failed = skipped = 0
    for name, fn in tests:
        marker = getattr(fn, "pytestmark", [])
        slow_marker = any(m.name == "skipif" for m in marker)
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"SKIP  {name}: {e}")
            skipped += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed, {skipped} skipped ===")
    sys.exit(0 if failed == 0 else 1)
