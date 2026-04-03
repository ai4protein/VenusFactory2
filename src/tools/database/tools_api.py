"""
Database API outer layer: FastAPI routes only, calling core download logic from each submodule.
Routes return the core result directly (status dict with file_info, etc.).
"""
import sys
from pathlib import Path

_REPO_ROOT = next((p for p in Path(__file__).absolute().parents if (p / "src").is_dir()), Path(__file__).absolute().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

# ========== Core layer imports (pure download logic) ==========
from .alphafold import (
    download_alphafold_structure_by_uniprot_id,
    download_alphafold_metadata_by_uniprot_id,
)
from .brenda import (
    download_brenda_km_values_by_ec_number,
    download_brenda_reactions_by_ec_number,
    download_brenda_enzymes_by_substrate,
    download_brenda_compare_organisms_by_ec_number,
    download_brenda_environmental_parameters_by_ec_number,
    download_brenda_kinetic_data_by_ec_number,
    download_brenda_pathway_report,
)
from .chembl import (
    download_chembl_molecule_by_id,
    download_chembl_similarity_by_smiles,
    download_chembl_substructure_by_smiles,
    download_chembl_drug_by_id,
)
from .foldseek import download_foldseek_results_by_pdb_file
from .kegg import (
    download_kegg_info_by_database,
    download_kegg_list_by_database,
    download_kegg_find_by_database,
    download_kegg_entry_by_id,
    download_kegg_conv_by_id,
    download_kegg_link_by_id,
    download_kegg_ddi_by_id,
)
from .interpro import (
    download_interpro_metadata_by_id,
    download_interpro_annotations_by_uniprot_id,
    download_interpro_proteins_by_id,
    download_interpro_uniprot_list_by_id,
)
from .ncbi import (
    download_ncbi_sequence,
    download_ncbi_metadata,
    download_ncbi_blast,
    download_ncbi_clinvar_variants,
    download_ncbi_gene_by_id,
    download_ncbi_gene_by_symbol,
    download_ncbi_batch_lookup_by_symbols,
)
from .rcsb import (
    download_rcsb_entry_metadata_by_pdb_id,
    download_rcsb_structure_by_pdb_id,
)
from .string import (
    download_string_map_ids,
    download_string_network,
    download_string_network_image,
    download_string_interaction_partners,
    download_string_enrichment,
    download_string_ppi_enrichment,
    download_string_homology,
)
from .uniprot import (
    download_uniprot_search_by_query,
    download_uniprot_retrieve_by_id,
    download_uniprot_mapping,
    download_uniprot_seq_by_id,
    download_uniprot_meta_by_id,
)
from src.web.utils.common_utils import get_save_path


router = APIRouter(prefix="/api/v1/database", tags=["database"])


# ---------- AlphaFold ----------
@router.get("/alphafold-structures/{uniprot_id}")
def api_get_alphafold_structure(
    uniprot_id: str,
    out_dir: str = Query(..., description="Output directory"),
    format: str = Query("pdb", description="pdb or cif"),
    version: str = Query("v6", description="v1, v2, v4, v6"),
    fragment: int = Query(1, ge=1, description="Fragment index"),
):
    """Download AlphaFold structure by UniProt ID."""
    try:
        result = download_alphafold_structure_by_uniprot_id(
            uniprot_id, out_dir, format=format, version=version, fragment=fragment
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alphafold-metadata/{uniprot_id}")
def api_get_alphafold_metadata(
    uniprot_id: str,
    out_dir: str = Query(..., description="Output directory"),
):
    """Download AlphaFold metadata by UniProt ID."""
    try:
        result = download_alphafold_metadata_by_uniprot_id(uniprot_id, out_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- BRENDA ----------
class BrendaKmBody(BaseModel):
    ec_number: str
    out_path: str
    organism: str = "*"
    substrate: str = "*"


@router.post("/brenda-km-values")
def api_brenda_km_values(body: BrendaKmBody):
    try:
        result = download_brenda_km_values_by_ec_number(
            body.ec_number, body.out_path, organism=body.organism, substrate=body.substrate
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaReactionsBody(BaseModel):
    ec_number: str
    out_path: str
    organism: str = "*"


@router.post("/brenda-reactions")
def api_brenda_reactions(body: BrendaReactionsBody):
    try:
        result = download_brenda_reactions_by_ec_number(
            body.ec_number, body.out_path, organism=body.organism
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaEnzymesBySubstrateBody(BaseModel):
    substrate: str
    out_path: str
    limit: int = 50


@router.post("/brenda-enzymes-by-substrate")
def api_brenda_enzymes_by_substrate(body: BrendaEnzymesBySubstrateBody):
    try:
        result = download_brenda_enzymes_by_substrate(
            body.substrate, body.out_path, limit=body.limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaCompareOrganismsBody(BaseModel):
    ec_number: str
    organisms: List[str]
    out_path: str


@router.post("/brenda-compare-organisms")
def api_brenda_compare_organisms(body: BrendaCompareOrganismsBody):
    try:
        result = download_brenda_compare_organisms_by_ec_number(
            body.ec_number, body.organisms, body.out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaEnvironmentalBody(BaseModel):
    ec_number: str
    out_path: str


@router.post("/brenda-environmental-parameters")
def api_brenda_environmental_parameters(body: BrendaEnvironmentalBody):
    try:
        result = download_brenda_environmental_parameters_by_ec_number(
            body.ec_number, body.out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaKineticBody(BaseModel):
    ec_number: str
    out_path: str
    format: str = "json"


@router.post("/brenda-kinetic-data")
def api_brenda_kinetic_data(body: BrendaKineticBody):
    try:
        result = download_brenda_kinetic_data_by_ec_number(
            body.ec_number, body.out_path, format=body.format
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BrendaPathwayReportBody(BaseModel):
    pathway: dict
    out_path: str


@router.post("/brenda-pathway-report")
def api_brenda_pathway_report(body: BrendaPathwayReportBody):
    try:
        result = download_brenda_pathway_report(body.pathway, body.out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- ChEMBL ----------
class ChemblMoleculeBody(BaseModel):
    mol_id: str
    out_path: str


@router.post("/chembl-molecules/{mol_id}")
def api_chembl_molecule(mol_id: str, out_path: str = Query(...)):
    try:
        result = download_chembl_molecule_by_id(mol_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChemblSimilarityBody(BaseModel):
    smiles: str
    out_path: str
    threshold: int = 70
    max_results: Optional[int] = None


@router.post("/chembl-similarity")
def api_chembl_similarity(body: ChemblSimilarityBody):
    try:
        result = download_chembl_similarity_by_smiles(
            body.smiles, body.out_path,
            threshold=body.threshold, max_results=body.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChemblSubstructureBody(BaseModel):
    smiles: str
    out_path: str
    max_results: Optional[int] = None


@router.post("/chembl-substructure")
def api_chembl_substructure(body: ChemblSubstructureBody):
    try:
        result = download_chembl_substructure_by_smiles(
            body.smiles, body.out_path, max_results=body.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChemblDrugBody(BaseModel):
    chembl_id: str
    out_path: str
    max_results: Optional[int] = None


@router.post("/chembl-drugs")
def api_chembl_drug(body: ChemblDrugBody):
    try:
        result = download_chembl_drug_by_id(
            body.chembl_id, body.out_path, max_results=body.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- FoldSeek ----------
class FoldSeekBody(BaseModel):
    pdb_file_path: str
    protect_start: int
    protect_end: int
    out_dir: Optional[str] = None


@router.post("/foldseek-results")
def api_foldseek_results(body: FoldSeekBody):
    try:
        out_dir = body.out_dir or str(get_save_path("FoldSeek", "Download_data"))
        result = download_foldseek_results_by_pdb_file(
            body.pdb_file_path, body.protect_start, body.protect_end, out_dir=out_dir
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- InterPro ----------
@router.get("/interpro-metadata/{interpro_id}")
def api_interpro_metadata(
    interpro_id: str,
    out_dir: str = Query(..., description="Output directory"),
):
    try:
        result = download_interpro_metadata_by_id(interpro_id, out_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interpro-annotations/{uniprot_id}")
def api_interpro_annotations(
    uniprot_id: str,
    out_dir: str = Query(..., description="Output directory"),
):
    try:
        result = download_interpro_annotations_by_uniprot_id(uniprot_id, out_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InterProProteinsBody(BaseModel):
    interpro_id: str
    out_dir: str
    max_results: Optional[int] = None


@router.post("/interpro-proteins")
def api_interpro_proteins(body: InterProProteinsBody):
    try:
        result = download_interpro_proteins_by_id(
            body.interpro_id, body.out_dir, max_results=body.max_results
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InterProUniprotListBody(BaseModel):
    interpro_id: str
    out_dir: str
    protein_name: str = ""
    chunk_size: int = 5000
    filter_name: Optional[str] = None
    page_size: int = 200
    max_results: Optional[int] = None


@router.post("/interpro-uniprot-list")
def api_interpro_uniprot_list(body: InterProUniprotListBody):
    try:
        result = download_interpro_uniprot_list_by_id(
            body.interpro_id, body.out_dir,
            protein_name=body.protein_name, chunk_size=body.chunk_size,
            filter_name=body.filter_name, page_size=body.page_size,
            max_results=body.max_results,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- KEGG ----------
@router.get("/kegg-info")
def api_kegg_info(
    database: str = Query(...),
    out_path: str = Query(...),
):
    try:
        result = download_kegg_info_by_database(database, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kegg-list")
def api_kegg_list(
    database: str = Query(...),
    out_path: str = Query(...),
    org_or_ids: Optional[str] = Query(None),
):
    try:
        result = download_kegg_list_by_database(
            database, out_path, org_or_ids=org_or_ids
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class KeggFindBody(BaseModel):
    database: str
    query: str
    out_path: str
    option: Optional[str] = None


@router.post("/kegg-find")
def api_kegg_find(body: KeggFindBody):
    try:
        result = download_kegg_find_by_database(
            body.database, body.query, body.out_path, option=body.option
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kegg-entries/{entry_id}")
def api_kegg_entry(
    entry_id: str,
    out_path: str = Query(...),
    format: Optional[str] = Query(None),
):
    try:
        result = download_kegg_entry_by_id(entry_id, out_path, format=format)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kegg-conv")
def api_kegg_conv(
    target_db: str = Query(...),
    source_id: str = Query(...),
    out_path: str = Query(...),
):
    try:
        result = download_kegg_conv_by_id(target_db, source_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kegg-link")
def api_kegg_link(
    target_db: str = Query(...),
    source_id: str = Query(...),
    out_path: str = Query(...),
):
    try:
        result = download_kegg_link_by_id(target_db, source_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kegg-ddi/{drug_id}")
def api_kegg_ddi(
    drug_id: str,
    out_path: str = Query(...),
):
    try:
        result = download_kegg_ddi_by_id(drug_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- NCBI ----------
@router.get("/ncbi-sequence/{ncbi_id}")
def api_ncbi_sequence(
    ncbi_id: str,
    out_path: str = Query(...),
    db: str = Query("protein"),
):
    try:
        result = download_ncbi_sequence(ncbi_id, out_path, db=db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ncbi-metadata/{ncbi_id}")
def api_ncbi_metadata(
    ncbi_id: str,
    out_path: str = Query(...),
    db: str = Query("protein"),
    rettype: str = Query("gb"),
):
    try:
        result = download_ncbi_metadata(
            ncbi_id, out_path, db=db, rettype=rettype
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NcbiBlastBody(BaseModel):
    sequence: str
    out_path: str
    program: str = "blastp"
    database: str = "swissprot"
    hitlist_size: int = 50
    alignments: int = 25
    format_type: str = "XML"
    entrez_query: Optional[str] = None


@router.post("/ncbi-blast")
def api_ncbi_blast(body: NcbiBlastBody):
    try:
        result = download_ncbi_blast(
            body.sequence, body.out_path,
            program=body.program, database=body.database,
            hitlist_size=body.hitlist_size, alignments=body.alignments,
            format_type=body.format_type, entrez_query=body.entrez_query,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NcbiClinvarBody(BaseModel):
    term: str
    out_path: str
    retmax: int = 20


@router.post("/ncbi-clinvar-variants")
def api_ncbi_clinvar(body: NcbiClinvarBody):
    try:
        result = download_ncbi_clinvar_variants(
            body.term, body.out_path, retmax=body.retmax
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ncbi-gene/{gene_id}")
def api_ncbi_gene_by_id(
    gene_id: str,
    out_path: str = Query(...),
):
    try:
        result = download_ncbi_gene_by_id(gene_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ncbi-gene-by-symbol")
def api_ncbi_gene_by_symbol(
    symbol: str = Query(...),
    taxon: str = Query(...),
    out_path: str = Query(...),
):
    try:
        result = download_ncbi_gene_by_symbol(symbol, taxon, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NcbiBatchLookupBody(BaseModel):
    gene_symbols: List[str]
    organism: str
    out_path: str


@router.post("/ncbi-batch-lookup")
def api_ncbi_batch_lookup(body: NcbiBatchLookupBody):
    try:
        result = download_ncbi_batch_lookup_by_symbols(
            body.gene_symbols, body.organism, body.out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- RCSB ----------
@router.get("/rcsb-metadata/{pdb_id}")
def api_rcsb_metadata(
    pdb_id: str,
    out_path: str = Query(...),
):
    try:
        result = download_rcsb_entry_metadata_by_pdb_id(pdb_id, out_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rcsb-structures/{pdb_id}")
def api_rcsb_structure(
    pdb_id: str,
    out_dir: str = Query(...),
    file_type: str = Query("pdb"),
):
    try:
        result = download_rcsb_structure_by_pdb_id(
            pdb_id, out_dir, file_type=file_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- STRING ----------
def _string_ids(identifiers: str) -> str | List[str]:
    ids_list = [x.strip() for x in identifiers.split(",") if x.strip()]
    return ids_list if len(ids_list) > 1 else (ids_list[0] if ids_list else identifiers.strip())


class StringMapIdsBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    limit: int = 1
    echo_query: int = 1
    filename: str = "map_ids.tsv"


@router.post("/string-map-ids")
def api_string_map_ids(body: StringMapIdsBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_map_ids(
            ids, body.out_dir,
            species=body.species, limit=body.limit,
            echo_query=body.echo_query, filename=body.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringNetworkBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    required_score: int = 400
    network_type: str = "functional"
    add_nodes: int = 0
    filename: str = "network.tsv"


@router.post("/string-network")
def api_string_network(body: StringNetworkBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_network(
            ids, body.out_dir,
            species=body.species, required_score=body.required_score,
            network_type=body.network_type, add_nodes=body.add_nodes,
            filename=body.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringNetworkImageBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    required_score: int = 400
    network_flavor: str = "evidence"
    add_nodes: int = 0
    filename: str = "network.png"


@router.post("/string-network-image")
def api_string_network_image(body: StringNetworkImageBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_network_image(
            ids, body.out_dir,
            species=body.species, required_score=body.required_score,
            network_flavor=body.network_flavor, add_nodes=body.add_nodes,
            filename=body.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringInteractionPartnersBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    required_score: int = 400
    limit: int = 10
    filename: str = "interaction_partners.tsv"


@router.post("/string-interaction-partners")
def api_string_interaction_partners(body: StringInteractionPartnersBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_interaction_partners(
            ids, body.out_dir,
            species=body.species, required_score=body.required_score,
            limit=body.limit, filename=body.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringEnrichmentBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    filename: str = "enrichment.tsv"


@router.post("/string-enrichment")
def api_string_enrichment(body: StringEnrichmentBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_enrichment(
            ids, body.out_dir, species=body.species, filename=body.filename
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringPpiEnrichmentBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    required_score: int = 400
    filename: str = "ppi_enrichment.json"


@router.post("/string-ppi-enrichment")
def api_string_ppi_enrichment(body: StringPpiEnrichmentBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_ppi_enrichment(
            ids, body.out_dir,
            species=body.species, required_score=body.required_score,
            filename=body.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StringHomologyBody(BaseModel):
    identifiers: str
    out_dir: str
    species: int = 9606
    filename: str = "homology.tsv"


@router.post("/string-homology")
def api_string_homology(body: StringHomologyBody):
    try:
        ids = _string_ids(body.identifiers)
        result = download_string_homology(
            ids, body.out_dir, species=body.species, filename=body.filename
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UniProt ----------
@router.post("/uniprot-search")
def api_uniprot_search(
    query: str = Query(...),
    out_path: str = Query(...),
    frmt: str = Query("tsv"),
    columns: Optional[str] = Query(None),
    limit: Optional[int] = Query(100),
    database: str = Query("uniprotkb"),
):
    try:
        result = download_uniprot_search_by_query(
            query=query, out_path=out_path, frmt=frmt,
            columns=columns, limit=limit, database=database,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uniprot-entries/{uniprot_id}")
def api_uniprot_retrieve(
    uniprot_id: str,
    out_path: str = Query(...),
    frmt: str = Query("fasta"),
):
    try:
        result = download_uniprot_retrieve_by_id(
            uniprot_id=uniprot_id, out_path=out_path, frmt=frmt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uniprot-mapping")
def api_uniprot_mapping(
    fr: str = Query(...),
    to: str = Query(...),
    query: str = Query(...),
    out_path: str = Query(...),
):
    try:
        result = download_uniprot_mapping(
            fr=fr, to=to, query=query, out_path=out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uniprot-sequences/{uniprot_id}")
def api_uniprot_seq(
    uniprot_id: str,
    out_path: str = Query(...),
):
    try:
        result = download_uniprot_seq_by_id(
            uniprot_id=uniprot_id, out_path=out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uniprot-metadata/{uniprot_id}")
def api_uniprot_meta(
    uniprot_id: str,
    out_path: str = Query(...),
):
    try:
        result = download_uniprot_meta_by_id(
            uniprot_id=uniprot_id, out_path=out_path
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
