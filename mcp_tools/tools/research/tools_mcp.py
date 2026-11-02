from fastmcp import FastMCP
from .tools_agent import ServiceName, public_analysis
from .basic_queries import Database, NCBIAction, GgetOperation, ncbi_query, gget_query, similarity_query
from typing import Literal

mcp=FastMCP('Venus_Public_Protein_Services')


@mcp.tool(name='query_public_protein_service')
async def query_public_protein_service(service: ServiceName, accession: str='', sequence: str='', structure_pdb: str='', structure_chain: str='') -> dict:
    """Run free remote NCBI BLAST or Foldseek, or query InterPro/Pfam, QuickGO, STRING, Reactome, AlphaFold DB and PDBe/PDB. Returns evidence, actual status and a remote job URL. BLAST sends a sequence to NCBI; Foldseek sends a protein structure. Use an exact UniProt ID for annotations. Pending jobs are not completed analyses. No local GPU computation."""
    return await public_analysis(service,accession,sequence,structure_pdb,structure_chain)


@mcp.tool(name='query_ncbi')
async def query_ncbi(database: Database='protein', action: NCBIAction='search', query: str='', ids: list[str] | None=None, limit: int=10) -> dict:
    """Query free NCBI Entrez: protein, nuccore, gene, taxonomy or pubmed. Search by gene/protein and organism, then use numeric UIDs for summary or abstracts; sequence also accepts accessions. Returns FASTA or records and source URLs. Up to 20 search hits or 10 known identifiers. No API key required."""
    return await ncbi_query(database=database, action=action, query=query, ids=ids or [], limit=limit)


@mcp.tool(name='query_ncbi_blast')
async def query_ncbi_blast(accession: str='', sequence: str='', job_id: str='') -> dict:
    """Run free remote NCBI BLAST (blastp, SwissProt) on a protein sequence or UniProt accession. Submission returns a job_id promptly. Check using only job_id, respecting next_poll; never resubmit to check status. Pending is not complete. Returns sequence similarity, not experimental function validation."""
    return await similarity_query('blast', accession=accession, sequence=sequence, job_id=job_id)


@mcp.tool(name='query_foldseek_remote')
async def query_foldseek_remote(accession: str='', structure_pdb: str='', structure_chain: str='', job_id: str='') -> dict:
    """Search the free public Foldseek web server using a UniProt accession or PDB text/chain. Submission returns job_id promptly; check using only job_id. Works online without local GPUs or large databases. Returns structural similarity evidence and a result URL; pending is not complete and similarity is not binding validation."""
    return await similarity_query('foldseek_remote', accession=accession, structure_pdb=structure_pdb, structure_chain=structure_chain, job_id=job_id)


@mcp.tool(name='query_gget')
async def query_gget(operation: GgetOperation, identifiers: list[str], resource: str='', species: Literal['human','mouse','fly','yeast','worm','fish']='human', translate: bool=True, limit: int=20) -> dict:
    """Use gget's public database queries. info/seq: up to 5 Ensembl gene/transcript IDs (seq translate=True returns protein); pdb: one PDB ID, resource entry/pubmed; archs4: one gene symbol, tissue/correlation; bgee: one Ensembl gene ID, orthologs/expression; opentargets: one Ensembl gene ID, diseases/drugs/expression/interactions/tractability/pharmacogenetics/depmap; enrichr: 2-200 gene symbols plus explicit library name in resource. Returns bounded real records with sources. No CLI, local training or paid model calls."""
    return await gget_query(operation=operation, identifiers=identifiers, resource=resource, species=species, translate=translate, limit=limit)


@mcp.tool(name='predict_s4pred')
def predict_s4pred(sequence: str) -> dict:
    """Predict single-sequence secondary structure with S4PRED on CPU. Returns helix/strand/coil probabilities and SS2 artifacts. Local and online; online up to 2,000 residues."""
    from mcp_tools.models.pretrained.s4pred import predict
    return predict(sequence)


@mcp.tool(name='query_interproscan')
async def query_interproscan(sequence: str='', job_id: str='') -> dict:
    """Submit a protein sequence to EBI InterProScan for domain/signature, GO and pathway annotation. Returns a task ID promptly; poll using only job_id after next_poll. Only this user's jobs can be read. Pending is not complete."""
    from .tools_agent import interproscan_query
    return await interproscan_query(sequence,job_id)

from .knowledge_queries import (
    OTOperation, OTResource, OTEntity, EnsemblOperation, SequenceType, PubChemNamespace,
    opentargets_query, ensembl_query, europepmc_query, pubchem_query,
)


@mcp.tool(name='query_opentargets')
async def query_opentargets(operation: OTOperation='search', query: str='', identifier: str='', entity: OTEntity='all', resource: OTResource='overview', limit: int=10, page: int=0) -> dict:
    """Open Targets: search gene/disease/drug names to resolve current IDs, then query target overview, diseases, drugs or tractability; disease-associated targets; or drug mechanisms. Human targets only. Bounded pages and release/source metadata. Associations do not prove causation."""
    return await opentargets_query(operation=operation, query=query, identifier=identifier, entity=entity, resource=resource, limit=limit, page=page)


@mcp.tool(name='query_ensembl')
async def query_ensembl(identifier: str, operation: EnsemblOperation='symbol', species: str='homo_sapiens', target_species: str='', sequence_type: SequenceType='protein', limit: int=20) -> dict:
    """Ensembl: resolve a gene symbol with species, then use its stable ID for lookup, xrefs, homology (orthologues), or sequence. Optional target_species narrows homology. Sequences are returned intact. No local genome database required."""
    return await ensembl_query(operation=operation, identifier=identifier, species=species, target_species=target_species, sequence_type=sequence_type, limit=limit)


@mcp.tool(name='query_europepmc')
async def query_europepmc(query: str, limit: int=6, cursor: str='*') -> dict:
    """Search Europe PMC for literature abstracts, DOI/PMID/PMCID, year, open-access and retraction status. Query EXT_ID:<PMID> AND SRC:MED for a known paper; use next_cursor for another page. Abstracts are not full text. No key needed."""
    return await europepmc_query(query=query, limit=limit, cursor=cursor)


@mcp.tool(name='query_pubchem')
async def query_pubchem(query: str, namespace: PubChemNamespace='name', limit: int=5) -> dict:
    """Look up compound properties by name, numeric CID or SMILES via PubChem: formula, mass, SMILES, InChIKey, XLogP, TPSA and H-bond descriptors. Bounded records and source links. Database descriptors are not ADMET or binding predictions."""
    return await pubchem_query(query=query, namespace=namespace, limit=limit)
