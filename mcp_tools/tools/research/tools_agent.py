"""Shared public-service tool for Agent and Scientist."""
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool
import asyncio
from .services import run_service, get_json, ACCESSION, sequence as clean_sequence, structure_chain as clean_structure

ServiceName = Literal['blast','foldseek_remote','interproscan','s4pred','interpro','quickgo','string','kegg','reactome','alphafold','pdbe']


async def prepare_context(service, accession='', sequence='', structure_pdb='', structure_chain=''):
    PublicProteinInput(service=service,accession=accession,sequence=sequence,structure_pdb=structure_pdb,structure_chain=structure_chain)
    if sequence:sequence=clean_sequence(sequence)
    if structure_pdb:
        structure_pdb,struct_seq,structure_chain=clean_structure(structure_pdb,structure_chain)
        if sequence and sequence!=struct_seq:raise ValueError('The sequence must match the selected structure chain')
        sequence=sequence or struct_seq
    context={'accession':accession,'sequence':sequence,'structure_pdb':structure_pdb,'structure_chain':structure_chain}
    if accession:
        if not ACCESSION.fullmatch(accession):raise ValueError('Use a UniProt accession')
        entry=await get_json('https://rest.uniprot.org/uniprotkb/'+accession+'.json')
        context['entry']=entry or {}
        if not sequence:context['sequence']=(entry or {}).get('sequence',{}).get('value','')
    return context


async def public_analysis(service, accession='', sequence='', structure_pdb='', structure_chain=''):
    return await run_service(service, await prepare_context(service, accession, sequence, structure_pdb, structure_chain))


class PublicProteinInput(BaseModel):
    service: ServiceName
    accession: str = Field(default='',max_length=20,description='Exact UniProt accession. Required for annotation lookups.')
    sequence: str = Field(default='',max_length=10000,description='Protein sequence for NCBI BLAST; sent to NCBI when selected.')
    structure_pdb: str = Field(default='',max_length=2_000_000,description='Optional protein PDB text for the remote Foldseek server. Otherwise use a mapped UniProt structure.')
    structure_chain: str = Field(default='',max_length=1)


@tool('query_public_protein_service',args_schema=PublicProteinInput)
def query_public_protein_service_tool(service: str, accession: str='', sequence: str='', structure_pdb: str='', structure_chain: str='') -> dict:
    """Run a bounded free public protein analysis: NCBI BLAST, remote Foldseek, InterPro/Pfam, QuickGO, STRING, KEGG, Reactome, AlphaFold DB, or PDBe/PDB. BLAST submits one protein sequence; Foldseek submits one protein structure. Returns actual status, evidence, and remote job URL. Remote similarity is not functional validation. No local GPU training or VenusMine execution. Jobs can take several minutes; pending means still processing, not complete."""
    return asyncio.run(public_analysis(service,accession,sequence,structure_pdb,structure_chain))


from langchain_core.tools import StructuredTool
from .basic_queries import NCBIInput, GgetInput, ncbi_query, gget_query, similarity_query


class BlastInput(BaseModel):
    accession: str = Field(default='', max_length=20, description='UniProt accession, or provide sequence.')
    sequence: str = Field(default='', max_length=10000, description='Protein sequence sent to NCBI BLAST against SwissProt.')
    job_id: str = Field(default='', max_length=100, description='To check an existing request, supply only its returned job_id; do not resubmit the sequence.')


class FoldseekInput(BaseModel):
    accession: str = Field(default='', max_length=20, description='UniProt accession used to find a mapped structure, or provide PDB text.')
    structure_pdb: str = Field(default='', max_length=2_000_000, description='Protein PDB text sent to the public Foldseek server.')
    structure_chain: str = Field(default='', max_length=1, description='Select the chain if the PDB has multiple protein chains.')
    job_id: str = Field(default='', max_length=100, description='To check an existing request, supply only its returned job_id.')


async def ncbi_blast_query(**kwargs):
    return await similarity_query('blast', **kwargs)


async def foldseek_remote_query(**kwargs):
    return await similarity_query('foldseek_remote', **kwargs)


def shared_tool(name, schema, coroutine, description):
    def sync(**kwargs):
        return asyncio.run(coroutine(**kwargs))
    return StructuredTool.from_function(func=sync, coroutine=coroutine, name=name, args_schema=schema, description=description)


query_ncbi_tool = shared_tool('query_ncbi', NCBIInput, ncbi_query,
    'Query free NCBI Entrez protein, nucleotide, gene, taxonomy or PubMed databases. Search by gene/protein and organism; retrieve summaries, FASTA sequences or PubMed abstracts using returned identifiers. Returns bounded records and source links, without local files or an API key.')
query_ncbi_blast_tool = shared_tool('query_ncbi_blast', BlastInput, ncbi_blast_query,
    'Submit a protein sequence or UniProt accession to the free NCBI BLAST web server (blastp, SwissProt). Returns a remote job_id promptly. To retrieve results, call again with only job_id, respecting next_poll; pending means not finished. Do not submit the same query again. Returns homology evidence, not functional validation.')
query_foldseek_remote_tool = shared_tool('query_foldseek_remote', FoldseekInput, foldseek_remote_query,
    'Search protein structural similarity using the free public Foldseek web server. Provide a UniProt accession or PDB text with chain. Returns a remote job_id promptly; call again with only job_id to retrieve hits. Available online, without local Foldseek databases or GPUs. Structural similarity is not binding or functional validation.')
query_gget_tool = shared_tool('query_gget', GgetInput, gget_query,
    'Use gget to query public genomics databases: info for Ensembl/UniProt gene annotations, seq for protein/nucleotide FASTA, pdb for RCSB metadata, archs4 for tissue expression/coexpression, bgee for orthologs/expression, opentargets for gene-disease/drug associations, enrichr for gene-set enrichment. Pass structured arguments, not shell commands. Returns real database results and sources; no LLM API or local model training.')

class SequencePredictionInput(BaseModel):
    sequence: str = Field(min_length=10,max_length=10000,description='Amino acid sequence, without a FASTA header.')

class InterProScanInput(BaseModel):
    sequence: str = Field(default='',max_length=10000,description='Protein sequence sent to the free EMBL-EBI InterProScan service.')
    job_id: str = Field(default='',max_length=32,description='Owned task ID returned at submission; provide only this to retrieve results.')

async def interproscan_query(sequence='',job_id=''):
    if job_id:
        if sequence: raise ValueError('Provide only job_id when checking a task')
        context={'remote_job_id':job_id,'poll_attempts':1}
    else:
        context={'sequence':clean_sequence(sequence),'submit_only':True,'poll_attempts':1}
    return await run_service('interproscan',context)

from mcp_tools.models.pretrained.s4pred import predict as s4pred_predict
predict_s4pred_tool=StructuredTool.from_function(func=s4pred_predict,name='predict_s4pred',args_schema=SequencePredictionInput,
    description='Predict per-residue helix, strand and coil probabilities from one protein sequence using the installed S4PRED ensemble. CPU-only; no MSA required. Available local and online (at most 2,000 residues online). Returns actual predictions and SS2/JSON artifacts; not a 3D structure.',
    metadata={'tool_category':'sequence_analysis','queue_resource':'cpu','timeout_seconds':600})
query_interproscan_tool=shared_tool('query_interproscan',InterProScanInput,interproscan_query,
    'Submit one protein sequence to the free EMBL-EBI InterProScan REST API for domain/signature, GO and pathway annotation. Returns an owned task ID promptly. Poll with only job_id after next_poll; do not resubmit. Pending is not complete. Results expire after 24 hours. No local GPU or database installation.')
query_interproscan_tool.metadata={'tool_category':'remote_analysis','queue_resource':'api:remote-analysis','timeout_seconds':90}


from .knowledge_queries import (
    OpenTargetsInput, EnsemblInput, EuropePMCInput, PubChemInput,
    opentargets_query, ensembl_query, europepmc_query, pubchem_query,
)
query_opentargets_tool = shared_tool('query_opentargets', OpenTargetsInput, opentargets_query,
    'Query Open Targets for human target-disease associations, drug candidates, tractability and drug mechanisms. Search a gene symbol, disease or drug name first to resolve current IDs; inspect target resource diseases/drugs/tractability, disease-associated targets or drug details. Returns source links, API/data release and bounded pages. Association scores are not causal or experimental validation.')
query_ensembl_tool = shared_tool('query_ensembl', EnsemblInput, ensembl_query,
    'Resolve gene symbols to Ensembl IDs with explicit species, retrieve gene/transcript/protein annotations, cross-references, orthologues and protein/cDNA/CDS/genomic sequences. Use symbol first, then the returned stable ID; target_species narrows orthologues. No local genome database or gget installation needed.')
query_europepmc_tool = shared_tool('query_europepmc', EuropePMCInput, europepmc_query,
    'Search Europe PMC literature and retrieve actual abstracts, DOI/PMID/PMCID, publication year, open-access and retraction metadata. Search with EXT_ID:<PMID> AND SRC:MED for a known paper. Use next_cursor to page. Abstracts are not full text; distinguish preprints and retracted work. No API key required.')
query_pubchem_tool = shared_tool('query_pubchem', PubChemInput, pubchem_query,
    'Look up one compound by name, PubChem CID or SMILES to retrieve formula, molecular weight, stereochemical/connectivity SMILES, InChIKey, XLogP, TPSA and hydrogen-bond descriptors. Returns compound source links. These database properties are not ADMET predictions or binding measurements.')
from mcp_tools.tools.database.kegg.query import KeggInput, kegg_query
query_kegg_tool = shared_tool('query_kegg', KeggInput, kegg_query,
    'Search KEGG genes, pathways, orthology and compounds; retrieve a resolved entry, map a UniProt accession to genes, or link an entry to a selected database. Returns bounded annotations and sources.')
KNOWLEDGE_TOOLS = [query_kegg_tool, query_opentargets_tool, query_ensembl_tool, query_europepmc_tool, query_pubchem_tool]
for _tool in KNOWLEDGE_TOOLS:
    _tool.metadata = {'tool_category': 'web_search' if _tool.name == 'query_europepmc' else 'protein_database',
                      'queue_resource': 'api:' + _tool.name.removeprefix('query_'), 'timeout_seconds': 90}

PUBLIC_PROTEIN_TOOLS = KNOWLEDGE_TOOLS + [query_public_protein_service_tool, query_ncbi_tool, query_ncbi_blast_tool, query_foldseek_remote_tool, query_gget_tool, predict_s4pred_tool, query_interproscan_tool]
