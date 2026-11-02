"""Shared, bounded public database queries for every conversational harness."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import tempfile
from typing import Literal
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .services import ServiceError, get_json, request, run_service

from .knowledge_queries import KNOWLEDGE_TOOL_NAMES

BASIC_PUBLIC_TOOL_NAMES = KNOWLEDGE_TOOL_NAMES + (
    'query_ncbi', 'query_ncbi_blast', 'query_foldseek_remote', 'query_gget',
    'query_public_protein_service',
)
Database = Literal['protein', 'gene', 'nuccore', 'pubmed', 'taxonomy']
NCBIAction = Literal['search', 'summary', 'sequence', 'abstracts']
GgetOperation = Literal['info', 'seq', 'pdb', 'archs4', 'bgee', 'opentargets', 'enrichr']
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
ROOT = Path(__file__).resolve().parents[3]
WORKER_TIMEOUT = 120
MAX_OUTPUT = 512_000


class NCBIInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    database: Database = 'protein'
    action: NCBIAction = 'search'
    query: str = Field(default='', max_length=500, description='NCBI Entrez search, e.g. HBA1[Gene Name] AND Homo sapiens[Organism].')
    ids: list[str] = Field(default_factory=list, max_length=10, description='NCBI UIDs, or protein/nucleotide accessions for sequence retrieval.')
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode='after')
    def check_arguments(self):
        if self.action == 'search':
            if not self.query.strip() or self.ids:
                raise ValueError('Search needs query text and no ids; use summary for known UIDs')
        elif not self.ids or self.query:
            raise ValueError('Use ids without query for summary, sequence or abstracts')
        pattern = r'[A-Za-z0-9_.-]{1,40}' if self.action == 'sequence' else r'[0-9]{1,20}'
        if any(not re.fullmatch(pattern, x) for x in self.ids):
            raise ValueError('Use NCBI numeric UIDs (sequence retrieval also accepts accessions)')
        if self.action == 'sequence' and self.database not in {'protein', 'nuccore'}:
            raise ValueError('Sequence retrieval needs protein or nuccore')
        if self.action == 'abstracts' and self.database != 'pubmed':
            raise ValueError('Abstract retrieval needs pubmed')
        return self


async def ncbi_query(**kwargs):
    arg = NCBIInput(**kwargs)
    params = {'db': arg.database, 'tool': 'VenusFactory2'}
    total = None
    ids = arg.ids
    if arg.action == 'search':
        found = await get_json(EUTILS+'esearch.fcgi', {**params, 'term': arg.query, 'retmax': arg.limit, 'retmode': 'json'})
        if not isinstance(found, dict) or 'error' in found or 'esearchresult' not in found:
            raise ServiceError('NCBI did not return a valid search result')
        found = found['esearchresult']
        ids = found.get('idlist', [])[:arg.limit]
        total = int(found.get('count', 0))
    sources = [{'title': 'NCBI '+arg.database, 'url': 'https://www.ncbi.nlm.nih.gov/'+arg.database+'/?'+urlencode({'term': arg.query or ','.join(ids)})}]
    result = {'provider': 'NCBI', 'database': arg.database, 'action': arg.action, 'ids': ids, 'sources': sources}
    if not ids:
        return {**result, 'status': 'empty', 'count': 0, 'total': total or 0, 'records': []}
    if arg.action == 'sequence':
        raw = await request('GET', EUTILS+'efetch.fcgi', params={**params, 'id': ','.join(ids), 'rettype': 'fasta', 'retmode': 'text'}, limit=500_000)
        fasta = raw.decode().strip()
        if not fasta.startswith('>'):
            raise ServiceError('NCBI did not return FASTA sequences for these identifiers')
        return {**result, 'status': 'completed', 'count': sum(x.startswith('>') for x in fasta.splitlines()), 'fasta': fasta}
    if arg.action == 'abstracts':
        raw = await request('GET', EUTILS+'efetch.fcgi', params={**params, 'id': ','.join(ids), 'retmode': 'xml'})
        if b'<!ENTITY' in raw.upper():
            raise ServiceError('Unsupported XML entities')
        tree = ET.fromstring(raw)
        records = []
        for article in tree.findall('.//PubmedArticle')[:10]:
            pmid = article.findtext('.//PMID')
            records.append({'pmid': pmid, 'title': ''.join(article.find('.//ArticleTitle').itertext()) if article.find('.//ArticleTitle') is not None else '',
                            'abstract': '\n'.join(''.join(p.itertext()) for p in article.findall('.//AbstractText')),
                            'url': 'https://pubmed.ncbi.nlm.nih.gov/'+str(pmid)+'/'})
    else:
        data = await get_json(EUTILS+'esummary.fcgi', {**params, 'id': ','.join(ids), 'retmode': 'json'})
        if not isinstance(data, dict) or 'error' in data or 'result' not in data:
            raise ServiceError('NCBI did not return valid record summaries')
        data = data['result']
        records = [data[k] for k in data.get('uids', []) if k in data and 'error' not in data[k]]
    return {**result, 'status': 'completed' if records else 'empty', 'count': len(records), 'total': total if total is not None else len(records), 'records': records}


class GgetInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: GgetOperation
    identifiers: list[str] = Field(min_length=1, max_length=200, description='Ensembl IDs for info/seq/bgee/opentargets, one PDB ID for pdb, or gene symbols for archs4/enrichr.')
    resource: str = Field(default='', max_length=80, description='pdb: entry/pubmed; archs4: tissue/correlation; bgee: orthologs/expression; opentargets: diseases/drugs/expression/interactions/tractability/pharmacogenetics/depmap; enrichr: library name (required).')
    species: Literal['human', 'mouse', 'fly', 'yeast', 'worm', 'fish'] = 'human'
    translate: bool = Field(default=True, description='For seq only: True returns amino acids, False returns nucleotide sequence.')
    limit: int = Field(default=20, ge=1, le=50, description='Maximum returned rows; never truncates sequence characters.')

    @model_validator(mode='after')
    def check_arguments(self):
        op = self.operation
        ids = self.identifiers
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,50}', x) for x in ids):
            raise ValueError('Use plain database identifiers or gene symbols')
        if op in {'info', 'seq'}:
            if len(ids) > 5 or any(not re.fullmatch(r'(?:ENS[A-Z]*[GTP][0-9]+(?:\.[0-9]+)?|WBGene[0-9]+|FBgn[0-9]+|FBtr[0-9]+)', x) for x in ids):
                raise ValueError('info/seq accepts up to five Ensembl, WormBase or FlyBase IDs')
        elif op != 'enrichr' and len(ids) != 1:
            raise ValueError('This operation accepts one identifier per call')
        if op in {'bgee', 'opentargets'} and not re.fullmatch(r'ENS[A-Z]*G[0-9]+', ids[0]):
            raise ValueError('Use an Ensembl gene ID for this operation')
        if op == 'pdb' and not re.fullmatch(r'[0-9][A-Za-z0-9]{3}', ids[0]):
            raise ValueError('Use a four-character PDB ID')
        allowed = {'info': {''}, 'seq': {''}, 'pdb': {'', 'entry', 'pubmed'}, 'archs4': {'', 'tissue', 'correlation'},
                   'bgee': {'', 'orthologs', 'expression'}, 'opentargets': {'', 'diseases', 'drugs', 'expression', 'interactions', 'tractability', 'pharmacogenetics', 'depmap'}}
        if op in allowed and self.resource not in allowed[op]:
            raise ValueError('Unsupported resource for '+op)
        if op == 'archs4' and self.species not in {'human', 'mouse'}:
            raise ValueError('ARCHS4 supports human and mouse')
        if op == 'enrichr' and (len(ids) < 2 or not re.fullmatch(r'[A-Za-z0-9_. -]{1,80}', self.resource)):
            raise ValueError('Enrichr needs at least two gene symbols and an explicit library name')
        return self


async def read_limited(stream, limit):
    data = bytearray()
    while chunk := await stream.read(8192):
        data.extend(chunk)
        if len(data) > limit:
            raise ServiceError('Public query output exceeded its size limit')
    return bytes(data)


async def gget_query(**kwargs):
    arg = GgetInput(**kwargs)
    from mcp_tools.common.installation import gget_python
    python = gget_python(ROOT)
    if not python.is_file():
        return {'status': 'unavailable', 'provider': 'gget', 'summary': 'The gget query runtime is not installed on this server.'}
    from backend.web_v2.job_queue import resource_slot
    async with resource_slot('api:gget', kind='public-genomics'):
        with tempfile.TemporaryDirectory(prefix='venus-gget-') as cwd:
            env = {'PATH': '/usr/bin:/bin', 'HOME': cwd, 'TMPDIR': cwd, 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MPLCONFIGDIR': cwd, 'MPLBACKEND': 'Agg'}
            proc = await asyncio.create_subprocess_exec(str(python), '-I', str(Path(__file__).with_name('gget_worker.py')),
                                                       stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, cwd=cwd, env=env)
            try:
                proc.stdin.write(arg.model_dump_json().encode())
                await proc.stdin.drain()
                proc.stdin.close()
                raw = await asyncio.wait_for(read_limited(proc.stdout, MAX_OUTPUT), WORKER_TIMEOUT)
                await asyncio.wait_for(proc.wait(), 5)
                if proc.returncode or not raw:
                    raise ServiceError('The gget query worker could not complete the request')
                return json.loads(raw)
            except asyncio.TimeoutError:
                return {'status': 'unavailable', 'provider': 'gget', 'operation': arg.operation, 'summary': 'The public database query timed out; retry later or query fewer identifiers.'}
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()


async def similarity_query(service, *, accession='', sequence='', structure_pdb='', structure_chain='', job_id=''):
    """Submit once, then retrieve using the returned remote job ID."""
    if service not in {'blast', 'foldseek_remote'}:
        raise ValueError('Unknown similarity service')
    if job_id:
        if not re.fullmatch(r'[A-Za-z0-9_-]{8,100}', job_id):
            raise ValueError('Use the remote job ID returned by this tool')
        if any((accession, sequence, structure_pdb, structure_chain)):
            raise ValueError('When checking a job, supply only job_id')
        context = {'remote_job_id': job_id, 'poll_attempts': 1}
    else:
        from .tools_agent import prepare_context
        context = await prepare_context(service, accession, sequence, structure_pdb, structure_chain)
        context.update(submit_only=True, poll_attempts=1)
    return await run_service(service, context)
