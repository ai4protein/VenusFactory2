"""Bounded public knowledge APIs shared by native tools, MCP and reports.

Fixed endpoints and GraphQL documents only. Returned records are evidence, never
instructions; upstream failures are distinct from successful empty searches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import json
import re
from typing import Literal
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .services import ServiceError, request

KNOWLEDGE_TOOL_NAMES = ('query_kegg', 'query_opentargets', 'query_ensembl', 'query_europepmc', 'query_pubchem')
OT_URL = 'https://api.platform.opentargets.org/api/v4/graphql'
ENSEMBL_URL = 'https://rest.ensembl.org/'
EUROPEPMC_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
PUBCHEM_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/'
OTOperation = Literal['search', 'target', 'disease', 'drug']
OTResource = Literal['overview', 'diseases', 'drugs', 'tractability', 'expression', 'genetic_constraint']
OTEntity = Literal['all', 'target', 'disease', 'drug']
EnsemblOperation = Literal['symbol', 'lookup', 'xrefs', 'homology', 'sequence']
SequenceType = Literal['protein', 'cdna', 'cds', 'genomic']
PubChemNamespace = Literal['name', 'cid', 'smiles']


class QueryInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class OpenTargetsInput(QueryInput):
    operation: OTOperation = 'search'
    query: str = Field(default='', max_length=200, description='Search text, e.g. BRCA2 or breast cancer. Search first to resolve the current identifier.')
    identifier: str = Field(default='', max_length=80, description='Target: human ENSG gene ID; disease: returned ontology ID; drug: CHEMBL ID.')
    entity: OTEntity = 'all'
    resource: OTResource = 'overview'
    limit: int = Field(default=10, ge=1, le=25)
    page: int = Field(default=0, ge=0, le=99)

    @model_validator(mode='after')
    def check(self):
        if self.operation == 'search':
            if not self.query or self.identifier or self.resource != 'overview':
                raise ValueError('Search needs query text, no identifier, and the overview resource')
        else:
            pattern = {'target': r'ENSG[0-9]{11}', 'disease': r'[A-Za-z][A-Za-z0-9]*_[0-9]+', 'drug': r'CHEMBL[0-9]+'}[self.operation]
            if self.query or not re.fullmatch(pattern, self.identifier) or self.entity != 'all':
                raise ValueError('Use a resolved identifier without query or entity filters')
            if self.operation != 'target' and self.resource != 'overview':
                raise ValueError('Resource selection applies only to a target')
            if self.page and (self.operation == 'drug' or self.operation == 'target' and self.resource in {'overview', 'tractability', 'genetic_constraint'}):
                raise ValueError('This resource is not paginated')
        return self


class EnsemblInput(QueryInput):
    operation: EnsemblOperation = 'symbol'
    identifier: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$', description='Gene symbol for symbol lookup; otherwise an Ensembl stable ID.')
    species: str = Field(default='homo_sapiens', max_length=60, pattern=r'^[a-z][a-z0-9_]*$')
    target_species: str = Field(default='', max_length=60, pattern=r'^(?:[a-z][a-z0-9_]*)?$', description='Optional homology target species, e.g. mus_musculus.')
    sequence_type: SequenceType = 'protein'
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode='after')
    def check(self):
        if self.operation != 'symbol' and not re.fullmatch(r'(?:ENS[A-Z]*[GTP][0-9]+|WBGene[0-9]+|FBgn[0-9]+|FBtr[0-9]+)(?:\.[0-9]+)?', self.identifier):
            raise ValueError('Resolve a symbol first, then use the returned stable ID')
        if self.target_species and self.operation != 'homology':
            raise ValueError('target_species applies only to homology')
        if self.operation != 'sequence' and self.sequence_type != 'protein':
            raise ValueError('sequence_type applies only to sequence retrieval')
        return self


class EuropePMCInput(QueryInput):
    query: str = Field(min_length=1, max_length=500, description='Europe PMC search expression; use EXT_ID:<PMID> AND SRC:MED for one paper. Returns abstracts, not full text.')
    limit: int = Field(default=6, ge=1, le=20)
    cursor: str = Field(default='*', min_length=1, max_length=2000, description='Use next_cursor from the previous page; * starts a new search.')


class PubChemInput(QueryInput):
    query: str = Field(min_length=1, max_length=2000, description='One compound name, numeric CID or SMILES.')
    namespace: PubChemNamespace = 'name'
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode='after')
    def check(self):
        if any(ord(c) < 32 for c in self.query):
            raise ValueError('Use a single-line compound identifier')
        if self.namespace == 'cid' and not re.fullmatch(r'[1-9][0-9]{0,11}', self.query):
            raise ValueError('Use one positive PubChem CID')
        if self.namespace == 'name' and len(self.query) > 200:
            raise ValueError('Use a compound name of at most 200 characters')
        return self


def public_query(provider):
    def decorate(fn):
        @wraps(fn)
        async def wrapped(**kwargs):
            try:
                return await fn(**kwargs)
            except (ServiceError, httpx.HTTPError, json.JSONDecodeError):
                return {'status': 'unavailable', 'provider': provider,
                        'summary': 'The public database could not complete this query. Retry later or narrow the query.'}
        return wrapped
    return decorate


async def fetch(url, params=None, body=None):
    raw = await request('POST' if body is not None else 'GET', url, params=params,
                        json_body=body, limit=1_000_000)
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, (dict, list)):
        raise ServiceError('Invalid database response')
    return value


def result(provider, records, sources, *, total=None, offset=0, **extra):
    total = len(records) if total is None else total
    return {'provider': provider, 'status': 'completed' if records else 'empty',
            'records': records, 'count': len(records), 'total': total,
            'has_more': offset + len(records) < total, 'sources': sources,
            'retrieved_at': datetime.now(timezone.utc).isoformat(), **extra}


def bounded_lists(value, limit=50):
    """Cap unpaginated nested arrays; expose truncation instead of hiding it."""
    truncated = []
    def visit(v, path):
        if isinstance(v, list):
            if len(v) > limit:
                truncated.append(path)
            return [visit(x, path + '[]') for x in v[:limit]]
        if isinstance(v, dict):
            return {k: visit(x, path + '.' + k) for k, x in v.items()}
        return v
    return visit(value, '$'), truncated


@public_query('Open Targets')
async def opentargets_query(**kwargs):
    arg = OpenTargetsInput(**kwargs)
    variables = {'id': arg.identifier, 'page': {'index': arg.page, 'size': arg.limit}}
    if arg.operation == 'search':
        declaration = '$query:String!,$entities:[String!],$page:Pagination'
        selection = 'search(queryString:$query,entityNames:$entities,page:$page){total hits{id name entity description}}'
        variables = {'query': arg.query, 'entities': None if arg.entity == 'all' else [arg.entity], 'page': variables['page']}
        url = 'https://platform.opentargets.org/search?' + urlencode({'q': arg.query})
    else:
        declaration = '$id:String!'
        url = 'https://platform.opentargets.org/' + arg.operation + '/' + arg.identifier
        if arg.operation == 'target':
            fields = 'id approvedSymbol approvedName biotype'
            if arg.resource in {'overview', 'tractability'}:
                fields += ' tractability{label modality value}'
            elif arg.resource == 'genetic_constraint':
                fields += ' geneticConstraint{constraintType exp obs oe oeLower oeUpper score}'
            elif arg.resource == 'expression':
                declaration += ',$page:Pagination'
                fields += ' baselineExpression(page:$page){count rows{targetId targetFromSourceId datatypeId datasourceId unit tissueBiosampleFromSource celltypeBiosampleFromSource min q1 median q3 max qualityControls distribution_score specificity_score}}'
            elif arg.resource == 'diseases':
                declaration += ',$page:Pagination'
                fields += ' associatedDiseases(page:$page){count rows{score disease{id name description}}}'
            elif arg.resource == 'drugs':
                # Current API uses clinical candidates, not the retired knownDrugs field.
                fields += ' drugAndClinicalCandidates{count rows{id maxClinicalStage drug{id name drugType}}}'
            selection = 'target(ensemblId:$id){' + fields + '}'
        elif arg.operation == 'disease':
            declaration += ',$page:Pagination'
            selection = 'disease(efoId:$id){id name description associatedTargets(page:$page){count rows{score target{id approvedSymbol approvedName}}}}'
        else:
            selection = 'drug(chemblId:$id){id name drugType description maximumClinicalStage mechanismsOfAction{rows{mechanismOfAction actionType targets{id approvedSymbol}}}}'
        if '$page' not in declaration:
            variables.pop('page')
    query = 'query(' + declaration + '){meta{apiVersion{x y z} dataVersion{year month iteration}} ' + selection + '}'
    obj = await fetch(OT_URL, body={'query': query, 'variables': variables})
    if not isinstance(obj, dict) or obj.get('errors') or not isinstance(obj.get('data'), dict):
        raise ServiceError('Open Targets query failed')
    data = obj['data']
    if arg.operation not in data:
        raise ServiceError('Open Targets returned an incomplete response')
    record = data[arg.operation]
    sources = [{'title': 'Open Targets', 'url': url}, {'title': 'Open Targets GraphQL API', 'url': OT_URL}]
    extra = {'operation': arg.operation, 'resource': arg.resource, 'version': data.get('meta', {}), 'page': arg.page}
    if record is None:
        return result('Open Targets', [], sources, **extra)
    if arg.operation == 'search':
        rows, total = record['hits'][:arg.limit], record['total']
    elif arg.operation == 'target' and arg.resource in {'diseases', 'drugs', 'expression'} or arg.operation == 'disease':
        field = 'associatedTargets' if arg.operation == 'disease' else {
            'diseases': 'associatedDiseases', 'drugs': 'drugAndClinicalCandidates',
            'expression': 'baselineExpression'}[arg.resource]
        group = record[field]
        start = arg.page * arg.limit if field == 'drugAndClinicalCandidates' else 0
        rows, total = group['rows'][start:start + arg.limit], group['count']
        extra['subject'] = {k: v for k, v in record.items() if k != field}
    else:
        rows, total = [record], 1
    rows, truncated = bounded_lists(rows)
    return result('Open Targets', rows, sources, total=total, offset=arg.page * arg.limit,
                  truncated_fields=truncated, **extra)


@public_query('Ensembl')
async def ensembl_query(**kwargs):
    arg = EnsemblInput(**kwargs)
    identity = arg.identifier.split('.')[0] if arg.operation != 'symbol' else arg.identifier
    params = {'content-type': 'application/json'}
    if arg.operation == 'symbol':
        path = 'lookup/symbol/' + arg.species + '/' + quote(identity, safe='')
    elif arg.operation == 'homology':
        path = 'homology/id/' + arg.species + '/' + identity
        params.update(type='orthologues', format='condensed')
        if arg.target_species:
            params['target_species'] = arg.target_species
    else:
        path = {'lookup': 'lookup/id/', 'xrefs': 'xrefs/id/', 'sequence': 'sequence/id/'}[arg.operation] + identity
        if arg.operation == 'sequence':
            params.update(type=arg.sequence_type, multiple_sequences=1)
    url = ENSEMBL_URL + path
    data = await fetch(url, params)
    if isinstance(data, dict) and data.get('error'):
        raise ServiceError('Ensembl query failed')
    if arg.operation == 'homology':
        rows = [h for item in (data or {}).get('data', []) for h in item.get('homologies', [])]
    else:
        rows = data if isinstance(data, list) else [data] if data else []
    return result('Ensembl', rows[:arg.limit], [{'title': 'Ensembl ' + arg.operation, 'url': url + '?' + urlencode(params)}],
                  total=len(rows), operation=arg.operation, identifier=arg.identifier, species=arg.species)


@public_query('Europe PMC')
async def europepmc_query(**kwargs):
    arg = EuropePMCInput(**kwargs)
    params = {'query': arg.query, 'format': 'json', 'resultType': 'core', 'pageSize': arg.limit, 'cursorMark': arg.cursor}
    obj = await fetch(EUROPEPMC_URL, params)
    if not isinstance(obj, dict) or 'hitCount' not in obj or 'resultList' not in obj:
        raise ServiceError('Europe PMC query failed')
    rows = []
    for paper in obj['resultList'].get('result', [])[:arg.limit]:
        row = {k: paper.get(k) for k in ('id', 'source', 'pmid', 'pmcid', 'doi', 'title', 'authorString', 'pubYear', 'abstractText', 'isOpenAccess', 'isRetracted')}
        row['url'] = 'https://europepmc.org/article/' + quote(str(paper['source']), safe='') + '/' + quote(str(paper['id']), safe='')
        row['evidence_level'] = 'abstract' if paper.get('abstractText') else 'bibliographic_record'
        rows.append(row)
    cursor = obj.get('nextCursorMark')
    more = bool(rows and cursor and cursor != arg.cursor and len(rows) == arg.limit and int(obj['hitCount']) > len(rows))
    value = result('Europe PMC', rows, [{'title': 'Europe PMC search', 'url': 'https://europepmc.org/search?' + urlencode({'query': arg.query})}],
                   total=int(obj['hitCount']), query=arg.query, version=obj.get('version'), next_cursor=cursor if more else None)
    value['has_more'] = more
    return value


@public_query('PubChem')
async def pubchem_query(**kwargs):
    arg = PubChemInput(**kwargs)
    properties = 'MolecularFormula,MolecularWeight,SMILES,ConnectivitySMILES,InChIKey,IUPACName,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount'
    # SMILES containing /, # or + is submitted as a form value, never a path.
    if arg.namespace == 'smiles':
        url = PUBCHEM_URL + 'smiles/property/' + properties + '/JSON'
        raw = await request('POST', url, data={'smiles': arg.query}, limit=1_000_000)
        obj = json.loads(raw)
    else:
        url = PUBCHEM_URL + arg.namespace + '/' + quote(arg.query, safe='') + '/property/' + properties + '/JSON'
        obj = await fetch(url)
    if obj is None:
        rows = []
    elif not isinstance(obj, dict) or 'PropertyTable' not in obj:
        raise ServiceError('PubChem query failed')
    else:
        rows = obj['PropertyTable'].get('Properties', [])
    sources = [{'title': 'PubChem CID ' + str(p['CID']), 'url': 'https://pubchem.ncbi.nlm.nih.gov/compound/' + str(p['CID'])} for p in rows[:arg.limit]]
    return result('PubChem', rows[:arg.limit], sources, total=len(rows), namespace=arg.namespace,
                  evidence_level='database_properties', note='Physicochemical descriptors are not an ADMET or binding prediction.')
