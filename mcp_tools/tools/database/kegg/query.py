"""Bounded KEGG evidence queries shared by MCP and Researcher."""
import re
from typing import Literal
from urllib.parse import quote

from pydantic import Field, model_validator
from mcp_tools.tools.research.knowledge_queries import QueryInput, public_query, result
from mcp_tools.tools.research.services import ACCESSION, ServiceError, request

Operation = Literal['search', 'get', 'convert', 'link']
Database = Literal['genes', 'pathway', 'ko', 'enzyme', 'module', 'disease', 'drug', 'compound']
ENTRY = re.compile(r'(?:[a-z][a-z0-9_]{1,9}:[A-Za-z0-9_.-]{1,60}|[KCMDH]\d{5}|[a-z]{2,5}\d{5}|(?:\d+\.){3}[\d-]+)')


class KeggInput(QueryInput):
    operation: Operation = 'search'
    query: str = Field(default='', max_length=120)
    identifier: str = Field(default='', max_length=80, description='One KEGG entry, e.g. hsa:7535 or K07360; convert takes a UniProt accession.')
    database: Database = 'genes'
    limit: int = Field(default=10, ge=1, le=25)

    @model_validator(mode='after')
    def check(self):
        if self.operation == 'search':
            if not self.query or self.identifier or any(ord(c) < 32 for c in self.query) or any(c in self.query for c in '/?#'):
                raise ValueError('Search with a protein, gene or pathway name')
        else:
            pattern = ACCESSION if self.operation == 'convert' else ENTRY
            if self.query or not pattern.fullmatch(self.identifier):
                raise ValueError('Use one resolved identifier, without query text')
            if self.operation == 'convert' and self.database != 'genes':
                raise ValueError('UniProt conversion maps to KEGG genes')
        return self


async def fetch(path):
    return (await request('GET', 'https://rest.kegg.jp/' + path, limit=1_000_000)).decode('utf-8')


def parse_entry(text, identifier):
    fields, current = {}, ''
    for line in text.splitlines():
        if line.startswith('///'):
            break
        field, value = line[:12].strip(), line[12:].strip()
        if field:
            current = field
        if current in {'ENTRY', 'NAME', 'SYMBOL', 'DEFINITION', 'ORGANISM', 'ORTHOLOGY', 'PATHWAY', 'MODULE', 'DISEASE', 'DBLINKS', 'REACTION'} and value:
            fields.setdefault(current, []).append(value[:1200])
    if not fields.get('ENTRY'):
        raise ServiceError('Invalid KEGG entry response')
    def annotations(key):
        return [{'id': parts[0], 'name': parts[1] if len(parts) > 1 else ''}
                for line in fields.get(key, [])[:50] if (parts := line.split(None, 1))]
    return {'id': identifier, 'name': ' '.join(fields.get('NAME', [])),
            'definition': ' '.join(fields.get('DEFINITION', [])), 'symbol': ' '.join(fields.get('SYMBOL', [])),
            'organism': ' '.join(fields.get('ORGANISM', [])), 'pathways': annotations('PATHWAY'),
            'orthology': annotations('ORTHOLOGY'), 'modules': annotations('MODULE'),
            'cross_references': fields.get('DBLINKS', [])[:30], 'url': 'https://www.kegg.jp/entry/' + identifier}


@public_query('KEGG')
async def kegg_query(**kwargs):
    args = KeggInput(**kwargs)
    if args.operation == 'get':
        path = 'get/' + args.identifier
    elif args.operation == 'convert':
        path = 'conv/genes/uniprot:' + args.identifier
    elif args.operation == 'link':
        path = 'link/' + args.database + '/' + args.identifier
    else:
        path = 'find/' + args.database + '/' + quote(args.query, safe='')
    raw = await fetch(path)
    if args.operation == 'get':
        rows = [parse_entry(raw, args.identifier)] if raw.strip() else []
    else:
        rows = []
        for line in raw.splitlines():
            parts = line.split('\t', 1)
            if len(parts) != 2:
                raise ServiceError('Invalid KEGG response')
            rows.append(dict(zip(('id', 'name') if args.operation == 'search' else ('source', 'target'), parts)))
    return result('KEGG', rows[:args.limit], [{'title': 'KEGG', 'url': 'https://rest.kegg.jp/' + path}],
                  total=len(rows), truncated=len(rows) > args.limit)


async def protein_annotations(context):
    accession = context['accession']
    ids = [r.get('id', '') for r in context.get('entry', {}).get('uniProtKBCrossReferences', []) if r.get('database') == 'KEGG']
    ids = list(dict.fromkeys(x for x in ids if ENTRY.fullmatch(x)))
    origin = 'UniProt KEGG cross-references'
    if not ids:
        mapped = await kegg_query(operation='convert', identifier=accession)
        if mapped['status'] == 'unavailable':
            raise ServiceError('KEGG identifier mapping unavailable')
        ids = list(dict.fromkeys(x['target'] for x in mapped['records'] if ENTRY.fullmatch(x['target'])))
        origin = 'KEGG UniProt conversion'
    genes, pathways, orthology, modules = [], {}, {}, {}
    for identifier in ids[:3]:
        value = await kegg_query(operation='get', identifier=identifier)
        if value['status'] == 'unavailable':
            raise ServiceError('KEGG entry unavailable')
        for gene in value['records']:
            genes.append(gene)
            for key, dest in [('pathways', pathways), ('orthology', orthology), ('modules', modules)]:
                for row in gene[key]:
                    item = dest.setdefault(row['id'], {**row, 'genes': [], 'url': 'https://www.kegg.jp/entry/' + row['id']})
                    item['genes'].append(identifier)
    return {'count': len(genes), 'accession': accession, 'mapping_source': origin,
            'genes': genes, 'pathways': list(pathways.values()), 'orthology': list(orthology.values()), 'modules': list(modules.values()),
            'truncated': len(ids) > 3, 'mapped_gene_count': len(ids),
            'interpretation': 'KEGG annotations map the reference protein to genes, pathways and orthology. Pathway membership does not demonstrate activity in the submitted construct or assay.'}
