"""Resolve a PDB target without substituting its reference protein sequence."""
import asyncio
import io
import re

from mcp_tools.tools.research.services import ServiceError, get_json, request, sequence


def pdb_identifier(value):
    value = value.strip()
    if re.fullmatch(r'[1-9][A-Za-z0-9]{3}', value):
        return value.upper()
    if re.fullmatch(r'pdb_[A-Za-z0-9]{8}', value, re.I):
        return value.lower()
    raise ValueError('Use a PDB ID such as 4HHB')


def pdb_query(value):
    """Return an explicit PDB lookup, otherwise leave name/UniProt search alone."""
    text = re.sub(r'^pdb\s*:\s*', '', value.strip(), flags=re.I)
    parts = text.split(':')
    if re.fullmatch(r'(?:[1-9][A-Za-z0-9]{3}|pdb_[A-Za-z0-9]{8})', parts[0], re.I):
        if len(parts) > 2 or len(parts) == 2 and not re.fullmatch(r'[A-Za-z0-9]{1,16}', parts[1]):
            raise ValueError('Use a PDB ID and optional chain, such as 4HHB:A')
        return pdb_identifier(parts[0]), parts[1] if len(parts) == 2 else ''
    if text != value.strip() or value.lower().startswith('pdb_'):
        raise ValueError('Use a PDB ID such as 4HHB or 4HHB:A')
    return None


async def resolve_pdb(identifier, chain=''):
    identifier = pdb_identifier(identifier)
    entry = await get_json('https://data.rcsb.org/rest/v1/core/entry/' + identifier)
    if not entry:
        return []
    entities = entry.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])
    if len(entities) > 64:
        raise ServiceError('This structure has too many entities; upload the target protein chain instead')
    slots = asyncio.Semaphore(4)
    async def read(entity_id):
        if not re.fullmatch(r'\d+', str(entity_id)):
            raise ServiceError('Invalid PDB entity identifier')
        async with slots:
            return await get_json(f'https://data.rcsb.org/rest/v1/core/polymer_entity/{identifier}/{entity_id}')
    rows = []
    for entity in await asyncio.gather(*(read(i) for i in entities)):
        entity = entity or {}
        poly = entity.get('entity_poly', {})
        if poly.get('rcsb_entity_polymer_type') != 'Protein':
            continue
        ids = entity.get('rcsb_polymer_entity_container_identifiers', {})
        accessions = ids.get('uniprot_ids', [])
        organism = ', '.join(dict.fromkeys(x.get('ncbi_scientific_name', '') for x in entity.get('rcsb_entity_source_organism', [])))
        for auth_chain in ids.get('auth_asym_ids', []):
            if chain and auth_chain != chain:
                continue
            rows.append({'id': identifier + ':' + auth_chain, 'pdb_id': identifier, 'source': 'pdb',
                         'structure_chain': auth_chain, 'entity_id': ids.get('entity_id'),
                         'accession': accessions[0] if len(accessions) == 1 else '', 'reference_accessions': accessions,
                         'name': entity.get('rcsb_polymer_entity', {}).get('pdbx_description', identifier),
                         'organism': organism, 'length': poly.get('rcsb_sample_sequence_length'),
                         'title': entry.get('struct', {}).get('title', ''),
                         'method': [x.get('method') for x in entry.get('exptl', [])],
                         'resolution_angstrom': entry.get('rcsb_entry_info', {}).get('resolution_combined', []),
                         'url': 'https://www.rcsb.org/structure/' + identifier})
    return rows


async def load_pdb_target(identifier, chain=''):
    rows = await resolve_pdb(identifier, chain)
    if len(rows) != 1:
        raise ServiceError('Select one protein chain from the PDB entry' if rows else 'No matching protein chain in this PDB entry')
    card = rows[0]
    raw = await request('GET', 'https://files.rcsb.org/download/' + card['pdb_id'] + '.cif', limit=10_000_000)
    if not raw:
        raise ServiceError('PDB coordinates are unavailable')
    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select
        from Bio.SeqUtils import seq1
        model = next(iter(MMCIFParser(QUIET=True).get_structure('target', io.StringIO(raw.decode()))))
        chosen = model[card['structure_chain']].copy()
        residues = [r for r in chosen if r.id[0] == ' ' and 'CA' in r]
        seq = sequence(''.join(seq1(r.resname) for r in residues))
        kept = {r.id for r in residues}
        # mmCIF author chain names can exceed the one-character PDB field.
        chosen.id = card['structure_chain'] if len(card['structure_chain']) == 1 else 'A'
        class ProteinOnly(Select):
            def accept_residue(self, residue):
                return residue.id in kept
        writer = PDBIO(); writer.set_structure(chosen)
        out = io.StringIO(); writer.save(out, ProteinOnly())
    except Exception as exc:
        raise ServiceError('Could not extract the selected protein coordinates') from exc
    return {'protein': {**card, 'entity_length': card.get('length'), 'length': len(seq)},
            'sequence': seq, 'structure_pdb': out.getvalue(), 'structure_chain': chosen.id,
            'source_chain': card['structure_chain'], 'pdb_id': card['pdb_id'],
            'interpretation': 'Calculations use observed residues with alpha-carbon coordinates in the selected first-model chain. Missing residues and ligands are excluded. Mapped UniProt/KEGG annotations describe reference proteins, not experimental validation of this construct.'}
