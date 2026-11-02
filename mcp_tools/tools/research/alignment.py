"""Bounded Foldseek retrieval for VenusREM; no shell/curl or duplicate submissions."""
import asyncio
import hashlib
import json
from pathlib import Path


def alignment_rows(hits, length):
    rows = {}
    for hit in hits:
        query, target = hit.get('qAln', ''), hit.get('dbAln', '')
        start, end = int(hit.get('qStartPos', 0)), int(hit.get('qEndPos', 0))
        if not query or len(query) != len(target) or not 1 <= start <= end <= length:
            continue
        aligned = ''.join(t for q, t in zip(query, target) if q != '-')
        if len(aligned) != end - start + 1 or not set(aligned.upper()) <= set('ACDEFGHIKLMNPQRSTVWYBXZOUJ-'):
            continue
        aligned = '-' * (start - 1) + aligned.upper() + '-' * (length - end)
        rows.setdefault(aligned, str(hit.get('target', 'hit')).replace('\n', ' ').replace('\r', ' '))
    return rows


def paths(pdb_file, output_dir):
    digest = hashlib.sha256(Path(pdb_file).read_bytes()).hexdigest()
    return Path(output_dir) / digest


def retrieval_metadata(pdb_file, output_dir):
    return json.loads((paths(pdb_file, output_dir) / 'retrieval.json').read_text())


def fetch_alignment_fasta(pdb_file, output_dir):
    from .services import run_service, structure_chain
    pdb, sequence, chain = structure_chain(Path(pdb_file).read_text())
    folder = paths(pdb_file, output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    context = {'structure_pdb': pdb, 'structure_chain': chain, 'include_alignments': True}
    result = asyncio.run(run_service('foldseek_remote', context, cache_dir=folder))
    if result['status'] not in {'completed', 'empty'}:
        raise RuntimeError('VenusREM retrieval did not complete: ' + str(result.get('summary', result['status']))
                           + (' · ' + result['url'] if result.get('job_id') else ''))
    hits = result.get('data', {}).get('hits', [])
    rows = alignment_rows(hits, len(sequence))
    if hits and not rows:
        raise RuntimeError('Foldseek returned hits without valid sequence alignments; VenusREM cannot score them')
    (folder / 'retrieval.json').write_text(json.dumps({
        'retrieval_status': 'aligned' if rows else 'no_hits',
        'retrieval_alignment_count': len(rows), 'retrieval_url': result.get('url'),
        'scoring_method': 'VenusREM' if rows else 'ProSST-2048 (no retrieval hits)',
    }))
    if not rows:
        return None
    target = folder / 'alignment.fasta'
    target.write_text(''.join('>' + name + '\n' + seq + '\n' for seq, name in rows.items()))
    return str(target)
