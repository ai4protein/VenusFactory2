"""Server-owned model/route policy shared by the UI and live request admission."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import os

DEFAULT_PATH = Path(__file__).resolve().parents[3] / 'configs/harness/free_execution_catalog.json'

def catalog_path():
    return Path(os.environ.get('VENUS_EXECUTION_CATALOG') or DEFAULT_PATH)

def free_routes(path=None):
    raw = Path(path or catalog_path()).read_bytes()
    data = json.loads(raw)
    routes = []
    identities = set()
    for row in data['models']:
        if row.get('enabled', True) is not True:
            continue
        price = row.get('pricing')
        if price is None:
            continue
        if set(price) != {'input_per_million', 'output_per_million'} or any(type(v) not in (int, float) or v != 0 for v in price.values()):
            continue
        binding = row['binding']
        url = urlsplit(binding['base_url'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid model route')
        protocols = row.get('protocols', [binding.get('protocol', 'chat_completions')])
        ceiling = row.get('max_output_tokens')
        if type(ceiling) is not int or ceiling < 1 or not protocols or len(set(protocols)) != len(protocols):
            raise ValueError('Invalid model limits')
        for protocol in protocols:
            if protocol not in {'chat_completions', 'responses'}:
                raise ValueError('Invalid protocol')
            identity = (binding['model'], binding['base_url'].rstrip('/'), protocol)
            if identity in identities:
                raise ValueError('Duplicate free route')
            identities.add(identity)
            routes.append(dict(model=identity[0], base_url=identity[1], protocol=protocol, max_output_tokens=ceiling))
    return routes, hashlib.sha256(raw).hexdigest()

def configured_free(model, base_url, protocol):
    try:
        routes, _ = free_routes()
        return any((r['model'], r['base_url'], r['protocol']) == (model, base_url.rstrip('/'), protocol) for r in routes)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
