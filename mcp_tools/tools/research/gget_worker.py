"""Isolated gget public-query worker. Input/output are JSON; no CLI commands."""
from __future__ import annotations
import contextlib
import io
import json
import sys
import time
import importlib
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit

HOSTS = {'rest.ensembl.org', 'rest.uniprot.org', 'www.ebi.ac.uk', 'data.rcsb.org',
         'maayanlab.cloud', 'bgee.org', 'www.bgee.org', 'api.platform.opentargets.org'}


def install_http_bounds():
    import requests
    original = requests.sessions.Session.request
    sources = []
    last_request = {}

    def bounded(session, method, url, **kwargs):
        u = urlsplit(url)
        if u.scheme == 'http' and u.hostname == 'rest.ensembl.org':
            u = u._replace(scheme='https')
            url = urlunsplit(u)
        if u.scheme == 'https' and u.netloc == 'bgee.org':
            # The official Bgee API redirects to its www host.
            u = u._replace(netloc='www.bgee.org')
            url = urlunsplit(u)
        if u.scheme != 'https' or u.hostname not in HOSTS or u.port not in {None, 443} or u.username or u.password:
            raise ValueError('Unsupported gget endpoint')
        if len(sources) >= 30:
            raise ValueError('gget request limit reached; query fewer identifiers')
        pause = last_request.get(u.hostname, 0)+0.4-time.monotonic()
        if pause > 0:
            time.sleep(pause)
        last_request[u.hostname] = time.monotonic()
        sources.append({'title': u.hostname, 'url': urlunsplit(u._replace(query='', fragment=''))})
        kwargs.update(timeout=(8, 25), allow_redirects=False, stream=True)
        response = original(session, method, url, **kwargs)
        # These are read-only lookups; never retry an Enrichr list submission.
        if response.status_code in {429, 502, 503, 504} and (method.upper() == 'GET' or u.hostname == 'rest.ensembl.org'):
            response.close()
            time.sleep(2)
            response = original(session, method, url, **kwargs)
        try:
            data = bytearray()
            for chunk in response.iter_content(8192):
                data.extend(chunk)
                if len(data) > 3_000_000:
                    raise ValueError('Public database response too large; narrow the query')
            response._content = bytes(data)
            response._content_consumed = True
            if 300 <= response.status_code < 400:
                raise ValueError('Public database endpoint moved; query is temporarily unavailable')
            return response
        finally:
            response.close()
    requests.sessions.Session.request = bounded
    return sources


def pdb_urlopen(url):
    """Route gget's urllib-based metadata query through the same HTTP bounds."""
    import requests
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPError(url, response.status_code, 'PDB query failed', {}, None)
    stream = io.BytesIO(response.content)
    stream.status = response.status_code
    return stream


def execute(arg, gget):
    op, ids, resource, limit = arg['operation'], arg['identifiers'], arg['resource'], arg['limit']
    if op == 'info':
        # Use Entrez via query_ncbi, rather than scrape NCBI HTML inside gget.
        return gget.info(ids, ncbi=False, pdb=False, json=True, verbose=False, save=False)
    if op == 'seq':
        return gget.seq(ids, translate=arg['translate'], isoforms=False, save=False, verbose=False)
    if op == 'pdb':
        importlib.import_module('gget.gget_pdb').urlopen = pdb_urlopen
        return gget.pdb(ids[0], resource=resource or 'entry', save=False)
    if op == 'archs4':
        return gget.archs4(ids[0], which=resource or 'tissue', gene_count=limit, species=arg['species'], json=True, save=False, verbose=False)
    if op == 'bgee':
        return gget.bgee(ids[0], type=resource or 'orthologs', json=True, verbose=False)
    if op == 'opentargets':
        resource = resource or 'diseases'
        if resource == 'drugs':
            import requests
            document = 'query($id:String!){target(ensemblId:$id){drugAndClinicalCandidates{count rows{id maxClinicalStage drug{id name drugType}}}}}'
            response = requests.post('https://api.platform.opentargets.org/api/v4/graphql',
                                     json={'query': document, 'variables': {'id': ids[0]}})
            response.raise_for_status()
            value = response.json()
            if value.get('errors') or 'target' not in value.get('data', {}):
                raise ValueError('Open Targets query failed')
            target = value['data']['target']
            return target['drugAndClinicalCandidates']['rows'] if target else []
        return gget.opentargets(ids[0], resource=resource, limit=None if resource in {'tractability', 'depmap'} else limit, json=True, verbose=False)
    if op == 'enrichr':
        return gget.enrichr(ids, database=resource, species=arg['species'], json=True, save=False, plot=False, verbose=False)
    raise ValueError('Unsupported gget operation')


def main():
    arg = json.loads(sys.stdin.read(100_000))
    sources = []
    result = {'provider': 'gget', 'operation': arg['operation'], 'identifiers': arg['identifiers']}
    try:
        # gget logs/progress never become a malformed MCP JSON response.
        with contextlib.redirect_stdout(io.StringIO()):
            import gget
            sources = install_http_bounds()
            data = execute(arg, gget)
        result['version'] = gget.__version__
        if arg['operation'] == 'seq':
            if not data or not any(str(x).startswith('>') for x in data):
                raise ValueError('No sequences returned for these identifiers')
            result.update(fasta='\n'.join(data), count=sum(str(x).startswith('>') for x in data), status='completed')
        else:
            count = len(data) if isinstance(data, list) else int(bool(data))
            result.update(data=data[:arg['limit']] if isinstance(data, list) else data, count=min(count, arg['limit']),
                          returned_total=count, truncated=count>arg['limit'], status='completed' if data else 'empty')
        if len(json.dumps(result).encode()) > 450_000:
            raise ValueError('Query result too large; request fewer identifiers or rows')
    except Exception as exc:
        # Worker environment contains no API keys; omit remote response bodies.
        result = {k: v for k, v in result.items() if k in {'provider', 'operation', 'identifiers', 'version'}}
        result.update(status='unavailable', summary='gget could not retrieve this public database result ('+type(exc).__name__+'). Try another resource or retry later.')
    result['sources'] = list({x['url']: x for x in sources}.values())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
