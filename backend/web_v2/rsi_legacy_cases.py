"""Local operator library records exported as reproducibility artifacts, never tools."""
from functools import lru_cache
import json
import time


@lru_cache(maxsize=4)
def _metadata(root, epoch):
    from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
    from mcp_tools.models.evolved.catalog import recorded_metadata
    return recorded_metadata(ScientistModelLibrary(root))['models']


def library():
    from mcp_tools.tools.runtime_tool_policy import is_local_mode
    if not is_local_mode():
        raise FileNotFoundError('Case not found')
    from mcp_tools.models.evolved.service import configured_library
    return configured_library()


def card(entry_id, key, origin):
    scope = key['scope']
    return {'id': 'scientist:' + entry_id,
            'title': str(scope.get('protein', 'Protein model')) + ' · ' + str(scope.get('target', 'Research model')),
            'summary': str(origin.get('stage', 'Recorded RSI model')), 'author_email': '',
            'visibility': 'private', 'status': 'recorded', 'created_at': 0, 'scope': scope,
            'origin': 'local_library'}


def cards(query=''):
    from mcp_tools.tools.runtime_tool_policy import is_local_mode
    if not is_local_mode():
        return []
    store = library()
    rows = _metadata(str(store.root), int(time.monotonic() // 60))
    words = query.casefold().split()
    return [card(r['entry_id'], r['key'], r['origin']) for r in rows
            if all(w in json.dumps(r['key'], ensure_ascii=False).casefold() for w in words)]


def documents(entry_id):
    publication = library().inspect_publication(entry_id)
    model, assessment = publication['packet']['model'], publication['packet']['assessment']
    item = card(entry_id, publication['key'], publication['origin'])
    readme = '# ' + item['title'] + '\n\nRecorded RSI model artifact from the local operator library.\n\n'
    readme += ('- `model.py`: generated model implementation.\n- `model.json`: recorded model recipe and parameters.\n'
               '- `evaluation.json`: recorded evaluation.\n- `scope.json`: frozen applicability scope.\n\n'
               'This export preserves the recorded model and evaluation. It does not package the full original dataset, '
               'execution environment or all study episodes. Inspect the source interface and recorded scope before reproducing it. '
               'No new training or numerical verification is performed by this download.\n')
    docs = {'README.md': readme, 'REPORT.md': '# Recorded evaluation\n\n' + json.dumps(assessment, ensure_ascii=False, indent=2),
            'model.py': model['recipe']['source_code'], 'model.json': json.dumps(model, ensure_ascii=False, indent=2),
            'evaluation.json': json.dumps(assessment, ensure_ascii=False, indent=2),
            'scope.json': json.dumps(publication['key']['scope'], ensure_ascii=False, indent=2)}
    return item, docs
