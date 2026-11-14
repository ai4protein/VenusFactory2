"""Model marketplace, private datasets and queued model/tool execution.

Marketplace identifiers are adapters over the existing registries, never a second
Scientist publication authority. Personal models have an independent lifecycle.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from mcp_tools.models.catalog_snapshot import catalog_snapshot

PROJECT = Path(__file__).resolve().parents[2]
AA = 'ACDEFGHIKLMNPQRSTVWY'
CATEGORIES = ['function', 'evolution', 'design', 'docking', 'admet', 'discovery', 'residue', 'properties', 'foundation', 'custom']

# Display priority only; keep registry identifiers and execution contracts stable.
from mcp_tools.models.platform_catalog import FEATURED_MODEL_FAMILIES, model_family

def prioritize_platform_models(cards):
    """Show one entry per in-house family first, then their remaining variants."""
    seen = {}
    ranked = []
    for index, card in enumerate(cards):
        family = card.get('family')
        if card.get('source') == 'platform' and str(card.get('id', '')).startswith('builtin:') and family in FEATURED_MODEL_FAMILIES:
            variant = seen.get(family, 0)
            seen[family] = variant + 1
            priority = (0, variant, FEATURED_MODEL_FAMILIES.index(family))
        else:
            priority = (1, index, 0)
        ranked.append((priority, card))
    return [card for _, card in sorted(ranked, key=lambda item: item[0])]


def catalog_card_summary(card):
    """Load complete input contracts only when a component is selected."""
    return {key: value for key, value in card.items() if key not in {"parameter_schema", "contract"}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.chmod(0o600)
    os.replace(temp, path)


class Platform:
    def __init__(self, root=None):
        default = Path.home() / '.local/share/venusfactory/model-platform' / hashlib.sha256(str(PROJECT).encode()).hexdigest()[:16]
        self.root = Path(root or os.environ.get('VENUS_MODEL_PLATFORM_ROOT', default)).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        from backend.web_v2.job_queue import JobQueue
        self.queue = JobQueue(self.root / 'queue' if root is not None else None)
        self._cache = None
        self._cache_time = 0
        from backend.web_v2.job_records import JobRecords
        self.records = JobRecords(self.queue)
        self.records.protect(self.root)
        marker = self.root / '.job-index-v1'
        if not marker.exists():
            from agent.rsi.engine.storage import file_lock
            with file_lock(self.root / '.jobs.lock'):
                if not marker.exists():
                    for path in (self.root / 'jobs').glob('*.json'):
                        self.register_job(read(path))
                    marker.touch()
        self._cache = None
        self._cache_time = 0

    def path(self, area, identity):
        require(area in {'sessions', 'datasets', 'models', 'jobs'}, 'Unknown collection')
        require(bool(re.fullmatch('[a-f0-9]{32,64}', identity)), 'Invalid identifier')
        return self.root / area / (identity + '.json')

    def owned(self, area, identity, owner):
        p = self.path(area, identity)
        if not p.is_file():
            raise FileNotFoundError('Item not found')
        item = read(p)
        if item.get('owner') != owner:
            raise FileNotFoundError('Item not found')
        return item

    def session(self, token):
        if token and re.fullmatch('[a-f0-9]{64}', token):
            owner = hashlib.sha256(token.encode()).hexdigest()
            if self.path('sessions', owner).is_file():
                return token, owner
        token = secrets.token_hex(32)
        owner = hashlib.sha256(token.encode()).hexdigest()
        write(self.path('sessions', owner), {'created': time.time()})
        return token, owner

    def upload(self, owner, name, content):
        from backend.web_v2.upload_limits import validate_upload_size
        require(bool(content), 'Choose a nonempty file')
        validate_upload_size(len(content))
        suffix = Path(name).suffix.lower()
        require(suffix in {'.csv', '.tsv', '.json', '.fasta', '.fa', '.faa', '.pdb', '.pdbqt', '.sdf', '.mol'}, 'Use CSV, TSV, JSON, FASTA, PDB, PDBQT, SDF or MOL')
        text = content.decode('utf-8-sig')
        if suffix in {'.pdb','.pdbqt'}:
            require(any(line.startswith(('ATOM  ', 'HETATM')) for line in text.splitlines()), 'PDB contains no atoms')
            rows, kind = [], 'structure'
        elif suffix in {'.sdf','.mol'}:
            require('M  END' in text or 'V3000' in text, 'Ligand file contains no molecule record')
            rows,kind=[], 'ligand'
        elif suffix in {'.fa', '.fasta', '.faa'}:
            rows = []
            for line in text.splitlines():
                if line.startswith('>'):
                    rows.append({'id': line[1:].strip(), 'sequence': ''})
                elif line.strip():
                    require(bool(rows), 'FASTA must start with a header')
                    rows[-1]['sequence'] += line.strip().upper()
            require(bool(rows) and all(r['sequence'] for r in rows), 'FASTA has empty sequences')
            kind = 'table'
        elif suffix == '.json':
            rows = json.loads(text)
            require(isinstance(rows, list), 'JSON must contain an array of records')
            kind = 'table'
        else:
            reader = csv.DictReader(io.StringIO(text), delimiter='\t' if suffix == '.tsv' else ',')
            require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Column names must be unique')
            rows = list(reader)
            kind = 'table'
        if kind == 'table':
            require(1 <= len(rows) <= 10000 and all(isinstance(r, dict) for r in rows), 'Use 1–10,000 records')
            require(all(r and len(r) <= 200 and all(isinstance(k, str) and k for k in r) for r in rows), 'Use 1–200 named columns')
            require(all(set(r) == set(rows[0]) for r in rows), 'Every record must have the same columns')
            json.dumps(rows, allow_nan=False)
        item = {'id': uuid.uuid4().hex, 'owner': owner, 'name': Path(name).name[:160], 'kind': kind,
                'rows': rows, 'columns': list(rows[0]) if rows else [], 'count': len(rows), 'created': time.time()}
        if kind == 'structure':
            item['structure'] = text
        elif kind == 'ligand':item['content']=text
        write(self.path('datasets', item['id']), item)
        return self.dataset_card(item)

    @staticmethod
    def dataset_card(item):
        return {**{k: item[k] for k in ['id', 'name', 'kind', 'columns', 'count']}, 'preview': item['rows'][:5]}

    def datasets(self, owner):
        return [self.dataset_card(d) for p in sorted((self.root / 'datasets').glob('*.json'), reverse=True)
                if (d := read(p))['owner'] == owner]

    @staticmethod
    def availability(card):
        from mcp_tools.models.platform_catalog import availability
        return availability(card)

    @staticmethod
    def require_available(card):
        from mcp_tools.models.evolved.policy import is_rsi_model, RSI_ARCHIVE_MESSAGE
        if is_rsi_model(card.get("id", "")) or card.get("adapter") in {"scientist", "capability"}:
            raise ValueError(RSI_ARCHIVE_MESSAGE)
        card = Platform.availability(card)
        if not card.get('available', True):
            if card.get('disabled_reason')=='admin_disabled':
                from mcp_tools.tools.capability_policy import CapabilityDisabled, unavailable_message
                raise CapabilityDisabled(unavailable_message())
            if card.get('disabled_reason')=='needs_revalidation':raise ValueError(card['name']+' needs operator revalidation')
            if card.get('disabled_reason')=='runtime_disabled':raise ValueError(card['name']+' runtime is disabled')
            if card.get('disabled_reason')=='not_configured':raise ValueError(card['name']+' runtime is not configured')
            raise ValueError(card['name'] + ' is available in local mode')
        return card

    @catalog_snapshot
    def official(self):
        from mcp_tools.models.platform_catalog import official_cards
        from backend.web_v2.model_parameters import parameter_schema
        cards = official_cards()
        for card in cards:
            schema = parameter_schema(card["adapter"])
            if schema: card["parameter_schema"] = schema
        return cards

    @catalog_snapshot
    def shared(self, refresh=False):
        from mcp_tools.tools.runtime_tool_policy import is_local_mode
        local = is_local_mode()
        if self._cache is not None and getattr(self, "_cache_local", None) == local and not refresh and time.monotonic() - self._cache_time < 30:
            return self._cache
        cards, issues = self.official(), []
        try:
            from mcp_tools.models.factory.training.model_registry import list_trained_models
            for m in list_trained_models()['models']:
                if m.get('status') == 'registered':
                    cards.append({'id': 'checkpoint:' + m['model_id'], 'name': m['model_id'], 'description': str(m.get('plm_model') or 'Registered checkpoint'),
                                  'category': 'custom', 'source': 'platform', 'visibility': 'public', 'adapter': 'checkpoint', 'predictable': True, 'input_kind': 'table',
                                   **{k:m[k] for k in ('model_type','runtime_id','backbone','training_dataset','training_method','provenance','artifact','version') if k in m}})
        except Exception:
            issues.append('checkpoints')
        self._cache, self._cache_time, self._cache_local = (cards, issues), time.monotonic(), local
        return self._cache

    @catalog_snapshot
    def catalog(self, owner, query='', category='', source='', mine=False, offset=0, limit=24, mode='advanced'):
        shared, issues = ([], []) if mine else self.shared()
        cards = [] if mine else [self.availability(c) for c in shared]
        for p in (self.root / 'models').glob('*.json'):
            m = read(p)
            if m['owner'] == owner or (not mine and m['visibility'] == 'public'):
                cards.append(self.availability(self.model_card(m, owner)))
        registered = len(cards)
        # Legacy mode parameters resolve to the complete catalog.
        counts = {c: sum(m['category'] == c for m in cards) for c in CATEGORIES}
        words = query.casefold().split()
        found = [m for m in cards if (not category or m['category'] == category) and (not source or m['source'] == source)
                 and all(w in json.dumps(m, ensure_ascii=False).casefold() for w in words)]
        found = prioritize_platform_models(found)
        return {'items': found[offset:offset+limit], 'total': len(found), 'registered': registered, 'categories': counts, 'unavailable_sources': issues}

    @catalog_snapshot
    def plaza_catalog(self, owner, query='', category='', source='', mine=False, kind='', offset=0, limit=24, model_type=''):
        from backend.web_v2.plaza_tools import catalog
        base = self.catalog(owner, mine=mine, limit=100000)
        tools, unavailable = ([], []) if mine else catalog(detail=False)
        cards = [{**m, 'node_kind': 'tool' if m['adapter'] == 'capability' else 'model'} for m in base['items']] + tools
        words = query.casefold().split()
        matching = [m for m in cards if (not source or m['source'] == source)
                    and (not kind or m['node_kind'] == kind)
                    and (not model_type or m.get('model_type', 'tool') == model_type)
                    and all(w in json.dumps(m, ensure_ascii=False).casefold() for w in words)]
        counts = {c: sum(m['category'] == c for m in matching) for c in [*CATEGORIES, 'processing', 'search']}
        found = [m for m in matching if not category or m['category'] == category]
        found = prioritize_platform_models(found)
        return {'items': [catalog_card_summary(card) for card in found[offset:offset+limit]], 'total': len(found), 'registered': len(cards),
                'categories': counts, 'unavailable_sources': base['unavailable_sources'] if not mine else [],
                'unavailable_tools': unavailable}

    @staticmethod
    def model_card(m, owner):
        keys = ['name', 'description', 'category', 'visibility', 'target', 'features', 'sequence_column', 'metrics', 'method', 'problem', 'created', 'split', 'training_dataset', 'training_method']
        return {**{k: m.get(k) for k in keys}, 'id': 'user:' + m['id'], 'source': 'community', 'adapter': 'personal',
                'owned': m['owner'] == owner, 'predictable': True, 'input_kind': 'table',
                'model_type':'trained', 'runtime_id':'tabular', 'provenance':'user_training'}

    def model(self, owner, identity, detail=False):
        from mcp_tools.models.evolved.policy import is_rsi_model
        if is_rsi_model(identity):
            raise FileNotFoundError("RSI cases are available in RSI Plaza")
        if identity.startswith('user:'):
            p = self.path('models', identity[5:])
            if not p.exists():
                raise FileNotFoundError('Model not found')
            m = read(p)
            if m['owner'] != owner and m['visibility'] != 'public':
                raise FileNotFoundError('Model not found')
            return self.availability(self.model_card(m, owner))
        cards, _ = self.shared()
        m = next((dict(c) for c in cards if c['id'] == identity), None)
        if m is None:
            raise FileNotFoundError('Model not found')
        return self.availability(m)

    @staticmethod
    def tool(identity):
        require(isinstance(identity, str) and identity.startswith('tool:'), 'Unknown tool')
        from backend.web_v2.plaza_tools import card
        return card(identity.removeprefix('tool:'))

    def visibility(self, owner, identity, visibility):
        m = self.owned('models', identity, owner)
        require(visibility == 'private', 'Request platform publication by email; public registration requires operator selection')
        m['visibility'] = visibility
        write(self.path('models', identity), m)
        return self.model_card(m, owner)


    def register_job(self, job):
        payload = job.get('payload', {})
        title = payload.get('name') or ''
        if not title and payload.get('tool_id'):
            from backend.web_v2.plaza_tools import SPECS
            title = SPECS.get(payload['tool_id'].removeprefix('tool:'), ('', 'Tool'))[1]
        if not title and payload.get('model_id'):
            try: title = self.model(job['owner'], payload['model_id'])['name']
            except (FileNotFoundError, ValueError, KeyError): title = 'Prediction'
        meta = self.records.register('platform', job['id'], job['owner'], job['kind'], title or job['kind'], job['created'])
        self.records.grant(self.root/'results'/(job['id']+'.json'), 'platform', job['owner'])
        self.records.grant(self.root/'work'/job['id'], 'platform', job['owner'])
        return meta

    def resolve_job(self, owner, identity):
        if identity.upper().startswith('VF-'):
            return self.records.resolve('platform', identity, owner)
        return identity

    def job(self, owner, identity, *, metadata=None):
        identity = self.resolve_job(owner, identity)
        job = self.owned('jobs', identity, owner)
        if job.get('queue_id'):
            schedule = self.queue.public(job['queue_id'], owner)
            job['queue'] = schedule
            if job['status'] in {'queued','running'}:
                if schedule['status'] in {'cancelled','interrupted','failed'}:
                    job.update(status=schedule['status'],message=schedule['message'])
                    write(self.path('jobs',identity),{k:v for k,v in job.items() if k!='queue'})
                elif schedule['status']=='queued':job['message']=schedule['message']
                elif schedule['status']=='completed' and job['status']!='completed':
                    # The worker may finish between the file read and queue read.
                    job = self.owned('jobs', identity, owner)
                    job['queue'] = schedule
                    if job['status'] in {'queued', 'running'}:
                        job.update(status='interrupted',message='Worker ended without results. Retry to continue.')
        elif job['status'] in {'queued', 'running'} and time.time() - job['created'] > 10:
            try: os.kill(job.get('pid', 0) or -999999, 0)
            except ProcessLookupError:
                job.update(status='interrupted', message='Run interrupted. Retry to continue.')
                write(self.path('jobs', identity), job)
        metadata = metadata or self.records.metadata('platform', identity, owner) or self.register_job(job)
        public = {**{k: v for k, v in job.items() if k not in {'owner', 'payload', 'pid', 'queue_id'}}, **metadata}
        return public

    def start(self, owner, kind, payload, launch=True):
        require(kind in {'train', 'tool', 'predict'}, 'Unknown job')
        if kind == 'tool':
            from backend.web_v2.plaza_tools import validate_input
            tool = self.require_available(self.tool(payload['tool_id']))
            data = self.owned('datasets', payload['dataset_id'], owner) if payload.get('dataset_id') else None
            validate_input(payload['tool_id'].removeprefix('tool:'), data, payload.get('config'))
        elif kind == 'predict':
            model = self.require_available(self.model(owner, payload['model_id']))
            from backend.web_v2.model_parameters import validated_parameters
            validated_parameters(model['adapter'], payload.get('config'))
            data=self.owned('datasets', payload['dataset_id'], owner)
            if model['adapter'].startswith('external:'):
                from backend.web_v2.external_models import validate_inputs
                validate_inputs(self,owner,model['adapter'].split(':',1)[1],payload.get('config'),data)
        else:
            from mcp_tools.tools.capability_policy import CapabilityStore
            CapabilityStore().assert_call('train_protein_model', {'plm_model':'classical'})
            self.validate_training(owner, payload)
        from agent.rsi.engine.storage import file_lock
        from backend.web_v2.compute_access import request_key_context
        admission_key = request_key_context.get()
        admission_hash = hashlib.sha256(json.dumps({'kind': kind, 'payload': payload}, sort_keys=True).encode()).hexdigest()
        with file_lock(self.root / '.jobs.lock'):
            if launch and admission_key:
                previous = self.queue.request(owner, admission_key, admission_hash)
                if previous:
                    original = self.queue.get(previous['id'], owner)
                    return self.job(owner, original['metadata']['platform_job_id'])
            job = {'id': uuid.uuid4().hex, 'owner': owner, 'kind': kind, 'payload': payload, 'status': 'queued', 'progress': 0,
                   'message': 'Queued', 'created': time.time(), 'steps': []}
            write(self.path('jobs', job['id']), job)
            if launch:
                env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'), VENUS_MODEL_PLATFORM_ROOT=str(self.root),
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                work = self.root / 'work' / job['id']
                work.mkdir(parents=True, mode=0o700)
                env['TEMP_OUTPUTS_DIR'] = str(work)
                from backend.web_v2.job_queue import launch
                try:
                    with (work / 'worker.log').open('ab') as log:
                        queue_id, proc = launch([sys.executable, '-m', 'backend.web_v2.model_platform_worker', job['id']], owner=owner,
                            kind='platform:'+kind, resource=self.resource_for(owner,kind,payload), env=env,cwd=PROJECT,stdout=log,
                            timeout=int(os.getenv('VENUS_QUEUE_MODEL_TIMEOUT','3600')), priority=10 if kind=='predict' else 0,
                            metadata={'platform_job_id':job['id'], 'compute_demands':self.compute_demands(owner,kind,payload)}, queue=self.queue,
                            request_key=admission_key, request_hash=admission_hash if admission_key else None)
                except Exception:
                    self.path('jobs', job['id']).unlink(missing_ok=True)
                    raise
                job.update(pid=proc.pid,queue_id=queue_id)
                write(self.path('jobs',job['id']),job)
            self.register_job(job)
        return self.job(owner, job['id'])

    def compute_demands(self, owner, kind, payload):
        from mcp_tools.scheduler.catalog import classify
        from backend.web_v2.compute_access import prediction_tool
        def prediction(identity):
            model = self.tool(identity) if identity.startswith('tool:') else self.model(owner, identity)
            if model.get('adapter') == 'registered-tool':
                from mcp_tools.scheduler.catalog import native_tools
                name = identity.removeprefix('tool:')
                tool = native_tools().get(name)
                return int(prediction_tool(name, classify(name, getattr(tool, 'metadata', None))['category']))
            return int(model.get('adapter', '').startswith('external:') or
                       (model.get('adapter') != 'descriptors' and model.get('model_type') != 'tool'))
        if kind == 'predict':
            return {'prediction': prediction(payload['model_id'])}
        if kind == 'tool':
            return {'prediction': prediction(payload['tool_id'])}
        return {}

    def resource_for(self, owner, kind, payload):
        if kind=='tool': return 'api:orchestration'
        if kind=='predict' and self.model(owner,payload['model_id'])['adapter'] in {'external:'+k for k in ('ligandmpnn','antibmpnn','vina','bindcraft','boltzgen','diffdock')}: return 'api:orchestration'
        if kind=='train': return 'cpu'
        ids=[payload['model_id']] if kind=='predict' else []
        gpu={'function','residue','evolution','structure-evolution','design','discovery','checkpoint','structure-property',
             'external:ligandmpnn','external:antibmpnn','external:bindcraft','external:boltzgen','external:diffdock'}
        return 'gpu' if any(self.model(owner, identity)['adapter'] in gpu for identity in ids) else 'cpu'

    def jobs(self, owner):
        return [self.job(owner, item['identity'], metadata={'job_number': item['number'], 'title': item['title']}) for item in self.records.entries('platform', owner)
                if item['kind'] in {'train', 'predict', 'tool'} and self.path('jobs', item['identity']).is_file()]

    def job_page(self, owner, **filters):
        from backend.web_v2.job_records import page_jobs
        return page_jobs(self.jobs(owner), **filters)

    def cancel_job(self, owner, identity):
        identity = self.resolve_job(owner, identity)
        job=self.owned('jobs',identity,owner)
        if job.get('queue_id'):self.queue.cancel(job['queue_id'],owner)
        elif job['status'] in {'queued','running'}:raise ValueError('This older run cannot be cancelled from the queue')
        return self.job(owner,identity)

    def retry_job(self, owner, identity):
        identity = self.resolve_job(owner, identity)
        job=self.owned('jobs',identity,owner)
        if self.job(owner,identity)['status'] not in {'failed','interrupted','cancelled'}:raise ValueError('Only stopped tasks can retry')
        return self.start(owner,job['kind'],job['payload'])

    def validate_training(self, owner, p, data=None):
        d = data if data is not None else self.owned('datasets', p.get('dataset_id', ''), owner)
        require(d['kind'] == 'table' and len(d['rows']) >= 12, 'Training requires at least 12 rows')
        require(0 < len(str(p.get('name', '')).strip()) <= 120, 'Enter a model name')
        require(p.get('target') in d['columns'], 'Select a target column')
        features = p.get('features', [])
        seq = p.get('sequence_column') or ''
        group = p.get('group_column') or ''
        require(bool(features) or bool(seq), 'Choose numeric features or a sequence column')
        require(isinstance(features, list) and len(features) == len(set(features)) and set(features) <= set(d['columns']), 'Invalid feature columns')
        require(not seq or seq in d['columns'], 'Invalid sequence column')
        require(not group or group in d['columns'], 'Invalid group column')
        require(p['target'] not in features and p['target'] != seq and (not group or (group not in features and group not in {seq, p['target']})), 'Target and group columns cannot be features')
        require(p.get('problem') in {'regression', 'classification'}, 'Choose a prediction type')
        require(p.get('method') in {'linear', 'forest', 'extra_trees'}, 'Choose a training method')
        require(p.get('category', 'custom') in CATEGORIES, 'Invalid category')
        return d

    @staticmethod
    def matrix(rows, features, sequence_column):
        import numpy as np
        result = []
        for row in rows:
            try:
                values = [float(row[k]) for k in features]
                if sequence_column:
                    seq = str(row[sequence_column]).strip().upper()
                    require(bool(seq) and set(seq) <= set(AA), 'Sequences must use the 20 standard amino acids')
                    values += [len(seq)] + [seq.count(a) / len(seq) for a in AA]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Check feature columns: numeric values and valid protein sequences are required') from exc
            require(all(np.isfinite(v) for v in values), 'Features must contain finite numbers')
            result.append(values)
        return np.asarray(result)

    def train(self, job, update, data=None):
        from backend.web_v2.job_queue import run_in_slot
        return run_in_slot('cpu', self._train, job, update, data)

    def _train(self, job, update, data=None):
        from mcp_tools.tools.capability_policy import CapabilityStore
        CapabilityStore().assert_call('train_protein_model', {'plm_model':'classical'})
        import joblib
        import numpy as np
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
        from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score, accuracy_score
        p, owner = job['payload'], job['owner']
        d = self.validate_training(owner, p, data)
        update(10, 'Preparing features')
        x = self.matrix(d['rows'], p.get('features', []), p.get('sequence_column'))
        regression = p['problem'] == 'regression'
        y = np.asarray([float(r[p['target']]) if regression else str(r[p['target']]) for r in d['rows']])
        require(all(np.isfinite(y)) if regression else all(str(v).strip() for v in y), 'Target column contains missing or invalid values')
        if not regression:
            _, counts = np.unique(y, return_counts=True)
            require(2 <= len(counts) <= min(50, len(y) // 3) and min(counts) >= 3, 'Classification needs at least three rows per class')
        group = p.get('group_column') or p.get('sequence_column')
        if group:
            groups = [str(r[group]) for r in d['rows']]
            require(len(set(groups)) >= 5, 'Use at least five independent groups for validation')
            train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(x, y, groups))
            split = {'kind': 'group', 'column': group, 'seed': 42}
        else:
            train, valid = train_test_split(np.arange(len(y)), test_size=.25, random_state=42, stratify=None if regression else y)
            split = {'kind': 'random', 'seed': 42}
        require(len(valid) >= 2 and (regression or set(y[valid]) <= set(y[train])), 'Validation contains classes absent from training; adjust grouping')
        method = p['method']
        if method == 'linear':
            model = make_pipeline(StandardScaler(), Ridge(alpha=1) if regression else LogisticRegression(max_iter=1000, random_state=42))
        else:
            cls = (RandomForestRegressor if regression else RandomForestClassifier) if method == 'forest' else (ExtraTreesRegressor if regression else ExtraTreesClassifier)
            model = cls(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=2)
        update(30, 'Training on the training partition')
        model.fit(x[train], y[train])
        update(70, 'Evaluating the held-out partition')
        prediction = model.predict(x[valid])
        metrics = {'MAE': float(mean_absolute_error(y[valid], prediction)), 'R2': float(r2_score(y[valid], prediction))} if regression else {'accuracy': float(accuracy_score(y[valid], prediction)), 'balanced_accuracy': float(balanced_accuracy_score(y[valid], prediction))}
        require(all(np.isfinite(v) for v in metrics.values()), 'Validation produced a non-finite metric')
        split.update(train_rows=len(train), validation_rows=len(valid))
        update(85, 'Fitting the reusable model on all rows')
        model.fit(x, y)
        identity = uuid.uuid4().hex
        checkpoint = self.root / 'weights' / (identity + '.joblib')
        checkpoint.parent.mkdir(exist_ok=True, mode=0o700)
        joblib.dump(model, checkpoint)
        checkpoint.chmod(0o600)
        m = {'id': identity, 'owner': owner, 'visibility': 'private', 'created': time.time(), 'metrics': metrics, 'split': split,
             **{k: p.get(k, '') for k in ['name', 'description', 'category', 'target', 'features', 'sequence_column', 'method', 'problem']}}
        m['category'] = p.get('category') or 'custom'
        write(self.path('models', identity), m)
        return {'model': self.model_card(m, owner)}

    def predict(self, owner, identity, data, config=None):
        m = self.require_available(self.model(owner, identity))
        key = m['adapter'].removeprefix('external:')
        if key in {'ligandmpnn','antibmpnn','vina','bindcraft','boltzgen','diffdock'}:
            from backend.web_v2.external_dispatch import predict
            return predict(self, owner, key, data, config or {})
        from backend.web_v2.job_queue import run_in_slot
        resource = self.resource_for(owner, 'predict', {'model_id': identity})
        return run_in_slot(resource, self._predict_local, owner, identity, data, config)

    def _predict_local(self, owner, identity, data, config=None):
        config = config or {}
        m = self.require_available(self.model(owner, identity))
        require(m['predictable'], 'Use this backbone in the training workspace')
        require(data['kind'] in m.get('accepted_input_kinds',[m['input_kind']]), 'This model requires ' + m['input_kind'] + ' input')
        rows = data.get('rows', [])
        require(data['kind'] in {'structure','ligand'} or bool(rows), 'Input contains no records')
        adapter = m['adapter']
        if adapter.startswith('external:'):
            from backend.web_v2.external_models import run
            return run(self,owner,adapter.split(':',1)[1],data,config)
        if adapter == 'personal':
            import joblib
            model = joblib.load(self.root / 'weights' / (identity[5:] + '.joblib'))
            x = self.matrix(rows, m['features'], m['sequence_column'])
            predictions = model.predict(x).tolist()
            name = str(config.get('output_column') or m['target'] + '_prediction')
            require(name not in rows[0], 'Choose an output column that does not already exist')
            output = [{**r, name: v} for r, v in zip(rows, predictions)]
        elif adapter == 'descriptors':
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            output = []
            for row in rows:
                seq = str(row.get(config.get('sequence_column', 'sequence'), '')).strip().upper()
                require(bool(seq) and set(seq) <= set(AA), 'Select a valid protein sequence column')
                a = ProteinAnalysis(seq)
                output.append({**row, 'length': len(seq), 'molecular_weight': a.molecular_weight(), 'isoelectric_point': a.isoelectric_point(), 'gravy': a.gravy(),
                               **{'fraction_' + aa: seq.count(aa) / len(seq) for aa in AA}})
        else:
            output = self.run_existing(m, data, config)
        return {'kind': 'table', 'rows': output}

    def run_existing(self, m, data, config):
        if m['adapter'] == 'checkpoint':
            import pandas as pd
            from mcp_tools.models.factory.training.train_operations import run_predict_tool
            from mcp_tools.common.common_utils import get_temp_outputs_base_dir
            folder = get_temp_outputs_base_dir() / ('checkpoint_' + uuid.uuid4().hex)
            folder.mkdir(parents=True, mode=0o700)
            seq_col = config.get('sequence_column', 'sequence')
            require(all(r.get(seq_col) for r in data['rows']), 'Select the protein sequence column')
            file = folder / 'inputs.csv'
            pd.DataFrame([{**r, 'aa_seq': r[seq_col]} for r in data['rows']]).to_csv(file, index=False)
            result = json.loads(run_predict_tool(m['id'].split(':', 1)[1], csv_file=str(file)))
            require(result.get('success') and (folder / 'predictions.csv').is_file(), 'Checkpoint prediction failed; check weights and input requirements')
            return json.loads(pd.read_csv(folder / 'predictions.csv').to_json(orient='records'))
        from backend.web_v2 import advanced_tools_api as advanced
        seq_col = config.get('sequence_column', 'sequence')
        if data['kind'] == 'structure':
            from mcp_tools.common.common_utils import get_web_v2_area_dir
            file = get_web_v2_area_dir('uploads', tool='model_platform') / (uuid.uuid4().hex + '.pdb')
            file.write_text(data['structure'])
            file_path = str(file)
        else:
            require(all(r.get(seq_col) for r in data['rows']), 'Select the protein sequence column')
            sequence = '\n'.join('>row_' + str(i) + '\n' + str(r[seq_col]) for i, r in enumerate(data['rows']))
        a = m['adapter']
        from backend.web_v2.model_parameters import validated_parameters
        parameters = validated_parameters(a, config)
        if a == 'function':
            result = asyncio.run(advanced.run_protein_function(advanced.ProteinFunctionBody(sequence=sequence, task=m['task'], model_name=m['model'], datasets=[m['dataset']], enable_ai=False)))
        elif a == 'residue':
            result = asyncio.run(advanced.run_functional_residue(advanced.FunctionalResidueBody(sequence=sequence, task=m['task'], model_name=m['model'], enable_ai=False)))
        elif a in {'evolution', 'structure-evolution'}:
            result = asyncio.run(advanced.run_directed_evolution(advanced.DirectedEvolutionBody(input_mode='structure' if a == 'structure-evolution' else 'sequence', file_path=file_path if a == 'structure-evolution' else None, sequence=sequence if a == 'evolution' else None, model_name=m['model'], enable_ai=False, **parameters)))
        elif a == 'design':
            result = asyncio.run(advanced.run_sequence_design(advanced.SequenceDesignBody(structure_file=file_path, **parameters)))
        elif a == 'discovery':
            result = asyncio.run(advanced.run_protein_discovery(advanced.ProteinDiscoveryBody(pdb_file=file_path, **parameters)))
        elif a == 'structure-property':
            from backend.web_v2.quick_tools_api import run_quick_tool_properties, PropertiesRunBody
            result = asyncio.run(run_quick_tool_properties(PropertiesRunBody(task=m['task'], file_path=file_path)))
        else:
            raise ValueError('Model adapter unavailable')
        output = result.get('table') or result.get('rows') or result.get('results') or result.get('data')
        if isinstance(output, dict):
            output = [{'id': k, **v} if isinstance(v, dict) else {'id': k, 'value': v} for k, v in output.items()]
        require(isinstance(output, list) and bool(output), 'The model returned no table. Check the model weights and input requirements.')
        if data['kind'] == 'table' and len(output) == len(data['rows']):
            return [{**r, **v} for r, v in zip(data['rows'], output)]
        return output

    def execute(self, identity):
        path = self.path('jobs', identity)
        job = read(path)
        def update(progress, message):
            job.update(status='running', progress=progress, message=message)
            write(path, job)
        try:
            update(1, 'Starting')
            if job['kind'] == 'train':
                result = self.train(job, update)
            elif job['kind'] == 'predict':
                p = job['payload']
                data = self.owned('datasets', p['dataset_id'], job['owner'])
                update(20, 'Running model')
                result = self.predict(job['owner'], p['model_id'], data, p.get('config'))
            elif job['kind'] == 'tool':
                from backend.web_v2.plaza_tools import run
                p = job['payload']
                self.require_available(self.tool(p['tool_id']))
                data = self.owned('datasets', p['dataset_id'], job['owner']) if p.get('dataset_id') else None
                result = run(p['tool_id'].removeprefix('tool:'), data, p.get('config'), self.root / 'work' / identity, progress=update)
            else:
                raise ValueError('This job type is no longer supported')
            result_path = self.root / 'results' / (identity + '.json')
            write(result_path, result)
            preview = result.get('rows') or next((v['rows'] for v in result.get('outputs', {}).values() if v.get('rows')), [])
            job.update(status='completed', progress=100, message='Completed', result={'model': result.get('model'), 'models': result.get('models', []), 'preview': preview[:20], 'rows': len(preview), 'download': '/api/model-platform/jobs/' + identity + '/result'})
            metadata=result.get('metadata') or next((v.get('metadata') for v in result.get('outputs',{}).values() if v.get('metadata')),None)
            if metadata:job['result']['metadata']=metadata
            if result.get('artifacts') or any(v.get('artifacts') for v in result.get('outputs',{}).values()):
                job['result']['artifacts_download']='/api/model-platform/jobs/'+identity+'/artifacts'
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if job['steps']:
                for step in job['steps']:
                    if step['status'] == 'running':
                        step.update(status='failed', finished=time.time())
                    elif step['status'] == 'queued': step.update(status='skipped', message='Run stopped before this step')
            message = str(exc) if isinstance(exc, (ValueError, KeyError)) else 'Run failed. Check the input schema and model availability.'
            job.update(status='failed', message=message[:300])
        job['finished'] = time.time()
        write(path, job)

    def artifact_archive(self,owner,identity):
        import zipfile
        identity=self.resolve_job(owner,identity)
        job=self.owned('jobs',identity,owner)
        require(job['status']=='completed','Results are not ready')
        result=read(self.root/'results'/(identity+'.json'))
        files=list(result.get('artifacts',[]))
        for value in result.get('outputs',{}).values():files.extend(value.get('artifacts',[]))
        require(bool(files),'No output files')
        work=(self.root/'work'/identity).resolve()
        archive=work/'model-results.zip'
        from agent.rsi.engine.storage import file_lock
        with file_lock(work/'.archive.lock'):
            temp=work/('archive-'+uuid.uuid4().hex+'.tmp')
            try:
                with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
                    for name in sorted(set(files)):
                        p=Path(name)
                        require(not p.is_symlink() and p.resolve().is_relative_to(work) and p.is_file(),'Output file is unavailable')
                        z.write(p,p.resolve().relative_to(work))
                os.replace(temp,archive)
            finally:
                if temp.exists():temp.unlink()
        return archive
