"""Model identity is independent of its execution environment and scientific task.

This is metadata over existing authoritative registries, not another publication
registry. Enrichment never changes an existing model ID, ownership or review state.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from mcp_tools.models.profiles import PROJECT, load_runtime_settings, repository_path

MODEL_FAMILIES=('pretrained','factory','rsi','foundation','utility')
MODEL_TYPES=('foundation','pretrained','finetuned','trained','tool')
MANAGED_RUNTIMES={'prosst','esm1v','esmif1','protssn','proteinmpnn','finetuned','plm_checkpoint'}
MUTATION_RUNTIMES={'ProSST':'prosst','ProSST-2048':'prosst','ESM-1v':'esm1v','ESM-IF1':'esmif1','ProtSSN':'protssn',
                   'ESM2-650M':'mutation_esm2','ESM-1b':'mutation_esm1b','SaProt':'mutation_saprot',
                   'MIF-ST':'mutation_mifst','VenusPLM':'mutation_venusplm','VenusREM (foldseek-based)':'mutation_venusrem'}


@lru_cache(maxsize=1)
def constants():
    return json.loads((PROJECT/'src/constant.json').read_text())

MUTATION_RUNTIMES.update({f'ProSST-{v}':'prosst' for v in (20,128,512,1024,2048,4096)})


def backbone(value):
    value=str(value or '')
    models=constants().get('plm_models',{})
    reverse={repo:name for name,repo in models.items()}
    aliases={'ProtBert':'Rostlab/prot_bert','ESM650M':'facebook/esm2_t33_650M_UR50D','ESM2-650M':'facebook/esm2_t33_650M_UR50D'}
    repo=models.get(value) or aliases.get(value) or (value if value in reverse or '/' in value else '')
    return {'name':value if value in models else reverse.get(repo,value),'repository':repo}


def trained_metadata(config, *, dataset=None, weights=None):
    base=backbone(config.get('plm_model') or config.get('backbone_name'))
    dataset=dataset or config.get('dataset_custom') or config.get('dataset_name') or config.get('dataset')
    if dataset is not None:
        dataset=str(dataset)
        if dataset.startswith(('/', '~')) or '\\' in dataset:dataset=dataset.replace('\\','/').rstrip('/').split('/')[-1]
    result={'schema_version':3,'model_family':'factory','model_type':'finetuned' if base['name'] else 'trained',
            'runtime_id':'plm_checkpoint' if base['name'] else 'tabular',
            'backbone':base if base['name'] else None,'training_dataset':dataset,
            'training_method':config.get('training_method') or config.get('method'),
            'target':config.get('target') or config.get('label_column') or config.get('problem_type'),
            'provenance':'factory_training'}
    if weights:
        path=Path(weights);digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        result['artifact']={'sha256':digest.hexdigest(),'bytes':path.stat().st_size}
        result['version']=digest.hexdigest()[:16]
    return result


def enrich(card):
    """Safe public metadata only: no installation paths, credentials or owner IDs."""
    c=dict(card);a=c.get('adapter','');base=c.get('backbone')
    c['schema_version']=3
    if a=='foundation':
        c.update(model_type='foundation',runtime_id='plm_training')
        c['backbone']=backbone(c.get('repository'))
    elif a in {'function','residue'}:
        dataset=c.get('dataset')
        if not dataset:
            datasets=constants()['web_ui']['residue_mapping_function'].get(c.get('task'),[])
            dataset=datasets[0] if datasets else None
        c.update(model_type='finetuned',runtime_id='finetuned',backbone=backbone(c.get('model')),
                 training_dataset=dataset,provenance='factory_reference_training')
        if dataset:c['name']=dataset+' · '+str(c.get('model',''))
    elif a in {'evolution','structure-evolution'}:
        c.update(model_type='pretrained',runtime_id=MUTATION_RUNTIMES.get(c.get('model'),'builtin_mutation'),provenance='pretrained_method')
    elif a=='design':c.update(model_type='pretrained',runtime_id='proteinmpnn',provenance='pretrained_method')
    elif a.startswith('external:'):
        key=a.split(':',1)[1];c.update(model_type='tool' if key=='vina' else 'pretrained',runtime_id=key,provenance='external_method')
    elif a=='checkpoint':
        c.setdefault('model_type','finetuned' if base and base.get('name') else 'trained')
        c.setdefault('runtime_id','plm_checkpoint');c.setdefault('provenance','factory_training')
    elif a=='personal':c.update(model_type='trained',runtime_id='tabular',provenance='user_training')
    elif a=='scientist':c.update(model_type='trained',runtime_id='reviewed_program',provenance='scientist_rsi')
    elif a=='capability':c.update(model_type='tool',runtime_id='session_sandbox',provenance='scientist_rsi')
    else:c.setdefault('model_type','tool');c.setdefault('runtime_id','builtin_tools')
    # Origin is orthogonal to scientific function, model architecture and queue.
    if a == 'foundation': family = 'foundation'
    elif a in {'function','residue','checkpoint','personal'}: family = 'factory'
    elif a in {'scientist','capability'}: family = 'rsi'
    elif a in {'evolution','structure-evolution','design'} or a.startswith('external:'): family = 'pretrained'
    elif c.get('id') == 'tool:predict_s4pred':
        family = 'pretrained'
        c.update(model_type='pretrained', runtime_id='s4pred', provenance='pretrained_method')
    else: family = 'utility'
    c['model_family'] = family
    return c


def runtime_availability(key, model=None):
    """Configuration readiness, not a claim of scientific validation."""
    if key not in MANAGED_RUNTIMES:return True,None
    try:
        settings=load_runtime_settings()
        cfg=settings.get(key,{})
        if key == 'prosst':
            from mcp_tools.models.pretrained.runners.mutation import prosst_settings
            size,cfg=prosst_settings(model or 'ProSST', settings)
            if cfg.get('structure_weights'):
                folder=repository_path(cfg['structure_weights'])
                if not all((folder/n).is_file() for n in ('AE.pt',f'{size}.joblib')):return False,'not_configured'
        if cfg.get('enabled') is False:return False,'runtime_disabled'
        for field in ['python','weights','structure_weights','config']:
            if cfg.get(field) and not repository_path(cfg[field]).exists():return False,'not_configured'
        if key in {'finetuned','plm_checkpoint'} and (cfg.get('weights') or cfg.get('model_id')):return False,'not_configured'
    except (ValueError,TypeError,OSError):return False,'not_configured'
    return True,None
