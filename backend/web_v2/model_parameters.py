"""Editable model settings share the existing API's validated request schemas."""
from __future__ import annotations
from functools import lru_cache
import math


@lru_cache(maxsize=8)
def request_model(adapter):
    from backend.web_v2 import model_contracts as api
    return {'design': api.SequenceDesignBody, 'discovery': api.ProteinDiscoveryBody,
            'evolution': api.DirectedEvolutionBody, 'structure-evolution': api.DirectedEvolutionBody}.get(adapter)


BOUND = {'structure_file', 'pdb_file', 'input_mode', 'file_path', 'sequence',
         'enable_ai', 'llm_provider', 'user_api_key'}
DERIVED = {'ca_only', 'use_soluble_model'}


def parameter_schema(adapter):
    if adapter.startswith('external:'):
        from backend.web_v2.external_models import schema
        return schema(adapter.split(':',1)[1])
    if adapter not in {'design','discovery','evolution','structure-evolution'}:return None
    model=request_model(adapter)
    skip=BOUND | (DERIVED if adapter=='design' else {'model_name'})
    properties={k:dict(v) for k,v in model.model_json_schema()['properties'].items() if k not in skip}
    for key,field in model.model_fields.items():
        if key in properties:
            properties[key]['default']=field.get_default(call_default_factory=True)
    if adapter=='design':
        from backend.web_v2.model_contracts import _proteinmpnn_model_options
        checkpoints=_proteinmpnn_model_options()
        properties['model_family']['enum']=['soluble','vanilla','ca']
        properties['model_name']['enum']=sorted(set(v for options in checkpoints.values() for v in options))
        properties['model_name']['choices_by']={'field':'model_family','options':checkpoints}
        properties['model_name']['default']='v_48_020' if 'v_48_020' in checkpoints['soluble'] else checkpoints['soluble'][0]
    if adapter=='discovery':
        properties['mmseqs_threads']['default']=4
        properties['cluster_threads']['default']=4
    return {'type':'object','properties':properties,'additionalProperties':False}


def validated_parameters(adapter, config=None):
    if adapter.startswith('external:'):
        from backend.web_v2.external_models import validate
        return validate(adapter.split(':',1)[1],config)
    schema=parameter_schema(adapter)
    if schema is None:return {}
    config=config or {}
    supplied=config.get('parameters')
    if supplied is None:supplied={k:v for k,v in config.items() if k not in {'columns','sequence_column'}}
    if not isinstance(supplied,dict):raise ValueError('Model parameters must be an object')
    unknown=set(supplied)-set(schema['properties'])
    if unknown:raise ValueError('Unknown model parameters: '+', '.join(sorted(unknown)))
    values={k:v['default'] for k,v in schema['properties'].items()}
    values.update(supplied)
    if adapter=='design':
        family=values['model_family']
        if family not in {'soluble','vanilla','ca'}:raise ValueError('Choose a ProteinMPNN model family')
        allowed=schema['properties']['model_name']['choices_by']['options'][family]
        if values['model_name'] not in allowed:raise ValueError('Choose a supported '+family+' ProteinMPNN checkpoint: '+', '.join(allowed))
        values.update(ca_only=family=='ca',use_soluble_model=family=='soluble')
        body=request_model(adapter)(structure_file='input.pdb',**values)
        if not body.temperatures or any(not math.isfinite(t) or t<=0 for t in body.temperatures):raise ValueError('Sampling temperatures must be positive numbers')
        if body.batch_size>body.num_sequences or body.num_sequences % body.batch_size:raise ValueError('Batch size must divide the number of sequences')
        if body.backbone_noise<0 or not math.isfinite(body.backbone_noise):raise ValueError('Backbone noise must be a non-negative number')
        from mcp_tools.tools.runtime_tool_policy import is_online_mode
        if is_online_mode():
            from backend.web_v2.advanced_tools_api import _ONLINE_SEQUENCE_DESIGN_LIMIT
            if body.num_sequences * len(body.temperatures) > _ONLINE_SEQUENCE_DESIGN_LIMIT:
                raise ValueError(f'Online design supports up to {_ONLINE_SEQUENCE_DESIGN_LIMIT} sequences across all temperatures')
    elif adapter=='discovery':body=request_model(adapter)(pdb_file='input.pdb',**values)
    else:body=request_model(adapter)(**values)
    return {k:v for k,v in body.model_dump().items() if k in values}
