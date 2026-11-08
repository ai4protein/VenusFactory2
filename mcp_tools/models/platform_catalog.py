"""Shared platform model identities and live availability for Plaza and MCP discovery."""
import hashlib
import json
import re
from pathlib import Path
from .catalog_snapshot import catalog_snapshot
PROJECT = Path(__file__).resolve().parents[2]

FEATURED_MODEL_FAMILIES = ('ProSST', 'VenusREM', 'VenusVaccine', 'ProtSolM', 'VenusX', 'VenusPLM', 'VenusMine')


def model_family(*values):
    for family in FEATURED_MODEL_FAMILIES:
        pattern = r'(?<![a-z0-9])' + re.escape(family.casefold()) + r'(?![a-z])'
        if any(re.search(pattern, str(value).casefold()) for value in values):
            return family
    return None


def availability(card):
    from mcp_tools.models.identity import enrich, runtime_availability
    card = enrich(card)
    from mcp_tools.tools.capability_policy import model_disabled, mode
    if model_disabled(card):
        return {**card,'available':False,'disabled_reason':'admin_disabled','runtime_mode':mode()}
    if card.get('validation_status') == 'needs_revalidation':
        return {**card, 'available': False, 'disabled_reason': 'needs_revalidation'}
    if card['adapter'].startswith('external:'):
        from mcp_tools.models.pretrained.design_docking.service import availability
        ready,reason=availability(card['adapter'].split(':',1)[1])
        return {**card,'available':ready,'disabled_reason':reason}
    if card.get('model') == 'VenusREM (foldseek-based)':
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        if not is_tool_allowed('query_foldseek_remote'):
            return {**card, 'available': False, 'disabled_reason': 'admin_disabled', 'runtime_mode': mode()}
        ready, reason = runtime_availability('prosst', model='ProSST')
        if not ready: return {**card, 'available': False, 'disabled_reason': reason}
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    disabled = is_online_mode() and (card['adapter'] in {'discovery', 'scientist'}
        or (card['adapter'] == 'foundation' and card.get('repository') != 'facebook/esm2_t6_8M_UR50D'))
    ready, reason = runtime_availability(card.get('runtime_id'), model=card.get('model'))
    return {**card, 'available': not disabled and ready, 'disabled_reason': 'local_only' if disabled else reason}


@catalog_snapshot
def official_cards():
    cfg = json.loads((PROJECT / 'src/constant.json').read_text())
    ui = cfg['web_ui']
    cards = []
    def add(name, category, adapter, **extra):
        identity = 'builtin:' + hashlib.sha256(json.dumps([name, category, extra], sort_keys=True).encode()).hexdigest()[:24]
        cards.append({'id': identity, 'name': name, 'category': category, 'source': 'platform', 'visibility': 'public',
                      'description': extra.get('task', name), 'adapter': adapter, 'workflow': adapter != 'foundation',
                      'input_kind': 'structure' if adapter in {'design', 'discovery', 'structure-evolution', 'structure-property'} else 'table', **extra})
    for task, datasets in ui['dataset_mapping_function'].items():
        for model in ui['model_mapping_function']:
            for dataset in datasets:
                add(f'{model} · {dataset}', 'function', 'function', task=task, model=model, dataset=dataset)
    for task, datasets in ui['residue_mapping_function'].items():
        for model in ui['model_residue_mapping_function']:
            add(f'{model} · {task}', 'residue', 'residue', task=task, model=model)
    for model in ui['model_mapping_zero_shot']:
        structure = model in {'ESM-IF1', 'MIF-ST', 'SaProt', 'ProtSSN', 'VenusREM (foldseek-based)'} or model.startswith('ProSST-')
        add(model, 'evolution', 'structure-evolution' if structure else 'evolution', model=model, task='Mutation scoring')
    add('ProteinMPNN', 'design', 'design', task='Structure-conditioned sequence design')
    add('VenusMine', 'discovery', 'discovery', task='Structure-guided protein discovery')
    add('Protein descriptors', 'properties', 'descriptors', task='Sequence composition and physicochemical properties')
    for prop in ui['protein_properties_function'][1:]:
        add(prop, 'properties', 'structure-property', task=prop)
    for name, repository in cfg['plm_models'].items():
        add(name + ' · Backbone', 'foundation', 'foundation', task='Protein language model for fine-tuning', repository=repository,
            route='/custom-model/training')
    for card in cards:
        datasets = ui['residue_mapping_function'].get(card.get('task'), []) if card['adapter'] == 'residue' else []
        family = model_family(card['name'], card.get('model', ''), card.get('dataset', ''), *datasets)
        if family:
            card['family'] = family
            # Residue cards previously omitted their VenusX model family.
            if card['adapter'] == 'function':
                card['name'] = card['dataset'] + ' · ' + card['model']
            elif family == 'VenusX' and datasets:
                card['name'] = datasets[0] + ' · ' + card['model']
    from mcp_tools.models.pretrained.design_docking.service import cards as external_cards
    return [availability(c) for c in cards + external_cards()]
