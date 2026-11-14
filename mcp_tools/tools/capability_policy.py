"""Live, mode-isolated administrator gates shared by APIs, tools and workers.

Switches control admission, not weight loading or the scientific validation of a
model. They never override the deployment's hard resource/online restrictions.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import sqlite3
import time
from mcp_tools.tools.tool_surface import WORKSPACE_ALIASES


class CapabilityDisabled(ValueError):
    pass


def mode():
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    return 'online' if is_online_mode() else 'local'


def unavailable_message():
    return 'Unavailable online' if mode() == 'online' else 'Disabled by administrator'


# MCP and native implementations sometimes use different public names.
ALIASES = {
    'proteinmpnn_design': 'proteinmpnn_sequence_design_from_structure',
    'proteinmpnn_score': 'proteinmpnn_sequence_scoring_from_structure',
    'analyze_alphafold_pae': 'analyze_alphafold_pae_by_pae_file',
    'analyze_alphafold_plddt': 'analyze_alphafold_plddt_by_metadata_file',
    'download_alphafold_metadata': 'download_alphafold_metadata_by_uniprot_id',
    'download_alphafold_structure': 'download_alphafold_structure_by_uniprot_id',
    'download_arxiv_paper': 'download_arxiv_paper_by_id',
    'download_blast_homologs': 'download_blast_homologs_by_sequence',
    'download_brenda_compare_organisms': 'download_brenda_compare_organisms_by_ec_number',
    'download_brenda_environmental_parameters': 'download_brenda_environmental_parameters_by_ec_number',
    'download_brenda_kinetic_data': 'download_brenda_kinetic_data_by_ec_number',
    'download_brenda_km_values': 'download_brenda_km_values_by_ec_number',
    'download_brenda_reactions': 'download_brenda_reactions_by_ec_number',
    'download_chembl_drug': 'download_chembl_drug_by_id',
    'download_chembl_molecule': 'download_chembl_molecule_by_id',
    'download_chembl_similarity': 'download_chembl_similarity_by_smiles',
    'download_chembl_substructure': 'download_chembl_substructure_by_smiles',
    'download_clustalo_msa': 'download_clustalo_msa_by_fasta',
    'download_interpro_annotations': 'download_interpro_annotations_by_uniprot_id',
    'download_interpro_metadata': 'download_interpro_metadata_by_id',
    'download_interpro_proteins': 'download_interpro_proteins_by_id',
    'download_interpro_uniprot_list': 'download_interpro_uniprot_list_by_id',
    'download_kegg_conv': 'download_kegg_conv_by_id',
    'download_kegg_ddi': 'download_kegg_ddi_by_id',
    'download_kegg_entry': 'download_kegg_entry_by_id',
    'download_kegg_find': 'download_kegg_find_by_database',
    'download_kegg_info': 'download_kegg_info_by_database',
    'download_kegg_link': 'download_kegg_link_by_id',
    'download_kegg_list': 'download_kegg_list_by_database',
    'download_mmseqs2_homologs': 'download_mmseqs2_homologs_by_sequence',
    'download_ncbi_batch_lookup': 'download_ncbi_batch_lookup_by_symbols',
    'download_openalex_entries': 'download_openalex_entries_by_query',
    'download_openalex_entry': 'download_openalex_entry_by_id',
    'download_pubmed_abstracts': 'download_pubmed_abstracts_by_pmids',
    'download_rcsb_metadata': 'download_rcsb_entry_metadata_by_pdb_id',
    'download_rcsb_search': 'download_rcsb_search_by_query',
    'download_rcsb_structure': 'download_rcsb_structure_by_pdb_id',
    'download_uniprot_entry': 'download_uniprot_retrieve_by_id',
    'download_uniprot_metadata': 'download_uniprot_meta_by_id',
    'download_uniprot_search': 'download_uniprot_search_by_query',
    'download_uniprot_sequence': 'download_uniprot_seq_by_id',
    'download_uniprot_sparql': 'download_uniprot_sparql_by_query',
    'download_foldseek_results': 'download_foldseek_results_by_pdb_file',
    'query_foldseek_search_by_pdb_file': 'download_foldseek_results_by_pdb_file',
    'calculate_physchem_from_sequence': 'calculate_physchem_from_fasta',
    'esmfold_structure_prediction': 'predict_structure_esmfold',
}


def canonical(name):
    return WORKSPACE_ALIASES.get(name, ALIASES.get(name, name))


def workspace_switch_rows(rows):
    """Merge old/new workspace switches conservatively until an admin saves."""
    result={}
    for row in rows:
        row=dict(row);key=row['key']
        if key.startswith('tool:'):
            key='tool:'+WORKSPACE_ALIASES.get(key[5:],key[5:])
        if key in result:
            row['enabled']=int(bool(row['enabled']) and bool(result[key]['enabled']))
            row['revision']=max(row['revision'],result[key]['revision'])
        row['key']=key;result[key]=row
    return result


ADAPTER_TOOLS = {
    'function': ['predict_protein_function'], 'residue': ['predict_residue_function'],
    'evolution': ['zero_shot_mutation_sequence_prediction'],
    'structure-evolution': ['zero_shot_mutation_structure_prediction'],
    'design': ['proteinmpnn_sequence_design_from_structure', 'proteinmpnn_sequence_scoring_from_structure'],
    'discovery': ['download_foldseek_results_by_pdb_file'],
    'descriptors': ['calculate_physchem_from_fasta'],
    'scientist': ['predict_with_evolved_model'], 'capability': ['run_generated_capability'],
    'checkpoint': ['protein_model_predict'], 'foundation': ['train_protein_model'],
    'external:admet_ai': ['predict_admet'],
    'external:ligandmpnn': ['ligandmpnn_design'],
    'external:antibmpnn': ['antibmpnn_design'],
    'external:vina': ['vina_dock'],
    'external:bindcraft': ['bindcraft_design'],
    'external:boltzgen': ['boltzgen_design'],
    'external:diffdock': ['diffdock_dock'],

}


def model_tools(card):
    if card['adapter'] == 'registered-tool': return [card['id'].split(':',1)[1]]
    if card['adapter'] == 'structure-property':
        task = card.get('task', '').lower()
        return ['calculate_rsa_from_pdb' if 'relative' in task or task == 'rsa' else
                'calculate_sasa_from_pdb' if 'surface' in task or task == 'sasa' else 'calculate_ss_from_pdb']
    return ADAPTER_TOOLS.get(card['adapter'], [])


def selector(card):
    return {k: card[k] for k in ('id','adapter','model','task','dataset','repository') if k in card} | {'tools': model_tools(card)}


class CapabilityStore:
    def __init__(self, path=None):
        from backend.web_v2.runtime_paths import runtime_root
        self.snapshot = Path(os.environ['VENUS_CAPABILITY_POLICY_SNAPSHOT']) if path is None and os.getenv('VENUS_CAPABILITY_POLICY_SNAPSHOT') else None
        self.path = self.snapshot or Path(path or runtime_root(mode()) / 'state/capabilities.sqlite3')

    def connect(self):
        if self.snapshot is not None:
            raise CapabilityDisabled("Frozen execution policies are read-only")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        c = sqlite3.connect(self.path, timeout=10)
        self.path.chmod(0o600)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA busy_timeout=10000')
        c.execute('CREATE TABLE IF NOT EXISTS switches (key TEXT PRIMARY KEY, enabled INTEGER NOT NULL, revision INTEGER NOT NULL, selector TEXT NOT NULL, updated REAL NOT NULL, actor TEXT NOT NULL)')
        c.execute('CREATE TABLE IF NOT EXISTS changes (id INTEGER PRIMARY KEY, key TEXT, enabled INTEGER, actor TEXT, at REAL)')
        return c

    def rows(self):
        if self.snapshot is not None:
            from .capability_snapshot import read
            return read(self.snapshot)
        if not self.path.exists(): return {}
        with self.connect() as c:
            return workspace_switch_rows(c.execute('SELECT * FROM switches'))

    def catalog_rows(self):
        from mcp_tools.models.catalog_snapshot import cached
        return cached(('capabilities', str(self.path)), self.rows)

    def enabled(self, key):
        return bool(self.catalog_rows().get(key, {}).get('enabled', True))

    def tool_enabled(self, name):
        name=canonical(name); rows=self.catalog_rows()
        if not rows.get('tool:'+name,{}).get('enabled',True): return False
        # Single-model tools disappear with their model. Multi-model tools stay
        # discoverable and validate the selected variant at invocation time.
        for key,row in rows.items():
            if row['enabled'] or not key.startswith('model:'): continue
            spec=json.loads(row['selector'])
            if (spec.get('adapter') in {'design','discovery','descriptors','structure-property','registered-tool'} or str(spec.get('adapter','')).startswith('external:')) and name in spec.get('tools',[]): return False
        return True

    def set(self, key, enabled, revision, spec, actor):
        if key.startswith('tool:'):key='tool:'+WORKSPACE_ALIASES.get(key[5:],key[5:])
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            old = workspace_switch_rows(c.execute('SELECT * FROM switches')).get(key)
            current = old['revision'] if old else 0
            if current != revision: raise ValueError('Availability changed. Refresh before saving.')
            for old_name,new_name in WORKSPACE_ALIASES.items():
                if key=='tool:'+new_name:c.execute('DELETE FROM switches WHERE key=?',('tool:'+old_name,))
            c.execute('INSERT INTO switches VALUES (?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET enabled=excluded.enabled, revision=excluded.revision, selector=excluded.selector, updated=excluded.updated, actor=excluded.actor',
                      (key, int(enabled), current+1, json.dumps(spec), time.time(), actor))
            c.execute('INSERT INTO changes(key,enabled,actor,at) VALUES (?,?,?,?)', (key,int(enabled),actor,time.time()))
        return current + 1

    def assert_call(self, name, arguments=None):
        name = canonical(name)
        for key, row in self.rows().items():
            if row['enabled']: continue
            if key == 'tool:' + name or (key.startswith('model:') and matches(json.loads(row['selector']), name, arguments or {})):
                raise CapabilityDisabled(unavailable_message())


def matches(spec, name, args):
    """Match an actual execution contract, never natural-language similarity."""
    if name not in spec.get('tools', []): return False
    adapter = spec.get('adapter')
    if adapter in {'scientist','capability','checkpoint'}:
        field = {'scientist':'entry_id','capability':'capability_id','checkpoint':'model_id'}[adapter]
        return str(args.get(field,'')) == spec['id'].split(':',1)[1]
    if adapter in {'function','residue','evolution','structure-evolution','foundation'}:
        model = args.get('model_name') or args.get('model') or args.get('plm_model')
        wanted = spec.get('model') or spec.get('repository')
        if model and wanted and str(model).casefold() != str(wanted).casefold():
            # Module aliases such as esm2 resolve to a configured backbone.
            from mcp_tools.models.profiles import PROJECT
            from mcp_tools.models.pretrained.runners.mutation import canonical_model
            cfg = json.loads((PROJECT / 'src' / 'constant.json').read_text())
            mapping = cfg.get('web_ui',{}).get('model_mapping_function',{})
            same_variant = adapter in {'evolution','structure-evolution'} and canonical_model(model) == canonical_model(wanted)
            if not same_variant and mapping.get(wanted) != model: return False
        if adapter in {'function','residue'}:
            if args.get('task') and args['task'] != spec.get('task'): return False
            datasets = args.get('datasets') or ([args['dataset']] if args.get('dataset') else [])
            if datasets and spec.get('dataset') and spec['dataset'] not in datasets: return False
            if args.get('adapter_path') and spec.get('dataset'):
                return spec['dataset'] in Path(args['adapter_path']).parts
        # Missing selectors can include this variant; do not permit a bypass.
    return True


def model_disabled(card):
    rows = CapabilityStore().catalog_rows()
    if not rows.get('model:'+card['id'],{}).get('enabled',True): return True
    return any(not rows.get('tool:'+canonical(t),{}).get('enabled',True) for t in model_tools(card))


def assert_model_enabled(identity):
    from mcp_tools.models.evolved.policy import is_rsi_model, RSI_ARCHIVE_MESSAGE
    if is_rsi_model(identity): raise CapabilityDisabled(RSI_ARCHIVE_MESSAGE)
    if not CapabilityStore().enabled('model:'+identity): raise CapabilityDisabled(unavailable_message())


def guard_operation(name):
    """Guard direct API/report calls as well as MCP/Agent calls to computation."""
    def decorate(fn):
        signature = inspect.signature(fn)
        def check(args, kwargs):
            bound = signature.bind(*args, **kwargs); bound.apply_defaults()
            values = dict(bound.arguments)
            body = values.get('body')
            if hasattr(body, 'model_dump'): values.update(body.model_dump())
            CapabilityStore().assert_call(name, values)
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def guarded(*args, **kwargs):
                check(args, kwargs)
                return await fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                check(args, kwargs)
                return fn(*args, **kwargs)
        return guarded
    return decorate


def guard_tool(tool):
    if getattr(tool, 'func', None) and not getattr(tool.func, '_availability_guard', False):
        tool.func = guard_operation(tool.name)(tool.func)
        tool.func._availability_guard = True
    if getattr(tool, 'coroutine', None) and not getattr(tool.coroutine, '_availability_guard', False):
        tool.coroutine = guard_operation(tool.name)(tool.coroutine)
        tool.coroutine._availability_guard = True
    return tool
