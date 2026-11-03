"""Dataflow adapters over registered tools; paths are managed by the workflow."""
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import json

# name: (registry module, display name, category, input kind, managed input)
SPECS = {
    'predict_s4pred': ('research','S4PRED','properties','table','sequence'),
    'query_interproscan': ('research','InterProScan','discovery','table','sequence'),
    'query_ncbi_blast': ('research', 'NCBI BLAST', 'discovery', 'table', 'sequence'),
    'query_foldseek_remote': ('research', 'Foldseek', 'discovery', 'structure', 'structure_pdb'),
    'calculate_physchem_from_fasta': ('predict', 'Sequence physicochemical properties', 'properties', 'table', 'fasta_file'),
    'calculate_rsa_from_pdb': ('predict', 'Relative solvent accessibility', 'properties', 'structure', 'pdb_file'),
    'calculate_sasa_from_pdb': ('predict', 'Solvent accessible surface area', 'properties', 'structure', 'pdb_file'),
    'calculate_ss_from_pdb': ('predict', 'Secondary structure', 'properties', 'structure', 'pdb_file'),
    'read_fasta': ('file', 'Read FASTA sequences', 'processing', 'table', 'file_path'),
    'pdb_chain_sequences': ('file', 'Extract PDB chain sequences', 'processing', 'structure', 'pdb_file'),
    'get_seq_from_pdb_chain_a': ('file', 'Extract PDB chain A', 'processing', 'structure', 'pdb_file'),
    'check_pdb_apo': ('file', 'Check apo structure', 'properties', 'structure', 'pdb_file'),
    'query_pubmed': ('search', 'PubMed search', 'search', 'table', None),
    'query_arxiv': ('search', 'arXiv search', 'search', 'table', None),
    'query_biorxiv': ('search', 'bioRxiv search', 'search', 'table', None),
    'query_duckduckgo': ('search', 'Web search', 'search', 'table', None),
    'query_github': ('search', 'GitHub search', 'search', 'table', None),
    'query_hugging_face': ('search', 'Models and datasets search', 'search', 'table', None),
    'query_semantic_scholar': ('search', 'Semantic Scholar search', 'search', 'table', None),
}
MANAGED = {'fasta_file', 'pdb_file', 'file_path', 'output_file', 'out_dir', 'api_key'}


def registered(name):
    if name not in SPECS:
        raise ValueError('Tool has no workflow adapter')
    tool = getattr(import_module('mcp_tools.tools.' + SPECS[name][0] + '.tools_agent'), name + '_tool')
    if tool.name != name:
        raise ValueError('Tool registration does not match')
    return tool


def card(name, *, detail=True):
    if name not in SPECS:
        raise ValueError('Unknown workflow tool')
    spec = SPECS[name]
    schema = None
    if detail:
        tool = registered(name)
        schema = tool.args_schema.model_json_schema()
        if name in {'query_ncbi_blast', 'query_foldseek_remote', 'query_interproscan', 'predict_s4pred'}:
            keep = {'structure_chain'} if name == 'query_foldseek_remote' else set()
            schema['properties'] = {k: v for k, v in schema.get('properties', {}).items() if k in keep}
        schema['properties'] = {k: v for k, v in schema.get('properties', {}).items() if k not in MANAGED}
        schema['required'] = [k for k in schema.get('required', []) if k in schema['properties']]
    from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
    reason=None
    if name=='predict_s4pred':
        try:
            from mcp_tools.models.pretrained.s4pred import runtime
            runtime()
        except (ValueError,OSError): reason='not_configured'
    if name=='query_interproscan':
        try:
            from mcp_tools.tools.research.interproscan import configuration
            configuration()
        except ValueError: reason='not_configured'
    from mcp_tools.tools.capability_policy import mode
    return {'available':is_tool_allowed(name) and reason is None,'disabled_reason':reason or (None if is_tool_allowed(name) else 'admin_disabled'),'runtime_mode':mode(),'id': 'tool:' + name, 'name': spec[1], 'description': {'predict_s4pred':'Secondary structure from a protein sequence', 'query_interproscan':'Protein families, domains and functional sites', 'query_ncbi_blast': 'Sequence similarity search · NCBI public API · SwissProt', 'query_foldseek_remote': 'Structure similarity search · Foldseek public API · PDB and AlphaFold SwissProt'}.get(name, spec[1]),
            'category': spec[2], 'source': 'platform', 'visibility': 'public', 'adapter': 'registered-tool',
            'workflow': True, 'node_kind': 'tool', 'input_kind': spec[3], 'output_kind': 'table',
            'standalone': spec[4] is None, **({'contract': schema} if detail else {})}


def catalog(*, detail=True):
    rows, unavailable = [], []
    for name in SPECS:
        try:
            rows.append(card(name, detail=detail))
        except (ImportError, AttributeError, ValueError):
            unavailable.append(name)
    return rows, unavailable


def result_rows(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return [{'result': result}]
    if isinstance(result, dict):
        if result.get('status') in {'error', 'failed'} or result.get('success') is False or result.get('error'):
            error = result.get('error') or result.get('message') or 'Tool failed'
            raise ValueError(str(error.get('message', error) if isinstance(error, dict) else error)[:300])
        if result.get('data') is not None:
            result = result['data']
        if isinstance(result, dict):
            sequences = result.get('chains') or result.get('sequences')
            if isinstance(sequences, dict):
                return [{'id': k, 'sequence': v} for k, v in sequences.items()]
            return [result]
    if isinstance(result, list):
        return [r if isinstance(r, dict) else {'result': r} for r in result]
    return [{'result': result}]


def run(name, data, config, work, progress=None):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed
    assert_tool_allowed(name)
    if name in {'query_ncbi_blast', 'query_foldseek_remote', 'query_interproscan', 'predict_s4pred'}:
        from backend.web_v2.remote_discovery import run as remote_search
        return remote_search(name, data, config or {}, work, progress)
    tool, spec = registered(name), SPECS[name]
    config = config or {}
    parameters, mapping = config.get('parameters', {}), config.get('columns', {})
    if not isinstance(parameters, dict) or not isinstance(mapping, dict):
        raise ValueError('Tool parameters and column mapping must be objects')
    allowed = set(card(name)['contract']['properties'])
    if not set(parameters) <= allowed or not set(mapping) <= allowed:
        raise ValueError('Use the documented tool parameters; files are supplied by connected data')
    if data and data['kind'] != spec[3]:
        raise ValueError('Tool requires ' + spec[3] + ' input')
    rows = data.get('rows', []) if data else []
    if spec[3] == 'structure':
        if not data:
            raise ValueError('Connect a structure input')
        rows = [{}]
    elif spec[4] is None and parameters.get('query') and not mapping:
        rows = [{}]
    if not rows:
        raise ValueError('Connect input data or enter a search query')
    if len(rows) > 100:
        raise ValueError('Use at most 100 records per tool step')
    work = Path(work); work.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = []
    with TemporaryDirectory(prefix='tool_', dir=work) as folder:
        folder = Path(folder)
        for i, row in enumerate(rows):
            if progress: progress(round(i / len(rows) * 100), f'Processing record {i + 1} of {len(rows)}')
            args = dict(parameters)
            for key, field in mapping.items():
                if not isinstance(field, str) or field not in row:
                    raise ValueError('Missing mapped column: ' + str(field))
                args[key] = row[field]
            if spec[4] is None and 'query' not in args and 'query' in row:
                args['query'] = row['query']
            if spec[4]:
                path = folder / (str(i) + ('.pdb' if spec[3] == 'structure' else '.fasta'))
                if spec[3] == 'structure':
                    content = data['structure']
                else:
                    sequence = str(row.get(config.get('sequence_column', 'sequence'), '')).strip().upper()
                    if not sequence or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
                        raise ValueError('Choose a valid protein sequence column')
                    content = '>row_' + str(i) + '\n' + sequence + '\n'
                path.write_text(content)
                args[spec[4]] = str(path)
            fields = tool.args_schema.model_fields
            outfile = folder / (str(i) + '.json')
            if 'output_file' in fields:
                args['output_file'] = str(outfile)
            if 'out_dir' in fields:
                args['out_dir'] = str(folder)
            if 'max_results' in args:
                args['max_results'] = min(50, max(1, int(args['max_results'])))
            checked = tool.args_schema.model_validate(args).model_dump()
            from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
            assert_tool_call_allowed(name, checked)
            from backend.web_v2.job_queue import run_in_slot
            from mcp_tools.scheduler.catalog import classify
            result = run_in_slot(classify(name, tool.metadata)['resource'], tool.invoke, checked)
            values = result_rows(result)  # Verify the tool status before consuming any file.
            if outfile.is_file():
                values = result_rows(json.loads(outfile.read_text()))
            output.extend({**row, **value} for value in values)
            if progress: progress(round((i + 1) / len(rows) * 100), f'Processed {i + 1} of {len(rows)} records')
    return {'kind': 'table', 'rows': output}
