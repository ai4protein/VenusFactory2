"""Queued plaza/workflow bridge to the shared public protein MCP services."""
import asyncio
from pathlib import Path

SERVICES = {'query_interproscan':'interproscan', 'predict_s4pred':'s4pred', 'query_ncbi_blast': 'blast', 'query_foldseek_remote': 'foldseek_remote'}


def run(name, data, config, work, progress=None):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
    from mcp_tools.tools.research.tools_agent import prepare_context
    from mcp_tools.tools.research.services import run_service
    assert_tool_call_allowed(name)
    service = SERVICES[name]
    parameters = config.get('parameters') or {}
    allowed = {'structure_chain'} if service == 'foldseek_remote' else set()
    if not isinstance(parameters, dict) or set(parameters) - allowed or config.get('columns'):
        raise ValueError('Use the sequence column or uploaded structure as the search input')
    expected = 'structure' if service == 'foldseek_remote' else 'table'
    if not data or data.get('kind') != expected:
        raise ValueError('Connect ' + expected + ' input')
    rows = [{}] if expected == 'structure' else data.get('rows', [])
    if not 1 <= len(rows) <= 5:
        raise ValueError('Use one to five protein sequences per public search job')

    async def search():
        # Validate the whole batch before submitting anything to a public server.
        contexts = []
        for row in rows:
            args = {'structure_pdb': data['structure'], **parameters} if expected == 'structure' else {
                'sequence': str(row.get(config.get('sequence_column') or 'sequence', ''))}
            contexts.append(await prepare_context(service, **args))
        output = [];artifacts=[]
        for index, (row, context) in enumerate(zip(rows, contexts)):
            def update(state):
                if progress:
                    status = state.get('status', 'running')
                    remote_id = state.get('job_id')
                    progress(round(index / len(rows) * 100),
                             f'{index + 1}/{len(rows)} · {status}' + (f' · {remote_id}' if remote_id else ''))
            result = await run_service(service, context, update, Path(work) / ('remote-' + str(index)))
            status = result.get('status')
            if status not in {'completed', 'empty'}:
                # A queued remote request is not a completed prediction or a zero-hit result.
                raise ValueError(f"{result.get('label', service)}: {result.get('summary') or status}. "
                                 + (f"Result: {result['url']}" if result.get('job_id') else ''))
            content=result.get('data', {})
            hits = content.get('hits', [])
            if service=='s4pred':
                hits=[{'secondary_structure':content['secondary_structure'],'residue_count':content['count']}]
                artifacts.extend(content.get('artifacts',[]))
            meta = {'query_id': row.get('id') or str(index + 1), 'search_status': status,
                    'remote_job_id': result.get('job_id'), 'source_url': result.get('url'), 'hit_count': len(hits)}
            output.extend({**row, **meta, **hit} for hit in (hits or [{}]))
            if progress: progress(round((index + 1) / len(rows) * 100), f'{index + 1}/{len(rows)} · {len(hits)} hits')
        return {'kind': 'table', 'rows': output, 'artifacts':artifacts}

    return asyncio.run(search())
