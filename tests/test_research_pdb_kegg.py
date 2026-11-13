from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.web_v2 import research_report as report
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_jobs import ResearchJobs
from mcp_tools.tools.database.rcsb import research_target as pdb
from mcp_tools.tools.database.kegg import query as kegg
from mcp_tools.tools.research import services


def cif():
    from Bio.PDB import PDBParser, MMCIFIO
    data = ''.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C  \n' for i in range(1, 13))
    structure = PDBParser(QUIET=True).get_structure('test', io.StringIO(data))
    structure[0]['A'].id = 'AA'
    writer = MMCIFIO(); writer.set_structure(structure); out = io.StringIO(); writer.save(out)
    return out.getvalue().encode()


def card(chain='AA'):
    return {'pdb_id': '4HHB', 'structure_chain': chain, 'accession': 'P69905', 'name': 'Hemoglobin',
            'organism': 'Homo sapiens', 'length': 141, 'url': 'https://www.rcsb.org/structure/4HHB'}


FLAT = 'ENTRY       7535              CDS\nNAME        kinase\nPATHWAY     hsa04660  T cell receptor signaling pathway\n            hsa04014  Ras signaling pathway\nORTHOLOGY   K07360  tyrosine kinase [EC:2.7.10.2]\nAASEQ       12\n            AAAAAAAAAAAA\n///\n'


class PdbKeggTests(unittest.IsolatedAsyncioTestCase):
    def test_identifier_recognition_and_path_rejection(self):
        self.assertEqual(pdb.pdb_query('PDB:4hhb:A'), ('4HHB', 'A'))
        self.assertEqual(pdb.pdb_query('pdb_00004hhb'), ('pdb_00004hhb', ''))
        for value in ['P69905', 'TP53', 'ABL1 mouse']:
            self.assertIsNone(pdb.pdb_query(value))
        for value in ['pdb:../../secret', '4HHB:../A', '4HHB:A:B']:
            with self.assertRaises(ValueError): pdb.pdb_query(value)
        for args in [dict(operation='get', identifier='../../env'), dict(operation='convert', identifier='hsa:7535'),
                     dict(operation='get', identifier='hsa:7535+7536'), dict(query='kinase\nHOST'), dict(query='x/y')]:
            with self.assertRaises(ValueError): kegg.KeggInput(**args)

    async def test_pdb_search_does_not_query_uniprot_by_name(self):
        with patch.object(pdb, 'resolve_pdb', new=AsyncMock(return_value=[card()])) as resolve, patch.object(report, 'get_json', new=AsyncMock()) as uniprot:
            self.assertEqual(len(await report.resolve_protein('4hhb:AA')), 1)
            resolve.assert_awaited_once_with('4HHB', 'AA'); uniprot.assert_not_called()

    async def test_metadata_preserves_all_chains_and_ambiguous_mapping(self):
        entry = {'rcsb_entry_container_identifiers': {'polymer_entity_ids': ['1']}}
        entity = {'entity_poly': {'rcsb_entity_polymer_type': 'Protein', 'rcsb_sample_sequence_length': 141},
                  'rcsb_polymer_entity_container_identifiers': {'auth_asym_ids': ['A', 'C'], 'uniprot_ids': ['P69905', 'P68871']}}
        with patch.object(pdb, 'get_json', new=AsyncMock(side_effect=[entry, entity])):
            rows = await pdb.resolve_pdb('4HHB')
        self.assertEqual([x['id'] for x in rows], ['4HHB:A', '4HHB:C'])
        self.assertEqual(rows[0]['accession'], '')
        with patch.object(pdb, 'resolve_pdb', new=AsyncMock(return_value=rows)), patch.object(pdb, 'request', new=AsyncMock()) as fetch:
            with self.assertRaisesRegex(ValueError, 'Select one'): await pdb.load_pdb_target('4HHB')
            fetch.assert_not_called()

    async def test_mmcif_chain_conversion_and_observed_sequence(self):
        with patch.object(pdb, 'resolve_pdb', new=AsyncMock(return_value=[card()])), patch.object(pdb, 'request', new=AsyncMock(return_value=cif())):
            target = await pdb.load_pdb_target('4HHB', 'AA')
        self.assertEqual(target['sequence'], 'A'*12)
        self.assertEqual((target['structure_chain'], target['source_chain']), ('A', 'AA'))
        self.assertEqual(target['protein']['entity_length'], 141)
        self.assertEqual(target['protein']['length'], 12)
        self.assertEqual(services.structure_chain(target['structure_pdb'])[1], 'A'*12)
        _, origin = await services.choose_structure(target)
        self.assertEqual(origin['origin'], 'experimental_structure')
        self.assertEqual(origin['source_chain'], 'AA')

    async def test_reference_sequence_never_replaces_pdb_construct(self):
        target = {'protein': card(), 'sequence': 'A'*12, 'structure_pdb': 'coordinates', 'structure_chain': 'A',
                  'source_chain': 'AA', 'interpretation': 'Observed construct, not full reference'}
        reference = {'primaryAccession': 'P69905', 'sequence': {'value': 'M'*142}, 'uniProtKBCrossReferences': []}
        contexts = []
        async def run(key, context, update, **kwargs):
            contexts.append(context)
            value = {'id': key, 'label': 'KEGG', 'status': 'empty', 'url': 'https://www.kegg.jp/', 'data': {'count': 0}, 'evidence_level': 'database_annotation'}
            update(value); return value
        with patch.object(pdb, 'load_pdb_target', new=AsyncMock(return_value=target)), patch.object(report, 'get_json', new=AsyncMock(return_value=reference)), patch.object(services, 'run_service', side_effect=run):
            state = await report.run_research({'pdb_id': '4HHB', 'structure_chain': 'AA', 'analyses': ['properties', 'kegg'], 'model_id': 'unused'}, lambda _: None, synthesize=False)
        self.assertEqual(state['sections'][0]['data']['length_aa'], 12)
        self.assertEqual(contexts[0]['sequence'], 'A'*12)
        self.assertEqual(contexts[0]['accession'], 'P69905')
        self.assertIn('reference_protein_annotation', [x['evidence_level'] for x in state['sources']])

    def test_job_accepts_pdb_and_rejects_mixed_targets(self):
        with tempfile.TemporaryDirectory() as tmp, patch('backend.web_v2.research_jobs.model_option'):
            jobs = ResearchJobs(Path(tmp)/'reports', JobQueue(Path(tmp)/'queue'))
            payload = {'pdb_id': '4hhb', 'structure_chain': 'AA', 'model_id': 'free', 'analyses': ['kegg']}
            jobs.validate(payload); self.assertEqual(payload['pdb_id'], '4HHB')
            for field in ['accession', 'sequence', 'structure_pdb']:
                with self.assertRaises(ValueError): jobs.validate({**payload, field: 'P69905' if field == 'accession' else 'A'*12})

    async def test_kegg_entry_parses_continuations_and_omits_raw_sequences(self):
        with patch.object(kegg, 'fetch', new=AsyncMock(return_value=FLAT)):
            value = await kegg.kegg_query(operation='get', identifier='hsa:7535')
        self.assertEqual(value['status'], 'completed')
        self.assertEqual(len(value['records'][0]['pathways']), 2)
        self.assertNotIn('AASEQ', json.dumps(value))
        self.assertTrue(value['sources'][0]['url'].endswith('/get/hsa:7535'))

    async def test_kegg_uniprot_mapping_empty_and_failure_distinct(self):
        with patch.object(kegg, 'fetch', new=AsyncMock(side_effect=['up:P43403\thsa:7535\n', FLAT])) as fetch:
            data = await kegg.protein_annotations({'accession': 'P43403'})
        self.assertEqual(fetch.call_args_list[0].args[0], 'conv/genes/uniprot:P43403')
        self.assertEqual(data['pathways'][0]['genes'], ['hsa:7535'])
        with patch.object(kegg, 'fetch', new=AsyncMock(return_value='')):
            self.assertEqual((await kegg.kegg_query(operation='convert', identifier='P43403'))['status'], 'empty')
        with patch.object(kegg, 'fetch', new=AsyncMock(side_effect=services.ServiceError('HTTP 503'))):
            self.assertEqual((await kegg.kegg_query(operation='convert', identifier='P43403'))['status'], 'unavailable')
            with self.assertRaises(ValueError): await kegg.protein_annotations({'accession': 'P43403'})
        self.assertEqual((await services.run_service('kegg', {'sequence': 'A'*12}))['status'], 'not_applicable')

    async def test_kegg_admin_gate_runs_before_cache(self):
        def gate(name, *args):
            if name == 'query_kegg': raise ValueError('Disabled')
        with patch('mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed', side_effect=gate), patch.object(services, 'annotation', new=AsyncMock()) as call:
            with self.assertRaisesRegex(ValueError, 'Disabled'): await services.run_service('kegg', {'accession': 'P43403'})
            call.assert_not_called()

    async def test_agent_and_mcp_share_kegg_queries(self):
        from fastmcp import Client
        from mcp_tools.tools.research.tools_mcp import mcp
        from mcp_tools.tools.research.tools_agent import query_kegg_tool
        with patch.object(kegg, 'fetch', new=AsyncMock(return_value=FLAT)) as fetch:
            direct = await query_kegg_tool.ainvoke({'operation': 'get', 'identifier': 'hsa:7535'})
            async with Client(mcp) as client:
                response = await client.call_tool('query_kegg', {'operation': 'get', 'identifier': 'hsa:7535'})
            self.assertFalse(response.is_error)
            self.assertIn('hsa04660', response.content[0].text)
            self.assertEqual(direct['status'], 'completed'); self.assertEqual(fetch.await_count, 2)

    def test_kegg_in_discovery_and_report_options(self):
        from mcp_tools.tools.tools_agent_hub import get_tools, PI_SEARCH_TOOL_NAMES
        self.assertIn('query_kegg', {t.name for t in get_tools()})
        self.assertIn('query_kegg', PI_SEARCH_TOOL_NAMES)
        self.assertTrue(next(x for x in report.analysis_options() if x['id'] == 'kegg')['available'])

    async def test_kegg_is_distributed_to_pathway_specialist(self):
        from backend.web_v2 import research_longform as longform
        source = {'id': 'S1', 'service': 'kegg', 'kind': 'public_service', 'title': 'KEGG',
                  'url': 'https://www.kegg.jp/entry/hsa:7535', 'evidence_level': 'database_annotation',
                  'text': json.dumps({'pathways': [{'id': 'hsa04660', 'name': 'T cell receptor signaling'}]})}
        async def collect(*args, **kwargs):
            return {'protein': {'name': 'ZAP70'}, 'sources': [source], 'steps': [], 'sections': [], 'partial': False}
        bundles = []
        async def complete(model, messages, tokens, **_review_metadata):
            bundle = json.loads(messages[1]['content']); bundles.append(bundle)
            return ('Measured evidence remains distinct from reference annotation. '*100 + '[S1]', False)
        with patch.object(report, 'run_research', side_effect=collect), patch.object(report, 'complete', side_effect=complete):
            await longform.run_long_report({'analyses': ['kegg'], 'model_id': 'unused'}, lambda _: None)
        pathway = next(x for x in bundles if x.get('chapter') == 'Function, catalysis & pathways')
        self.assertIn('hsa04660', pathway['sources'][0]['text'])

    async def test_failed_public_task_preserves_other_evidence(self):
        async def fail(key, *args, **kwargs):
            if key == 'kegg': raise RuntimeError('MCP worker connection failed')
            return {'id': key, 'label': 'InterPro', 'status': 'empty', 'url': 'https://www.ebi.ac.uk/interpro/', 'data': {'count': 0}}
        with patch.object(services, 'run_service', side_effect=fail):
            state = await report.run_research({'analyses': ['properties', 'kegg', 'interpro'], 'sequence': 'A'*12, 'model_id': 'unused'}, lambda _: None, synthesize=False)
        self.assertTrue(state['partial'])
        self.assertEqual(next(x for x in state['remote_services'] if x['id'] == 'kegg')['status'], 'unavailable')
        self.assertEqual(state['sections'][0]['data']['length_aa'], 12)
        import asyncio
        with patch.object(services, 'run_service', new=AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await report.run_research({'analyses': ['kegg'], 'model_id': 'unused'}, lambda _: None, synthesize=False)
