import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.web_v2 import research_report as r
from backend.web_v2.research_jobs import ResearchJobs, write
from backend.web_v2.job_queue import JobQueue


class ReportPolicyTests(unittest.TestCase):
    def test_literature_is_anchored_to_the_selected_protein(self):
        entry = {'primaryAccession': 'P69905', 'proteinDescription': {'recommendedName': {'fullName': {'value': 'Hemoglobin subunit alpha'}}},
                 'genes': [{'geneName': {'value': 'HBA1'}}], 'organism': {'scientificName': 'Homo sapiens'},
                 'references': [{'citation': {'citationCrossReferences': [{'database': 'PubMed', 'id': '123'}]}}]}
        queries = r.literature_queries(entry, 'protein', ['random OR everything binding'])
        self.assertIn('SRC:MED', queries[0])
        self.assertIn('EXT_ID:123', queries[0])
        self.assertIn('TITLE_ABS:"Hemoglobin subunit alpha"', queries[1])
        self.assertIn('"Homo sapiens"', queries[1])
        self.assertNotIn('everything', queries[1])

    def test_search_feed_is_bounded_and_does_not_accept_unsafe_urls(self):
        raw = b'<rss><channel><item><title>A</title><link>https://example.org/a</link><description>evidence</description></item><item><title>Bad</title><link>javascript:alert(1)</link></item></channel></rss>'
        rows = r.parse_search_feed(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['evidence_level'], 'search_snippet')
        with self.assertRaises(ValueError): r.parse_search_feed(b'<!DOCTYPE x><rss/>')

    def test_model_plaza_keeps_online_disabled_cards(self):
        from backend.web_v2.model_platform import Platform
        with patch('mcp_tools.tools.runtime_tool_policy.is_online_mode', return_value=True):
            mine = Platform.availability({'id': 'test:venusmine', 'name': 'VenusMine', 'adapter': 'discovery'})
            self.assertEqual(mine['name'], 'VenusMine')
            self.assertFalse(mine['available'])
            with self.assertRaises(ValueError): Platform.require_available(mine)
            small = Platform.availability({'id': 'test:esm8m', 'name': 'ESM2 8M', 'adapter': 'foundation', 'repository': 'facebook/esm2_t6_8M_UR50D'})
            large = Platform.availability({'id': 'test:esm650m', 'name': 'ESM2 650M', 'adapter': 'foundation', 'repository': 'facebook/esm2_t33_650M_UR50D'})
            self.assertTrue(small['available'])
            self.assertFalse(large['available'])

    def test_online_visible_disabled_predictions(self):
        with patch.object(r, 'is_online_mode', return_value=True):
            rows = {o['id']: o for o in r.analysis_options()}
            self.assertFalse(rows['mutation']['available'])
            self.assertTrue(rows['properties']['available'])
            self.assertTrue(rows['ai_predictions']['available'])
            self.assertEqual(r.validate_scopes(['ai_predictions']), ['ai_predictions'])
            with self.assertRaises(ValueError): r.validate_scopes(['mutation'])
        with patch.object(r, 'is_online_mode', return_value=False):
            self.assertEqual(r.validate_scopes(['mutation']), ['mutation'])

    def test_protein_text_is_not_silently_turned_into_sequence(self):
        with self.assertRaises(ValueError): r.parse_sequences('ABL1 mouse protein')
        with self.assertRaises(ValueError): r.parse_sequences('>one\nMKTAYIAKQRQISFVK\n>one\nMKTAYIAKQRQISFVK')
        self.assertEqual(len(r.parse_sequences('>a\nMKTAYIAKQRQISFVK\n>b\nMKTAYIAKQRQISFVK')), 2)
        with self.assertRaises(ValueError): r.clean_sequence('M' * 5001)

    def test_citations_only_link_collected_sources(self):
        sources = [{'id': 'S1', 'url': 'https://uniprot.org/a'}]
        result = r.cite_report('Fact [S1]. Other [S9]. [bad](https://evil.example) ![x](http://evil.example/a)', sources)
        self.assertIn('[S1](https://uniprot.org/a)', result)
        self.assertIn('[source unavailable]', result)
        self.assertNotIn('evil.example', result)
        self.assertEqual(r.safe_url('javascript:alert(1)'), '')

    def test_failed_synthesis_retains_evidence_and_properties(self):
        payload = {'analyses': ['properties'], 'model_id': 'mock', 'sequence': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL', 'protein': 'test'}
        updates = []
        with patch.object(r, 'complete', new=AsyncMock(side_effect=ValueError('Free model returned HTTP 403'))):
            result = asyncio.run(r.run_research(payload, lambda state: updates.append(json.loads(json.dumps(state)))))
        self.assertTrue(result['partial'])
        self.assertEqual(result['sections'][0]['id'], 'properties')
        self.assertTrue(result['sources'])
        self.assertEqual(result['steps'][-1]['status'], 'unavailable')
        self.assertGreater(len(updates), 2)

    def test_queue_privacy_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = JobQueue(Path(tmp)/'queue'); store = ResearchJobs(Path(tmp)/'reports', q)
            identity = 'a'*32
            row = {'id': identity, 'owner': 'alice', 'created': 1, 'title': 'protein', 'payload': {'model_id': 'free', 'analyses': ['properties']}}
            write(store.folder(identity)/'request.json', row)
            store.records.register('report', identity, 'alice', 'research', 'protein', 1)
            store.records.grant(store.folder(identity), 'report', 'alice')
            self.assertEqual(store.get('alice', identity)['title'], 'protein')
            with self.assertRaises(FileNotFoundError): store.get('bob', identity)
            self.assertEqual(store.history('bob'), [])
            with self.assertRaises(FileNotFoundError): store.records.check_artifact(store.folder(identity)/'request.json', {'report': 'bob'})

    def test_paid_unknown_models_rejected_before_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = JobQueue(Path(tmp)/'queue'); store = ResearchJobs(Path(tmp)/'reports', q)
            with patch('backend.web_v2.research_jobs.model_option', side_effect=ValueError('Choose a configured free model')):
                with self.assertRaises(ValueError): store.start('alice', {'analyses': ['properties'], 'model_id': 'paid'})
            self.assertEqual(list((Path(tmp)/'reports').iterdir()), [])

    def test_http_boundary_cannot_bypass_online_policy(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web_v2.report_api import router
        app = FastAPI(); app.include_router(router)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'VENUS_REPORT_ROOT': tmp + '/reports', 'VENUS_JOB_QUEUE_ROOT': tmp + '/queue'}), patch.object(r, 'is_online_mode', return_value=True), TestClient(app) as client:
            body = {'model_id': 'free', 'accession': 'P69905', 'analyses': ['mutation']}
            response = client.post('/api/report/jobs', json=body)
            self.assertEqual(response.status_code, 422)
            self.assertIn('local mode', response.json()['detail'])
            self.assertEqual(client.get('/api/report/jobs/' + 'f'*32).status_code, 404)
            self.assertEqual(client.post('/api/report/jobs', json=body, headers={'Origin': 'https://other.example'}).status_code, 403)
            self.assertEqual(client.post('/api/report/generate', json={'sequence_map': {'a': 'M'*20}, 'selected_analyses': ['mutation']}).status_code, 422)


if __name__ == '__main__': unittest.main()
