import asyncio
from contextlib import asynccontextmanager
import json
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from backend.web_v2 import research_report as r
from backend.web_v2 import research_longform as lf


def evidence():
    return {'protein': {'name': 'Public protein'}, 'sources': [{'id': 'S1', 'title': 'A study', 'url': 'https://example.org/paper',
            'kind': 'literature', 'evidence_level': 'abstract', 'text': 'Evidence for this protein.'}],
            'steps': [], 'sections': [], 'partial': False, 'report': ''}


class LongReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_eight_independent_writers_and_editor(self):
        calls, active, maximum = [], 0, 0
        async def collect(payload, update, **kwargs):
            value = evidence(); update(value); return value
        async def complete(model, messages, tokens):
            nonlocal active, maximum
            active += 1; maximum = max(maximum, active)
            calls.append(messages)
            await asyncio.sleep(.001)
            active -= 1
            return ('## Evidence\n' + 'measured protein evidence ' * 160 + '[S1]', False)
        payload = {'model_id': 'free-model', 'analyses': ['properties'], 'lang': 'en'}
        with patch.object(r, 'run_research', side_effect=collect), patch.object(r, 'complete', side_effect=complete):
            result = await lf.run_long_report(payload, lambda x: None)
        self.assertEqual(len(result['chapters']), 8)
        self.assertEqual(result['completed_chapters'], 8)
        self.assertTrue(result['minimum_met'])
        self.assertEqual(result['model_calls'], 9)
        self.assertEqual(len({m[0]['content'] for m in calls[:8]}), 8)
        self.assertLessEqual(maximum, 2)
        for i in range(1, 9): self.assertIn(f'## {i}.', result['report'])

    async def test_fewer_than_five_complete_is_not_success(self):
        async def collect(payload, update, **kwargs):
            value = evidence(); update(value); return value
        async def complete(model, messages, tokens):
            if any(spec[3] in messages[0]['content'] for spec in lf.CHAPTERS[:4]):
                raise ValueError('Unavailable')
            return ('protein evidence ' * 230 + '[S1]', False)
        with patch.object(r, 'run_research', side_effect=collect), patch.object(r, 'complete', side_effect=complete):
            result = await lf.run_long_report({'model_id': 'free', 'analyses': ['properties']}, lambda x: None)
        self.assertEqual(result['completed_chapters'], 4)
        self.assertFalse(result['minimum_met'])
        self.assertEqual(result['phase'], 'incomplete')
        self.assertTrue(result['partial'])

    async def test_failed_short_chapters_have_one_bounded_revision(self):
        async def collect(payload, update, **kwargs):
            value = evidence(); update(value); return value
        with patch.object(r, 'run_research', side_effect=collect), patch.object(r, 'complete', new=AsyncMock(return_value=('Short [S1]', False))):
            result = await lf.run_long_report({'model_id': 'free', 'analyses': ['properties']}, lambda x: None)
        self.assertEqual(result['model_calls'], 17)
        self.assertLessEqual(result['model_calls'], lf.MAX_MODEL_CALLS)
        self.assertTrue(all(c['attempts'] == 2 for c in result['chapters']))
        self.assertFalse(result['minimum_met'])

    async def test_tavily_is_explicit_basic_search_without_answer_generation(self):
        sent = []
        def handle(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={'results': [{'title': 'Study', 'url': 'https://example.org/paper', 'content': 'Evidence'}]})
        real_client = httpx.AsyncClient
        @asynccontextmanager
        async def slot(*args, **kwargs): yield 'test'
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-only', 'VENUS_REPORT_SEARCH_PROVIDER': 'auto'}), patch('backend.web_v2.job_queue.resource_slot', slot), patch.object(r.httpx, 'AsyncClient', side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs)):
            rows = await r.web_search('protein structure')
        self.assertEqual(rows[0]['provider'], 'tavily')
        self.assertEqual(sent[0]['search_depth'], 'basic')
        self.assertFalse(sent[0]['include_answer'])
        self.assertFalse(sent[0]['auto_parameters'])
        self.assertNotIn('api_key', sent[0])

    async def test_revision_failure_keeps_the_first_draft(self):
        async def collect(payload, update, **kwargs):
            value = evidence(); update(value); return value
        async def complete(model, messages, tokens):
            if len(messages) > 2:
                raise ValueError('Temporary provider failure')
            return ('Retained first draft [S1]', True)
        with patch.object(r, 'run_research', side_effect=collect), patch.object(r, 'complete', side_effect=complete):
            result = await lf.run_long_report({'model_id': 'free', 'analyses': ['properties']}, lambda x: None)
        self.assertTrue(all(c['status'] == 'partial' and c['draft'] == 'Retained first draft [S1]' for c in result['chapters']))
        self.assertFalse(result['minimum_met'])
        self.assertIn('Retained first draft', result['report'])

    async def test_resume_reuses_collected_evidence_and_completed_chapters(self):
        checkpoint = evidence() | {'workflow': 'specialist_research_v2', 'phase': 'chapters', 'model_calls': 6,
            'partial': False, 'chapters': []}
        for i, (identity, title, *_) in enumerate(lf.CHAPTERS):
            checkpoint['chapters'].append({'id': identity, 'title': title, 'status': 'completed' if i < 3 else 'pending',
                'draft': 'Already verified chapter' if i < 3 else '', 'source_ids': ['S1'], 'attempts': 1, 'length': 500})
        collect = AsyncMock(side_effect=AssertionError('must not recollect'))
        complete = AsyncMock(return_value=('protein evidence ' * 230 + '[S1]', False))
        with patch.object(r, 'run_research', collect), patch.object(r, 'complete', complete):
            result = await lf.run_long_report({'model_id': 'free', 'analyses': ['properties']}, lambda x: None, checkpoint=checkpoint)
        collect.assert_not_called()
        self.assertEqual(complete.await_count, 6)  # five unfinished chapters and one editor
        self.assertEqual(result['model_calls'], 12)
        self.assertTrue(all(c['draft'] == 'Already verified chapter' for c in result['chapters'][:3]))
        self.assertEqual(result['completed_chapters'], 8)
        self.assertEqual(checkpoint['model_calls'], 6)  # no mutation of the saved checkpoint

    async def test_resume_keeps_cumulative_model_budget(self):
        checkpoint = evidence() | {'workflow': 'specialist_research_v2', 'phase': 'chapters', 'model_calls': lf.MAX_MODEL_CALLS,
            'chapters': [{'id': i, 'title': title, 'status': 'pending', 'draft': '', 'source_ids': [], 'attempts': 0, 'length': 0} for i,title,*_ in lf.CHAPTERS]}
        complete = AsyncMock(side_effect=AssertionError('budget must prevent requests'))
        with patch.object(r, 'complete', complete):
            result = await lf.run_long_report({'model_id': 'free', 'analyses': ['properties']}, lambda x: None, checkpoint=checkpoint)
        complete.assert_not_called()
        self.assertEqual(result['model_calls'], lf.MAX_MODEL_CALLS)
        self.assertFalse(result['minimum_met'])

    def test_citation_audit_rejects_unknown_and_too_short_chapters(self):
        result = lf.audit_chapter('unsupported [S9]', ['S1'], 'en')
        self.assertFalse(result['substantial'])
        self.assertEqual(result['unknown_citations'], ['S9'])

    def test_assembly_does_not_validate_a_citation_from_another_context(self):
        state = evidence()
        state['sources'].append({'id': 'S9', 'title': 'Other chapter source', 'url': 'https://example.org/other', 'evidence_level': 'abstract'})
        state['chapters'] = [{'title': 'Chapter', 'source_ids': ['S1'], 'draft': 'Evidence [S1]. Unsupported [S9].', 'status': 'partial'}]
        text = lf.assemble(state, 'en')
        self.assertIn('Unsupported [source unavailable]', text)
        self.assertNotIn('Unsupported [S9](https://example.org/other)', text)


if __name__ == '__main__': unittest.main()
