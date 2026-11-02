import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeHistoryActivity, submissionHistoryRow } from '../src/lib/optimisticHistory.ts';

test('all modes appear before the session creation response', () => {
  for (const mode of ['science_agent', 'science_expert', 'researcher']) {
    const row = submissionHistoryRow(mode, 'request', '  Protein\n activity  ');
    assert.equal(row.title, 'Protein activity'); assert.equal(row.running, true); assert.equal(row.pending, true);
    assert.equal(mergeHistoryActivity([], { request: row }).length, 1);
  }
});
test('server identity replaces the temporary row without duplication', () => {
  const row = submissionHistoryRow('science_agent', 'request', 'Analyze my protein', 'session');
  const merged = mergeHistoryActivity([{ ...row, running:false, title:'New Chat' }], { request:row });
  assert.equal(merged.length, 1); assert.equal(merged[0].title, 'Analyze my protein'); assert.equal(merged[0].running, true); assert.equal(merged[0].pending, false);
});
test('stale empty list does not erase an in-flight submission', () => {
  const row = submissionHistoryRow('science_expert', 'request', 'Develop a solubility model', 'session');
  assert.equal(mergeHistoryActivity([], {request:row})[0].running, true);
});
test('follow-up retains existing title and starts spinning immediately', () => {
  const row = submissionHistoryRow('science_agent', 'request', 'Continue please', 'session');
  const merged = mergeHistoryActivity([{ ...row, title:'Original protein study', running:false }], {request:row});
  assert.equal(merged[0].title, 'Original protein study'); assert.equal(merged[0].running, true);
});
test('failure cleanup leaves no temporary conversations', () => {
  assert.deepEqual(mergeHistoryActivity([], {}), []);
});
test('identical ids in different modes and parallel submissions stay distinct', () => {
  const a=submissionHistoryRow('science_agent','a','Question','same'), b=submissionHistoryRow('researcher','b','Report','same');
  assert.equal(mergeHistoryActivity([a], {a,b}).length,2);
});
