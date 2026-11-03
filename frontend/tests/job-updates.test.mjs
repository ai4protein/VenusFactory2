import test from 'node:test';
import assert from 'node:assert/strict';
import { mergeJobUpdate, jobPollDelay } from '../src/lib/jobUpdates.ts';

test('completion updates the existing row and badge immediately and is idempotent', () => {
  const job = { id: 'a', status: 'running', title: 'ProSST-2048' };
  const page = { items: [job], active: 1, total: 1, offset: 0, limit: 10 };
  const done = { id: 'a', status: 'completed', progress: 100 };
  const next = mergeJobUpdate(page, done);
  assert.equal(next.items[0].status, 'completed');
  assert.equal(next.items[0].title, 'ProSST-2048');
  assert.equal(next.active, 0);
  assert.equal(mergeJobUpdate(next, done).active, 0);
  assert.equal(page.items[0].status, 'running');
});

test('unseen and different jobs do not overwrite the filtered list', () => {
  const page = { items: [{ id: 'a', status: 'running' }], active: 2 };
  assert.equal(mergeJobUpdate(page, { id: 'b', status: 'completed' }), page);
  assert.equal(mergeJobUpdate(undefined, { id: 'b' }), undefined);
});

test('active visible jobs refresh quickly and background tabs back off', () => {
  assert.equal(jobPollDelay(true, true, false), 1000);
  assert.equal(jobPollDelay(false, true, false), 3000);
  assert.ok(jobPollDelay(true, false, false) > 1000);
  assert.equal(jobPollDelay(true, true, true), 30000);
});
