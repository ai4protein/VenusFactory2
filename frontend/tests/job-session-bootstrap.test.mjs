import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ensureJobSession } from '../src/lib/jobSession.ts';

test('catalog can bootstrap the shared workspace without a redundant session request', async () => {
  let resolve, calls = 0;
  const pending = new Promise(done => { resolve = done; });
  const first = ensureJobSession('platform', async () => { calls++; await pending; });
  const second = ensureJobSession('platform', async () => { throw new Error('duplicate identity'); });
  assert.equal(first, second);
  assert.equal(calls, 1);
  resolve(); await Promise.all([first, second]);
});

test('failed bootstrap can be retried', async () => {
  await assert.rejects(ensureJobSession('custom', async () => { throw new Error('offline'); }), /offline/);
  let retried = false;
  await ensureJobSession('custom', async () => { retried = true; });
  assert.equal(retried, true);
});
