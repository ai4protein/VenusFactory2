import test from 'node:test';
import assert from 'node:assert/strict';
import { shareRead, invalidateRead } from '../src/lib/readRequests.ts';

test('simultaneous reads share a flight, but completed snapshots are never cached', async () => {
  let finish, calls = 0;
  const loader = () => { calls++; return new Promise(resolve => { finish = resolve; }); };
  const one = shareRead('session-A', loader), two = shareRead('session-A', loader);
  assert.equal(one, two);
  await Promise.resolve(); assert.equal(calls, 1);
  finish({ status: 'running' });
  await Promise.all([one, two]);
  assert.deepEqual(await shareRead('session-A', async () => ({ status: 'completed' })), { status: 'completed' });
});
test('different sessions and credentials do not share requests', async () => {
  const values = await Promise.all([
    shareRead('session-A:owner-1', async () => 'one'),
    shareRead('session-A:owner-2', async () => 'two'),
    shareRead('session-B:owner-1', async () => 'three'),
  ]);
  assert.deepEqual(values, ['one', 'two', 'three']);
});
test('a failed read can be retried immediately', async () => {
  await assert.rejects(shareRead('failure', async () => { throw Error('offline'); }));
  assert.equal(await shareRead('failure', async () => 'restored'), 'restored');
});

test('an invalidated in-flight read cannot absorb new reads or evict their pending request', async () => {
  let finishOld, finishNew;
  const old = shareRead('changed', () => new Promise(resolve => { finishOld = resolve; }));
  await Promise.resolve();
  invalidateRead('changed');
  const fresh = shareRead('changed', () => new Promise(resolve => { finishNew = resolve; }));
  await Promise.resolve();
  assert.notEqual(old, fresh);
  finishOld('before mutation');
  assert.equal(await old, 'before mutation');
  assert.equal(shareRead('changed', () => { throw Error('must share the fresh request'); }), fresh);
  finishNew('after mutation');
  assert.equal(await fresh, 'after mutation');
});
