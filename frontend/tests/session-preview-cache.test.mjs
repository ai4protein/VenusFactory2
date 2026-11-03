import test from 'node:test';
import assert from 'node:assert/strict';
import { SessionPreviewCache } from '../src/lib/sessionPreviewCache.ts';

test('previews are isolated per mounted workspace and keyed by session', () => {
  const cache = new SessionPreviewCache();
  cache.remember({session_id: 'a', history: ['A']});
  cache.remember({session_id: 'b', history: ['B']});
  assert.deepEqual(cache.peek('a').history, ['A']);
  assert.equal(new SessionPreviewCache().peek('a'), null);
  cache.forget('a');
  assert.equal(cache.peek('a'), null);
});

test('fresh state replaces preview; least recently read and oversized data are evicted', () => {
  const cache = new SessionPreviewCache(2, 160);
  cache.remember({session_id: 'a', status: 'running'});
  cache.remember({session_id: 'b'});
  cache.peek('a');
  cache.remember({session_id: 'c'});
  assert.equal(cache.peek('b'), null);
  cache.remember({session_id: 'a', status: 'completed'});
  assert.equal(cache.peek('a').status, 'completed');
  cache.remember({session_id: 'a', history: 'x'.repeat(161)});
  assert.equal(cache.peek('a'), null);
  assert.equal(cache.peek('c').session_id, 'c');
});
