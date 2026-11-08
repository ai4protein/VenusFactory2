import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ChatTokenQueue } from '../src/lib/chatTokenQueue.ts';
import { chatStreamReducer } from '../src/lib/chatStreamState.ts';

function harness(deliver) {
  const frames = new Map(), cancelled = [];
  let id = 0;
  const queue = new ChatTokenQueue(deliver,
    callback => { frames.set(++id, callback); return id; },
    handle => { cancelled.push(handle); });
  return { queue, frames, cancelled };
}

test('a burst renders in one batch, retaining spaces and message identities', () => {
  const delivered = [];
  const { queue, frames } = harness(batch => delivered.push(batch));
  const deltas = Array.from({ length: 80 }, (_, i) => ({ content: i + ' ', role_id: 'pi', turn_id: 'one' }));
  for (const delta of deltas) queue.push(delta);
  assert.equal(frames.size, 1);
  assert.equal(delivered.length, 0);
  frames.get(1)();
  assert.deepEqual(delivered, [deltas]);
});

test('discarding a session prevents its cancelled frame from draining the next session', () => {
  const delivered = [];
  const { queue, frames, cancelled } = harness(batch => delivered.push(batch));
  queue.push({ content: 'Old session' });
  const stale = frames.get(1);
  queue.reset();
  queue.push({ content: 'New session' });
  stale();
  assert.deepEqual(delivered, []);
  assert.deepEqual(cancelled, [1]);
  frames.get(2)();
  assert.deepEqual(delivered, [[{ content: 'New session' }]]);
});

test('flushing before a checkpoint applies queued text exactly once', () => {
  const delivered = [];
  const { queue, frames } = harness(batch => delivered.push(batch));
  queue.push({ content: 'Partial' });
  queue.flush();
  frames.get(1)();
  queue.flush();
  assert.deepEqual(delivered, [[{ content: 'Partial' }]]);
});

test('text queued before a new assistant message cannot append to the new message', () => {
  let state = { snapshot: { session_id: 'one', status: 'running',
    history: [{ role: 'user', content: 'Question' }] }, streamingIdx: -1 };
  const dispatch = action => { state = chatStreamReducer(state, action); };
  const { queue, frames } = harness(deltas => dispatch({ type: 'tokens', deltas }));
  dispatch({ type: 'start', info: { role_id: 'pi' } });
  queue.push({ content: 'PI output', role_id: 'pi' });
  queue.flush();
  dispatch({ type: 'start', info: { role_id: 'ml' } });
  queue.push({ content: 'ML output', role_id: 'ml' });
  frames.get(1)();
  frames.get(2)();
  assert.deepEqual(state.snapshot.history.slice(1).map(item => [item.role_id, item.content]),
    [['pi', 'PI output'], ['ml', 'ML output']]);
});

test('a completed snapshot replaces the queued partial answer and stays completed', () => {
  let state = { snapshot: { session_id: 'one', status: 'running',
    history: [{ role: 'user', content: 'Question' }] }, streamingIdx: -1 };
  const dispatch = action => { state = chatStreamReducer(state, action); };
  const { queue, frames } = harness(deltas => dispatch({ type: 'tokens', deltas }));
  queue.push({ content: 'A partial answer' });
  queue.flush();
  dispatch({ type: 'state', snapshot: { session_id: 'one', status: 'completed',
    history: [{ role: 'user', content: 'Question' }, { role: 'assistant', content: 'Final answer' }] } });
  frames.get(1)();
  assert.equal(state.snapshot.history.at(-1).content, 'Final answer');
  assert.equal(state.streamingIdx, -1);
});

test('empty deltas and repeated cleanup schedule no animation work', () => {
  const delivered = [];
  const { queue, frames } = harness(batch => delivered.push(batch));
  queue.push({ content: '' }); queue.reset(); queue.reset(); queue.flush();
  assert.equal(frames.size, 0);
  assert.deepEqual(delivered, []);
});
