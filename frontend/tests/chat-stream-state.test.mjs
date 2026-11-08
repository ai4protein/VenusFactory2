import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chatStreamReducer, mergeChatSnapshot, sanitizeChatSnapshot } from '../src/lib/chatStreamState.ts';

const user = (content = 'Predict this sequence', turn_id = 'a') => ({ role: 'user', content, turn_id });
const answer = (content, turn_id = 'a') => ({ role: 'assistant', content, turn_id, kind: 'text' });
const saved = (history, extra = {}) => ({
  session_id: 'one', model_name: 'test', status: 'running', history,
  plan: [], tool_executions: [], conversation_log: [], clarification_questions: [],
  created_at: '', waiting_for: '', ...extra,
});
const render = snapshot => ({ snapshot, streamingIdx: -1 });
function freeze(value) {
  if (value && typeof value === 'object') {
    Object.freeze(value);
    for (const child of Object.values(value)) freeze(child);
  }
  return value;
}

test('same-turn server updates preserve streamed text and update other server state', () => {
  const previous = freeze(saved([user(), answer('A partially streamed answer')]));
  const incoming = freeze(saved([user(), answer('A partially')], { tool_executions: [{ name: 'search', status: 'running' }] }));
  const result = mergeChatSnapshot(previous, incoming);
  assert.equal(result.history.at(-1).content, 'A partially streamed answer');
  assert.equal(result.tool_executions, incoming.tool_executions);
  assert.equal(incoming.history.at(-1).content, 'A partially');
});

test('a new user question never inherits the preceding long answer', () => {
  const previous = saved([user('First question'), answer('The complete previous answer must stay in its original turn.')]);
  const incoming = freeze(saved([...previous.history, user('A new question', 'b'),
    { ...answer('Thinking', 'b'), phase: 'thinking' }]));
  assert.equal(mergeChatSnapshot(previous, incoming), incoming);
  const state = chatStreamReducer({ snapshot: previous, streamingIdx: 1 }, { type: 'state', snapshot: incoming });
  assert.equal(state.streamingIdx, -1);
});

test('session, run, turn and expert identities prevent stale tail merging', () => {
  const previous = saved([user(), { ...answer('Long earlier answer'), role_id: 'pi' }],
    { active_run: { id: 'old-run', kind: 'agent', status: 'running' } });
  for (const incoming of [
    saved([user(), answer('')], { session_id: 'two' }),
    saved([user(), answer('')], { active_run: { id: 'new-run', kind: 'agent', status: 'running' } }),
    saved([user('Predict this sequence', 'b'), answer('', 'b')]),
    saved([user(), { ...answer(''), role_id: 'ml' }]),
    saved([user('Different question'), answer('')]),
  ]) assert.equal(mergeChatSnapshot(previous, incoming), incoming);
});

test('compacted or unrelated snapshots remain authoritative', () => {
  const previous = saved([user(), answer('Long earlier answer')]);
  for (const incoming of [
    saved([answer('')]),
    saved([user(), answer('A different conclusion')]),
    saved([user(), { ...answer('Review'), kind: 'checkpoint' }]),
    saved([user(), { ...answer('Context'), kind: 'compaction' }]),
  ]) assert.equal(mergeChatSnapshot(previous, incoming), incoming);
});

test('restoring and streaming a waiting or terminal checkpoint clean identical placeholders', () => {
  const previous = saved([user(), answer('A long incomplete answer')]);
  for (const status of ['waiting_for_plan_confirmation', 'waiting_for_user', 'completed',
    'cancelled', 'failed', 'error', 'stopped', 'planning_failed', 'execution_failed', '']) {
    const incoming = freeze(saved([
      user('I am preparing clarification for my experiment'),
      { ...answer('Thinking'), kind: 'thinking', phase: 'thinking' },
      { ...answer('Preparing clarification'), phase: 'preparing' },
      { ...answer('Approve this step'), kind: 'checkpoint' },
      answer('Final.'),
    ], { status }));
    const restored = sanitizeChatSnapshot(incoming);
    const streamed = chatStreamReducer({ snapshot: previous, streamingIdx: 1 }, { type: 'state', snapshot: incoming });
    assert.deepEqual(streamed.snapshot, restored);
    assert.deepEqual(restored.history.map(item => item.content),
      ['I am preparing clarification for my experiment', 'Approve this step', 'Final.']);
    assert.equal(streamed.streamingIdx, -1);
  }
});

test('running placeholders are preserved until their stream begins', () => {
  const snapshot = freeze(saved([user(), { ...answer('Thinking'), phase: 'thinking' }]));
  assert.equal(sanitizeChatSnapshot(snapshot), snapshot);
  let state = chatStreamReducer(render(snapshot), { type: 'start', info: { role_id: 'pi', turn_id: 'a' } });
  state = chatStreamReducer(state, { type: 'tokens', deltas: [{ content: 'First ' }, { content: 'result' }] });
  assert.equal(state.snapshot.history.length, 2);
  assert.equal(state.snapshot.history[1].content, 'First result');
  assert.equal(state.snapshot.history[1].role_id, 'pi');
  assert.equal(state.snapshot.history[1].turn_id, 'a');
  assert.equal(state.snapshot.history[1].phase, undefined);
  assert.equal(state.streamingIdx, 1);
});

test('legacy text placeholders stay compatible without confusing user text', () => {
  for (const marker of ['Thinking', '正在总结', 'writing sub-report']) {
    const state = chatStreamReducer(render(saved([user(), answer(marker)])), { type: 'start', info: {} });
    assert.equal(state.snapshot.history.length, 2);
    assert.equal(state.snapshot.history.at(-1).content, '');
  }
  const state = chatStreamReducer(render(saved([user('Thinking about enzymes')])), { type: 'start', info: {} });
  assert.equal(state.snapshot.history[0].content, 'Thinking about enzymes');
  assert.equal(state.snapshot.history.length, 2);
});

test('reasoning, answers and subsequent turns retain separate messages and turn IDs', () => {
  let state = render(freeze(saved([user()])));
  for (const action of [
    { type: 'start', info: { kind: 'thinking', turn_id: 'a' } },
    { type: 'thinking', delta: { content: 'Inspect ', turn_id: 'a' } },
    { type: 'thinking', delta: { content: 'data.', turn_id: 'a' } },
    { type: 'start', info: { turn_id: 'a' } },
    { type: 'tokens', deltas: [{ content: 'Answer.' }] },
    { type: 'thinking', delta: { content: 'Another turn', turn_id: 'b' } },
  ]) state = chatStreamReducer(state, action);
  assert.deepEqual(state.snapshot.history.slice(1).map(item => [item.kind, item.turn_id, item.content]), [
    ['thinking', 'a', 'Inspect data.'], ['text', 'a', 'Answer.'], ['thinking', 'b', 'Another turn'],
  ]);
});

test('tokens arriving without a start event create text without corrupting reasoning or checkpoints', () => {
  for (const last of [
    { ...answer('Inspect'), kind: 'thinking' },
    { ...answer('Approve'), kind: 'checkpoint' },
    { ...answer('Compressed context'), kind: 'compaction' },
  ]) {
    const state = chatStreamReducer(render(freeze(saved([user(), last]))),
      { type: 'tokens', deltas: [{ content: 'Reply', turn_id: 'a' }] });
    assert.equal(state.snapshot.history[1], last);
    assert.equal(state.snapshot.history[2].content, 'Reply');
  }
});

test('stopping preserves partial output and replacing a session resets the cursor', () => {
  let state = render(saved([user()]));
  state = chatStreamReducer(state, { type: 'tokens', deltas: [{ content: 'Partial answer' }] });
  state = chatStreamReducer(state, { type: 'stop' });
  assert.equal(state.snapshot.history.at(-1).content, 'Partial answer');
  assert.equal(state.streamingIdx, -1);
  state = chatStreamReducer(state, { type: 'replace', value: null });
  assert.equal(chatStreamReducer(state, { type: 'tokens', deltas: [{ content: 'late' }] }), state);
  assert.equal(state.snapshot, null);
});

test('functional plan updates remain supported without another copy of history state', () => {
  const state = chatStreamReducer(render(saved([user(), answer('Existing result')])),
    { type: 'replace', value: previous => ({ ...previous, status: 'waiting_for_plan_confirmation', plan: [{ step: 1 }] }) });
  assert.equal(state.snapshot.history.at(-1).content, 'Existing result');
  assert.deepEqual(state.snapshot.plan, [{ step: 1 }]);
});
