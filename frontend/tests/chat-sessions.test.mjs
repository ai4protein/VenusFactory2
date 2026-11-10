import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activeSessionStorageKey, sessionsForMode, sessionCandidates, abbreviateSessionTitle, titleFromHistory,
  readSessionCache, writeSessionCache, sessionHistoryState, reduceSessionHistory } from '../src/lib/chatSessions.ts';

const sessions = [
  {session_id: 'agent-recent', chat_mode: 'science_agent'},
  {session_id: 'scientist-new', chat_mode: 'science_expert'},
  {session_id: 'agent-old', chat_mode: 'science_agent'},
  {session_id: 'scientist-old', chat_mode: 'science_expert'},
];
test('Agent and Scientist have disjoint histories and active-session keys', () => {
  assert.deepEqual(sessionsForMode(sessions, 'science_agent').map(s=>s.session_id), ['agent-recent','agent-old']);
  assert.deepEqual(sessionsForMode(sessions, 'science_expert').map(s=>s.session_id), ['scientist-new','scientist-old']);
  assert.notEqual(activeSessionStorageKey('science_agent'), activeSessionStorageKey('science_expert'));
});
test('restoring a mode never uses a remembered session from the other mode', () => {
  assert.deepEqual(sessionCandidates(sessions, 'science_expert', 'agent-recent'), ['scientist-new','scientist-old']);
  assert.deepEqual(sessionCandidates(sessions, 'science_agent', 'agent-old'), ['agent-old','agent-recent']);
});
test('legacy selection is migrated only into its own mode', () => {
  assert.equal(sessionCandidates(sessions, 'science_agent', null, 'scientist-old')[0], 'agent-recent');
  assert.equal(sessionCandidates(sessions, 'science_expert', null, 'scientist-old')[0], 'scientist-old');
});
test('new drafts, empty modes and deleted selections never fall into another mode', () => {
  assert.deepEqual(sessionCandidates(sessions, 'science_agent', ''), []);
  assert.deepEqual(sessionCandidates(sessions.slice(0,1), 'science_expert', null), []);
  assert.equal(sessionCandidates(sessions, 'science_expert', 'deleted')[0], 'scientist-new');
});

const meta = (id, overrides = {}) => ({ session_id: id, created_at: "2026-11-10T00:00:00Z",
  model_name: "free-model", history_size: 2, status: "completed", chat_mode: "science_agent", ...overrides });

function historyFixture(rows = []) {
  let state = sessionHistoryState(rows);
  return { get state() { return state; }, send(action) { state = reduceSessionHistory(state, action); return state; } };
}

test('a delayed history response cannot replace a newer response', () => {
  const f = historyFixture();
  f.send({ type: 'request', id: 1 });
  f.send({ type: 'request', id: 2 });
  f.send({ type: 'received', id: 2, sessions: [meta('new')] });
  f.send({ type: 'received', id: 1, sessions: [meta('old')] });
  assert.deepEqual(f.state.sessions.map(row => row.session_id), ['new']);
});

test('refresh cannot erase a newly submitted title and running indicator', () => {
  const f = historyFixture([meta('active', { title: '' })]);
  f.send({ type: 'request', id: 1 });
  f.send({ type: 'update', update: rows => rows.map(row => ({ ...row, title: 'Predict protein solubility', status: 'running' })) });
  f.send({ type: 'received', id: 1, sessions: [meta('active', { title: '' })] });
  assert.equal(f.state.sessions[0].status, 'running');
  assert.equal(f.state.sessions[0].title, 'Predict protein solubility');
  f.send({ type: 'request', id: 2 });
  f.send({ type: 'received', id: 2, sessions: [meta('active', { title: '' })] });
  assert.equal(f.state.sessions[0].status, 'completed');
  assert.equal(f.state.sessions[0].title, 'Predict protein…');
});

test('a deleted conversation is not resurrected by an in-flight list', () => {
  const f = historyFixture([meta('deleted'), meta('retained')]);
  f.send({ type: 'request', id: 1 });
  f.send({ type: 'update', update: rows => rows.filter(row => row.session_id !== 'deleted') });
  f.send({ type: 'received', id: 1, sessions: [meta('deleted'), meta('retained')] });
  assert.deepEqual(f.state.sessions.map(row => row.session_id), ['retained']);
});

test('successful authenticated refresh removes obsolete cached identities', () => {
  const f = historyFixture([meta('old-owner')]);
  f.send({ type: 'request', id: 1 });
  f.send({ type: 'received', id: 1, sessions: [meta('current'), meta('empty', { has_content: false })] });
  assert.deepEqual(f.state.sessions.map(row => row.session_id), ['current']);
  f.send({ type: 'request', id: 2 });
  f.send({ type: 'received', id: 2, sessions: [] });
  assert.deepEqual(f.state.sessions, []);
});

test('temporary activity invalidates old refreshes and never enters the persisted cache', () => {
  const f = historyFixture([meta('saved'), meta('unaccepted', { history_size: 0, title: 'New question' })]);
  f.send({ type: 'request', id: 1 });
  f.send({ type: 'activity', key: 'draft', row: { id: 'pending:draft', mode: 'science_expert', title: 'Create model',
    date: 1, running: true, pending: true, canDelete: false } });
  f.send({ type: 'received', id: 1, sessions: [] });
  assert.equal(f.state.activity.draft.running, true);
  let cached;
  writeSessionCache({ setItem: (_key, value) => { cached = JSON.parse(value); } }, f.state.sessions);
  assert.deepEqual(cached.map(row => row.session_id), ['saved']);
  f.send({ type: 'activity', key: 'draft', row: null });
  assert.deepEqual(f.state.activity, {});
});

test('cache restores accepted Scientist topics but tolerates invalid or unavailable storage', () => {
  const scientist = meta('topic', { chat_mode: 'science_expert', history_size: 0, has_content: true });
  let stored;
  writeSessionCache({ setItem: (_key, value) => { stored = value; } }, [scientist]);
  assert.deepEqual(readSessionCache({ getItem: () => stored }), [scientist]);
  assert.deepEqual(readSessionCache({ getItem: () => '{broken' }), []);
  assert.deepEqual(readSessionCache({ getItem: () => JSON.stringify([null, {}, { ...scientist, title: {} }]) }), []);
  assert.deepEqual(readSessionCache({ getItem: () => { throw new Error('storage denied'); } }), []);
  assert.doesNotThrow(() => writeSessionCache({ setItem: () => { throw new Error('storage full'); } }, [scientist]));
});

test('sidebar labels preserve original user intent across quotations and multipart messages', () => {
  assert.equal(abbreviateSessionTitle('> quoted result\nDesign an enzyme. Compare variants'), 'Design an enzyme');
  assert.equal(titleFromHistory([
    { role: 'assistant', content: 'Planning' }, { role: 'user', content: 'Clarification details: retry' },
    { role: 'user', content: [{ type: 'text', text: 'Compare enzyme variants' }] },
  ]), 'Compare enzyme…');
  assert.equal(abbreviateSessionTitle('预测蛋白质溶解度。然后筛选突变'), '预测蛋白质溶解度');
});
