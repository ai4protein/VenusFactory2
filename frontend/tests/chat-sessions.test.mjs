import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activeSessionStorageKey, sessionsForMode, sessionCandidates } from '../src/lib/chatSessions.ts';

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
