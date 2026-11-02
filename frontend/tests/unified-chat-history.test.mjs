import { test } from 'node:test';
import assert from 'node:assert/strict';
import { claimNewChatDraft, visibleChatHistory, historyRowKey, newChatLocation } from '../src/lib/unifiedChatHistory.ts';
import { chatSessionLocation } from '../src/lib/chatSessionRoute.ts';

const emptyPins = { science_agent: [], science_expert: [], researcher: [] };
const row = (mode, id, date, title = id) => ({ mode, id, date, title, running: false, canDelete: true });

test('three modes share chronological history without colliding identities', () => {
  const rows = [row('science_agent', 'same', 100), row('science_expert', 'topic', 300), row('researcher', 'same', 200)];
  const result = visibleChatHistory(rows, '', emptyPins);
  assert.deepEqual(result.map(historyRowKey), ['science_expert:topic', 'researcher:same', 'science_agent:same']);
  assert.equal(rows[0].date, 100);
});
test('existing mode-scoped pins remain pinned in the shared list', () => {
  const rows = [row('researcher', 'same', 300), row('science_agent', 'same', 100), row('science_expert', 'topic', 200)];
  assert.deepEqual(visibleChatHistory(rows, '', { ...emptyPins, science_agent: ['same'] }).map(historyRowKey),
    ['science_agent:same', 'researcher:same', 'science_expert:topic']);
});
test('search covers the conversation title and its mode', () => {
  const rows = [row('science_agent', 'a', 100, 'PETase question'), row('science_expert', 'b', 200, 'Solubility'), row('researcher', 'c', 300, 'PETase report')];
  assert.deepEqual(visibleChatHistory(rows, 'PETase', emptyPins).map(r => r.id), ['c', 'a']);
  assert.deepEqual(visibleChatHistory(rows, ' Scientist ', emptyPins).map(r => r.id), ['b']);
});
test('each New Chat is a fresh route and never includes a saved identity', () => {
  for (const mode of Object.keys(emptyPins)) {
    const a = newChatLocation('en', mode, 'draft-a'), b = newChatLocation('en', mode, 'draft-b');
    assert.notEqual(a, b);
    const url = new URL(a, 'http://localhost');
    assert.equal(url.pathname, '/en/agent/chat');
    assert.equal(url.searchParams.get('new'), 'draft-a');
    assert.equal(url.searchParams.get('mode'), mode === 'science_agent' ? null : mode === 'science_expert' ? 'scientist' : 'researcher');
    const saved = chatSessionLocation('en', 'saved', mode, url.searchParams);
    assert.equal(new URL(saved, 'http://localhost').searchParams.has('new'), false);
  }
});
test('refresh preserves a research draft, while another New Chat clears it', () => {
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  assert.equal(claimNewChatDraft('researcher', 'first', storage), true);
  assert.equal(claimNewChatDraft('researcher', 'first', storage), false);
  assert.equal(claimNewChatDraft('science_expert', 'first', storage), true);
  assert.equal(claimNewChatDraft('researcher', 'next', storage), true);
});
