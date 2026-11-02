import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chatSessionLocation, chatRouteKey } from '../src/lib/chatSessionRoute.ts';

test('stable conversation URLs identify the session independently of cached selection', () => {
  assert.equal(chatSessionLocation('en', 'saved-session', 'science_agent'), '/en/agent/chat/saved-session');
  assert.equal(chatSessionLocation('zh', 'saved-session', 'science_expert'), '/zh/agent/chat/saved-session?mode=scientist');
  assert.notEqual(chatRouteKey('same-id', 'science_agent'), chatRouteKey('same-id', 'science_expert'));
});
test('new chats clear identity and Researcher uses its own report identity', () => {
  const previous = new URLSearchParams('mode=scientist&session_id=old');
  assert.equal(chatSessionLocation('en', '', 'science_agent', previous), '/en/agent/chat');
  assert.equal(chatSessionLocation('en', 'old', 'researcher', previous), '/en/agent/chat/old?mode=researcher');
  assert.equal(previous.get('session_id'), 'old');
});
test('session identifiers cannot change the route or inject query parameters', () => {
  assert.equal(chatSessionLocation('en', 'a/b?mode=researcher', 'science_agent'), '/en/agent/chat/a%2Fb%3Fmode%3Dresearcher');
});
