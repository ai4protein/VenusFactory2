import test from 'node:test';
import assert from 'node:assert/strict';
import { createChatSession, deleteChatSession, listChatSessions } from '../src/lib/api.ts';

for (const action of ['create', 'delete']) {
  test(`${action} followed by history refresh starts a new authenticated read generation`, async () => {
    const prior = { fetch: globalThis.fetch, window: globalThis.window, localStorage: globalThis.localStorage };
    const values = new Map(), reads = [], mutations = [];
    const storage = { getItem: key => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
    globalThis.localStorage = storage;
    globalThis.window = Object.assign(new EventTarget(), { localStorage: storage });
    globalThis.fetch = (url, options = {}) => {
      if (!options.method || options.method === 'GET') return new Promise(resolve => reads.push(resolve));
      mutations.push([url, options.method]);
      return Promise.resolve(Response.json({ success: true, session_id: 'topic', model_name: 'free-model',
        created_at: '2026-11-10T00:00:00Z', session_access_token: 'synthetic-test-owner' }));
    };
    try {
      const old = listChatSessions();
      await Promise.resolve();
      if (action === 'create') await createChatSession('science_expert');
      else await deleteChatSession('topic');
      const fresh = listChatSessions();
      await Promise.resolve();
      assert.equal(mutations.length, 1);
      assert.equal(reads.length, 2, 'the post-mutation request must not reuse the old list');
      reads[0](Response.json({ sessions: [{ session_id: 'before' }] }));
      assert.equal((await old).sessions[0].session_id, 'before');
      const shared = listChatSessions();
      await Promise.resolve();
      assert.equal(reads.length, 2, 'ordinary simultaneous reads still coalesce');
      reads[1](Response.json({ sessions: [{ session_id: 'after' }] }));
      assert.equal((await fresh).sessions[0].session_id, 'after');
      assert.equal((await shared).sessions[0].session_id, 'after');
    } finally {
      for (const [key, value] of Object.entries(prior)) {
        if (value === undefined) delete globalThis[key]; else globalThis[key] = value;
      }
    }
  });
}
