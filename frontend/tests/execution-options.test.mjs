import test from 'node:test';
import assert from 'node:assert/strict';
import { getAgentExecutionOptions } from '../src/lib/api.ts';

const catalog = {
  default: { harness: 'codex', model_id: 'fixture' }, shared_default_model: 'fixture',
  harnesses: [{ id: 'codex', label: 'Codex', ready: true, model_catalog: 'shared' }],
  models: [{ id: 'fixture', label: 'Fixture', model: 'fixture', provider: 'fixture', ready: true }],
  combinations: [{ harness: 'codex', model_id: 'fixture', ready: true, reasons: [] }],
};

test('a catalog outage can be retried without reloading the page or starting a model request', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push([url, options]);
    return requests.length === 1 ? new Response('unavailable', { status: 503 }) : Response.json(catalog);
  });
  await assert.rejects(getAgentExecutionOptions(), /HTTP 503/);
  assert.deepEqual(await getAgentExecutionOptions(), catalog);
  assert.equal(requests.length, 2);
  assert.ok(requests.every(([url, options]) => url === '/api/chat/agent-execution-options' && options.cache === 'no-store'));
});

for (const bad of [null, {}, { ...catalog, default: null }, { ...catalog, models: [null] },
  { ...catalog, combinations: [{ ...catalog.combinations[0], ready: 'true' }] },
  { ...catalog, combinations: [{ ...catalog.combinations[0], reasons: null }] }]) {
  test(`malformed catalogs are recoverable: ${JSON.stringify(bad)}`, async t => {
    t.mock.method(globalThis, 'fetch', async () => Response.json(bad));
    await assert.rejects(getAgentExecutionOptions(), /Invalid model options/);
  });
}

test('an unconfigured installation and unavailable models remain valid catalog responses', async t => {
  const empty = { ...catalog, harnesses: [], models: [], combinations: [] };
  t.mock.method(globalThis, 'fetch', async () => Response.json(empty));
  assert.deepEqual(await getAgentExecutionOptions(), empty);
  const unavailable = structuredClone(catalog);
  unavailable.models[0].ready = false;
  unavailable.combinations[0] = { ...unavailable.combinations[0], ready: false, reasons: ['credential_missing'] };
  globalThis.fetch = async () => Response.json(unavailable);
  assert.deepEqual(await getAgentExecutionOptions(), unavailable);
});

function pendingFetch(t) {
  let observed;
  t.mock.method(globalThis, 'fetch', (_url, { signal }) => {
    observed = signal;
    return new Promise((_resolve, reject) => {
      if (signal.aborted) reject(signal.reason);
      else signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    });
  });
  return () => observed;
}

test('leaving the page cancels the pending options request', async t => {
  const observed = pendingFetch(t), controller = new AbortController();
  const request = getAgentExecutionOptions(controller.signal);
  controller.abort();
  await assert.rejects(request, { name: 'AbortError' });
  assert.equal(observed().aborted, true);
});

test('an already cancelled load cannot start an active request', async t => {
  const observed = pendingFetch(t), controller = new AbortController();
  controller.abort();
  await assert.rejects(getAgentExecutionOptions(controller.signal), { name: 'AbortError' });
  assert.equal(observed().aborted, true);
});

test('a stalled options endpoint times out and the next load succeeds', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const observed = pendingFetch(t);
  const failure = assert.rejects(getAgentExecutionOptions(), { name: 'TimeoutError' });
  t.mock.timers.tick(15_000);
  await failure;
  assert.equal(observed().aborted, true);
  globalThis.fetch = async () => Response.json(catalog);
  assert.deepEqual(await getAgentExecutionOptions(), catalog);
});
