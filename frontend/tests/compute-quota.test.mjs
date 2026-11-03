import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeQuotaError, COMPUTE_QUOTA_CHANGED } from '../src/lib/computeQuota.ts';

test('shared compute allowance errors show required/remaining counts and reset without opening chat credits', () => {
  const events = []; const old = globalThis.window;
  globalThis.window = { location: { pathname: '/en/workflows' }, dispatchEvent: event => events.push(event) };
  try {
    const quota = { enforced: true, mode: 'online', prediction: { limit: 20, used: 19, remaining: 1 }, resets_at: '2099-01-02T00:00:00+08:00', timezone: 'Asia/Shanghai' };
    const text = computeQuotaError({ detail: { code: 'COMPUTE_DAILY_LIMIT_REACHED', resource: 'prediction', required: 2, quota } });
    assert.match(text, /needs 2 prediction runs; 1 remain today/);
    assert.match(text, /00:00/);
    assert.equal(events[0].type, COMPUTE_QUOTA_CHANGED);
    assert.equal(events[0].detail, quota);
    globalThis.window.location.pathname = '/zh/workflows';
    assert.match(computeQuotaError({ detail: { code: 'COMPUTE_DAILY_LIMIT_REACHED', resource: 'workflow', required: 1, quota: { ...quota, workflow: { remaining: 0 } } } }), /今日剩余 0 次/);
    assert.equal(computeQuotaError({ detail: 'Invalid dataset' }), null);
  } finally { globalThis.window = old; }
});
