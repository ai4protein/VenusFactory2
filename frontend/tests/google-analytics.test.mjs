import test from 'node:test';
import assert from 'node:assert/strict';
import { initializeGoogleAnalytics } from '../src/lib/googleAnalytics.ts';

function browser() {
  const scripts = [];
  return {
    scripts,
    document: {
      getElementById: id => scripts.find(script => script.id === id) ?? null,
      createElement: tag => { assert.equal(tag, 'script'); return {}; },
      head: { appendChild: script => scripts.push(script) },
    },
  };
}

test('local, unknown, missing and invalid configuration never load Google', () => {
  for (const config of [
    { mode: 'local', google_analytics_id: 'G-TEST123456' },
    { google_analytics_id: 'G-TEST123456' },
    { mode: 'online' },
    { mode: 'online', google_analytics_id: 'G-TEST123456&x=1' },
  ]) {
    const target = browser();
    initializeGoogleAnalytics(config, target);
    assert.equal(target.scripts.length, 0);
    assert.equal(target.dataLayer, undefined);
  }
});

test('online loads asynchronously once and does not add manual page views', () => {
  const target = browser();
  const config = { mode: 'online', google_analytics_id: 'G-TEST123456' };
  initializeGoogleAnalytics(config, target);
  initializeGoogleAnalytics(config, target);
  assert.equal(target.scripts.length, 1);
  assert.equal(target.scripts[0].src, 'https://www.googletagmanager.com/gtag/js?id=G-TEST123456');
  assert.equal(target.scripts[0].async, true);
  const calls = target.dataLayer.map(value => Array.from(value));
  assert.equal(calls.length, 2);
  assert.equal(calls[0][0], 'js');
  assert.ok(calls[0][1] instanceof Date);
  assert.deepEqual(calls[1], ['config', 'G-TEST123456']);
});

test('existing dataLayer is preserved and runtime IDs are not baked into source', () => {
  const target = browser();
  const prior = { event: 'existing' };
  target.dataLayer = [prior];
  initializeGoogleAnalytics({ mode: 'online', google_analytics_id: 'G-SECOND1234' }, target);
  assert.equal(target.dataLayer[0], prior);
  assert.ok(target.scripts[0].src.endsWith('G-SECOND1234'));
});
