import test from 'node:test';
import assert from 'node:assert/strict';
import { canonicalManualPath, MANUAL_SECTIONS } from '../src/lib/manualCatalog.ts';

test('manual chapter URLs remain canonical in both languages', () => {
  for (const lang of ['en', 'zh']) {
    for (const key of ['index', ...MANUAL_SECTIONS.map(section => section.key)]) {
      const href = `/${lang}/manual/${key}`;
      assert.equal(canonicalManualPath(href, lang), href);
      assert.equal(canonicalManualPath(`${href}/`, lang), href);
    }
    assert.equal(canonicalManualPath(`/${lang}/manual`, lang), `/${lang}/manual/index`);
  }
});

test('previously accumulated navigation URLs recover the requested final chapter', () => {
  assert.equal(canonicalManualPath('/en/manual/index/index/start/agent/models/training/evaluation', 'en'), '/en/manual/evaluation');
  assert.equal(canonicalManualPath('/en/manual/index/faq/agent', 'en'), '/en/manual/agent');
  assert.equal(canonicalManualPath('/zh/manual/index/start/scientist', 'zh'), '/zh/manual/scientist');
  assert.equal(canonicalManualPath('/en/manual/agent/index', 'en'), '/en/manual/index');
});

test('historical section aliases resolve to their current guide', () => {
  for (const lang of ['en', 'zh']) {
    for (const section of MANUAL_SECTIONS) {
      for (const alias of section.aliases) {
        assert.equal(canonicalManualPath(`/${lang}/manual/${alias}`, lang), `/${lang}/manual/${section.key}`);
      }
    }
  }
});

test('unknown chapters do not silently resolve to a different guide', () => {
  for (const path of ['/en/manual/no-such-guide', '/en/manual/no-such-guide/agent', '/en/models/agent', '/zh/manual/agent']) {
    assert.equal(canonicalManualPath(path, 'en'), null);
  }
});
