import test from 'node:test';
import assert from 'node:assert/strict';
import { MANUAL_SECTIONS, getManualSection, manualQuestion, manualChatHref } from '../src/lib/manualCatalog.ts';

test('manual legacy routes resolve to current topics and both languages', () => {
  for (const [oldKey,newKey] of [['workspace','library'],['report','researcher'],['quick-tools','models'],['advanced-tools','models'],['download','data']]) {
    assert.equal(getManualSection(oldKey)?.key,newKey);
  }
  for (const s of MANUAL_SECTIONS) {
    assert.ok(s.title.en && s.title.zh);
    assert.ok(s.files.English.startsWith('en/') && s.files.Chinese.startsWith('zh/'));
  }
});
test('ask Agent produces an editable topic question, never a run or existing session URL', () => {
  assert.match(manualQuestion('training','zh'),/训练/);
  assert.match(manualQuestion('training','en'),/Train a model/);
  assert.equal(manualQuestion('../../.env','en'),'');
  const url=new URL(manualChatHref('prediction','en'),'http://localhost');
  assert.equal(url.pathname,'/en/agent/chat');
  assert.equal(url.searchParams.get('mode'),'agent');
  assert.equal(url.searchParams.get('help'),'prediction');
  assert.equal(url.searchParams.has('send'),false);
});
