import test from 'node:test';
import assert from 'node:assert/strict';
import { submitsComposer } from '../src/lib/composerKeys.ts';
test('Enter sends, Shift+Enter and Chinese IME composition do not send', () => {
  assert.equal(submitsComposer({key:'Enter'}),true);
  assert.equal(submitsComposer({key:'Enter',shiftKey:true}),false);
  assert.equal(submitsComposer({key:'Enter',isComposing:true}),false);
  assert.equal(submitsComposer({key:'Enter',keyCode:229}),false);
  assert.equal(submitsComposer({key:'a'}),false);
});
