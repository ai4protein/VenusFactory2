import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseHistoryPins, updateHistoryPin, pinnedFirst, historyTime} from '../src/lib/historyPreferences.ts';

test('pin, reload and unpin preserve independent mode preferences', () => {
  let pins = updateHistoryPin({}, 'science_agent', 'a', true);
  pins = updateHistoryPin(pins, 'science_expert', 's', true);
  pins = updateHistoryPin(pins, 'researcher', 'r', true);
  const restored = parseHistoryPins(JSON.stringify(pins));
  assert.deepEqual(restored, {science_agent:['a'],science_expert:['s'],researcher:['r']});
  assert.deepEqual(updateHistoryPin(restored, 'science_agent', 'a', false), {science_agent:[],science_expert:['s'],researcher:['r']});
});
test('pinning moves a row without mutating histories or changing unpinned order', () => {
  const items = [{id:'new'}, {id:'middle'}, {id:'old'}];
  const result = pinnedFirst(items, ['old','missing'], item=>item.id);
  assert.deepEqual(result.map(item=>item.id), ['old','new','middle']);
  assert.deepEqual(items.map(item=>item.id), ['new','middle','old']);
  assert.deepEqual(pinnedFirst(items, [], item=>item.id), items);
});
test('invalid stored preferences and invalid dates are handled without breaking history', () => {
  for (const value of ['broken','[]','null']) assert.deepEqual(parseHistoryPins(value), {});
  assert.deepEqual(parseHistoryPins('{"science_agent":["a","a",null,9]}'), {science_agent:['a']});
  assert.equal(historyTime('invalid', 'en'), '');
  const formatted = historyTime('2026-10-31T12:34:56Z', 'en');
  assert.match(formatted, /2026/);
  assert.match(formatted, /\d{2}:\d{2}:\d{2}/);
  assert.match(historyTime('2026-10-31T12:34:56Z', 'zh'), /2026/);
});
