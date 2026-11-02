import test from 'node:test';
import assert from 'node:assert/strict';
import { nextPickerItem, matchPickerItem, pickerPlacement } from '../src/lib/pickerNavigation.ts';

const items = [
  { value: 'a', label: 'Claude Code', disabled: true },
  { value: 'b', label: 'Codex' },
  { value: 'c', label: 'DeepSeek Harness', disabled: true },
  { value: 'd', label: 'Kimi Code' },
];
test('keyboard navigation skips unavailable harnesses and wraps in both directions', () => {
  assert.equal(nextPickerItem(items, -1, 1), 1);
  assert.equal(nextPickerItem(items, 1, 1), 3);
  assert.equal(nextPickerItem(items, 3, 1), 1);
  assert.equal(nextPickerItem(items, 1, -1), 3);
  assert.equal(nextPickerItem(items, 0, -1), 3);
});
test('empty or fully unavailable catalog has no selectable item', () => {
  assert.equal(nextPickerItem([], -1, 1), -1);
  assert.equal(nextPickerItem(items.map(item => ({ ...item, disabled: true })), 0, -1), -1);
});
test('typeahead ignores unavailable entries and matches names without case sensitivity', () => {
  assert.equal(matchPickerItem(items, 'c', -1), 1);
  assert.equal(matchPickerItem(items, 'KIM', 1), 3);
  assert.equal(matchPickerItem(items, 'Deep', 1), -1);
  assert.equal(matchPickerItem(items, 'co', 0), 1);
});
test('composer menus open above the input and stay within a narrow viewport', () => {
  const p = pickerPlacement({ left: 260, top: 640, bottom: 672, width: 120 }, { width: 375, height: 720 });
  assert.equal(p.bottom, 86);
  assert.equal(p.left + p.width, 363);
  assert.ok(p.maxHeight + p.bottom < 720);
});
test('menus near the top open below, and small windows constrain width and height', () => {
  const p = pickerPlacement({ left: 5, top: 10, bottom: 42, width: 80 }, { width: 240, height: 200 });
  assert.equal(p.top, 48);
  assert.equal(p.width, 216);
  assert.equal(p.left, 12);
  assert.ok(p.top + p.maxHeight <= 188);
});
