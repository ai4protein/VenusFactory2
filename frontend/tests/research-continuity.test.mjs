import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ResearchSelection, readResearchDraft, writeResearchDraft, moveResearchDraft } from '../src/lib/researchContinuity.ts';
const storage = () => { const rows = new Map(); return { getItem: k => rows.get(k) ?? null, setItem: (k,v) => rows.set(k,v), removeItem: k => rows.delete(k) }; };
test('late restoration and polling cannot replace a newer research or a new draft', async () => {
  const view = new ResearchSelection(); let resolve;
  const old = view.select('A');
  const pending = new Promise(r => { resolve = r; }).then(() => view.current('A', old));
  const current = view.select('B'); resolve();
  assert.equal(await pending, false); assert.equal(view.current('B', current), true);
  view.select(''); assert.equal(view.current('B', current), false);
});
test('rejoining the same research invalidates older network responses', () => {
  const view = new ResearchSelection(), old = view.select('A'), latest = view.select('A');
  assert.equal(view.current('A', old), false); assert.equal(view.current('A', latest), true);
});
test('Scientist creation migrates its objective and uploaded resources; modes remain isolated', () => {
  const s = storage(), scientist = { objective: 'PETase solubility', attachments: [{ id: 'data' }], reply: 'Use public data' };
  writeResearchDraft('scientist:draft', scientist, s);
  writeResearchDraft('researcher:draft', { focus: 'Catalytic mechanism', sequences: { chain: 'MKWV' } }, s);
  moveResearchDraft('scientist:draft', 'scientist:topic', s);
  assert.deepEqual(readResearchDraft('scientist:topic', {}, s), scientist);
  assert.deepEqual(readResearchDraft('scientist:draft', {}, s), {});
  assert.equal(readResearchDraft('researcher:draft', {}, s).focus, 'Catalytic mechanism');
});
test('refresh restores plan edits and malformed or unavailable storage does not break editing', () => {
  const s = storage(); writeResearchDraft('revision:g1', { text: 'New control', plan: { steps: ['A'] } }, s);
  assert.equal(readResearchDraft('revision:g1', {}, s).text, 'New control');
  const broken = { getItem() { throw Error('blocked'); }, setItem() { throw Error('full'); } };
  assert.deepEqual(readResearchDraft('draft', { focus: '' }, broken), { focus: '' });
  assert.doesNotThrow(() => writeResearchDraft('draft', {}, broken));
});
