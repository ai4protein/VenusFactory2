import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isRunActive, planProgress } from '../src/lib/runProgress.ts';
test('running status works across all modes without spinning for paused, terminal or input gates', () => {
  for (const s of ['started','active','queued','collecting','chapters','reviewing','stopping']) assert.equal(isRunActive(s),true,s);
  for (const s of ['','completed','stopped','paused','waiting_for_input','interrupted','incomplete','budget_exhausted','error','cancelled']) assert.equal(isRunActive(s),false,s);
});
test('a model checkpoint cannot mark a plan step complete before host review', () => {
  const p=planProgress([{id:'a',status:'completed'},{id:'b',status:'pending'},{id:'c',status:'pending'}],{completed_steps:['a','b','unknown']});
  assert.equal(p.complete,1); assert.equal(p.total,3); assert.deepEqual([...p.reviewing],['b']);
});
