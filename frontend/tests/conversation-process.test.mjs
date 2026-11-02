import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildConversationProcesses, isProcessItem, toolState } from '../src/lib/conversationProcess.ts';

test('tools appear once before the reply of their own user turn', () => {
  const items = [
    {role:'user',content:'First'},
    {role:'assistant',kind:'thinking',content:'Inspect data',turn_id:'a'},
    {role:'assistant',content:'First result',turn_id:'a'},
    {role:'assistant',content:'More detail',turn_id:'a'},
    {role:'user',content:'Second'},
    {role:'assistant',content:'Second result',turn_id:'b'},
  ];
  const tools = [{tool_call_id:'one',turn_id:'a',status:'ok'}, {tool_call_id:'two',turn_id:'b',status:'ok'}];
  const groups=buildConversationProcesses(items,tools,false);
  assert.deepEqual(groups.map(g=>[g.anchor,g.tools.map(t=>t.tool_call_id)]),[[1,['one']],[5,['two']]]);
  assert.equal(groups[0].messages.length,1);
  assert.ok(groups.every(g=>!g.live));
});

test('a running tool is visible before any assistant response and survives its update', () => {
  const items=[{role:'user',content:'Run model'}];
  const groups=buildConversationProcesses(items,[
    {tool_call_id:'one',turn_id:'pending-turn',status:'running',name:'run_model'},
    {tool_call_id:'one',status:'ok',output:{score:0.8}},
  ],true);
  assert.equal(groups[0].anchor,1);
  assert.equal(groups[0].tools.length,1);
  assert.equal(groups[0].tools[0].turn_id,'pending-turn');
  assert.equal(groups[0].tools[0].status,'ok');
  assert.equal(groups[0].live,true);
});

test('orphan tools stay in the latest turn and only that turn animates', () => {
  const items=[{role:'user',content:'A'},{role:'assistant',kind:'thinking',content:'A',turn_id:'a'},
    {role:'user',content:'B'},{role:'assistant',kind:'thinking',content:'B',turn_id:'b'}];
  const groups=buildConversationProcesses(items,[{name:'read_file'}],true);
  assert.equal(groups[0].tools.length,0);
  assert.equal(groups[0].live,false);
  assert.equal(groups[1].tools.length,1);
  assert.equal(groups[1].live,true);
});

test('idle conversations have no invented thinking state; interruption is not completion', () => {
  assert.deepEqual(buildConversationProcesses([{role:'user',content:'A'}],[],false),[]);
  assert.equal(toolState({status:'running'},false),'stopped');
  assert.equal(toolState({status:'failed',output:'failed'},false),'error');
  assert.equal(toolState({status:'cancelled'},false),'stopped');
  assert.equal(toolState({status:'completed'}),'complete');
  assert.equal(toolState({outputs:{value:2}}),'complete');
  assert.equal(toolState({status:'running',output:'partial'},true),'running');
});

test('checkpoints, user input and compaction remain outside the process fold', () => {
  assert.equal(isProcessItem({role:'assistant',kind:'checkpoint',content:'Review this'}),false);
  assert.equal(isProcessItem({role:'assistant',kind:'compaction',content:'Compressed'}),false);
  assert.equal(isProcessItem({role:'user',kind:'thinking',content:'My question'}),false);
  assert.equal(isProcessItem({role:'assistant',phase:'sub_report_checkpoint',content:'Continue?'}),false);
  assert.equal(isProcessItem({role:'assistant',phase:'thinking',content:'Working'}),true);
});

test('an unanswered earlier turn keeps its tools before the next user message', () => {
  const groups=buildConversationProcesses([
    {role:'user',content:'A',turn_id:'a'},
    {role:'user',content:'B',turn_id:'b'},
    {role:'assistant',content:'B reply',turn_id:'b'},
  ],[{name:'read_file',turn_id:'a',status:'error'}],false);
  assert.equal(groups[0].anchor,1);
});
