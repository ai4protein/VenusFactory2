import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chatHistoryWithPending, hasAcceptedTurn } from '../src/lib/pendingChatTurn.ts';
import { buildConversationProcesses } from '../src/lib/conversationProcess.ts';

const draft = (changes = {}) => ({sessionId:'', text:'Inspect this protein', attachments:[], previousUsers:[], ...changes});
const user = (content, turn_id) => ({role:'user',content,turn_id});

test('a first submission displays the question and a live process before session creation returns', () => {
  const pending = draft();
  const history = [];
  const visible = chatHistoryWithPending(history, pending, '');
  assert.deepEqual(visible, [{role:'user',content:pending.text,kind:'text'}]);
  assert.equal(history.length, 0, 'optimistic display must not mutate server history');
  const processes = buildConversationProcesses(visible, [], true);
  assert.equal(processes.length, 1);
  assert.equal(processes[0].anchor, 1);
  assert.equal(processes[0].live, true);
});

test('slow upload and stale empty snapshots preserve attachment-only submissions', () => {
  const pending = draft({sessionId:'a',text:'',attachments:['sample.fasta','measurements.csv']});
  for (let i=0;i<3;i++) assert.match(chatHistoryWithPending([], pending, 'a')[0].content, /sample.fasta[\s\S]*measurements.csv/);
  assert.deepEqual(chatHistoryWithPending([], pending, 'b'), [], 'a different session must not show this draft');
  assert.deepEqual(chatHistoryWithPending([], null, 'a'), [], 'rejected or detached drafts disappear');
});

test('a repeated question waits for a new user turn and server acknowledgement removes the optimistic duplicate', () => {
  const earlier = user('Inspect this protein', 'old');
  const pending = draft({sessionId:'a',previousUsers:[earlier]});
  const oldHistory = [earlier, {role:'assistant',content:'Earlier answer'}];
  assert.equal(chatHistoryWithPending(oldHistory,pending,'a').length, 3);
  assert.equal(hasAcceptedTurn(oldHistory,pending), false);
  const accepted = [...oldHistory,user(pending.text,'new'),{role:'assistant',content:'Current answer'}];
  assert.equal(chatHistoryWithPending(accepted,pending,'a'), accepted);
  assert.equal(hasAcceptedTurn(accepted,pending), true);
});

test('acknowledgement still works if context compaction removes earlier turns or appends attachment text', () => {
  const previousUsers = [user('Earlier question','one'),user('Another question','two')];
  const pending = draft({sessionId:'a',previousUsers,attachments:['sample.fasta']});
  assert.equal(hasAcceptedTurn([user(pending.text,'three')],pending), true);
  assert.equal(hasAcceptedTurn([user(pending.text+'\n\nAttached: sample.fasta')],pending), true);
  assert.equal(hasAcceptedTurn([previousUsers[1]],pending), false);
});
