import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSseFrame, streamSSEFromPost } from '../src/lib/sse.ts';

test('SSE preserves multiline text and ignores heartbeat comments', () => {
  assert.equal(parseSseFrame(': heartbeat'), null);
  assert.deepEqual(parseSseFrame('id: run:2\nevent: token\ndata: first \ndata:  second'),
    { id:'run:2', event:'token', data:'first \n second' });
});

test('stream reconnects with cursor and does not replay a delivered delta', async () => {
  const original=globalThis.fetch;const requests=[];const events=[];
  globalThis.fetch=async (url,init) => {
    requests.push({url,init});
    const text=requests.length===1 ? 'id: run:1\nevent: token\ndata: one\n\n' :
      'id: run:1\nevent: token\ndata: one\n\nid: run:2\nevent: token\ndata: two\n\nevent: done\ndata: {}\n\n';
    return new Response(text,{headers:{'X-Run-ID':'run','X-Event-Stream':'/api/chat/sessions/s/runs/run/events'}});
  };
  try {
    await streamSSEFromPost('/api/chat/sessions/s/messages/stream',{execution:{harness:'test'}},event=>events.push(event));
    assert.deepEqual(events.filter(e=>e.event==='token').map(e=>e.data),['one','two']);
    assert.equal(requests[1].init.method,'GET');
    assert.equal(requests[1].init.headers['Last-Event-ID'],'run:1');
    assert.ok(requests[0].init.headers['Idempotency-Key']);
  } finally {globalThis.fetch=original;}
});

test('lost submission response retries with the same idempotency identity', async () => {
  const original=globalThis.fetch;const keys=[];
  globalThis.fetch=async (_url,init) => {
    keys.push(init.headers['Idempotency-Key']);
    if(keys.length===1)throw new TypeError('Connection interrupted');
    return new Response('event: done\ndata: {}\n\n',{headers:{'X-Run-ID':'run'}});
  };
  try {
    await streamSSEFromPost('/api/chat/sessions/s/messages/stream',{execution:{harness:'test'}},()=>{});
    assert.equal(keys.length,2);assert.equal(keys[0],keys[1]);
  } finally {globalThis.fetch=original;}
});
