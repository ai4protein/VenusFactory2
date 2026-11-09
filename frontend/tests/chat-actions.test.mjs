import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { runChatStreamAction, runChatUpdateAction, cancelChatStreamAction } from '../src/lib/chatActions.ts';
import { streamSSEFromPost } from '../src/lib/sse.ts';

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function streamFixture(request) {
  const stream = { current: null }, trace = [];
  const view = { active: true };
  const action = {
    stream, current: () => view.active, request,
    onStart: () => trace.push('start'), onEvent: event => trace.push(event.event),
    onError: error => trace.push(error.message), onFinish: () => trace.push('finish'),
    onSuccess: () => trace.push('success'),
  };
  return { action, stream, trace, view };
}

test('a streamed response keeps session authentication and clears its own busy state', async () => {
  const fetchBefore = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, init });
    return new Response('event: state\ndata: {}\n\nevent: done\ndata: {}\n\n');
  };
  try {
    const f = streamFixture((signal, onEvent) => streamSSEFromPost('/session/one/response', { answer: 'continue' }, onEvent, signal, { 'X-Session-Token': 'test-owner' }));
    assert.equal(await runChatStreamAction(f.action), true);
    assert.deepEqual(f.trace, ['start', 'state', 'done', 'success', 'finish']);
    assert.equal(f.stream.current, null);
    assert.equal(requests.length, 1);
    assert.equal(requests[0].init.headers['X-Session-Token'], 'test-owner');
    assert.ok(requests[0].init.headers['Idempotency-Key']);
    assert.deepEqual(JSON.parse(requests[0].init.body), { answer: 'continue' });
  } finally { globalThis.fetch = fetchBefore; }
});

test('an error SSE frame followed by done does not accept an interaction', async () => {
  const fetchBefore = globalThis.fetch;
  globalThis.fetch = async () => new Response('event: error\ndata: {"message":"answer expired"}\n\nevent: done\ndata: {}\n\n');
  try {
    const f = streamFixture((signal, onEvent) => streamSSEFromPost('/response', {}, onEvent, signal));
    assert.equal(await runChatStreamAction(f.action), false);
    assert.deepEqual(f.trace, ['start', 'error', 'done', 'finish']);
  } finally { globalThis.fetch = fetchBefore; }
});

test('double submission before React updates makes only one request', async () => {
  const wait = deferred(); let requests = 0;
  const f = streamFixture(() => { requests++; return wait.promise; });
  const first = runChatStreamAction(f.action);
  assert.equal(await runChatStreamAction(f.action), false);
  assert.equal(requests, 1);
  wait.resolve();
  assert.equal(await first, true);
  assert.equal(f.stream.current, null);
});

for (const failure of [false, true]) {
  test(`late ${failure ? 'errors' : 'events'} from a detached stream cannot change a new session`, async () => {
    const wait = deferred(); let emit;
    const f = streamFixture((_signal, callback) => { emit = callback; return wait.promise; });
    const old = runChatStreamAction(f.action);
    const newController = new AbortController();
    f.view.active = false; f.stream.current = newController;
    emit({ event: 'state', data: '{"session_id":"old"}' });
    if (failure) wait.reject(new Error('old connection lost')); else wait.resolve();
    assert.equal(await old, false);
    assert.deepEqual(f.trace, ['start']);
    assert.equal(f.stream.current, newController);
    assert.equal(newController.signal.aborted, false);
  });
}

test('aborting the active stream releases it without a misleading error notice', async () => {
  const f = streamFixture(signal => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))));
  const action = runChatStreamAction(f.action);
  f.stream.current.abort();
  assert.equal(await action, false);
  assert.deepEqual(f.trace, ['start', 'finish']);
  assert.equal(f.stream.current, null);
});

test('current request failures release the stream and retain the error', async () => {
  const f = streamFixture(async () => { throw new Error('connection lost'); });
  assert.equal(await runChatStreamAction(f.action), false);
  assert.deepEqual(f.trace, ['start', 'connection lost', 'finish']);
  assert.equal(f.stream.current, null);
});

test('late iteration acknowledgments cannot apply an old plan or report an old error', async () => {
  for (const failure of [false, true]) {
    const wait = deferred(), updates = []; let current = true;
    const pending = runChatUpdateAction({ current: () => current, request: () => wait.promise,
      onSuccess: result => updates.push(result), onError: error => updates.push(error) });
    current = false;
    if (failure) wait.reject(new Error('old plan failed')); else wait.resolve({ plan: ['old plan'] });
    assert.equal(await pending, false);
    assert.deepEqual(updates, []);
  }
});

// Exercise the actual page handlers with their normal dependencies replaced by
// deferred requests. This covers navigation during the HTTP acknowledgment gap.
const pageText = readFileSync(new URL('../src/pages/ChatPage.tsx', import.meta.url), 'utf8');
const pageSource = ts.createSourceFile('ChatPage.tsx', pageText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function pageHandler(name, bindings) {
  let declaration;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) declaration = node;
    if (ts.isVariableDeclaration(node) && node.name?.text === name && ts.isArrowFunction(node.initializer)) declaration = node;
    ts.forEachChild(node, visit);
  }
  visit(pageSource);
  assert.ok(declaration, `Missing page handler ${name}`);
  const definition = `${ts.isVariableDeclaration(declaration) ? 'const ' : ''}${declaration.getText(pageSource)};`;
  const code = ts.transpileModule(definition, { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText;
  return new Function(...Object.keys(bindings), `${code}; return ${name};`)(...Object.values(bindings));
}
function pageFixture() {
  const sessionLoad = { current: { target: 'first', revision: 1, pending: false } };
  const activeModeRef = { current: 'science_agent' }, visibleModeRef = { current: 'agent' };
  const viewMounted = { current: true };
  const currentSessionView = pageHandler('currentSessionView', { sessionId: 'first', sessionLoad, activeModeRef, visibleModeRef, viewMounted });
  const controller = new AbortController(), abortRef = { current: controller };
  const updates = [], refreshes = [], timers = [];
  const bindings = {
    sessionId: 'first', agentWorking: false, abortRef, currentSessionView,
    runChatUpdateAction, cancelChatStreamAction,
    setError: value => updates.push(['error', value]), setRunStatus: value => updates.push(['status', value]),
    setRunning: value => updates.push(['running', value]),
    refreshCurrentSession: async (...args) => refreshes.push(args),
    setTimeout: callback => timers.push(callback), t: { errStopRun: 'stop failed', errIteration: 'iteration failed' },
  };
  const navigate = (mode = 'science_agent') => {
    sessionLoad.current = { target: 'second', revision: 2, pending: false };
    activeModeRef.current = mode;
    visibleModeRef.current = mode === 'researcher' ? 'researcher' : mode === 'science_expert' ? 'scientist' : 'agent';
  };
  return { bindings, controller, abortRef, updates, refreshes, timers, navigate, viewMounted };
}

for (const mode of ['science_agent', 'science_expert', 'researcher']) {
  test(`stopping an old conversation cannot abort a newly opened ${mode} stream`, async () => {
    const f = pageFixture(), wait = deferred(), cancelled = [];
    const stop = pageHandler('abortRun', { ...f.bindings, cancelChatSession: sid => { cancelled.push(sid); return wait.promise; } });
    const pending = stop();
    f.navigate(mode);
    const newController = new AbortController(); f.abortRef.current = newController;
    wait.resolve(); await pending;
    assert.deepEqual(cancelled, ['first']);
    assert.equal(f.controller.signal.aborted, true);
    assert.equal(newController.signal.aborted, false);
    assert.deepEqual(f.updates, [['status', 'stopping']]);
    assert.deepEqual(f.refreshes, []);
    assert.deepEqual(f.timers, []);
  });
}

test('the delayed stop refresh cannot reopen a conversation after navigation', async () => {
  const f = pageFixture();
  const stop = pageHandler('abortRun', { ...f.bindings, cancelChatSession: async () => {} });
  await stop();
  assert.equal(f.controller.signal.aborted, true);
  assert.equal(f.timers.length, 1);
  f.abortRef.current = null;
  f.navigate(); f.timers[0]();
  assert.deepEqual(f.refreshes, []);
});

test('a failed cancellation preserves the live stream and shows a retryable notice', async () => {
  const f = pageFixture();
  const stop = pageHandler('abortRun', { ...f.bindings, cancelChatSession: async () => { throw new Error('temporarily unavailable'); } });
  await stop();
  assert.equal(f.controller.signal.aborted, false);
  assert.deepEqual(f.updates, [['status', 'stopping'], ['status', 'running'], ['error', 'temporarily unavailable']]);
  assert.deepEqual(f.timers, []);
});

test('iteration updates revalidate only the original visible session', async () => {
  for (const navigate of [false, true]) {
    const f = pageFixture(), wait = deferred(), submitted = [];
    f.abortRef.current = null;
    const iterate = pageHandler('handleIterationDecision', { ...f.bindings,
      iterationDecide: (sid, action) => { submitted.push([sid, action]); return wait.promise; } });
    const pending = iterate('modify_plan');
    if (navigate) f.navigate('science_expert');
    wait.resolve({ status: 'waiting_for_plan_confirmation', plan: ['old plan'] });
    await pending;
    assert.deepEqual(submitted, [['first', 'modify_plan']]);
    assert.deepEqual(f.refreshes, navigate ? [] : [['first', { syncComposer: false }]]);
  }
});


test('all response handlers preserve their endpoint, payload, notices and result contract', async () => {
  const url = name => sid => `${name}/${sid}`;
  const answers = [{ question: 'which assay?', answer: 'solubility' }];
  const plan = [{ title: 'Fit a baseline' }];
  const cases = [
    ['retryLastMessage', [], '/api/chat/sessions/first/messages/retry/stream', {}, 'retry'],
    ['submitClarification', [answers], 'clarification/first', { answers }, 'clarification'],
    ['submitInteraction', [{ id: 'interaction-1', answers }], '/api/chat/sessions/first/interactions/respond/stream', { id: 'interaction-1', answers }, 'ask', true],
    ['submitAskUser', [answers], 'ask/first', { answers }, 'ask'],
    ['submitApproval', [{ decision: 'approved', selected_label: 'allow', feedback: 'one run' }], 'approval/first', { decision: 'approved', selected_label: 'allow', feedback: 'one run' }, 'approval'],
    ['confirmPlan', [plan, false], 'plan/first', { plan, auto_execute: false }, 'plan'],
    ['handleStepDecision', ['abort'], 'step/first', { action: 'abort' }, 'step'],
    ['handleSubReportDecision', ['rewrite', 'add evidence'], 'subreport/first', { action: 'rewrite', comment: 'add evidence' }, 'subreport'],
  ];
  for (const [name, input, endpoint, body, error, refresh] of cases) {
    const calls = [], notices = []; let quotas = 0;
    const action = pageHandler(name, {
      sessionId: 'first', agentWorking: false, running: false,
      snapshot: { history: [{ role: 'user', content: 'goal' }] },
      submitSessionAction: async (...args) => { calls.push(args); return false; },
      setError: () => {}, setModelSwitchNotice: text => notices.push(text),
      refreshChatQuota: async () => { quotas++; },
      getClarificationRespondUrl: url('clarification'), getAskUserRespondUrl: url('ask'),
      getApprovalDecideUrl: url('approval'), getPlanConfirmUrl: url('plan'),
      getStepDecideUrl: url('step'), getSubReportDecideUrl: url('subreport'),
      t: { errRetryMsg: 'retry', errClarification: 'clarification', errAskUser: 'ask', errApproval: 'approval',
        errConfirmPlan: 'plan', errStepDecision: 'step', errSubReport: 'subreport',
        noticeAskUserSubmitted: 'submitted', noticeApprovalApproved: 'approved' },
    });
    const result = await action(...input);
    assert.deepEqual(calls, [[endpoint, body, error, ...(refresh ? [true] : [])]], name);
    assert.equal(quotas, name === 'retryLastMessage' ? 1 : 0, name);
    if (name === 'submitAskUser') assert.deepEqual(notices, ['submitted']);
    if (name === 'submitApproval') assert.deepEqual(notices, ['approved']);
    if (name === 'handleSubReportDecision') assert.equal(result, false);
  }
});


test('leaving Chat prevents a late iteration response from reconnecting a detached reader', async () => {
  const f = pageFixture(), wait = deferred();
  f.abortRef.current = null;
  const iterate = pageHandler('handleIterationDecision', { ...f.bindings, iterationDecide: () => wait.promise });
  const pending = iterate('continue');
  f.viewMounted.current = false;
  wait.resolve({ status: 'running' }); await pending;
  assert.deepEqual(f.refreshes, []);
  assert.deepEqual(f.updates, [['error', '']]);
});

test('browser history restoration detaches the frozen reader before rejoining', () => {
  for (const persisted of [true, false]) {
    const f = pageFixture(), pulls = [];
    const restore = pageHandler('onPageShow', { ...f.bindings, stopStream: () => {}, pull: force => pulls.push(force) });
    restore({ persisted });
    assert.equal(f.controller.signal.aborted, persisted);
    assert.equal(f.abortRef.current, persisted ? null : f.controller);
    assert.deepEqual(pulls, [true]);
  }
});
