import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isFreeResearchModel, sessionMatches, availabilityReason } from '../src/lib/researchUi.ts';

test('use backend free-route admission without guessing from provider or model name', () => {
  assert.ok(isFreeResearchModel({ provider:'custom', model:'configured-model', free:true }));
  assert.equal(isFreeResearchModel({ provider:'dmx', model:'glm-4-flash', free:false }),false);
  assert.equal(isFreeResearchModel({ provider:'openrouter', model:'vendor/model:free' }),false);
});
test('search combines title and status, supports Chinese and multiple words', () => {
  assert.ok(sessionMatches('  enzyme COMPLETED ', 'Enzyme activity', 'completed'));
  assert.ok(sessionMatches('蛋白', '蛋白溶解度', 'idle'));
  assert.ok(sessionMatches('', 'New chat'));
  assert.equal(sessionMatches('missing', 'Enzyme activity'), false);
});
test('unavailable UI gives concise next steps', () => {
  assert.match(availabilityReason('adapter_pending',false),/Not available in Web/);
  assert.match(availabilityReason('web_interactions_pending',true),/Web 使用/);
  assert.match(availabilityReason('model_protocol_incompatible',true),/兼容模型/);
  assert.ok(availabilityReason(undefined,false));
});


test("model availability keeps missing keys disabled and switches to a compatible free model", async () => {
  const { modelAvailability, preferredModel } = await import("../src/lib/researchUi.ts");
  const choices = [
    { harness: "langgraph", model_id: "dmx", ready: true, reasons: [] },
    { harness: "codex", model_id: "dmx", ready: false, reasons: ["model_protocol_incompatible"] },
    { harness: "codex", model_id: "openrouter", ready: true, reasons: [] },
    { harness: "codex", model_id: "official", ready: false, reasons: ["credential_unavailable", "free_model_required"] },
  ];
  assert.equal(modelAvailability(choices, "codex", "official", true).label, "未配置 Key");
  assert.equal(modelAvailability(choices, "codex", "official", false).ready, false);
  assert.equal(preferredModel(choices, "codex", "dmx"), "openrouter");
  assert.equal(preferredModel(choices, "codex", "openrouter"), "openrouter");
});

test('model menus keep model names and stable route IDs', async () => {
  const { modelName } = await import('../src/lib/researchUi.ts');
  assert.equal(modelName({ label: 'Unisound U2 Flash · DMXAPI Free' }), 'Unisound U2 Flash');
  assert.equal(modelName({ label: 'Google: Gemma 4 31B · OpenRouter Free' }), 'Gemma 4 31B');
  assert.equal(modelName({ model: 'vendor/model:free' }), 'model');
  assert.equal(modelName({ label: 'My protein assistant', model: 'custom-model' }), 'My protein assistant');
});
test('online hides unsupported pairs; local hides missing credentials without guessing provider keys', async () => {
  const { visibleExecutionModels } = await import('../src/lib/researchUi.ts');
  const models = ['ready', 'incompatible', 'missing', 'other'].map(id => ({ id }));
  const choices = [
    { harness: 'codex', model_id: 'ready', ready: true, reasons: [] },
    { harness: 'codex', model_id: 'incompatible', ready: false, reasons: ['model_protocol_incompatible'] },
    { harness: 'codex', model_id: 'missing', ready: false, reasons: ['credential_unavailable'] },
    { harness: 'kimi-code', model_id: 'other', ready: true, reasons: [] },
  ];
  assert.deepEqual(visibleExecutionModels(models, choices, 'codex', true).map(m => m.id), ['ready']);
  assert.deepEqual(visibleExecutionModels(models, choices, 'codex', false).map(m => m.id), ['ready', 'incompatible']);
  assert.deepEqual(visibleExecutionModels(models, [], 'codex', false), []);
});
test('connection notices are short and preserve actionable validation messages', async () => {
  const { connectionNotice } = await import('../src/lib/researchUi.ts');
  assert.equal(connectionNotice('TypeError: Failed to fetch', false), 'Connection lost. Please try again.');
  assert.equal(connectionNotice('Connection lost. Please reconnect and retry.', true), '连接已中断，请重试。');
  assert.equal(connectionNotice('Each file must be 10 MB or smaller.', false), 'Each file must be 10 MB or smaller.');
});
