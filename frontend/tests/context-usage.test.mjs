import test from "node:test";
import assert from "node:assert/strict";
import { contextUsage } from "../src/lib/contextUsage.ts";

test("prepared input estimate uses its own configured budget", () => {
  assert.deepEqual(contextUsage({ estimated_tokens: 6000, budget: 24000, compacted: true }, { context_usage: .9 }),
    { percent: 25, used: 6000, limit: 24000, estimated: true, compacted: true });
});
test("legacy runtime counts take precedence over a rounded ratio", () => {
  assert.equal(contextUsage(null, { context_tokens: 5000, max_context_tokens: 20000, context_usage: .9 }).percent, 25);
});
test("ratio-only legacy snapshots are supported", () => {
  assert.deepEqual(contextUsage(null, { context_usage: .625 }), { percent: 63, estimated: false, compacted: false });
});
test("missing and invalid data remain unknown instead of showing a false zero", () => {
  for (const context of [undefined, null, {}, { estimated_tokens: 0, budget: 0 }, { estimated_tokens: NaN, budget: 100 },
    { estimated_tokens: -1, budget: 100 }, { estimated_tokens: 1, budget: Infinity }]) assert.equal(contextUsage(context), null);
  assert.equal(contextUsage(null, { context_usage: NaN }), null);
});
test("zero is valid, full estimates retain counts, and the ring never overflows", () => {
  assert.equal(contextUsage({ estimated_tokens: 0, budget: 100 }).percent, 0);
  const over = contextUsage({ estimated_tokens: 150, budget: 100 });
  assert.equal(over.percent, 100);
  assert.equal(over.used, 150);
});
