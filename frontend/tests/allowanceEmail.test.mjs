import test from 'node:test';
import assert from 'node:assert/strict';
import { allowanceEmailLink } from '../src/lib/allowanceEmail.ts';

test('feedback email preserves contact details and both allowances without injecting recipients', () => {
  const request = {
    name: ' Researcher ', institution: 'Protein Lab', email: 'researcher@example.org',
    contact: 'Email', conversations: 6, predictions: 30,
    feedback: 'More filters & clearer units?\n建议：保留多行反馈 &bcc=other@example.org',
  };
  const url = new URL(allowanceEmailLink(request));
  assert.equal(url.protocol, 'mailto:');
  assert.equal(url.pathname, 'tanyang.august@sjtu.edu.cn');
  assert.deepEqual([...url.searchParams.keys()], ['subject', 'body']);
  const body = url.searchParams.get('body');
  assert.ok(body.includes('Name: Researcher'));
  assert.ok(body.includes('Email: researcher@example.org'));
  assert.ok(body.includes('Extra conversations per day: 6'));
  assert.ok(body.includes('Extra model predictions per day: 30'));
  assert.ok(body.endsWith(request.feedback));
});
