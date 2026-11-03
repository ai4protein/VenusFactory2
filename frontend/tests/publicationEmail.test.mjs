import test from 'node:test';
import assert from 'node:assert/strict';
import { publicationEmail } from '../src/lib/publicationEmail.ts';

test('publication uses an encoded email draft and includes author attribution fields', () => {
  const url = new URL(publicationEmail('Model &bcc=attacker@example.org', 'goal/1'));
  assert.equal(url.protocol, 'mailto:');
  assert.equal(url.pathname, 'tanyang.august@sjtu.edu.cn');
  assert.equal(url.searchParams.get('bcc'), null);
  const body = url.searchParams.get('body');
  assert.ok(body.includes('Author email (to display on the published model):'));
  assert.ok(body.includes('goal/1'));
  assert.ok(body.includes('test report'));
});
