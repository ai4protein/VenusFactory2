import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialProteinModel, modelPlazaSection, modelWorkspaceRoute } from '../src/lib/modelPlazaNavigation.ts';

test('training and experimental models share one plaza section in both languages', () => {
  assert.equal(modelPlazaSection('/en/models'), '');
  assert.equal(modelPlazaSection('/zh/models/'), '');
  assert.equal(modelPlazaSection('/en/models/new'), 'training');
  assert.equal(modelPlazaSection('/zh/models/training'), 'training');
  assert.equal(modelPlazaSection('/en/models/leaderboards'), 'leaderboards');
});
test('foundation cards carry the chosen model into training; old routes remain inside the plaza', () => {
  const path = modelWorkspaceRoute({ name: 'ESM2-8M · Backbone', category: 'foundation' });
  const url = new URL(path, 'http://localhost');
  assert.equal(url.pathname, '/models/training');
  assert.equal(url.searchParams.get('plm'), 'ESM2-8M');
  assert.equal(modelWorkspaceRoute({ name: 'Saved model', route: '/custom-model/predict?model=a%2Fb' }), '/models/predict?model=a%2Fb');
});
test('deep links cannot select a model excluded by online training policy', () => {
  const allowed = { 'ESM2-8M': 'facebook/esm2_t6_8M_UR50D' };
  assert.equal(initialProteinModel('ESM2-650M', allowed), 'ESM2-8M');
  assert.equal(initialProteinModel('ESM2-8M', allowed), 'ESM2-8M');
  assert.equal(initialProteinModel('constructor', allowed), 'ESM2-8M');
  assert.equal(initialProteinModel(null, {}), '');
});
