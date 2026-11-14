import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const read=path=>fs.readFileSync(new URL('../src/'+path,import.meta.url),'utf8');
test('retired Workflow has no executable client route or model-plaza action',()=>{
 for(const path of ['App.tsx','lib/pageLoaders.ts','components/AppSidebar.tsx','pages/ModelPlazaPage.tsx','components/RegisteredToolDrawer.tsx','components/ModelPredictionDrawer.tsx']) {
  const source=read(path);
  assert.doesNotMatch(source,/ModelWorkflowPage|\/workflows\/run|\/workflow-components|to=.*?\/workflows/);
 }
 assert.match(read('App.tsx'),/path="workflows\/\*" element=\{<UnavailablePage/);
 assert.match(read('components/RegisteredToolDrawer.tsx'),/post<Job>\('\/tools\/run'/);
});
