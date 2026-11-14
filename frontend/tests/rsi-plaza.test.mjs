import test from 'node:test';
import assert from 'node:assert/strict';
import { repositoryTree, firstCaseFile, caseEndpoint, caseSession, caseDate } from '../src/lib/rsiPlaza.ts';
import { modelPlazaSection } from '../src/lib/modelPlazaNavigation.ts';

test('RSI repository IDs remain one encoded route component and retain private session identity', () => {
 const id='research:12345678-abcd-abcd-abcd-123456789012:abcdef0123456789';
 assert.equal(caseSession(id),'12345678-abcd-abcd-abcd-123456789012');
 assert.equal(caseSession('capability:abcdef'), '');
 assert.equal(caseSession('research:../../other:abc'), '');
 assert.equal(caseEndpoint(id), '/api/rsi-plaza/cases/'+encodeURIComponent(id));
 assert.equal(caseEndpoint('unexpected/path'), '/api/rsi-plaza/cases/unexpected%2Fpath');
 assert.equal(modelPlazaSection('/en/models/rsi'), 'rsi');
 assert.equal(modelPlazaSection('/zh/models/rsi'), 'rsi');
});
test('repository directory navigation preserves exact paths without object-property collisions', () => {
 const tree=repositoryTree(['README.md','src/model.py','src/utils/data.py','constructor/file.txt','__proto__/note.md','../private.txt','bad//file'].map(path=>({path,size:1})));
 assert.deepEqual(tree.files.map(f=>f.path),['README.md']);
 assert.equal(tree.folders.find(f=>f.name==='src').folders[0].files[0].path,'src/utils/data.py');
 assert.equal(tree.folders.find(f=>f.name==='__proto__').files[0].path,'__proto__/note.md');
 assert.equal(Object.prototype.hasOwnProperty.call({},'note.md'),false);
 assert.equal(tree.folders.some(f=>['..','bad'].includes(f.name)),false);
});
test('README defaults use available files only and timestamps do not invent dates', () => {
 const files=[{path:'src/a.py',size:3},{path:'README.md',size:50}];
 assert.equal(firstCaseFile({files,readme:'missing.md'}),'README.md');
 assert.equal(firstCaseFile({files,readme:'src/a.py'}),'src/a.py');
 assert.equal(firstCaseFile({files:[]}), '');
 assert.equal(caseDate('invalid','en'),'');
 assert.equal(caseDate(undefined,'zh'),'');
 assert.ok(caseDate(1700000000,'en'));
});
