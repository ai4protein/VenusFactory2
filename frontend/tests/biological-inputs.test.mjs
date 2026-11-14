import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sequenceColumn, numericColumns, parseProteinInput, resultRows, tableCsv } from '../src/lib/biologicalInputs.ts';

test('wrapped raw protein and multi-record FASTA retain identity and custom column', () => {
  assert.deepEqual(parseProteinInput('mktay\nia kqr'), [{ name: 'protein_1', sequence: 'MKTAYIAKQR' }]);
  assert.deepEqual(parseProteinInput('>enzyme A\nACDE\nFG\n>enzyme B\nMKTAY', 'aa_seq'), [{ name: 'enzyme A', aa_seq: 'ACDEFG' }, { name: 'enzyme B', aa_seq: 'MKTAY' }]);
});
test('invalid and empty FASTA records fail before submission', () => {
  for (const s of ['', '>a\n>b\nACDE', 'ACD*', 'ACTG\n>other\nACDE', 'MKTX']) assert.throws(() => parseProteinInput(s));
});
test('sequence inference is unambiguous and numerical blanks are not zero', () => {
  assert.equal(sequenceColumn({columns:['aa_seq','activity'],preview:[{aa_seq:'ACDEF',activity:3}]}),'aa_seq');
  assert.equal(sequenceColumn({columns:['protein','label'],preview:[{protein:'ACDEF',label:'high'}]}),'protein');
  assert.equal(sequenceColumn({columns:['one','two'],preview:[{one:'ACDEF',two:'MKTAY'}]}),'');
  assert.deepEqual(numericColumns({columns:['a','b','c'],preview:[{a:'2',b:null,c:'high'},{a:4,b:'',c:'low'}]}),['a']);
});
test('CSV includes full result rather than preview, quotes text, preserves negative numbers', () => {
  const rows = Array.from({length:40}, (_,i) => ({sample:`enzyme, ${i}`,value:-i}));
  assert.equal(resultRows({rows}).length,40);
  const csv=tableCsv([{name:'a"b\nc',value:-2,text:'=1+1',meta:{x:1}}]);
  assert.ok(csv.includes('"a""b\nc"')); assert.ok(csv.includes('"-2"')); assert.ok(csv.includes('"\'=1+1"'));
  assert.ok(csv.includes('{""x"":1}'));
});
test('multi-output result CSV identifies each result and does not overwrite data', () => {
  const rows=resultRows({outputs:{a:{rows:[{value:1,result_output:'original'}]},b:{rows:[{value:2}]}}});
  assert.equal(rows[0]._result_output,'a'); assert.equal(rows[0].result_output,'original');
  assert.equal(rows[1]._result_output,'b');
  assert.deepEqual(resultRows({outputs:{a:{rows:[{value:1}]}}}),[{value:1}]);
});
