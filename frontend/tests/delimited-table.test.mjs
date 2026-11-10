import test from 'node:test';
import assert from 'node:assert/strict';
import { parseDelimitedTable, tableRecords } from '../src/lib/delimitedTable.ts';

const example = 'sample,measurement,notes\r\n001,0.9942129558371191,"pH 7, 25 C"\r\n002,1.2,"repeat A\nrepeat B"\r\n';
test('one quoted multiline observation remains one row without changing identifiers or precision', () => {
  assert.deepEqual(parseDelimitedTable(example), {headers:['sample','measurement','notes'], rows:[
    ['001','0.9942129558371191','pH 7, 25 C'],['002','1.2','repeat A\nrepeat B']], truncated:false});
});
test('CSV and TSV preserve delimiters, escaped quotes, BOM, CRLF and empty cells', () => {
  for (const separator of [',','\t']) {
    const source='\uFEFFid'+separator+'note'+separator+'value\r\n0001'+separator+'"a'+separator+'b ""quoted""\r\nnext"'+separator+'\r\n';
    assert.deepEqual(parseDelimitedTable(source, separator).rows, [['0001','a'+separator+'b "quoted"\r\nnext','']]);
  }
});
test('empty documents, header-only tables and single-column data have explicit empty states', () => {
  assert.deepEqual(parseDelimitedTable(''),{headers:[],rows:[],truncated:false});
  assert.deepEqual(parseDelimitedTable('\uFEFF\r\n\n'),{headers:[],rows:[],truncated:false});
  assert.deepEqual(parseDelimitedTable('id'),{headers:['id'],rows:[],truncated:false});
  assert.deepEqual(parseDelimitedTable('id\n001\n\n""\n').rows,[['001'],['']]);
  assert.deepEqual(parseDelimitedTable('x,y\n,\n').rows,[['','']]);
});
test('cell whitespace and scientific notation are display strings', () => {
  assert.deepEqual(parseDelimitedTable('id,value\n" 001 ",1.234567890123456789e-123\n').rows,[[' 001 ','1.234567890123456789e-123']]);
});
test('preview limits retain whole records only, including quoted multiline cells', () => {
  const table=parseDelimitedTable(example,',',{maxRows:1});
  assert.equal(table.rows.length,1); assert.equal(table.truncated,true);
  for (let end=example.indexOf('002,');end<example.length-2;end++) {
    const part=parseDelimitedTable(example,',',{maxChars:end});
    assert.equal(part.rows.length,1,`cut at ${end}`); assert.equal(part.truncated,true);
  }
  assert.equal(parseDelimitedTable(example,',',{maxRows:2}).truncated,false);
  assert.equal(parseDelimitedTable(example,',',{maxRows:0}).rows.length,0);
});
test('malformed or uneven records are not silently turned into scientific tables', () => {
  for(const source of ['id,value\n001,"open','id,value\n001,2,3','id,value\n001','id,value\n001,"x"oops','id,value\n001,ab"cd']) {
    assert.throws(()=>parseDelimitedTable(source));
  }
});
test('duplicate, unnamed and object-special headers do not overwrite measurements', () => {
  const table=parseDelimitedTable('score,score,score (2),,column_4,__proto__,constructor\n0.1,0.2,0.3,x,y,z,k\n');
  const [row]=tableRecords(table);
  assert.deepEqual(Object.keys(row),['score','score (3)','score (2)','column_4 (2)','column_4','__proto__','constructor']);
  assert.deepEqual(Object.values(row),['0.1','0.2','0.3','x','y','z','k']);
  assert.equal(Object.getPrototypeOf(row),Object.prototype);
});
test('preview budgets reject invalid values', () => {
  for(const options of [{maxRows:-1},{maxRows:NaN},{maxRows:1.5},{maxChars:Infinity},{maxChars:-1}]) {
    assert.throws(()=>parseDelimitedTable(example,',',options));
  }
});
