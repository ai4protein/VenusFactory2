import test from 'node:test';
import assert from 'node:assert/strict';
import { parseArrayParameter, parameterChoices, updateParameter } from '../src/lib/modelParameters.ts';
test('model controls keep chain labels and parse multiple temperatures',()=>{
  assert.deepEqual(parseArrayParameter('A, B C',false),['A','B','C']);
  assert.deepEqual(parseArrayParameter('0.1， 0.2',true),[.1,.2]);
  assert.deepEqual(parseArrayParameter('',false),[]);
  assert.throws(()=>parseArrayParameter('0.1, nope',true));
});
test('changing ProteinMPNN family replaces an unsupported checkpoint',()=>{
  const schema={properties:{model_family:{default:'soluble'},model_name:{default:'v_48_020',choices_by:{field:'model_family',options:{soluble:['v_48_020'],ca:['v_48_002']}}}}};
  assert.deepEqual(parameterChoices(schema.properties.model_name,schema,{}),['v_48_020']);
  assert.deepEqual(updateParameter(schema,{model_name:'v_48_020'},'model_family','ca'),{model_name:'v_48_002',model_family:'ca'});
});
