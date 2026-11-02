import { test } from 'node:test';
import assert from 'node:assert/strict';
import { connectGraph } from '../src/lib/modelPlatform.ts';
const graph = { nodes: [{id:'a',kind:'data'},{id:'b',kind:'model'},{id:'c',kind:'model'},{id:'d',kind:'output'}], edges: [] };
test('connect and branch without mutating the saved graph',()=>{
 const a=connectGraph(graph,'a','b'); const b=connectGraph(a,'b','c'); const c=connectGraph(b,'b','d');
 assert.equal(graph.edges.length,0); assert.equal(c.edges.length,3);
});
test('replace the old input when reconnecting a model',()=>{
 const a=connectGraph(connectGraph(graph,'a','b'),'b','c'); const b=connectGraph(a,'a','c');
 assert.deepEqual(b.edges,[{source:'a',target:'b'},{source:'a',target:'c'}]);
});
test('reject cycles, self connections and reversed ports',()=>{
 const a=connectGraph(connectGraph(graph,'a','b'),'b','c');
 assert.throws(()=>connectGraph(a,'c','b'),/cycle/);
 assert.throws(()=>connectGraph(graph,'b','b'));
 assert.throws(()=>connectGraph(graph,'d','b'));
 assert.throws(()=>connectGraph(graph,'b','a'));
});

test('connections reject wrong kinds and ports without dropping the previous edge',()=>{
 const g={nodes:[{id:'data',kind:'data',dataset_id:'table'},{id:'pdb',kind:'data',dataset_id:'structure'},{id:'calc',kind:'tool',input_kinds:['structure']},{id:'train',kind:'train'},{id:'predict',kind:'predict'}],edges:[]};
 const datasets=[{id:'table',kind:'table'},{id:'structure',kind:'structure'}];
 assert.throws(()=>connectGraph(g,'data','calc','data',datasets),/requires structure/);
 assert.equal(connectGraph(g,'pdb','calc','data',datasets).edges.length,1);
 assert.throws(()=>connectGraph(g,'pdb','train','data',datasets),/requires table/);
 assert.throws(()=>connectGraph(g,'data','predict','bogus',datasets),/Unknown input port/);
 assert.throws(()=>connectGraph(g,'train','predict','data',datasets));
 assert.throws(()=>connectGraph(g,'data','calc','model',datasets));
 assert.deepEqual(g.edges,[]);
});
