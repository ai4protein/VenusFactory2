import { test } from 'node:test';
import assert from 'node:assert/strict';
import { connectGraph, removeGraphNode } from '../src/lib/modelPlatform.ts';

const base = { nodes: [{id:'data',kind:'data'}, {id:'other',kind:'data'}, {id:'train',kind:'train'}, {id:'predict',kind:'predict'}, {id:'out',kind:'output'}], edges: [] };
test('training and data connections coexist on distinct prediction inputs', () => {
  let g=connectGraph(base,'data','train');g=connectGraph(g,'train','predict');g=connectGraph(g,'data','predict');
  assert.equal(g.edges.filter(e=>e.target==='predict').length,2);
  g=connectGraph(g,'other','predict');
  assert.deepEqual(g.edges.filter(e=>e.target==='predict'),[{source:'train',target:'predict',target_port:'model'},{source:'other',target:'predict',target_port:'data'}]);
});
test('wrong prediction port types are rejected',()=>{
  assert.throws(()=>connectGraph(base,'data','predict','model'));
  assert.throws(()=>connectGraph(base,'train','predict','data'));
  assert.throws(()=>connectGraph(base,'train','train'));
});
test('deleting a card removes only its incident wires and is reversible',()=>{
  let g=connectGraph(connectGraph(connectGraph(base,'data','train'),'train','predict'),'other','predict');
  g=connectGraph(g,'predict','out');
  const next=removeGraphNode(g,'train');
  assert.equal(next.nodes.length,4);assert.equal(next.edges.length,2);
  assert.equal(g.nodes.length,5);assert.equal(g.edges.length,4);
  assert.ok(next.edges.every(e=>e.source!=='train'&&e.target!=='train'));
});
