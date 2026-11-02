import { test } from 'node:test';
import assert from 'node:assert/strict';
import { appendWorkflowStep, workflowIssue, workflowName, workflowSignature } from '../src/lib/workflowEditor.ts';
const labels={data:'Input data',output:'Results'};
const node=(id,kind='model')=>({id,kind,label:id,x:40,y:80,model_id:'fixture:'+id});
test('first model gets input and result; a later model is inserted without losing the result',()=>{
 const first=appendWorkflowStep({nodes:[],edges:[]},node('m1'),'',labels);
 assert.deepEqual(first.nodes.map(n=>n.kind),['data','model','output']);
 assert.deepEqual(first.edges,[{source:'m1_data',target:'m1'},{source:'m1',target:'m1_result'}]);
 const next=appendWorkflowStep(first,node('m2'),'m1',labels);
 assert.deepEqual(next.edges,[{source:'m1_data',target:'m1'},{source:'m1',target:'m2'},{source:'m2',target:'m1_result'}]);
 assert.equal(first.nodes.length,3);assert.equal(first.edges[1].source,'m1');
 assert.equal(new Set(next.nodes.map(n=>n.x+','+n.y)).size,next.nodes.length);
});
test('standalone searches have results without fabricating required input data',()=>{
 const g=appendWorkflowStep({nodes:[],edges:[]},node('search','tool'),'',labels,true);
 assert.deepEqual(g.nodes.map(n=>n.kind),['tool','output']);assert.equal(workflowIssue(g,false),undefined);
});
test('training output goes to prediction model port, not its data port',()=>{
 const g=appendWorkflowStep({nodes:[],edges:[]},node('train','train'),'',labels);
 const next=appendWorkflowStep(g,node('predict','predict'),'train',labels);
 assert.ok(next.edges.some(e=>e.target==='predict'&&e.target_port==='model'));
 assert.ok(next.edges.some(e=>e.source==='predict'&&e.target==='train_result'));
});
test('incomplete drafts can be named and saved independently of run readiness',()=>{
 const w={name:'',graph:appendWorkflowStep({nodes:[],edges:[]},node('ProSST'),'',labels)};
 assert.equal(workflowName(w,false),'ProSST');
 assert.equal(workflowIssue(w.graph,false).id,'ProSST_data');
 const signature=workflowSignature(w);assert.equal(workflowSignature({...w,id:'saved'}),signature);
 assert.notEqual(workflowSignature({...w,name:'Renamed'}),signature);
});
