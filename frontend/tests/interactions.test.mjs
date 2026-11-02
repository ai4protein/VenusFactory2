import test from 'node:test';
import assert from 'node:assert/strict';
import { canAnswer, makeResponse } from '../src/lib/interactions.ts';

test('free text, choices and unanswered questions require explicit user input', () => {
  const interaction={id:'i',questions:[{id:'protein',question:'Protein?',options:[],allow_text:true,multiple:false}]};
  const draft={answers:{},feedback:''};
  assert.equal(canAnswer(interaction,draft),false);
  draft.answers.protein={selected:[],text:' PETase '};
  assert.equal(canAnswer(interaction,draft),true);
  assert.deepEqual(makeResponse(interaction,draft,'answer').answers,[{question_id:'protein',selected:[],text:'PETase'}]);
  interaction.questions.push({id:'property',question:'Properties?',options:['Stability','Solubility'],allow_text:false,multiple:true});
  assert.equal(canAnswer(interaction,draft),false);
  draft.answers.property={selected:[0,1],text:''};
  assert.equal(canAnswer(interaction,draft),true);
  assert.deepEqual(makeResponse(interaction,draft,'cancel').answers,[]);
  assert.equal(makeResponse(interaction,draft,'reject').decision,'reject');
});
