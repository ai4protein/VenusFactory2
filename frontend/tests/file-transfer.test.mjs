import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bindFileTransfer, hasTransferredFiles, transferredFiles, mergeFiles, proteinFileError } from '../src/lib/fileTransfer.ts';
import { uploadSizeError, ONLINE_UPLOAD_BYTES } from '../src/lib/uploadLimits.ts';

const file = (name='protein.fasta', content='>protein\nACDEFGHIK') => new File([content],name,{type:'text/plain',lastModified:42});
const transfer = files => ({files,items:[],types:['Files'],dropEffect:'none'});
function fixture() {
  const root = new EventTarget(), view = new EventTarget(), classes = new Set(), attrs = new Map(), uploads=[], errors=[];
  root.classList={add:c=>classes.add(c),remove:c=>classes.delete(c)};
  root.setAttribute=(k,v)=>attrs.set(k,v);root.removeAttribute=k=>attrs.delete(k);
  root.ownerDocument={defaultView:view};root.contains=node=>node===root||node==='child';
  const options={disabled:false,label:'Drop files',unavailable:'Busy',unreadable:'Choose files',onFiles:files=>uploads.push(files),onError:e=>errors.push(e)};
  const cleanup=bindFileTransfer(root,()=>options);
  function emit(type,data,extra={}) { const e=new Event(type,{cancelable:true});Object.assign(e,{[type==='paste'?'clipboardData':'dataTransfer']:data,...extra});root.dispatchEvent(e);return e; }
  return {root,view,classes,attrs,options,uploads,errors,cleanup,emit};
}

test('ordinary text, HTML and URLs retain native paste/drop behavior',()=>{
  const f=fixture(); const text={files:[],items:[{kind:'string'}],types:['text/plain','text/html']};
  assert.equal(f.emit('paste',text).defaultPrevented,false);
  assert.equal(f.emit('drop',text).defaultPrevented,false);
  assert.deepEqual(f.uploads,[]);f.cleanup();
});
test('pasted screenshots and clipboard file items use the same upload handler',()=>{
  const f=fixture(), image=new File(['png'],'image.png',{type:'image/png'});
  const clip={files:[],types:['Files'],items:[{kind:'file',getAsFile:()=>image}]};
  assert.equal(f.emit('paste',clip).defaultPrevented,true);
  assert.deepEqual(f.uploads,[[image]]);f.cleanup();
});
test('multiple dropped files upload once and repeated entries are deduplicated',()=>{
  const f=fixture(), a=file(),b=file('data.csv','sequence,label\nACDE,1');
  assert.equal(f.emit('drop',transfer([a,b,a])).defaultPrevented,true);
  assert.deepEqual(f.uploads,[[a,b]]);
  assert.deepEqual(mergeFiles([a],[a,b]),[a,b]);f.cleanup();
});
test('protected dragover identifies Files without reading unavailable contents',()=>{
  const f=fixture(), data={files:[],items:[],types:['Files']};
  assert.equal(hasTransferredFiles(data),true);
  assert.equal(f.emit('dragenter',data).defaultPrevented,true);
  assert.ok(f.classes.has('is-file-dragover'));
  f.emit('dragover',data);assert.equal(data.dropEffect,'copy');
  f.emit('dragleave',data,{relatedTarget:'child'});assert.ok(f.classes.has('is-file-dragover'));
  f.emit('dragleave',data,{relatedTarget:null});assert.equal(f.classes.size,0);f.cleanup();
});
test('busy composers consume files without navigating away or submitting work',()=>{
  const f=fixture();f.options.disabled=true;
  assert.equal(f.emit('drop',transfer([file()])).defaultPrevented,true);
  assert.equal(f.emit('paste',transfer([file()])).defaultPrevented,true);
  assert.deepEqual(f.uploads,[]);assert.deepEqual(f.errors,['Busy','Busy']);f.cleanup();
});
test('folder-only and unreadable transfers are reported rather than discarded',()=>{
  const f=fixture();f.emit('drop',{files:[],types:['Files'],items:[{kind:'file',getAsFile:()=>null}]});
  assert.deepEqual(f.errors,['Choose files']);assert.deepEqual(transferredFiles(null),[]);f.cleanup();
});
test('mode switches clean listeners and drag feedback, and callbacks stay current',()=>{
  const f=fixture();f.options.onFiles=()=>f.errors.push('new mode');
  f.emit('paste',transfer([file()]));assert.deepEqual(f.errors,['new mode']);
  f.emit('dragenter',transfer([]));f.cleanup();
  assert.equal(f.classes.size,0);assert.equal(f.attrs.size,0);
  assert.equal(f.emit('drop',transfer([file()])).defaultPrevented,false);
  assert.deepEqual(f.errors,['new mode']);
});
test('cancelled drags clear feedback and async upload failures are reported',async()=>{
  const f=fixture();f.emit('dragenter',transfer([]));f.view.dispatchEvent(new Event('blur'));assert.equal(f.classes.size,0);
  f.options.onFiles=async()=>{throw new Error('Upload failed')};f.emit('paste',transfer([file()]));
  await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(f.errors,['Upload failed']);f.cleanup();
});
test('Researcher validates the whole selection instead of ignoring extra files',()=>{
  assert.equal(proteinFileError([file()],false),'');assert.equal(proteinFileError([file('1CRN.PDB')],true),'');
  assert.ok(proteinFileError([file('paper.pdf')],false));assert.ok(proteinFileError([file(),file('other.fasta')],true));
});
test('the shared online 10 MB rule applies to any file transfer without limiting local files',()=>{
  const exact={name:'data.csv',size:ONLINE_UPLOAD_BYTES},large={...exact,size:ONLINE_UPLOAD_BYTES+1};
  assert.equal(uploadSizeError([exact],'online','en'),'');assert.match(uploadSizeError([large],'online','en'),/10 MB/);
  assert.equal(uploadSizeError([large],'local','en'),'');
});
