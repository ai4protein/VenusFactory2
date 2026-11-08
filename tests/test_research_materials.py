from research_source_review_fixture import with_source_review
import io
import json
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile
from backend.web_v2 import research_materials as materials

@pytest.mark.asyncio
async def test_owner_scoped_materials_extraction_snapshot_and_download(tmp_path):
    row=await materials.upload(tmp_path,'alice',UploadFile(filename='assay.csv',file=io.BytesIO(b'protein,solubility\nABC,0.8\n')))
    assert row['extraction']['format']=='csv'
    with pytest.raises(FileNotFoundError):materials.sources(tmp_path,'bob',[row['id']],'a'*32)
    sources=materials.sources(tmp_path,'alice',[row['id']],'a'*32)
    assert '[Row 2]' in sources[0]['text']
    materials.snapshot(tmp_path,'alice','a'*32,sources)
    class Service:
        def owned(self,owner,job):
            if owner!='alice':raise FileNotFoundError()
            return job,{'payload':{'material_sources':sources}}
        def folder(self,job):return tmp_path/job
    path,name=materials.download(Service(),'alice','a'*32,row['id'])
    assert path.read_bytes().startswith(b'protein') and name=='assay.csv'
    with pytest.raises(FileNotFoundError):materials.download(Service(),'bob','a'*32,row['id'])
    with pytest.raises(FileNotFoundError):materials.download(Service(),'alice','a'*32,'b'*32)

@pytest.mark.asyncio
async def test_online_actual_byte_limit_and_failed_extraction_cleanup(tmp_path,monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    with pytest.raises(HTTPException) as e:
        await materials.upload(tmp_path,'alice',UploadFile(filename='data.txt',file=io.BytesIO(b'a'*(10*1024**2+1))))
    assert e.value.status_code==413
    with pytest.raises(ValueError):await materials.upload(tmp_path,'alice',UploadFile(filename='bad.pdf',file=io.BytesIO(b'not a PDF')))
    assert not list(materials.owner_folder(tmp_path,'alice').glob('*/metadata.json'))

def test_pdf_and_spreadsheet_extraction(tmp_path):
    from pypdf import PdfWriter
    blank=tmp_path/'scan.pdf';w=PdfWriter();w.add_blank_page(width=100,height=100);w.write(blank)
    with pytest.raises(ValueError,match='OCR'):materials.extract(blank)
    from openpyxl import Workbook
    xlsx=tmp_path/'assays.xlsx';book=Workbook();book.active.title='Assays';book.active.append(['protein','value']);book.active.append(['ABC',0.8]);book.save(xlsx)
    result=materials.extract(xlsx)
    assert '[Sheet Assays]' in result['text'] and '0.8' in result['text']

@pytest.mark.asyncio
async def test_materials_reach_specialists_and_cited_report_without_entering_web_queries(monkeypatch):
    from backend.web_v2 import research_longform as module
    source={'id':'S1','title':'Assay.csv','url':'/api/report/jobs/'+'a'*32+'/materials/'+'b'*32,'text':'[Row 2] solubility condition pH 7.5 at 25 C; measurement 0.8','kind':'user_material','evidence_level':'user_supplied_material','extraction':{'truncated':False}}
    async def collect(payload,update,**kwargs):
        return {'sources':[source],'protein':{'name':'fixture'},'steps':[],'sections':[],'remote_services':[]}
    observed=[]
    @with_source_review
    async def complete(model,messages,*args):
        bundle=json.loads(messages[-1]['content'])
        if 'chapter' in bundle:
            observed.append(bundle)
            assert 'pH 7.5' in bundle['sources'][0]['text'] and bundle['sources'][0]['extraction']['truncated'] is False
        return ('Evidence [S1]. '+'analysis '*450,False)
    monkeypatch.setattr(module.research,'run_research',collect)
    monkeypatch.setattr(module.research,'complete',complete)
    result=await module.run_long_report({'model_id':'fixture','analyses':[]},lambda s:None)
    assert len(observed)==8 and result['minimum_met']
    assert source['url'] in result['report']

def test_all_twelve_materials_are_assigned_across_chapters():
    sources=[{'id':f'S{i}','kind':'user_material','text':f'file{i} assay'} for i in range(12)]
    read={s['id'] for i in range(8) for s in materials.chapter_materials(sources,i,'assay')}
    assert len(read)==12
