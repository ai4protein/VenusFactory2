"""Run in the isolated ADMET environment with its real weights."""
import importlib.util
from pathlib import Path
import json
import sys
import tempfile

spec=importlib.util.spec_from_file_location('admet_runner',Path(__file__).resolve().parents[1]/'mcp_tools/models/pretrained/admet/runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def main():
    from rdkit import Chem
    smiles=['CC(=O)Oc1ccccc1C(=O)O','invalid molecule','CC(=O)Oc1ccccc1C(=O)O','Cn1c(=O)c2c(ncn2C)n(C)c1=O']
    result=runner.predict({'kind':'table','rows':[{'id':str(i),'smiles':s} for i,s in enumerate(smiles)]},{'smiles_column':'smiles','include_physchem':True},32)
    assert len(result['rows'])==4 and [r['id'] for r in result['rows']]==['0','1','2','3']
    assert result['rows'][1]['admet_status']=='invalid' and 'hERG' not in result['rows'][1]
    assert result['rows'][0]['hERG']==result['rows'][2]['hERG']
    assert all(0<=r['hERG']<=1 for r in result['rows'] if r['admet_status']=='predicted')
    assert not any('DrugBank' in c for c in result['rows'][0])
    assert {'Absorption','Distribution','Metabolism','Excretion','Toxicity','Physicochemical'} <= {e['category'] for e in result['metadata']['endpoints']}
    mol=Chem.MolFromSmiles('CCO');block=Chem.MolToMolBlock(mol)
    rows=runner.input_rows({'kind':'ligand','name':'ethanol.sdf','content':block+'\n$$$$\n'+block+'\n$$$$\n'},'smiles',32)
    assert len(rows)==2 and rows[0]['smiles']==rows[1]['smiles']=='CCO'
    try:runner.input_rows({'kind':'ligand','name':'many.sdf','content':(block+'\n$$$$\n')*33},'smiles',32)
    except ValueError:pass
    else:raise AssertionError('SDF limit not enforced')
    try:runner.predict({'kind':'table','rows':[{'smiles':'not smiles'}]},{'smiles_column':'smiles','include_physchem':True},32)
    except ValueError as exc:assert 'No valid' in str(exc)
    else:raise AssertionError('invalid input accepted')
    path=Path(sys.argv[1]);path.write_text(json.dumps(result,indent=2));print('Runtime checks passed; endpoints',len(result['metadata']['endpoints']))

if __name__=='__main__':main()
