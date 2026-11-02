"""Executed by the dedicated ADMET-AI environment, never in a Web process."""
from pathlib import Path
import csv
import io
import json
import math
import sys
from importlib.metadata import version


def input_rows(data, column, limit):
    from rdkit import Chem
    if data['kind'] == 'table':
        return [dict(row) for row in data['rows']]
    content = data['content']
    if data['name'].lower().endswith('.sdf'):
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(content.encode()), sanitize=True, removeHs=True)
    else:
        supplier = [Chem.MolFromMolBlock(content, sanitize=True, removeHs=True)]
    rows = []
    for i, mol in enumerate(supplier):
        if i >= limit:
            raise ValueError(f'Use at most {limit} molecules per prediction')
        rows.append({'id':str(i+1),'name':mol.GetProp('_Name') if mol is not None and mol.HasProp('_Name') else '',
                     column:Chem.MolToSmiles(mol) if mol is not None else ''})
    return rows


def predict(data, parameters, limit, model_factory=None):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    column = parameters['smiles_column']
    rows = input_rows(data, column, limit)
    if not 1 <= len(rows) <= limit: raise ValueError(f'Use 1–{limit} molecules per prediction')
    if any(str(key).startswith('admet_') for row in rows for key in row):
        raise ValueError('Input columns beginning with admet_ are reserved for prediction results')
    canonical = []
    for i, row in enumerate(rows):
        row['admet_row'] = i+1
        raw = row.get(column, '')
        mol = Chem.MolFromSmiles(raw.strip()) if isinstance(raw,str) and raw.strip() and len(raw)<=4096 else None
        if mol is None or mol.GetNumAtoms() == 0:
            row.update(admet_status='invalid', admet_error='Invalid SMILES or molecule record')
        elif mol.GetNumHeavyAtoms() > 256:
            row.update(admet_status='unsupported', admet_error='Use a small molecule with at most 256 heavy atoms')
        else:
            s=Chem.MolToSmiles(mol, isomericSmiles=True);canonical.append(s)
            row.update(admet_status='predicted', admet_smiles=s, admet_error='')
    if not canonical: raise ValueError('No valid small molecules were found')
    if model_factory is None:
        from admet_ai import ADMETModel
        model_factory = ADMETModel
    model = model_factory(include_physchem=parameters['include_physchem'], drugbank_path=None, num_workers=0)
    unique = list(dict.fromkeys(canonical))
    predicted = model.predict(smiles=unique)
    if len(predicted)!=len(unique) or list(predicted.index)!=unique:
        raise ValueError('Prediction rows do not match molecular inputs')
    if any(key in row for key in predicted.columns for row in rows):
        raise ValueError('Input columns overlap prediction properties; rename these columns before predicting')
    by_smiles = {}
    for s, values in predicted.iterrows():
        by_smiles[s] = {k:float(v) if math.isfinite(float(v)) else None for k,v in values.items()}
    for row in rows:
        if row['admet_status']=='predicted':row.update(by_smiles[row['admet_smiles']])
    from admet_ai.constants import DEFAULT_ADMET_PATH
    with Path(DEFAULT_ADMET_PATH).open() as f:
        metadata = [{k:r[k] for k in ['id','name','category','task_type','units','species','url']} for r in csv.DictReader(f) if r['id'] in predicted.columns]
    metadata_ids={m['id'] for m in metadata}
    for key in predicted.columns:
        if key not in metadata_ids:
            metadata.append({'id':key,'name':key,'category':'Other','task_type':'unknown','units':'','species':'','url':''})
    return {'kind':'table','rows':rows,'metadata':{'engine':'ADMET-AI','version':version('admet-ai'),'device':'cpu',
            'predicted':len(canonical),'invalid':len(rows)-len(canonical),'endpoints':metadata,
            'interpretation':'Classification values are positive-class probabilities, not confidence or a safety score. Regression units and assay species are endpoint-specific.',
            'reference':'https://github.com/swansonk14/admet_ai'}}


def main():
    request=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
    result=predict(request['data'],request['parameters'],request['limit'])
    (out/'result.json').write_text(json.dumps(result,allow_nan=False))
    (out/'endpoints.json').write_text(json.dumps(result['metadata'],indent=2))
    fields=list(dict.fromkeys(k for row in result['rows'] for k in row))
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result['rows'])


if __name__=='__main__':
    try:main()
    except ValueError as exc:
        (Path(sys.argv[2])/'error.json').write_text(json.dumps({'message':str(exc)}))
        raise
