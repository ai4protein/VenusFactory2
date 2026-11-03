"""Server-configured design/docking runtimes behind the platform job queue.

Request bodies contain scientific settings and owned dataset identifiers only.
Executable paths, weights, device allocation and output directories are server-owned.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT = Path(__file__).resolve().parents[4]


class Parameters(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class LigandParameters(Parameters):
    chains_to_design: str = Field('', title='Chains to design', max_length=100)
    fixed_residues: str = Field('', title='Fixed residues (A12 A13)', max_length=10000)
    redesigned_residues: str = Field('', title='Design only these residues (A12 A13)', max_length=10000)
    temperature: float = Field(.1, gt=0, le=10)
    batch_size: int = Field(1, ge=1, le=32)
    number_of_batches: int = Field(8, ge=1, le=512)
    seed: int = Field(0, ge=0, le=2147483647)
    omit_AA: str = Field('X', title='Exclude amino acids', pattern=r'^[ACDEFGHIKLMNPQRSTVWYX]*$')
    bias_AA: str = Field('', title='Amino acid bias (A:0.5,G:-0.5)', max_length=1000)
    bias_AA_per_residue_text: str = Field('', title='Residue-specific bias (JSON)', max_length=100000)
    omit_AA_per_residue_text: str = Field('', title='Residue-specific exclusions (JSON)', max_length=100000)
    symmetry_residues: str = Field('', title='Linked residues (A1,B1|A2,B2)', max_length=10000)
    symmetry_weights: str = Field('', title='Symmetry weights (0.5,0.5|0.5,0.5)', max_length=10000)
    homo_oligomer: bool = False
    ligand_mpnn_use_atom_context: bool = True
    ligand_mpnn_use_side_chain_context: bool = False
    ligand_mpnn_cutoff_for_score: float = Field(8, gt=0, le=100)
    parse_these_chains_only: str = Field('', max_length=100)
    parse_atoms_with_zero_occupancy: bool = False
    pack_side_chains: bool = False
    number_of_packs_per_design: int = Field(1, ge=1, le=16)
    sc_num_denoising_steps: int = Field(3, ge=1, le=20)
    sc_num_samples: int = Field(16, ge=1, le=128)
    repack_everything: bool = False
    pack_with_ligand_context: bool = True

    @model_validator(mode='after')
    def check(self):
        if self.batch_size*self.number_of_batches>512:raise ValueError('Use at most 512 designs')
        if self.fixed_residues and self.redesigned_residues:raise ValueError('Choose fixed residues or residues to redesign')
        for value in (self.fixed_residues,self.redesigned_residues):
            if value and not re.fullmatch(r'[A-Za-z0-9\s-]+',value):raise ValueError('Use chain and residue labels, such as A12 A13')
        for value in (self.bias_AA_per_residue_text,self.omit_AA_per_residue_text):
            if value and not isinstance(json.loads(value),dict):raise ValueError('Residue constraints must be JSON objects')
        return self


class AntibodyParameters(Parameters):
    chains_to_design: str = Field('H', title='Antibody chains to design', pattern=r'^[A-Za-z0-9](?: [A-Za-z0-9])*$')
    design_positions: str = Field(..., title='Design positions per chain (e.g. 102 103 104)', min_length=1, max_length=10000)
    num_sequences: int = Field(8, ge=1, le=512)
    temperatures: list[float] = Field(default_factory=lambda:[.1], min_length=1, max_length=8)
    batch_size: int = Field(1, ge=1, le=32)
    backbone_noise: float = Field(0, ge=0, le=5)
    seed: int = Field(0, ge=0, le=2147483647)
    omit_aas: str = Field('X', pattern=r'^[ACDEFGHIKLMNPQRSTVWYX]*$')
    max_length: int = Field(2000, ge=1, le=20000)

    @model_validator(mode='after')
    def check(self):
        groups=self.design_positions.split(',')
        if len(groups)!=len(self.chains_to_design.split()) or any(not re.fullmatch(r'\s*[1-9]\d*(?:\s+[1-9]\d*)*\s*',g) for g in groups):
            raise ValueError('Enter positive residue positions for each chain; separate chains with commas')
        if self.num_sequences%self.batch_size or self.num_sequences*len(self.temperatures)>512:raise ValueError('Batch size must divide sequences; use at most 512 designs')
        if any(t<=0 or t>10 for t in self.temperatures):raise ValueError('Temperatures must be between 0 and 10')
        return self


class BindCraftParameters(Parameters):
    chains: str = Field('A', title='Target chains', pattern=r'^[A-Za-z0-9](?:,[A-Za-z0-9])*$')
    target_hotspot_residues: str = Field('', title='Target hotspots (A56,A58)', max_length=10000)
    min_length: int = Field(65, ge=10, le=400)
    max_length: int = Field(150, ge=10, le=400)
    number_of_final_designs: int = Field(2, ge=1, le=100)

    @model_validator(mode='after')
    def check(self):
        if self.min_length>self.max_length:raise ValueError('Minimum length must not exceed maximum length')
        if self.target_hotspot_residues and not re.fullmatch(r'[A-Za-z0-9,\s-]+',self.target_hotspot_residues):raise ValueError('Invalid hotspot residues')
        return self


class BoltzGenParameters(Parameters):
    target_chain: str = Field('A', pattern=r'^[A-Za-z0-9]+$', max_length=8)
    binder_chain: str = Field('B', pattern=r'^[A-Za-z0-9]+$', max_length=8)
    min_length: int = Field(80, ge=5, le=400)
    max_length: int = Field(140, ge=5, le=400)
    binding_residues: str = Field('', title='Target binding positions (5..7,13)', pattern=r'^[0-9.,]*$')
    protocol: Literal['protein-anything','peptide-anything'] = 'protein-anything'
    num_designs: int = Field(10, ge=1, le=1000)
    budget: int = Field(2, ge=1, le=100)
    diffusion_batch_size: int = Field(1, ge=1, le=16)

    @model_validator(mode='after')
    def check(self):
        if self.target_chain==self.binder_chain:raise ValueError('Target and binder chain IDs must differ')
        if self.min_length>self.max_length:raise ValueError('Minimum length must not exceed maximum length')
        if self.budget>self.num_designs:raise ValueError('Final design budget must not exceed generated designs')
        return self


class VinaParameters(Parameters):
    center: list[float] = Field(..., min_length=3, max_length=3, title='Box centre X, Y, Z (Å)')
    size: list[float] = Field(default_factory=lambda:[20,20,20], min_length=3, max_length=3, title='Box size X, Y, Z (Å)')
    exhaustiveness: int = Field(8, ge=1, le=128)
    num_modes: int = Field(9, ge=1, le=50)
    energy_range: float = Field(3, gt=0, le=100)
    seed: int = Field(42, ge=0, le=2147483647)
    scoring: Literal['vina','vinardo'] = 'vina'

    @model_validator(mode='after')
    def check(self):
        if any(s<=0 or s>100 for s in self.size):raise ValueError('Box dimensions must be between 0 and 100 Å')
        if self.size[0]*self.size[1]*self.size[2]>125000:raise ValueError('Use a docking box smaller than 125,000 Å³')
        return self


class DiffDockParameters(Parameters):
    ligand_smiles: str = Field('', title='Ligand SMILES (or upload a ligand)', max_length=10000)
    samples_per_complex: int = Field(10, ge=1, le=100)
    inference_steps: int = Field(20, ge=1, le=100)
    batch_size: int = Field(5, ge=1, le=32)
    no_final_step_noise: bool = True
    save_visualisation: bool = False


from mcp_tools.models.pretrained.admet.service import ADMETParameters

MODELS = {
    'admet_ai':('ADMET-AI','admet','Small-molecule absorption, distribution, metabolism, excretion and toxicity','https://github.com/swansonk14/admet_ai',ADMETParameters),
    'ligandmpnn':('LigandMPNN','design','Ligand-conditioned sequence design','https://github.com/dauparas/LigandMPNN',LigandParameters),
    'antibmpnn':('AntiBMPNN','design','Antibody sequence design','https://github.com/zeysun/AntiBMPNN',AntibodyParameters),
    'bindcraft':('BindCraft','design','Protein binder design','https://github.com/martinpacesa/BindCraft',BindCraftParameters),
    'boltzgen':('BoltzGen','design','Generative protein and peptide binder design','https://github.com/HannesStark/boltzgen',BoltzGenParameters),
    'vina':('AutoDock Vina','docking','Protein–ligand docking','https://github.com/ccsb-scripps/AutoDock-Vina',VinaParameters),
    'diffdock':('DiffDock','docking','Diffusion-based protein–ligand docking','https://github.com/gcorso/DiffDock',DiffDockParameters),
}


def settings():
    from mcp_tools.models.profiles import load_runtime_settings
    return load_runtime_settings(root=PROJECT)


def runtime(key):
    value=settings().get(key,{})
    if not value.get('enabled'):raise ValueError('Model runtime is not configured')
    required={'ligandmpnn':['python','root','weights'], 'antibmpnn':['python','root','weights'],
              'bindcraft':['python','root','filters','advanced','installation'], 'boltzgen':['executable','cache','installation'],
              'admet_ai':['python','installation'], 'vina':['executable'], 'diffdock':['python','root','model_dir','confidence_model_dir','config','installation']}[key]
    for name in required:
        if not value.get(name) or not Path(value[name]).is_absolute() or not Path(value[name]).exists():
            raise ValueError('Model runtime is not configured: '+name)
    entry={'ligandmpnn':'run.py','antibmpnn':'Running_AntiBMPNN_run.py','bindcraft':'bindcraft.py','diffdock':'inference.py'}.get(key)
    if entry and not (Path(value['root'])/entry).is_file():raise ValueError('Model entry point is missing')
    if key=='antibmpnn' and not (Path(value['weights'])/'antibmpnn_000.pt').is_file():raise ValueError('AntiBMPNN weights are missing')
    if key=='admet_ai':
        receipt=json.loads(Path(value['installation']).read_text())
        if not receipt.get('verified') or receipt.get('version')!='2.0.1':raise ValueError('ADMET-AI runtime is not verified')
    if key in {'bindcraft','boltzgen','diffdock'}:
        receipt=json.loads(Path(value['installation']).read_text())
        if not receipt.get('verified') or receipt.get('runtime')!=key:raise ValueError('Model runtime has not passed inference verification')
    if key=='diffdock':
        for folder,checkpoint in [('model_dir',value.get('ckpt','best_ema_inference_epoch_model.pt')),('confidence_model_dir',value.get('confidence_ckpt','best_model.pt'))]:
            if not (Path(value[folder])/checkpoint).is_file():raise ValueError('DiffDock weights are missing')
    return value


def availability(key):
    from mcp_tools.tools.capability_policy import model_disabled
    if model_disabled({'id':'external:'+key,'adapter':'external:'+key}):return False,'admin_disabled'
    from mcp_tools.tools.runtime_tool_policy import is_online_mode
    if is_online_mode() and key in {'bindcraft','boltzgen'}:return False,'local_only'
    try:runtime(key)
    except (ValueError,OSError,json.JSONDecodeError):return False,'not_configured'
    return True,None


def schema(key):
    cls=MODELS[key][4];result=cls.model_json_schema()
    for name,field in cls.model_fields.items():
        if not field.is_required():result['properties'][name]['default']=field.get_default(call_default_factory=True)
    return result


def cards():
    result=[]
    for key,(name,category,description,url,_) in MODELS.items():
        ready,reason=availability(key)
        c={'id':'external:'+key,'name':name,'category':category,'description':description,'source':'platform',
           'visibility':'public','adapter':'external:'+key,'workflow':True,'input_kind':'structure',
           'parameter_schema':schema(key),'available':ready,'disabled_reason':reason,'reference_url':url,
           'input_accept':'.pdbqt' if key=='vina' else '.pdb'}
        if key=='admet_ai':
            c.update(input_kind='table',accepted_input_kinds=['table','ligand'],input_accept='.csv,.tsv,.json,.sdf,.mol',input_format='SMILES · CSV · SDF',version='2.0.1')
        if key in {'vina','diffdock'}:
            c['extra_inputs']=[{'key':'ligand_dataset_id','label':'Ligand','accept':'.pdbqt' if key=='vina' else '.sdf,.mol', 'required':key=='vina'}]
        result.append(c)
    return result


def validate(key,config):
    raw=(config or {}).get('parameters',{})
    return MODELS[key][4].model_validate(raw).model_dump()


def validate_inputs(platform,owner,key,config,data=None):
    config=config or {};p=validate(key,config)
    if key=='admet_ai':
        from mcp_tools.models.pretrained.admet.service import validate_input
        if data:validate_input(data,p)
        return None
    if data:
        ext=Path(data.get('name','')).suffix.lower()
        if ext!=('.pdbqt' if key=='vina' else '.pdb'):raise ValueError('Choose a '+('receptor PDBQT' if key=='vina' else 'PDB')+' file')
    ligand=None
    if config.get('ligand_dataset_id'):
        ligand=platform.owned('datasets',config['ligand_dataset_id'],owner)
        if Path(ligand['name']).suffix.lower() not in ({'.pdbqt'} if key=='vina' else {'.sdf','.mol'}):raise ValueError('Choose a compatible ligand file')
    if key=='vina' and not ligand:raise ValueError('Upload a ligand PDBQT file')
    if key=='diffdock':
        if not ligand and not p['ligand_smiles'].strip():raise ValueError('Upload a ligand or enter SMILES')
        if p['ligand_smiles'] and (any(ch.isspace() for ch in p['ligand_smiles']) or p['ligand_smiles'].startswith(('/','\\')) or '://' in p['ligand_smiles']):raise ValueError('Enter a SMILES string, not a file path or URL')
    return ligand


def write_json(folder,name,data):
    path=folder/name;path.write_text(json.dumps(data));return str(path)


def commands(key,p,cfg,source,ligand,work,out):
    """Construct argv arrays using the upstream programs' documented interfaces."""
    root=Path(cfg.get('root',work));py=cfg.get('python')
    if key=='vina':
        cmd=[cfg['executable'],'--receptor',str(source),'--ligand',str(ligand),'--out',str(out/'poses.pdbqt')]
        for axis,value in zip('xyz',p['center']):cmd+=['--center_'+axis,str(value)]
        for axis,value in zip('xyz',p['size']):cmd+=['--size_'+axis,str(value)]
        for k in ['exhaustiveness','num_modes','energy_range','seed','scoring']:cmd+=['--'+k,str(p[k])]
        return [cmd+['--cpu',str(max(1,min(int(cfg.get('cpus',2)),8)))]]
    if key=='ligandmpnn':
        cmd=[py,str(root/'run.py'),'--model_type','ligand_mpnn','--checkpoint_ligand_mpnn',cfg['weights'],'--pdb_path',str(source),'--out_folder',str(out)]
        for k,v in p.items():
            if v=='':continue
            if k.endswith('_text'):
                k=k[:-5];v=write_json(work,k+'.json',json.loads(v))
            cmd+=['--'+k,str(int(v) if isinstance(v,bool) else v)]
        if p['pack_side_chains']:
            if not cfg.get('packing_weights') or not Path(cfg['packing_weights']).is_file():raise ValueError('Side-chain packing weights are not configured')
            cmd+=['--checkpoint_path_sc',cfg['packing_weights']]
        return [cmd]
    if key=='antibmpnn':
        parsed=str(work/'parsed.jsonl');assigned=str(work/'chains.jsonl');fixed=str(work/'positions.jsonl')
        pre=[ [py,str(root/'helper_scripts/parse_multiple_chains.py'),'--input_path',str(source.parent)+'/', '--output_path',parsed],
              [py,str(root/'helper_scripts/assign_fixed_chains.py'),'--input_path',parsed,'--output_path',assigned,'--chain_list',p['chains_to_design']],
              [py,str(root/'helper_scripts/make_fixed_positions_dict.py'),'--input_path',parsed,'--output_path',fixed,'--chain_list',p['chains_to_design'],'--position_list',p['design_positions'],'--specify_non_fixed'] ]
        cmd=[py,str(root/'Running_AntiBMPNN_run.py'),'--jsonl_path',parsed,'--chain_id_jsonl',assigned,'--fixed_positions_jsonl',fixed,'--out_folder',str(out),
             '--path_to_model_weights',cfg['weights'].rstrip('/')+'/','--model_name','antibmpnn_000','--num_seq_per_target',str(p['num_sequences']),
             '--sampling_temp',' '.join(map(str,p['temperatures'])),'--omit_AAs',p['omit_aas']]
        for k in ['batch_size','backbone_noise','seed','max_length']:cmd+=['--'+k,str(p[k])]
        return pre+[cmd]
    if key=='bindcraft':
        target={'design_path':str(out)+'/', 'binder_name':'binder','starting_pdb':str(source), 'chains':p['chains'],
                'target_hotspot_residues':p['target_hotspot_residues'] or None,'lengths':[p['min_length'],p['max_length']], 'number_of_final_designs':p['number_of_final_designs']}
        return [[py,str(root/'bindcraft.py'),'--settings',write_json(work,'target.json',target),'--filters',cfg['filters'],'--advanced',cfg['advanced']]]
    if key=='boltzgen':
        target={'path':str(source),'include':[{'chain':{'id':p['target_chain']}}]}
        if p['binding_residues']:target['binding_types']=[{'chain':{'id':p['target_chain'],'binding':p['binding_residues']}}]
        spec={'entities':[{'protein':{'id':p['binder_chain'],'sequence':str(p['min_length'])+'..'+str(p['max_length'])}},{'file':target}]}
        spec_path=write_json(work,'design.yaml',spec) # JSON is a strict subset of YAML.
        cmd=[cfg['executable'],'run',spec_path,'--output',str(out),'--protocol',p['protocol'],'--devices','1','--num_workers','2']
        if cfg.get('cache'):cmd+=['--cache',cfg['cache']]
        for k in ['num_designs','budget','diffusion_batch_size']:cmd+=['--'+k,str(p[k])]
        return [cmd]
    if key=='diffdock':
        import yaml
        pre=[]
        if ligand is None:
            ligand=work/'ligand.sdf'
            prepare="from rdkit import Chem; from rdkit.Chem import AllChem; import sys; m=Chem.MolFromSmiles(sys.argv[1]); assert m is not None, 'Invalid ligand SMILES'; m=Chem.AddHs(m); status=AllChem.EmbedMolecule(m,randomSeed=42); assert status==0, 'Could not embed ligand'; w=Chem.SDWriter(sys.argv[2]); w.write(m); w.close()"
            pre=[[py,'-c',prepare,p['ligand_smiles'],str(ligand)]]
        defaults=yaml.safe_load(Path(cfg['config']).read_text()) or {}
        if not isinstance(defaults,dict):raise ValueError('DiffDock runtime config must be an object')
        defaults.update({k:v for k,v in p.items() if k!='ligand_smiles'})
        defaults['actual_steps']=min(int(defaults.get('actual_steps') or p['inference_steps']),p['inference_steps'])
        defaults.update(protein_path=str(source),protein_ligand_csv=None,ligand_description=str(ligand),
                        complex_name='complex',out_dir=str(out),model_dir=cfg['model_dir'],confidence_model_dir=cfg['confidence_model_dir'],
                        ckpt=cfg.get('ckpt','best_ema_inference_epoch_model.pt'),confidence_ckpt=cfg.get('confidence_ckpt','best_model.pt'),
                        gnina_minimize=False,gnina_full_dock=False)
        # Upstream YAML overrides CLI values; put the final request in the config.
        return pre+[[py,str(root/'inference.py'),'--config',write_json(work,'inference.yaml',defaults)]]
    raise ValueError('Unknown external model')


def collect(key,out):
    rows=[]
    if key in {'ligandmpnn','antibmpnn'}:
        for f in sorted(out.rglob('*.fa'))+sorted(out.rglob('*.fasta')):
            current=None
            for line in f.read_text().splitlines():
                if line.startswith('>'):
                    current={'description':line[1:],'sequence':''};rows.append(current)
                elif current is not None:current['sequence']+=line.strip()
    elif key=='vina':
        pose=0
        for line in (out/'poses.pdbqt').read_text().splitlines():
            if line.startswith('MODEL'):pose=int(line.split()[1])
            if line.startswith('REMARK VINA RESULT:'):
                values=line.split(':',1)[1].split();rows.append({'pose':pose,'affinity_kcal_mol':float(values[0]),'rmsd_lb':float(values[1]),'rmsd_ub':float(values[2])})
    elif key=='diffdock':
        for f in sorted(out.rglob('rank*.sdf')):
            row={'pose':f.name};match=re.search(r'confidence([-+0-9.e]+)',f.stem)
            if match:
                try:row['confidence']=float(match.group(1))
                except ValueError:pass
            rows.append(row)
    else:
        preferred=list(out.rglob('*final*.csv'))+list(out.rglob('*rank*.csv'))+list(out.rglob('*accepted*.csv'))
        for f in (preferred or sorted(out.rglob('*.csv')))[:10]:
            with f.open() as stream:rows.extend(list(csv.DictReader(stream))[:1000])
        if not rows:
            structures=[f for f in sorted(out.rglob('*')) if f.suffix.lower() in {'.pdb','.cif'}]
            if key=='boltzgen':
                # This root file visualizes input constraints, not a generated model.
                structures=[f for f in structures if f!=out/'design.cif']
            rows=[{'structure':str(f.relative_to(out))} for f in structures][:1000]
            if key=='bindcraft':
                for row in rows:
                    row['accepted']=row['structure'].startswith('Accepted/')
                    row['outcome']='accepted' if row['accepted'] else 'candidate_not_accepted'
    if not rows:raise ValueError('The runtime finished without model output; see the job log')
    return rows


def execute_command(cmd,*,cwd,env,log,timeout):
    import psutil
    proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    try:
        code=proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:children=psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:children=[]
        for child in children:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        proc.terminate()
        _,alive=psutil.wait_procs(children,timeout=3)
        for child in alive:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise ValueError('Model runtime exceeded its time budget') from None
    if code:raise ValueError('Model runtime exited with code '+str(code)+'; see the job log')


def run(platform,owner,key,data,config):
    available,reason=availability(key)
    if not available:
        if reason=='admin_disabled':
            from mcp_tools.tools.capability_policy import CapabilityDisabled, unavailable_message
            raise CapabilityDisabled(unavailable_message())
        raise ValueError('Local runtime required' if reason=='local_only' else 'Model runtime is not configured')
    if key=='admet_ai':
        from mcp_tools.models.pretrained.admet.service import run_prediction
        return run_prediction(data,validate(key,config))
    cfg=runtime(key);p=validate(key,config);ligand_data=validate_inputs(platform,owner,key,config,data)
    work=Path(os.environ['TEMP_OUTPUTS_DIR'])/('external_'+key+'_'+uuid.uuid4().hex)
    inputs=work/'inputs';out=work/'outputs';inputs.mkdir(parents=True,mode=0o700);out.mkdir(mode=0o700)
    source=inputs/('receptor'+Path(data['name']).suffix.lower());source.write_text(data['structure'])
    ligand=None
    if ligand_data:
        ligand=work/('ligand'+Path(ligand_data['name']).suffix.lower());ligand.write_text(ligand_data.get('content') or ligand_data['structure'])
    argv=commands(key,p,cfg,source,ligand,work,out)
    env=dict(os.environ)
    if cfg.get('cuda_visible_devices') is not None:env['CUDA_VISIBLE_DEVICES']=str(cfg['cuda_visible_devices'])
    # Runtimes never inherit provider secrets.
    env={k:v for k,v in env.items() if not any(t in k.upper() for t in ['API_KEY','TOKEN','SECRET','PASSWORD'])}
    env.update(OMP_NUM_THREADS=str(max(1,min(int(cfg.get('cpus',2)),8))),OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if key=='diffdock' and cfg.get('cache'):env['TORCH_HOME']=cfg['cache']
    if key=='boltzgen' and cfg.get('cache'):env['HF_HUB_OFFLINE']='1'
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with (work/'runtime.log').open('ab') as log:
        for cmd in argv:
            execute_command(cmd,cwd=cfg.get('root',work),env=env,log=log,timeout=max(1,min(int(cfg.get('timeout_seconds',3600)),86400)))
    rows=collect(key,out)
    artifacts=[str(f) for f in out.rglob('*') if f.is_file() and not f.is_symlink() and f.suffix.lower() in {'.fa','.fasta','.pdb','.cif','.pdbqt','.sdf','.csv','.json','.png','.pdf','.html'}]
    return {'kind':'table','rows':rows,'artifacts':artifacts,'parameters':p,'runtime':key}
