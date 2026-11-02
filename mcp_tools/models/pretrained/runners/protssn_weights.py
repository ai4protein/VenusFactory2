"""Resolve ProtSSN checkpoints without shell commands or CWD-dependent paths."""
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def model_directory(configured=None, ensemble=True, neighbors=10):
    required=[f'protssn_k{k}_h{h}.pt' for k in ([10,20,30] if ensemble else [neighbors])
              for h in ([512,768,1280] if ensemble else [512])]
    cache=Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))/'hub')
    folder=Path(configured) if configured else cache/'models--tyang816--ProtSSN/model'
    if all((folder/name).is_file() for name in required):return str(folder)
    if configured:raise FileNotFoundError('Configured ProtSSN checkpoint set is incomplete')
    from huggingface_hub import hf_hub_download
    archive=hf_hub_download('tyang816/ProtSSN','ProtSSN.zip',token=False,
                           local_files_only=os.environ.get('HF_HUB_OFFLINE','').lower() in {'1','true','yes'})
    folder.mkdir(parents=True,exist_ok=True)
    # Extract only expected weight files to fixed names, never archive-supplied paths.
    with zipfile.ZipFile(archive) as z:
        for name in required:
            matches=[entry for entry in z.infolist() if not entry.is_dir() and Path(entry.filename).name==name]
            if len(matches)!=1:raise ValueError('ProtSSN archive has missing or ambiguous weights')
            with tempfile.NamedTemporaryFile(dir=folder,delete=False) as out:
                temp=Path(out.name)
                try:
                    with z.open(matches[0]) as source:shutil.copyfileobj(source,out,1024*1024)
                except BaseException:temp.unlink(missing_ok=True);raise
            os.replace(temp,folder/name)
    return str(folder)
