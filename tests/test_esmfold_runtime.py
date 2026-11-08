import json
import sys
from pathlib import Path
import pytest
from mcp_tools.models.pretrained.runners import esmfold

def config(**extra):
    return {"esmfold": {"python": sys.executable, **extra}}

def test_disabled_runtime_never_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(esmfold, "execute", lambda _: pytest.fail("disabled runtime launched"))
    with pytest.raises(ValueError, match="disabled"):
        esmfold.run("MKT", tmp_path, settings=config(enabled=False))

def test_input_validation_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(esmfold, "execute", lambda _: pytest.fail("invalid input launched"))
    for sequence in ["", "MKT;ls", "A"*6]:
        with pytest.raises(ValueError):
            esmfold.run(sequence, tmp_path, settings=config(max_sequence_length=5))

def test_isolated_runtime_preserves_custom_paths(tmp_path, monkeypatch):
    output=tmp_path/"results with spaces"/"fold.pdb"
    weights=tmp_path/"weights";weights.mkdir()
    def execute(command):
        request=json.loads(Path(command.argv[-1]).read_text())
        assert command.argv[:3]==[sys.executable, "-m", esmfold.__name__]
        assert request["sequence"]=="MKT"
        assert request["weights"]==str(weights)
        assert command.env["PYTHON_DOTENV_DISABLED"]=="1"
        Path(request["output_file"]).write_text("ATOM  fixture\nEND\n")
        Path(request["result_file"]).write_text(json.dumps({"plddt":80.0}))
    monkeypatch.setattr(esmfold, "execute", execute)
    path, info=esmfold.run(" m k t ",tmp_path,output,settings=config(weights=str(weights)))
    assert path==str(output) and info["plddt"]==80.0

def test_exit_success_without_structure_is_not_success(tmp_path, monkeypatch):
    def execute(command):
        request=json.loads(Path(command.argv[-1]).read_text())
        Path(request["result_file"]).write_text("{}")
    monkeypatch.setattr(esmfold, "execute", execute)
    with pytest.raises(RuntimeError,match="no structure"):
        esmfold.run("MKT",tmp_path,settings=config())



def test_pdb_confidence_uses_standard_zero_to_hundred_scale():
    import torch
    import numpy as np
    from types import SimpleNamespace
    from mcp_tools.models.pretrained.structure.esmfold import _convert_outputs_to_pdb
    outputs={"positions":torch.zeros(1,1,2,14,3),"atom37_atom_exists":torch.ones(1,2,37),
             "aatype":torch.zeros(1,2,dtype=torch.long),"residue_index":torch.arange(2)[None],
             "plddt":torch.full((1,2,37),0.85)}
    records=[]
    _convert_outputs_to_pdb(outputs,lambda protein:records.append(protein) or "PDB",
                            SimpleNamespace,lambda *args:torch.zeros(1,2,37,3))
    assert np.allclose(records[0].b_factors,85.0)
