import pytest
from mcp_tools.models.pretrained.design_docking.service import validate, validate_inputs, collect
from mcp_tools.models.pretrained.design_docking.inputs import validate_ligand_design

def backbone(chain="C", number=1, occupancy=1.0):
    return "\n".join(f"ATOM  {i:5d} {atom:^4s} ALA {chain}{number:4d}    {1.:8.3f}{2.:8.3f}{3.:8.3f}{occupancy:6.2f}{20.:6.2f}           C"
                     for i, atom in enumerate(["N", "CA", "C", "O"], 1))

def params(**values):
    return validate("ligandmpnn", {"parameters": values})

def test_ligand_design_accepts_complete_protein_backbone():
    assert validate_ligand_design(backbone(), params(chains_to_design="C")) == 1

@pytest.mark.parametrize("values", [
    {"chains_to_design":"A"},
    {"fixed_residues":"C1"},
    {"redesigned_residues":"C99"},
    {"parse_these_chains_only":"A"},
])
def test_ligand_design_rejects_empty_or_wrong_masks(values):
    with pytest.raises(ValueError):
        validate_inputs(None, "owner", "ligandmpnn", {"parameters":values},
                        {"name":"input.pdb", "structure":backbone()})

def test_zero_occupancy_is_only_used_when_explicitly_requested():
    with pytest.raises(ValueError):
        validate_ligand_design(backbone(occupancy=0), params())
    assert validate_ligand_design(backbone(occupancy=0),
                                 params(parse_atoms_with_zero_occupancy=True)) == 1

def test_ligand_output_with_zero_design_residues_is_not_success(tmp_path):
    (tmp_path/"output.fa").write_text(">input, num_res=0, num_ligand_res=0\nACD\n>sample, seq_rec=nan\nACD\n")
    with pytest.raises(ValueError, match="did not design"):
        collect("ligandmpnn", tmp_path)


def test_download_header_hides_server_path_and_preserves_raw_evidence(tmp_path):
    from mcp_tools.models.pretrained.design_docking.service import normalize_ligand_fasta
    out = tmp_path / "out"; out.mkdir()
    private = tmp_path / "private"
    raw = ">native, num_res=3, model_path=/private/runtime/weights/model.pt" + chr(10) + "ACD" + chr(10)
    path = out / "result.fa"; path.write_text(raw)
    normalize_ligand_fasta(out, private)
    assert "/private/runtime" not in path.read_text()
    assert "model_checkpoint=model.pt" in path.read_text()
    assert (private / "result.fa").read_text() == raw
    assert path.read_text().splitlines()[-1] == "ACD"
