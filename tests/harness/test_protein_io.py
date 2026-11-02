import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ['mcp_tools.tools.protein_io', 'mcp_tools.tools.protein_io'])
def test_protein_io_import_does_not_load_ui_or_models(module):
    root = Path(__file__).resolve().parents[2]
    code = (
        "import importlib,sys; importlib.import_module("
        + repr(module)
        + "); assert not {'gradio','torch','transformers','Bio'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_first_fasta_retains_descriptive_header_and_single_file(tmp_path, monkeypatch):
    from mcp_tools.tools.protein_io import extract_first_sequence_from_fasta_file

    monkeypatch.setenv("TEMP_OUTPUTS_DIR", str(tmp_path / "outputs"))
    source = tmp_path / "two.fasta"
    source.write_text(">sample1 protein description\nACDEFG\nHIK\n>sample2\nYYYY\n")
    selected = Path(extract_first_sequence_from_fasta_file(str(source)))
    assert selected != source and selected.is_file()
    assert selected.read_text() == ">sample1 protein description\nACDEFGHIK"
    assert extract_first_sequence_from_fasta_file(str(selected)) == str(selected)
    assert ">sample2" in source.read_text()


def test_first_pdb_chain_is_selected_and_renamed(tmp_path, monkeypatch):
    from mcp_tools.tools.protein_io import extract_first_chain_from_pdb_file

    monkeypatch.setenv("TEMP_OUTPUTS_DIR", str(tmp_path / "outputs"))
    source = tmp_path / "two.pdb"
    source.write_text(
        "HEADER    FIXTURE\nATOM      1  CA  ALA B   1      10.000  10.000  10.000  1.00 20.00           C  \nATOM      2  CA  GLY C   1      20.000  20.000  20.000  1.00 20.00           C  \nEND\n"
    )
    selected = Path(extract_first_chain_from_pdb_file(str(source)))
    atoms = [line for line in selected.read_text().splitlines() if line.startswith("ATOM")]
    assert len(atoms) == 1 and atoms[0][21] == "A" and atoms[0][17:20] == "ALA"
    assert selected != source and "GLY C" in source.read_text()
