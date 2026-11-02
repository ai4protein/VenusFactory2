"""Parameter-level checks reject corruption in a prepared HREX topology."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_phl7_hrex_preparation import molecular_atoms, parse_topology, verify_scaling

SOURCE = """*** published force-field attribution ***
[ defaults ]
1 2 yes 0.5 0.8333
[ atomtypes ]
C 6 12.0 0 A 0.34 0.36
C_ C 12.0 0 A 0.34 0.36
[ moleculetype ]
Protein 3
[ atoms ]
1 C_ 1 ALA CA 1 0.2 12.0
2 C 1 ALA CB 2 -0.2 12.0
[ bonds ]
1 2 1 0.2 1000
[ dihedrals ]
1 2 2 1 9 180 2.0 1
[ molecules ]
Protein 1
"""


def scaled():
    value = parse_topology(SOURCE)
    value[("GLOBAL", "atomtypes")][1][-1] = "0.09"
    value[("Protein", "atoms")][0][6] = "0.1"
    value[("Protein", "dihedrals")][0][6] = "0.5"
    return value


def test_charge_lj_dihedral_scaling_and_preamble_preservation():
    original = parse_topology(SOURCE)
    result = verify_scaling(original, scaled(), 0.25)
    assert result["molecular_atoms"] == molecular_atoms(original) == 2
    assert result["numeric_parameters_checked"] == 3 and result["max_relative_scaling_error"] == 0
    assert ("GLOBAL", "preamble") in original


@pytest.mark.parametrize(
    "key,column,value",
    [(("Protein", "atoms"), 6, "0.2"), (("Protein", "bonds"), 4, "500"), (("Protein", "dihedrals"), 6, "2.0")],
)
def test_wrong_charge_unscaled_connectivity_or_torsion_is_rejected(key, column, value):
    actual = scaled()
    actual[key][0][column] = value
    with pytest.raises(ValueError):
        verify_scaling(parse_topology(SOURCE), actual, 0.25)


def test_preprocessor_and_unrecognized_preamble_are_not_ignored():
    with pytest.raises(ValueError, match="preprocessor"):
        parse_topology('#include "missing.itp"\n' + SOURCE)
    with pytest.raises(ValueError, match="outside"):
        parse_topology("unrecognized data\n" + SOURCE)


def test_accidental_restart_does_not_overwrite_historical_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import run_phl7_hrex_preparation as module

    output = tmp_path / "completed"
    output.mkdir()
    report = output / "report.json"
    report.write_text('{"status":"completed"}')
    before = report.read_bytes()
    monkeypatch.setattr(module, "sha", lambda path: module.ARCHIVE_SHA256)
    with pytest.raises(FileExistsError):
        module.execute(SimpleNamespace(output=output, archive=tmp_path / "archive", plumed=tmp_path / "plumed"))
    assert report.read_bytes() == before
