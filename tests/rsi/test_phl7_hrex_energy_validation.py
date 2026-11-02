"""Energy and coordinate corruption must fail fixed-frame PHL7 validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.run_phl7_hrex_energy_validation import (
    SCALED_TERMS,
    UNCHANGED_TERMS,
    build_index,
    check_energies,
    energies,
    mark_atoms,
)

TOP = """[ atomtypes ]
C 6 12 0 A .3 .2
[ moleculetype ]
Protein_chain_A 3
[ atoms ]
1 C_ 1 ALA CA 1 0.2 12 ; preserve comment
[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.834 16
[ molecules ]
Protein_chain_A 1
SOL 1
"""


def gro():
    return (
        "title\n2\n"
        + "".join(
            f"{i:5d}{res:<5s}{name:>5s}{i:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
            for i, res, name in [(1, "ALA", "CA"), (2, "SOL", "OW")]
        )
        + "1 1 1\n"
    )


def fixture_energies():
    original = dict.fromkeys(SCALED_TERMS, -100.0) | dict.fromkeys(UNCHANGED_TERMS, 20.0)
    original["Potential"] = sum(original.values())
    half = {k: v * 0.5 if k in SCALED_TERMS else v for k, v in original.items()}
    half["Potential"] = sum(v for k, v in half.items() if k != "Potential")
    return original, dict(original), half


def test_marking_changes_only_atom_type_suffixes():
    unmarked, n = mark_atoms(TOP, "unmark")
    assert n == 1 and unmarked == TOP.replace("1 C_", "1 C")
    all_hot, n = mark_atoms(unmarked, "all_hot")
    assert n == 2 and "1 C_ 1 ALA" in all_hot and "1 OW_ 1 SOL" in all_hot
    assert "C 6 12 0 A .3 .2" in all_hot and "0.2 12 ; preserve comment" in all_hot


def test_index_has_exact_molecule_and_coordinate_identity():
    index, counts = build_index(TOP, gro())
    assert counts == {"Protein_PET": 1, "Water_and_Ions": 1}
    assert index == "[ Protein_PET ]\n1\n[ Water_and_Ions ]\n2\n"


@pytest.mark.parametrize(
    "text", [gro().replace("ALA", "VAL"), gro().replace("   CA", "   CB"), gro().replace("\n2\n", "\n3\n")]
)
def test_coordinate_identity_or_count_corruption_fails(text):
    with pytest.raises(ValueError):
        build_index(TOP, text)


def test_exact_scaling_and_energy_decomposition():
    checks = check_energies(*fixture_energies())
    assert len(checks) == 2 * (len(SCALED_TERMS) + len(UNCHANGED_TERMS) + 1)
    assert all(x["absolute_error"] == 0 for x in checks)


@pytest.mark.parametrize("which,term", [(1, "Potential"), (2, "LJ (SR)"), (2, "Bond"), (2, "Coul. recip.")])
def test_wrong_energy_component_or_unit_scaling_is_rejected(which, term):
    values = fixture_energies()
    values[which][term] += 1
    with pytest.raises(ValueError):
        check_energies(*values)


def test_missing_and_nonfinite_energy_are_rejected():
    values = fixture_energies()
    del values[2]["Angle"]
    with pytest.raises(ValueError):
        check_energies(*values)
    values = fixture_energies()
    values[1]["Potential"] = float("nan")
    with pytest.raises(ValueError):
        check_energies(*values)


def test_only_one_finite_energy_frame_is_accepted(tmp_path):
    p = tmp_path / "e.xvg"
    p.write_text('@ s0 legend "Potential"\n0.0 -12.3\n')
    assert energies(p) == {"Potential": -12.3}
    p.write_text(p.read_text() + "1.0 -12.2\n")
    with pytest.raises(ValueError):
        energies(p)


def test_optional_mass_column_in_ion_atoms_is_supported():
    text = "[ atoms ]\n1 Cl 1 CL CL 1 -1.0\n"
    result, count = mark_atoms(text, "all_hot")
    assert result == "[ atoms ]\n1 Cl_ 1 CL CL 1 -1.0\n" and count == 1


def test_total_charge_includes_molecule_multiplicity():
    from scripts.rsi.cases.run_phl7_hrex_energy_validation import topology_charge

    assert topology_charge(TOP) == pytest.approx(0.2 - 0.834)
    assert topology_charge(TOP.replace("SOL 1\n", "SOL 2\n")) == pytest.approx(0.2 - 2 * 0.834)


def test_charge_precision_diagnostic_preserves_other_parameters():
    import math

    from scripts.rsi.cases.run_phl7_hrex_energy_validation import restore_scaled_charge_precision
    from scripts.rsi.cases.run_phl7_hrex_preparation import parse_topology

    all_hot, _ = mark_atoms(TOP, "all_hot")
    result = parse_topology(restore_scaled_charge_precision(all_hot, TOP, 0.5))
    original = parse_topology(all_hot)
    for key, rows in result.items():
        if key[1] == "atoms":
            for a, b in zip(rows, original[key], strict=True):
                assert a[:6] == b[:6] and a[7:] == b[7:]
                assert float(a[6]) == pytest.approx(float(b[6]) * math.sqrt(0.5), abs=1e-15)
        else:
            assert rows == original[key]


def test_untempered_explicit_pair_changes_only_coulomb_sr_and_potential_expectations():
    original, unit, half = fixture_energies()
    pair = -246.0
    half["Coulomb (SR)"] += 0.5 * pair
    half["Potential"] += 0.5 * pair
    with pytest.raises(ValueError):
        check_energies(original, unit, half)
    checks = check_energies(original, unit, half, untempered_pair_coulomb=pair)
    assert all(row["absolute_error"] == 0 for row in checks)


def test_explicit_pair_oracle_uses_distance_and_rejects_nonzero_lj():
    from scripts.rsi.cases.run_phl7_hrex_energy_validation import explicit_pair_coulomb

    top = """[ moleculetype ]
Protein_chain_A 3
[ atoms ]
1 C 1 ALA CA 1 .3 12
2 C 1 ALA CB 2 -.4 12
[ pairs_nb ]
1 2 1 .3 -.4 0 0
[ molecules ]
Protein_chain_A 1
"""
    coords = (
        "pair\n2\n"
        + "".join(
            f"{1:5d}{'ALA':<5s}{name:>5s}{i:5d}{x:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
            for i, name, x in [(1, "CA", 0.0), (2, "CB", 0.4)]
        )
        + "2 2 2\n"
    )
    result = explicit_pair_coulomb(top, coords)
    assert result["distance_nm"] == pytest.approx(0.4)
    assert result["energy_kj_mol"] == pytest.approx(-41.68063729331459)
    assert result["both_outside_published_hot_region"] is True
    with pytest.raises(ValueError, match="nonzero LJ"):
        explicit_pair_coulomb(top.replace(".3 -.4 0 0", ".3 -.4 .2 .1"), coords)
