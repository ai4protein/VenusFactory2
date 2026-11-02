import pytest

from agent.rsi.rfantibody_targets import (
    exact_ca_subset,
    pdb_residues,
    resolve_cif_hotspots,
    validate_target_input,
    write_observed_chain,
)
from agent.rsi.storage import sha256_file


def pdb(rows):
    lines = []
    serial = 0
    for chain, number, residue, offset in rows:
        for atom, dx in [("N", -1.0), ("CA", 0.0), ("C", 1.0)]:
            serial += 1
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{number:4d}    {offset + dx:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}           C\n"
            )
    return "".join(lines) + "END\n"


def test_whole_observed_chain_preserves_gaps_and_coordinates_without_filling(tmp_path):
    source = tmp_path / "raw.pdb"
    source.write_text(pdb([("A", 1, "ALA", 0), ("E", 492, "LEU", 10), ("E", 497, "PHE", 30)]))
    out = tmp_path / "target.pdb"
    report = write_observed_chain(source, out, "E", "B")
    assert list(pdb_residues(out)) == [("B", 492, ""), ("B", 497, "")]
    assert report["observed_residues"] == 2
    assert not report["publication_crop_equivalence_established"]
    assert all(r["ca_distance"] == 0 for r in report["coordinate_mapping"])
    with pytest.raises(ValueError, match="overwrite"):
        write_observed_chain(source, out, "E", "B")


def test_coordinate_only_similarity_cannot_admit_an_amino_acid_change():
    reference = {("A", 1, ""): {"aa": "A", "atoms": {"CA": (1.0, 2.0, 3.0)}}}
    candidate = {("T", 10, ""): {"aa": "V", "atoms": {"CA": (1.0, 2.0, 3.0)}}}
    with pytest.raises(ValueError, match="identity"):
        exact_ca_subset(reference, candidate)


def test_duplicate_coordinates_are_ambiguous_even_if_sequence_matches():
    reference = {
        ("A", 1, ""): {"aa": "A", "atoms": {"CA": (1.0, 2.0, 3.0)}},
        ("B", 1, ""): {"aa": "A", "atoms": {"CA": (1.0, 2.0, 3.0)}},
    }
    with pytest.raises(ValueError, match="ambiguous"):
        exact_ca_subset(reference, {("T", 10, ""): reference[("A", 1, "")]})


def test_cif_chain_namespace_is_explicit_not_silently_guessed(tmp_path):
    path = tmp_path / "raw.cif"
    path.write_text(
        "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\nATOM CA . LEU B E 174 492 ? 1.0 2.0 3.0\n#\n"
    )
    result = resolve_cif_hotspots(path, "B", [492], chain_namespace="label")
    assert result[0]["auth_chain"] == "E" and result[0]["auth_residue_number"] == 492
    assert result[0]["label_residue_number"] == 174
    with pytest.raises(ValueError, match="exactly one"):
        resolve_cif_hotspots(path, "B", [492], chain_namespace="auth")
    with pytest.raises(ValueError, match="namespace"):
        resolve_cif_hotspots(path, "B", [492], chain_namespace="auto")


def test_catalog_rejects_modified_coordinates_and_wrong_hotspot(tmp_path):
    path = tmp_path / "research_runs/rsi/inputs/target.pdb"
    path.parent.mkdir(parents=True)
    path.write_text(pdb([("B", 492, "LEU", 10)]))
    record = {
        "path": str(path.relative_to(tmp_path)),
        "sha256": sha256_file(path),
        "residues": 1,
        "hotspots": [{"chain": "B", "number": 492, "aa": "L"}],
    }
    assert validate_target_input(tmp_path, record)["hotspots"] == 1
    record["hotspots"][0]["number"] = 491
    with pytest.raises(ValueError, match="Hotspot"):
        validate_target_input(tmp_path, record)
    record["hotspots"][0]["number"] = 492
    path.write_text(pdb([("B", 492, "LEU", 11)]))
    with pytest.raises(ValueError, match="changed"):
        validate_target_input(tmp_path, record)


def test_incomplete_backbone_is_not_silently_repaired(tmp_path):
    source = tmp_path / "raw.pdb"
    source.write_text(
        "".join(line for line in pdb([("A", 1, "ALA", 0)]).splitlines(True) if line[12:16].strip() != "N")
    )
    with pytest.raises(ValueError, match="backbone"):
        write_observed_chain(source, tmp_path / "output.pdb", "A", "T")


def test_only_first_model_is_admitted(tmp_path):
    path = tmp_path / "models.pdb"
    path.write_text(pdb([("A", 1, "ALA", 0)]).replace("END\n", "ENDMDL\n") + pdb([("A", 1, "VAL", 0)]))
    assert pdb_residues(path)[("A", 1, "")]["aa"] == "A"


def test_input_symlink_cannot_replace_admitted_regular_file(tmp_path):
    original = tmp_path / "research_runs/rsi/original.pdb"
    original.parent.mkdir(parents=True)
    original.write_text(pdb([("B", 492, "LEU", 10)]))
    link = original.with_name("link.pdb")
    link.symlink_to(original)
    record = {
        "path": str(link.relative_to(tmp_path)),
        "sha256": sha256_file(original),
        "residues": 1,
        "hotspots": [{"chain": "B", "number": 492, "aa": "L"}],
    }
    with pytest.raises(ValueError, match="changed"):
        validate_target_input(tmp_path, record)
