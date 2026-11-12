"""Synthetic format fixtures verify admission boundaries; not biological results."""
import csv
import io
import json
import math
import tarfile

import pytest

from agent.rsi.datasets import skempi_source as source
from agent.rsi.datasets import skempi_structures as structures
from agent.rsi.datasets.skempi_protocol import homology_groups


def row(**changes):
    value = dict.fromkeys(source.COLUMNS, "")
    value.update({"#Pdb": "1ABC_A_B", "Mutation(s)_PDB": "AA42AC", "Mutation(s)_cleaned": "AA1C",
                  "Affinity_mut (M)": "2E-9", "Affinity_mut_parsed": "2E-9",
                  "Affinity_wt (M)": "1E-9", "Affinity_wt_parsed": "1E-9", "Temperature": "298",
                  "Reference": "111", "Protein 1": "Synthetic partner A", "Protein 2": "Synthetic partner B",
                  "Method": "SPR", "SKEMPI version": "2"})
    value.update(changes)
    return value


def snapshot(tmp_path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=source.COLUMNS, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    (tmp_path / source.FILE).write_bytes(raw)
    asset = {"source": "skempi2", "file": source.FILE, "url": source.URL, "final_url": source.URL,
             "bytes": len(raw), "sha256": source.sha256(raw), "rows": len(rows), "status": "downloaded_not_admitted"}
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"schema": "property_source_snapshot.v1", "files": [asset]}))
    return path, source.sha256(path.read_bytes())


def atom(name="ALA", chain="A", number=1, record="ATOM", insertion=" "):
    return f"{record:<6}{1:5d}  CA  {name:3s} {chain}{number:4d}{insertion}   {1.0:8.3f}{2.0:8.3f}{3.0:8.3f}\n"


def coordinate_pair():
    return (atom() + atom("GLY", "B")).encode(), b"ALA A  42A 1\nGLY B   5  1\n"


@pytest.mark.parametrize("raw,kind", [
    ("<2E-9", "upper_bound"), (">2E-9", "lower_bound"), ("~2E-9", "approximate"),
    ("n.b", "no_detectable_binding"), ("n.b.", "no_detectable_binding"), ("unf", "unfolded"),
    ("", "missing"), ("0", "invalid_nonpositive"), ("unknown", "unparsed"),
])
def test_censoring_is_not_converted_into_an_exact_training_target(raw, kind):
    item = row(**{"Affinity_mut (M)": raw, "Affinity_mut_parsed": "2E-9"})
    normalized = source.normalize(item, 2)
    assert normalized["raw"] == item
    assert normalized["measurements"]["kd"]["mut"]["kind"] == kind
    assert normalized["derived_ddg"]["value"] is None


def test_exact_ddg_sign_units_and_original_temperature():
    normalized = source.normalize(row(), 2)
    assert normalized["derived_ddg"]["value"] == pytest.approx(8.314 / 4184 * 298 * math.log(2))
    assert normalized["derived_ddg"]["unit"] == "kcal mol^-1"
    assert normalized["measurements"]["kon"]["wt"]["unit"] == "M^-1 s^-1"
    assert normalized["measurements"]["entropy"]["wt"]["unit"] == "cal mol^-1 K^-1"
    assert normalized["temperature"]["kelvin"] == 298
    assert source.measurement("-12.5", "kcal mol^-1", positive=False)["kind"] == "exact"


@pytest.mark.parametrize("change", [
    {"Temperature": "298(assumed)"}, {"Temperature": ""}, {"Affinity_wt_parsed": "9E-9"}
])
def test_assumptions_and_disagreeing_source_parsing_are_excluded(change):
    normalized = source.normalize(row(**change), 2)
    assert normalized["derived_ddg"]["value"] is None
    assert normalized["derived_ddg"]["excluded_reasons"]


def test_negative_numbers_insertions_and_repeated_sites_are_not_silently_repaired():
    values = source.mutation_list("WA-1F,AA42AC,AA42AC")
    assert values[0]["residue_number"] == -1
    assert values[1]["insertion_code"] == "A"
    assert values[2]["repeated_site"]
    with pytest.raises(ValueError, match="Unrecognized"):
        source.mutation_list("A42")


def test_export_preserves_repeats_and_never_admits_training(tmp_path):
    item = row()
    path, checksum = snapshot(tmp_path, [item, item])
    raw = (tmp_path / source.FILE).read_bytes()
    result = source.export_source(path, checksum, tmp_path / "normalized")
    normalized = [json.loads(line) for line in (tmp_path / "normalized/observations.jsonl").read_text().splitlines()]
    assert len(normalized) == 2 and normalized[0]["id"] != normalized[1]["id"]
    assert result["summary"]["exact_duplicate_excess"] == 1
    assert result["summary"]["admitted_tasks"] == 0 and result["status"] == "normalized_not_admitted"
    assert result["attribution"]["license"] == "CC-BY-4.0"
    assert normalized[0]["raw"] == item
    assert raw == (tmp_path / source.FILE).read_bytes()
    with pytest.raises(ValueError, match="new non-symlink"):
        source.export_source(path, checksum, tmp_path / "normalized")
    (tmp_path / source.FILE).write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        source.read_snapshot(path, checksum)


def test_source_manifest_and_symlink_boundaries(tmp_path):
    path, checksum = snapshot(tmp_path, [row()])
    with pytest.raises(ValueError, match="manifest checksum"):
        source.read_snapshot(path, "0" * 64)
    link = tmp_path / "linked"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlinks"):
        source.read_snapshot(link, checksum)


def test_holdout_and_original_study_relations_are_transitive():
    rows = [source.normalize(row(**{"#Pdb": "1ABC_A_B", "Hold_out_proteins": "2ABC_A_B"}), 2),
            source.normalize(row(**{"#Pdb": "2ABC_A_B", "Reference": "222"}), 3),
            source.normalize(row(**{"#Pdb": "3ABC_A_B", "Reference": "222"}), 4),
            source.normalize(row(**{"#Pdb": "4ABC_A_B", "Reference": "444"}), 5)]
    result = source.exclusion_groups(rows)
    assert rows[0]["exclusion_group"] == rows[1]["exclusion_group"] == rows[2]["exclusion_group"]
    assert rows[3]["exclusion_group"] != rows[0]["exclusion_group"]
    assert result["count"] == 2 and not result["split_assigned"]


def test_cleaned_structure_mapping_verifies_wt_and_insertion_without_solvent_collision():
    pdb, mapping = coordinate_pair()
    parsed = structures.parse_pair(pdb + atom("HOH", "A", record="HETATM").encode(), mapping)
    assert parsed["original"][("A", 42, "A")]["cleaned"] == ("A", 1, "")
    assert parsed["chains"]["A"]["observed_sequence"] == "A"
    normalized = source.normalize(row(), 2)
    structures.attach_mapping([normalized], {"PDBs/1ABC.pdb": pdb, "PDBs/1ABC.mapping": mapping})
    assert normalized["sequence_mapping_status"] == "verified_observed_residues"
    assert normalized["mutation_mapping"][0]["sequence_index"] == 0
    assert normalized["training_eligible"] is False


@pytest.mark.parametrize("fault", ["wt", "insertion", "cleaned", "duplicate", "nonstandard"])
def test_mapping_errors_never_create_a_verified_sequence(fault):
    pdb, mapping = coordinate_pair()
    item = row()
    if fault == "wt":
        item.update({"Mutation(s)_PDB": "GA42AC", "Mutation(s)_cleaned": "GA1C"})
    elif fault == "insertion":
        item["Mutation(s)_PDB"] = "AA42BC"
    elif fault == "cleaned":
        item["Mutation(s)_cleaned"] = "AA2C"
    elif fault == "duplicate":
        item.update({"Mutation(s)_PDB": "AA42AC,AA42AC", "Mutation(s)_cleaned": "AA1C,AA1C"})
    else:
        pdb = pdb.replace(b"ALA", b"UNK")
        mapping = mapping.replace(b"ALA", b"UNK")
    normalized = source.normalize(item, 2)
    structures.attach_mapping([normalized], {"PDBs/1ABC.pdb": pdb, "PDBs/1ABC.mapping": mapping})
    assert normalized["sequence_mapping_status"] == "requires_review"
    assert normalized["sequence_mapping_issues"] and normalized["training_eligible"] is False


def test_duplicate_mapping_is_rejected_not_deduplicated():
    pdb, mapping = coordinate_pair()
    with pytest.raises(ValueError, match="Non-bijective"):
        structures.parse_pair(pdb, mapping + mapping)


def test_homology_links_shared_partner_complexes_without_using_labels():
    rows = []
    for index, sequence in enumerate(["A" * 30, "A" * 25 + "C" * 5, "W" * 30], 1):
        r = source.normalize(row(**{"#Pdb": f"{index}ABC_A_B", "Reference": str(index)}), index+1)
        r.update(sequence_mapping_status="verified_observed_residues",
                 partner_sequences={"A": {"observed_sequence": sequence}})
        rows.append(r)
    result = homology_groups(rows)
    assert result["complete"] and result["comparisons"] == 3
    assert rows[0]["exclusion_group"] == rows[1]["exclusion_group"]
    assert rows[0]["exclusion_group"] != rows[2]["exclusion_group"]
    assert result["admitted_tasks"] == 0
    before = [r["exclusion_group"] for r in rows]
    for r in rows:
        r["derived_ddg"]["value"] = 1e6
    homology_groups(rows)
    assert before == [r["exclusion_group"] for r in rows]
    with pytest.raises(ValueError, match="Incomplete homology"):
        homology_groups(rows, wall_seconds=-1)


@pytest.mark.parametrize("unsafe", [False, True])
def test_archive_is_hash_bound_and_never_extracts_paths(tmp_path, unsafe):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        name = "../escape.pdb" if unsafe else "PDBs/1ABC.pdb"
        body = coordinate_pair()[0]
        item = tarfile.TarInfo(name)
        item.size = len(body)
        archive.addfile(item, io.BytesIO(body))
    assets = []
    for name, body, url in [
        (structures.ARCHIVE, payload.getvalue(), structures.URL),
        ("terms.html", b"Synthetic license fixture", source.LICENSE),
        ("faq.html", b"Synthetic format fixture", source.SOURCE + "info/faq_and_help"),
    ]:
        (tmp_path / name).write_bytes(body)
        assets.append({"file": name, "url": url, "final_url": url, "bytes": len(body), "sha256": source.sha256(body)})
    path = tmp_path / "structures.json"
    path.write_text(json.dumps({"schema": "skempi_structure_snapshot.v1", "license": "CC-BY-4.0", "files": assets}))
    checksum = source.sha256(path.read_bytes())
    if unsafe:
        with pytest.raises(ValueError, match="Unsafe archive"):
            structures.read_structures(path, checksum)
    else:
        _, members = structures.read_structures(path, checksum)
        assert set(members) == {"PDBs/1ABC.pdb"}
        (tmp_path / "terms.html").write_bytes(b"changed")
        with pytest.raises(ValueError, match="asset changed"):
            structures.read_structures(path, checksum)
    assert not (tmp_path / "escape.pdb").exists()
