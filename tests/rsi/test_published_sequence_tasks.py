import copy
import csv
import io
import json
import zipfile

import pytest

from agent.rsi.published_sequence_tasks import (
    ESOL_ENDPOINTS,
    audit_catalog,
    bound,
    deduplicate,
    designed_records,
    esol_records,
    group_assignments,
    partition_rows,
    write_task,
)


def records():
    return [
        {
            "id": f"sample{i}",
            "sequence": "A" * (20 + i) + "C",
            "group": f"parent{i // 2}",
            "endpoints": {"x": float(i), "y": float(-i)},
        }
        for i in range(100)
    ]


def test_assignments_do_not_depend_on_values_order_or_endpoint():
    source = records()
    expected = group_assignments(source)
    changed = copy.deepcopy(source[::-1])
    for row in changed:
        row["endpoints"] = {"x": 9999999}
    assert group_assignments(changed) == expected
    for i in range(0, 100, 2):
        assert expected[f"sample{i}"] == expected[f"sample{i + 1}"]
    x, _ = partition_rows(source, "x", expected, (12, 8, 8))
    y, _ = partition_rows(source, "y", expected, (12, 8, 8))
    assert {p: [r["id"] for r in rows] for p, rows in x.items()} == {
        p: [r["id"] for r in rows] for p, rows in y.items()
    }
    with pytest.raises(ValueError, match="exactly"):
        group_assignments(source, {"sample0": "group"})


def test_every_duplicate_member_is_excluded_without_averaging():
    source = records()[:4]
    source[1]["sequence"] = source[0]["sequence"]
    source[3]["id"] = source[2]["id"]
    kept, excluded = deduplicate(source)
    assert not kept and len(excluded) == 4


def esol_fixture(tmp_path, variant=None):
    columns = ["JW_ID", "ECK number", "B number", "Type of gene product", "Gene product description"]
    columns += [item[3] for item in ESOL_ENDPOINTS]
    row = dict.fromkeys(columns, "")
    row.update(
        {
            "JW_ID": "JW0001",
            "ECK number": "ECK0002",
            "B number": "b0002",
            "Type of gene product": "e",
            "Solubility (%)": "25.2",
            "Yield (uM)": "nan",
        }
    )
    cds = {"note": ["JW0001:ECK0002:b0002"], "sequence": "A" * 25, "locus_tag": "test", "protein_id": ["test"]}
    annotations = [cds]
    if variant == "ambiguous":
        annotations.append(cds.copy())
    elif variant == "mismatch":
        cds["note"] = ["JW0001:ECK0003:b0002"]
    elif variant == "noncanonical":
        cds["sequence"] += "X"
    elif variant == "putative":
        row["Type of gene product"] = "pe"
    with (tmp_path / "esol.csv").open("w") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerow(row)
    (tmp_path / "annotations.json").write_text(json.dumps(annotations))
    return esol_records(tmp_path / "esol.csv", tmp_path / "annotations.json")


def test_exact_w3110_identifiers_and_finite_measurements(tmp_path):
    kept, excluded = esol_fixture(tmp_path)
    assert not excluded and len(kept) == 1
    assert kept[0]["endpoints"]["Solubility (%)"] == 25.2
    assert kept[0]["endpoints"]["Yield (uM)"] is None


@pytest.mark.parametrize("variant", ["ambiguous", "mismatch", "noncanonical", "putative"])
def test_esol_exclusions(tmp_path, variant):
    kept, excluded = esol_fixture(tmp_path, variant)
    assert not kept
    assert len(excluded) == (0 if variant == "putative" else 1)


def csv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_design_quality_and_source_filtered_targets(tmp_path):
    parents = [
        {"name": "design", "DMS_group": "G0", "WT_cluster": "HHH"},
        {"name": "poor", "DMS_group": "G5", "WT_cluster": "HHH"},
        {"name": "natural", "DMS_group": "G0", "WT_cluster": "19"},
    ]
    data = [
        {"name": f"row{i}", "WT_name": parent, "WT_cluster": topology, "aa_seq": "A" * (20 + i), "dG_ML": target}
        for i, (parent, topology, target) in enumerate(
            [("design", "HHH", "1.2"), ("design", "HHH", "-"), ("poor", "HHH", "2.0"), ("natural", "19", "1.3")]
        )
    ]
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Processed_K50_dG_datasets/Single_DMS_list.csv", csv_text(parents))
        z.writestr("Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv", csv_text(data))
    kept, excluded, admitted = designed_records(archive)
    assert len(kept) == 1 and kept[0]["endpoints"]["dG_ML"] == 1.2
    assert set(admitted) == {"design"}
    assert excluded["counts"] == {"author_filtered_or_nonfinite_dG_ML": 1}


def test_admission_and_global_cross_endpoint_audit(tmp_path):
    source = records()
    assignments = group_assignments(source)
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic source fixture")
    entries = []
    for endpoint in ("x", "y"):
        entries.append(
            write_task(
                tmp_path,
                identity=f"fixture_{endpoint}",
                stage="solubility",
                paper="esol2009",
                protein="synthetic fixture",
                endpoint=endpoint,
                unit="arbitrary",
                records=source,
                assignments=assignments,
                references=[bound(reference)],
            )
        )
    audit = audit_catalog(entries)
    assert audit["tasks"] == 2 and audit["new_autonomous_models"] == 0
    assert not audit["homology_independence_established"]
    task_path = tmp_path / "fixture_y/task.json"
    task = json.loads(task_path.read_text())
    task["data"]["validation"], task["data"]["future"] = task["data"]["future"], task["data"]["validation"]
    task_path.write_text(json.dumps(task))
    entries[1]["task"] = bound(task_path)
    with pytest.raises(ValueError, match="Cross-task"):
        audit_catalog(entries)


def test_prepared_task_rejects_source_tampering(tmp_path):
    source = records()
    reference = tmp_path / "reference.txt"
    reference.write_text("fixture")
    entry = write_task(
        tmp_path,
        identity="fixture",
        stage="stability",
        paper="tsuboyama2023",
        protein="fixture",
        endpoint="x",
        unit="arbitrary",
        records=source,
        assignments=group_assignments(source),
        references=[bound(reference)],
    )
    reference.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        audit_catalog([entry])
