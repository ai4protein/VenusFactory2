import copy
import json
from pathlib import Path

import pytest

from agent.rsi.engine.case_evidence import canonical_claims, render_evidence_report
from agent.rsi.campaigns.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.cases.phl7_case import (
    ANALYSES,
    ARCHIVE_SHA256,
    TOLERANCES,
    admit,
    allocation_ids,
    canonical_phl7,
    compare_fes,
    read_table,
    verify_artifacts,
)
from agent.rsi.engine.storage import sha256_file

GRID = """#! FIELDS D1 D2 fes_cv1_cv2 dfes_cv1_cv2_D1 dfes_cv1_cv2_D2
#! SET min_D1 0
#! SET max_D1 1
0 0 1 0 0
1 0 2 0.1 0.2
0 1 inf 0 0
1 1 3 0.3 0.4
"""


def test_grid_compares_support_derivatives_and_does_not_hide_constant_shift():
    c = compare_fes(GRID, GRID)
    assert c["matches_reference"] and c["reference_finite_energies"] == 3
    for changed in [GRID.replace("0 0 1 0 0", "0 0 1.01 0 0"),
                    GRID.replace("0 1 inf", "0 1 4"),
                    GRID.replace("0.3 0.4", "0.3 0.5")]:
        assert not compare_fes(GRID, changed)["matches_reference"]
    shifted = GRID.replace("0 0 1 0 0", "0 0 2 0 0").replace("1 0 2 0.1", "1 0 3 0.1").replace("1 1 3 0.3", "1 1 4 0.3")
    c = compare_fes(GRID, shifted)
    assert c["mean_energy_offset"] == 1 and c["offset_removed_max_abs_error"] == 0
    assert not c["matches_reference"]


@pytest.mark.parametrize("changed", [GRID.replace("0 0 1", "0.01 0 1"),
                                      GRID.replace("inf", "nan"), GRID.replace("inf", "-inf"),
                                      GRID.replace("#! SET min_D1 0", "#! SET min_D1 0.1")])
def test_invalid_or_changed_grid_is_not_comparable(changed):
    with pytest.raises(ValueError):
        compare_fes(GRID, changed)


def test_missing_or_duplicate_table_fields_rejected():
    for text in ["0 0 1", GRID + "#! FIELDS a b c d e\n", GRID.replace("D1 D2", "D1 D1")]:
        with pytest.raises(ValueError):
            read_table(text)


def test_wrong_archive_rejected_before_parsing(tmp_path):
    p = tmp_path / "fake.zip"
    p.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="identity"):
        admit(p)


def fixture_report(tmp_path):
    protocol = {"archive_sha256": ARCHIVE_SHA256, "comparison_tolerances": TOLERANCES,
                "allocations": [], "method_discrepancies": ["source discrepancy"]}
    rows = []
    for ident in allocation_ids():
        analysis, _, _ = ident.split("/")
        p = tmp_path / "allocations" / ident
        p.mkdir(parents=True)
        names = ["CV_", "reweight.dat", "reference.dat", ANALYSES[analysis], "driver.log"]
        if analysis == "93_open_closed":
            names.append("DD_")
        for name in names:
            (p / name).write_text(GRID if name in ["reference.dat", ANALYSES[analysis]] else "input or log")
        hashes = {name: sha256_file(p / name) for name in names}
        inputs = {name: "archive/" + name for name in names if name in ["CV_", "DD_", "reweight.dat"]}
        protocol["allocations"].append({"id": ident, "analysis": analysis, "input_members": inputs,
                                       "input_sha256": {k: hashes[k] for k in inputs}, "reference_sha256": hashes["reference.dat"]})
        rows.append({"id": ident, "status": "completed", "comparison": compare_fes(GRID, GRID), "artifact_hashes": hashes})
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    return {"status": "completed", "execution_origin": "new_analysis_of_author_outputs",
            "archive_sha256": ARCHIVE_SHA256, "protocol_sha256": sha256_file(tmp_path / "protocol.json"),
            "new_molecular_dynamics_performed": False, "full_paper_reproduced": False,
            "allocations": rows, "method_discrepancies": protocol["method_discrepancies"]}


def test_allocation_failures_and_forged_agreement_are_not_success(tmp_path):
    report = fixture_report(tmp_path)
    c = canonical_claims("phl7.opes_analysis.v1", report)
    assert c["completed_analyses"] == 24 and c["reference_agreement_count"] == 24
    assert not c["new_molecular_dynamics_performed"] and not c["full_paper_reproduced"]
    assert "24 free-energy" in render_evidence_report("test", c)
    for change in ["duplicate", "failed", "false_agreement", "full_claim"]:
        bad = copy.deepcopy(report)
        if change == "duplicate":
            bad["allocations"][0]["id"] = bad["allocations"][1]["id"]
        elif change == "failed":
            bad["allocations"][0]["status"] = "failed"
        elif change == "false_agreement":
            bad["allocations"][0]["comparison"]["free_energy_max_abs_error"] = 1.0
        else:
            bad["full_paper_reproduced"] = True
        with pytest.raises(ValueError):
            canonical_phl7(bad)


def test_discrepancies_are_valid_results_not_dropped_allocations(tmp_path):
    report = fixture_report(tmp_path)
    row = report["allocations"][4]
    row["comparison"]["free_energy_max_abs_error"] = 0.1
    row["comparison"]["matches_reference"] = False
    c = canonical_phl7(report)
    assert c["reference_agreement_count"] == 23 and c["analysis_discrepancies"] == [row["id"]]


def test_artifact_replay_detects_changes_even_after_file_hash_is_updated(tmp_path):
    report = fixture_report(tmp_path)
    assert verify_artifacts(report, tmp_path)["reference_agreement_count"] == 24
    row = report["allocations"][0]
    p = tmp_path / "allocations" / row["id"] / "CV_"
    p.write_text("altered input")
    row["artifact_hashes"]["CV_"] = sha256_file(p)
    with pytest.raises(ValueError, match="admitted archive"):
        verify_artifacts(report, tmp_path)


def test_plugin_keeps_all_allocations_and_is_cpu_only(tmp_path):
    plugin = "phl7.opes_analysis.v1"
    assert next(x for x in list_case_plugins() if x["id"] == plugin)["parameters"]["properties"] == {}
    assert resource_for(plugin, {}) == "cpu"
    args = {"root": tmp_path, "output": tmp_path / "research_runs/rsi/run", "controller_python": Path("/python")}
    assert "run_phl7_case.py" in invocation(plugin, {}, **args)[2]
    for bad in [{"replicates": 1}, {"source": "/tmp/elsewhere"}, {"operation": "new_md"}]:
        with pytest.raises(ValueError):
            invocation(plugin, bad, **args)


@pytest.mark.parametrize("mutation", ["none", "input", "comparison"])
def test_independent_auditor_uses_archive_and_rejects_tampering(tmp_path, monkeypatch, mutation):
    import hashlib
    import importlib.util
    import zipfile

    script = Path(__file__).resolve().parents[2] / "scripts/rsi/cases/audit_phl7_case.py"
    spec = importlib.util.spec_from_file_location("independent_phl7_audit", script)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    report = fixture_report(campaign)
    protocol = json.loads((campaign / "protocol.json").read_text())
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        seen = set()
        for ident in allocation_ids():
            analysis, variant, replica = ident.split("/")
            prefix = f"MD_simulations_PHL7/OPES/{analysis}/{variant}/"
            folder = campaign / "allocations" / ident
            members = {"CV_": prefix + replica + "/CV_", "reweight.dat": prefix + "reweight.dat",
                       "reference.dat": prefix + replica + "/" + ANALYSES[analysis]}
            if analysis == "93_open_closed":
                members["DD_"] = prefix + replica + "/DD_"
            for name, member in members.items():
                if member not in seen:
                    z.writestr(member, (folder / name).read_bytes())
                    seen.add(member)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(auditor, "ARCHIVE", archive_hash)
    protocol["archive_sha256"] = archive_hash
    protocol["plumed_version"] = "2.9.0"
    for name, key in [("phl7_case.py", "module_sha256"), ("run_phl7_case.py", "entry_sha256")]:
        (campaign / name).write_text("source fixture")
        protocol[key] = sha256_file(campaign / name)
    (campaign / "protocol.json").write_text(json.dumps(protocol))
    report["protocol_sha256"] = sha256_file(campaign / "protocol.json")
    report["archive_sha256"] = archive_hash
    for row in report["allocations"]:
        row["exit_code"] = 0
    if mutation == "input":
        p = campaign / "allocations" / report["allocations"][0]["id"] / "CV_"
        p.write_text("changed data")
        report["allocations"][0]["artifact_hashes"]["CV_"] = sha256_file(p)
    elif mutation == "comparison":
        report["allocations"][0]["comparison"]["free_energy_max_abs_error"] = 0.1
    (campaign / "report.json").write_text(json.dumps(report))
    if mutation == "none":
        result = auditor.audit(campaign, archive, tmp_path / "audit")
        assert result["bitwise_matches"] == 24 and result["numeric_matches"] == 24
    else:
        with pytest.raises(ValueError):
            auditor.audit(campaign, archive, tmp_path / "audit")
