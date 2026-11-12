"""SC receives verified computations, not inferred targets or preview-only values."""
import hashlib
import json
import shutil

import pytest

from agent.harness import research_analysis_provenance as provenance
from agent.harness.research_analysis_evidence import capture
from agent.harness.research_workspace import ResearchWorkspace
from mcp_tools.scheduler.analysis_evidence import AnalysisCapture
from mcp_tools.scheduler.receipt_binding import value_digest

SEQUENCE = "MKWVTFISLLFLFSSAYS"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def setup(tmp_path):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "private", {"steps": []})
    raw = (">control\n" + SEQUENCE + "\n").encode()
    (service.workspace / "control.fasta").write_bytes(raw)
    inputs = [{"id": "upload-one", "path": "control.fasta", "sha256": sha(raw), "size": len(raw)}]
    return service, inputs


def compute(service, value=None, arguments=None, output=False):
    arguments = {"sequence": SEQUENCE} if arguments is None else arguments
    spec = {"route": "native", "workspace": str(service.workspace), "session_id": "session-one",
            "name": "scientific_calculation", "arguments": arguments}
    if output:
        arguments["output_file"] = str(service.workspace / "result.json")
    worker = AnalysisCapture(spec, service.workspace, "sequence_analysis")
    if value is None:
        value = json.dumps({
            "status": "success", "content_preview": "truncated/incorrect preview: length 999",
            "data": {"sequence_length": 18, "gravy": 1.233, "binding": {"mean": 0.3}, "zero": 0},
            "units": {"sequence_length": "amino acids"},
        })
    if output:
        (service.workspace / "result.json").write_text('{"actual":true}')
    evidence = worker.finish({"value": value, "is_error": False})
    assert evidence is not None
    identity = format(len(service.receipts) + 1, "032x")
    job = {"id": identity, "status": "completed", "tool": spec["name"],
           "category": "sequence_analysis", "job_number": "VF-fixture",
           "parent_job": "b" * 32, "session_id": "session-one",
           "arguments_sha256": value_digest(arguments),
           "result": {"value": value, "is_error": False, "analysis_evidence": evidence}}
    return capture(service, job=job, name=spec["name"], arguments=arguments, value=value,
                   owner="owner-one", parent_job=job["parent_job"], session_id=job["session_id"])


def build(service, inputs, checkpoint=None, source_reader=None):
    return provenance.build_evidence_index(service.workspace, receipts=service.receipts,
                                          inputs=inputs, checkpoint=checkpoint, source_reader=source_reader)


def test_real_capture_projects_full_result_and_registered_fasta(setup):
    service, inputs = setup
    artifact = compute(service)
    report = b"Length: 18\n"
    (service.workspace / "report.md").write_bytes(report)
    result = build(service, inputs, {"artifacts": [{"path": "report.md", "sha256": sha(report)}]})
    material = result["materials"]["records"][0]
    assert material["status"] == "verified"
    assert material["sequences"] == [{"record_index": 0, "id": "control", "length": 18,
                                     "sequence_sha256": sha(SEQUENCE.encode())}]
    computation = result["computations"][0]
    assert computation["artifact"] == {key: artifact[key] for key in ("path", "sha256")}
    fields = {row["pointer"]: row for row in computation["result_fields"]}
    assert fields["/decoded_result/data/gravy"]["value"] == 1.233
    assert fields["/decoded_result/data/sequence_length"]["unit"] == "amino acids"
    assert fields["/decoded_result/data/zero"]["value"] == 0
    assert fields["/decoded_result/data/binding/mean"]["unit_status"] == "unknown"
    assert computation["sequence_relations"]["records"][0]["relation"] == "exact_match"
    assert computation["input_files"] == {}  # Literal sequence, not a falsely inferred file input.
    assert result["checkpoint_artifacts"]["records"][0]["status"] == "unchanged"
    assert "scientific_acceptance" not in result
    assert str(service.workspace) not in json.dumps(result)


def test_unmatched_sequence_is_evidence_not_automatic_scientific_rejection(setup):
    service, inputs = setup
    compute(service, arguments={"sequence": "MAESGPGQV"})
    computation = build(service, inputs)["computations"][0]
    assert computation["status"] == "verified"
    relation = computation["sequence_relations"]["records"][0]
    assert relation["relation"] == "unmatched"
    assert relation["length"] == 9
    assert relation["comparison_complete"] is True
    assert "valid derived" in relation["interpretation"]
    assert "passed" not in computation


def test_workspace_files_are_not_invented_as_original_uploads(setup):
    service, _ = setup
    compute(service)
    result = build(service, [])
    assert result["materials"] == {"status": "not_registered", "records": [], "omissions": []}
    assert result["computations"][0]["sequence_relations"]["records"][0]["relation"] == "unknown"


def test_text_does_not_guess_a_target_sequence(setup):
    service, _ = setup
    raw = ("Suggested example: " + SEQUENCE).encode()
    (service.workspace / "notes.txt").write_bytes(raw)
    compute(service)
    result = build(service, [{"path": "notes.txt", "sha256": sha(raw)}])
    assert result["materials"]["records"][0]["sequence_status"] == "not_parsed"
    assert result["computations"][0]["sequence_relations"]["records"][0]["relation"] == "unknown"


@pytest.mark.parametrize("raw", [b"not fasta", b">id\n", b">\nAAA\n", b">id\nABC12\n", b">id\n\xff"])
def test_bad_fasta_has_explicit_unknown(setup, raw):
    service, _ = setup
    (service.workspace / "bad.fasta").write_bytes(raw)
    row = build(service, [{"path": "bad.fasta", "sha256": sha(raw)}])["materials"]["records"][0]
    assert row["status"] == "verified"
    assert row["sequence_status"] == "unknown"
    assert row["reason"] == "fasta_parse_failed"


def test_protected_original_reader_validates_hash_without_exposing_labels(setup):
    service, _ = setup
    original = b"sequence,hidden_test_label\nMKWVTFIS,SECRET_TARGET_398\n"
    (service.workspace / "data.csv").write_text('{"data_access":"protected_research_table"}')
    seen = []

    def reader(path):
        seen.append(path)
        return original

    result = build(service, [{"path": "data.csv", "sha256": sha(original)}], source_reader=reader)
    assert seen == ["data.csv"]
    assert result["materials"]["records"][0]["status"] == "verified"
    serialized = json.dumps(result)
    assert "SECRET_TARGET" not in serialized and "hidden_test_label" not in serialized


@pytest.mark.parametrize("path", ["../foreign.fasta", "/tmp/foreign.fasta", ".env", "linked/control.fasta"])
def test_material_paths_never_escape_or_follow_links(setup, tmp_path, path):
    service, inputs = setup
    other = tmp_path / "other"
    other.mkdir()
    (other / "control.fasta").write_text(">other\nAAA\n")
    (service.workspace / "linked").symlink_to(other, target_is_directory=True)
    result = build(service, [{**inputs[0], "path": path}])
    assert result["materials"]["records"][0]["status"] == "unknown"
    assert result["materials"]["records"][0]["reason"] == "registered_input_unavailable"


def test_external_path_is_rejected_before_protected_reader(setup):
    service, inputs = setup
    calls = []
    result = build(service, [{**inputs[0], "path": "../private"}], source_reader=lambda p: calls.append(p))
    assert calls == []
    assert result["materials"]["records"][0]["status"] == "unknown"


@pytest.mark.parametrize("fault", ["artifact", "output", "foreign_workspace", "symlink", "receipt"])
def test_changed_or_foreign_evidence_is_not_reported_as_verified(setup, tmp_path, fault):
    service, inputs = setup
    artifact = compute(service, output=True)
    if fault == "artifact":
        (service.workspace / artifact["path"]).write_text('{"value":999}')
    elif fault == "output":
        (service.workspace / "result.json").write_text('{"actual":false}')
    elif fault == "symlink":
        path = service.workspace / artifact["path"]
        raw = path.read_bytes()
        path.unlink()
        target = tmp_path / "foreign.json"
        target.write_bytes(raw)
        path.symlink_to(target)
    elif fault == "receipt":
        service.receipts[0]["result"]["evidence"]["result_sha256"] = "invalid"
    else:
        other = tmp_path / "copied"
        shutil.copytree(service.workspace, other)
        service.workspace = other
    result = build(service, inputs)
    assert result["computations"][0]["status"] == "unverified"
    assert "result_fields" not in result["computations"][0]


@pytest.mark.parametrize("value", [0, False, 1.25, {"score": 0.5}, {"data": [0.0, 1.0]}])
def test_scalar_and_structured_results_do_not_require_output_files(setup, value):
    service, inputs = setup
    compute(service, value=value)
    row = build(service, inputs)["computations"][0]
    assert row["status"] == "verified"
    assert row["output_files"] == {}
    assert row["result_complete"] is True
    assert row["result_fields"]


def test_json_pointers_escape_keys_and_report_units_only_as_recorded(setup):
    service, inputs = setup
    compute(service, value={"data": {"a/b~c": 3, "temperature": -0.97}, "units": {"a/b~c": "nm"}})
    fields = build(service, inputs)["computations"][0]["result_fields"]
    assert fields[0]["pointer"] == "/result/data/a~1b~0c"
    assert fields[0]["unit"] == "nm"
    assert fields[1]["unit"] is None  # No invented degrees Celsius.


def test_report_tampering_is_visible(setup):
    service, inputs = setup
    (service.workspace / "report.md").write_text("changed")
    result = build(service, inputs, {"artifacts": [{"path": "report.md", "sha256": sha(b"original")}]})
    assert result["checkpoint_artifacts"]["records"][0]["status"] == "changed"


def test_fields_strings_and_sequences_have_explicit_omissions(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_FIELDS", 2)
    monkeypatch.setattr(provenance, "MAX_VALUE_CHARS", 10)
    compute(service, value={"data": {"long": "A" * 80, "score": 0, "omitted": 3}})
    row = build(service, inputs)["computations"][0]
    assert row["result_complete"] is False
    assert {r["reason"] for r in row["result_omissions"]} == {"value_limit", "field_limit"}
    assert row["result_fields"][0]["value_sha256"]


def test_large_input_is_unknown_without_reading(setup):
    service, inputs = setup
    calls = []
    result = build(service, [{**inputs[0], "size": provenance.MAX_INPUT_BYTES + 1}],
                   source_reader=lambda p: calls.append(p))
    assert calls == []
    assert result["materials"]["records"][0]["reason"] == "input_too_large"


def test_total_index_has_explicit_computation_and_material_limits(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_MATERIAL_CHARS", 30)
    monkeypatch.setattr(provenance, "MAX_INDEX_CHARS", 1300)
    compute(service)
    result = build(service, inputs)
    assert result["materials"]["omissions"][0]["reason"] == "material_summary_limit"
    assert result["omissions"][0]["reason"] == "index_size_limit"
    assert result["computations"] == []
    assert len(json.dumps(result)) < 1500


def test_sensitive_parameters_are_not_exposed_by_new_projection(setup):
    service, inputs = setup
    secret = "opaque-credential-that-has-no-recognizable-prefix"
    compute(service, arguments={"sequence": SEQUENCE, "api_key": secret, "credentials": {"token": secret}})
    result = build(service, inputs)
    assert secret not in json.dumps(result)
    projection = result["computations"][0]["arguments"]
    assert projection["complete"] is False
    assert all(row["reason"] == "sensitive_field_redacted" for row in projection["omissions"])


def test_material_sequence_limit_is_explicit_and_comparison_not_complete(setup, monkeypatch):
    service, _ = setup
    monkeypatch.setattr(provenance, "MAX_SEQUENCES", 1)
    raw = (">one\n" + SEQUENCE + "\n>two\nAAA\n").encode()
    (service.workspace / "many.fasta").write_bytes(raw)
    compute(service)
    result = build(service, [{"path": "many.fasta", "sha256": sha(raw)}])
    material = result["materials"]["records"][0]
    assert material["sequences_complete"] is False
    assert material["omissions"] == [{"reason": "sequence_limit"}]
    relation = result["computations"][0]["sequence_relations"]["records"][0]
    assert relation["relation"] == "exact_match"
    assert relation["comparison_complete"] is False


def test_receipt_count_limit_does_not_hide_remaining_computations(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_COMPUTATIONS", 1)
    compute(service)
    compute(service)
    result = build(service, inputs)
    assert len(result["computations"]) == 1
    assert result["omissions"] == [{"reason": "computation_limit", "count": 1}]


def test_registered_material_hash_change_is_unknown(setup):
    service, inputs = setup
    (service.workspace / "control.fasta").write_text(">control\nAAA\n")
    compute(service)
    result = build(service, inputs)
    assert result["materials"]["records"][0]["reason"] == "original_hash_mismatch"
    assert result["computations"][0]["sequence_relations"]["records"][0]["relation"] == "unknown"


def test_many_large_material_descriptors_still_obey_summary_limit(setup, monkeypatch):
    service, _ = setup
    monkeypatch.setattr(provenance, "MAX_MATERIAL_CHARS", 400)
    entries = [{"path": "long-name-" + str(i) + ".fasta", "sha256": "a" * 64,
                "size": provenance.MAX_INPUT_BYTES + 1} for i in range(30)]
    result = build(service, entries)
    assert result["materials"]["omissions"][0]["reason"] == "material_summary_limit"
    assert len(json.dumps(result)) < 2000



def capture_file_result(service, arguments, action):
    spec = {"route": "native", "workspace": str(service.workspace), "session_id": "session-one",
            "name": "calculate_physchem_from_fasta", "arguments": arguments}
    worker = AnalysisCapture(spec, service.workspace, "sequence_analysis")
    value = action()
    evidence = worker.finish({"value": value, "is_error": False})
    assert evidence is not None
    job = {"id": format(len(service.receipts) + 1, "032x"), "status": "completed", "tool": spec["name"],
           "category": "sequence_analysis", "parent_job": "b" * 32, "session_id": "session-one",
           "arguments_sha256": value_digest(arguments),
           "result": {"value": value, "is_error": False, "analysis_evidence": evidence}}
    capture(service, job=job, name=spec["name"], arguments=arguments, value=value,
            owner="owner-one", parent_job=job["parent_job"], session_id=job["session_id"])
    return value


def file_results(service, outputs, *, inputs=None):
    arguments = {"sequence": SEQUENCE, "output_file": [str(service.workspace / name) for name in outputs]}
    arguments.update(inputs or {})

    def action():
        for name, data in outputs.items():
            path = service.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {"status": "success", "content_preview": "not an authoritative measurement"}

    return capture_file_result(service, arguments, action)


def test_real_fasta_file_wrapper_projects_complete_bound_output(setup):
    from mcp_tools.tools.predict.features.features_operations import calculate_physchem_from_fasta

    service, inputs = setup
    arguments = {"fasta_file": str(service.workspace / "control.fasta"),
                 "output_file": str(service.workspace / "physchem.json")}
    # Exercise the actual CPU calculation and public response shape; policy admission is tested separately.
    operation = calculate_physchem_from_fasta.__wrapped__
    response = json.loads(capture_file_result(service, arguments, lambda: operation(**arguments)))
    assert response["status"] == "success" and "data" not in response
    raw = (service.workspace / "physchem.json").read_bytes()
    expected = json.loads(raw)
    assert len(raw) > len(response["content_preview"])
    computation = build(service, inputs)["computations"][0]
    fields = {row["pointer"]: row for row in computation["result_fields"]}
    assert fields["/output_files/physchem.json/molecular_weight"]["value"] == expected["molecular_weight"]
    assert fields["/output_files/physchem.json/amino_acid_composition/Y"]["value"] == expected["amino_acid_composition"]["Y"]
    assert not any("content_preview" in name for name in fields)
    assert any(row["reason"] == "preview_not_measurement" for row in computation["result_omissions"])
    receipt = computation["output_results"]["records"][0]
    assert receipt["status"] == "verified" and receipt["sha256"] == sha(raw)
    assert receipt["bytes"] == len(raw) and receipt["complete"] is True


def test_changed_bound_json_is_not_available_for_claims(setup):
    service, inputs = setup
    file_results(service, {"result.json": b'{"score":1.5}'})
    (service.workspace / "result.json").write_bytes(b'{"score":999}')
    row = build(service, inputs)["computations"][0]
    assert row["status"] == "unverified" and "result_fields" not in row


def test_output_read_rechecks_bound_hash_after_receipt_validation(setup, monkeypatch):
    service, inputs = setup
    file_results(service, {"result.json": b'{"score":1.5}'})
    read = provenance.read_bytes

    def replaced(workspace, path, **kwargs):
        return b'{"score":999}' if path == "result.json" else read(workspace, path, **kwargs)

    monkeypatch.setattr(provenance, "read_bytes", replaced)
    row = build(service, inputs)["computations"][0]
    assert row["output_results"]["records"][0]["status"] == "unknown"
    assert not any(field["pointer"].startswith("/output_files/") for field in row["result_fields"])


def test_json_output_copied_from_registered_input_does_not_expose_labels(setup, monkeypatch):
    service, inputs = setup
    raw = b'{"held_out_labels":[123456789,987654321]}'
    (service.workspace / "private-table.json").write_bytes(raw)
    inputs.append({"path": "private-table.json", "sha256": sha(raw)})
    # Even material summaries omitted from SC retain their protection in the host.
    monkeypatch.setattr(provenance, "MAX_MATERIALS", 1)
    file_results(service, {"copy.json": raw})
    index = build(service, inputs)
    row = index["computations"][0]
    assert row["output_results"]["records"][0]["reason"] == "input_copy_not_exposed"
    assert "123456789" not in json.dumps(index) and "held_out_labels" not in json.dumps(index)


def test_json_output_copied_from_task_input_does_not_expose_labels(setup):
    service, inputs = setup
    raw = b'{"held_out_labels":[123456789]}'
    (service.workspace / "task-input.json").write_bytes(raw)
    file_results(service, {"copy.json": raw}, inputs={"input_file": str(service.workspace / "task-input.json")})
    index = build(service, inputs)
    assert index["computations"][0]["output_results"]["records"][0]["reason"] == "input_copy_not_exposed"
    assert "123456789" not in json.dumps(index)


def test_output_bytes_are_shared_across_all_computations(setup, monkeypatch):
    service, inputs = setup
    raw = b'{"score":1.5}'
    monkeypatch.setattr(provenance, "MAX_OUTPUT_BYTES", len(raw))
    file_results(service, {"one.json": raw})
    file_results(service, {"two.json": raw})
    first, second = build(service, inputs)["computations"]
    assert first["output_results"]["records"][0]["status"] == "verified"
    assert second["output_results"]["records"][0]["reason"] == "output_byte_limit"
    assert not any(field["pointer"].startswith("/output_files/") for field in second["result_fields"])


def test_oversized_json_output_is_explicitly_unknown(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_OUTPUT_BYTES", 5)
    file_results(service, {"result.json": b'{"score":123456789}'})
    row = build(service, inputs)["computations"][0]
    assert row["output_results"]["records"][0]["reason"] == "output_unavailable_invalid_or_too_large"
    assert row["result_complete"] is False
    assert "123456789" not in json.dumps(row)


def test_output_fields_share_limit_with_returned_result(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_FIELDS", 3)
    file_results(service, {"first.json": b'{"a":1,"b":2,"c":3}', "last.json": b'{"d":4}'})
    row = build(service, inputs)["computations"][0]
    assert len(row["result_fields"]) == 3  # status + two measurements, across all roots.
    assert row["result_fields"][-1]["pointer"] == "/output_files/first.json/b"
    assert row["output_results"]["records"][-1]["reason"] == "field_limit"


def test_output_file_limit_is_explicit(setup, monkeypatch):
    service, inputs = setup
    monkeypatch.setattr(provenance, "MAX_OUTPUT_FILES", 1)
    file_results(service, {"first.json": b'{"a":1}', "last.json": b'{"b":2}'})
    row = build(service, inputs)["computations"][0]
    assert len(row["output_results"]["records"]) == 1
    assert {"reason": "output_file_limit", "count": 1} in row["output_results"]["omissions"]


@pytest.mark.parametrize("name,data,reason", [
    ("result.csv", b"score\n1.5\n", "unsupported_output_format"),
    ("result.json", b'{"score":1,"score":2}', "output_unavailable_invalid_or_too_large"),
    ("result.json", b'{"score":NaN}', "output_unavailable_invalid_or_too_large"),
    ("result.json", b'not json', "output_unavailable_invalid_or_too_large"),
])
def test_unsupported_or_ambiguous_outputs_do_not_fabricate_values(setup, name, data, reason):
    service, inputs = setup
    file_results(service, {name: data})
    row = build(service, inputs)["computations"][0]
    assert row["output_results"]["records"][0]["reason"] == reason
    assert not any(field["pointer"].startswith("/output_files/") for field in row["result_fields"])


def test_output_pointer_escaping_and_sensitive_fields(setup):
    service, inputs = setup
    file_results(service, {"nested/a~b.json": b'{"a/b":1.5,"api_key":"opaque-secret-field"}'})
    row = build(service, inputs)["computations"][0]
    fields = {item["pointer"]: item for item in row["result_fields"]}
    assert fields["/output_files/nested~1a~0b.json/a~1b"]["value"] == 1.5
    assert "opaque-secret-field" not in json.dumps(row)
    assert any(item["reason"] == "sensitive_field_redacted" for item in row["result_omissions"])
