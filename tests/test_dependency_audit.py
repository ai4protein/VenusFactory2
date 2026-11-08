"""The audit must not turn incomplete scans or new advisories into a pass."""
from datetime import date
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest

spec = importlib.util.spec_from_file_location(
    "dependency_audit", Path(__file__).resolve().parents[1] / "scripts/audit_dependencies.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ID = "GHSA-abcd-efgh-jkmn"
TODAY = date(2026, 11, 8)


def baseline(**changes):
    entry = {
        "name": "torch", "version": "2.8.0", "ids": [ID], "profiles": ["linux"],
        "reason": "Awaiting ABI migration", "evidence": "docs/security/review.md",
        "reviewed": "2026-11-08", "review_by": "2026-12-08",
    }
    entry.update(changes)
    return entry


def inventory(tmp_path, entries):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"schema": 1, "entries": entries}))
    return path


def test_exact_pins_normalize_names_and_collapse_duplicates(tmp_path):
    path = tmp_path / "resolved.txt"
    path.write_text("# uv output\nMy_Package==1.2.3\nmy-package==1.2.3 # duplicate\n")
    assert audit.read_pins(path) == [("my-package", "1.2.3")]


@pytest.mark.parametrize("text", [
    "", "# no dependencies\n", "torch>=2.8", "torch==2.*",
    "torch @ https://example.org/torch.whl", "--index-url https://example.org/",
    "torch==not-a-version", "torch==2..8", "torch==2.8.0-nightly",
    "torch==2.8; python_version > '3.10'", "torch==2.8\ntorch==2.9",
])
def test_unresolved_or_ambiguous_inputs_cannot_pass(tmp_path, text):
    path = tmp_path / "resolved.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        audit.read_pins(path)


def test_all_pages_are_audited_without_requerying_completed_packages():
    calls = []
    def send(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"results": [
                {"vulns": [{"id": ID}], "next_page_token": "second"},
                {},
            ]}
        assert payload["queries"] == [{
            "package": {"name": "torch", "ecosystem": "PyPI"},
            "version": "2.8.0", "page_token": "second",
        }]
        return {"results": [{"vulns": [{"id": "PYSEC-2026-1"}]}]}
    result = audit.query([("torch", "2.8.0"), ("other", "1.0")], send)
    assert result[("torch", "2.8.0")] == {ID, "PYSEC-2026-1"}
    assert result[("other", "1.0")] == set()


@pytest.mark.parametrize("reply", [
    {}, {"results": []}, {"results": [None]}, {"results": [{"error": "failed"}]},
    {"results": [{"vulns": None}]}, {"results": [{"vulns": [{}]}]},
    {"results": [{"vulns": [{"id": "::error::spoof"}]}]},
])
def test_incomplete_or_untrusted_response_is_not_a_clean_scan(reply):
    with pytest.raises(ValueError):
        audit.query([("torch", "2.8.0")], lambda _: reply)


def test_pagination_must_advance():
    with pytest.raises(ValueError, match="pagination"):
        audit.query([("torch", "2.8.0")],
                    lambda _: {"results": [{"next_page_token": "repeated"}]})


def test_batching_keeps_every_package():
    batches = []
    def send(payload):
        batches.append(len(payload["queries"]))
        return {"results": [{} for _ in payload["queries"]]}
    pins = [(f"package-{i}", "1.0") for i in range(205)]
    assert len(audit.query(pins, send)) == 205
    assert batches == [100, 100, 5]


@pytest.mark.parametrize("changes", [
    {"version": "*"}, {"ids": ["*"]}, {"reason": ""}, {"evidence": ""},
    {"review_by": "2030-01-01"}, {"review_by": "2026-11-08"}, {"profiles": ["*"]},
])
def test_inventory_has_no_wildcards_or_indefinite_exemptions(tmp_path, changes):
    with pytest.raises(ValueError):
        audit.read_baseline(inventory(tmp_path, [baseline(**changes)]))


def test_duplicate_inventory_is_invalid(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        audit.read_baseline(inventory(tmp_path, [baseline(), baseline()]))


def test_recorded_risks_remain_in_report():
    report = audit.classify({("torch", "2.8.0"): {ID}}, [baseline()], "linux", TODAY)
    assert report["status"] == "known_open"
    assert report["findings"][0]["state"] == "known_open"
    assert ID in audit.render(report)
    assert "not the GitHub alert count" in audit.render(report)


@pytest.mark.parametrize("name,version,identifier,profile", [
    ("another", "2.8.0", ID, "linux"), ("torch", "2.9.0", ID, "linux"),
    ("torch", "2.8.0", "PYSEC-2026-2", "linux"), ("torch", "2.8.0", ID, "macos"),
])
def test_inventory_never_covers_another_package_version_advisory_or_profile(name, version, identifier, profile):
    report = audit.classify({(name, version): {identifier}}, [baseline()], profile, TODAY)
    assert report["status"] == "action_required"
    assert report["new_count"] == 1


def test_review_deadline_blocks_only_still_affected_dependencies():
    affected = audit.classify({("torch", "2.8.0"): {ID}}, [baseline()], "linux", date(2026, 12, 8))
    assert affected["status"] == "action_required"
    assert affected["overdue_count"] == 1
    patched = audit.classify({("torch", "2.13.0"): set()}, [baseline()], "linux", date(2026, 12, 8))
    assert patched["status"] == "clear"


def test_network_failure_produces_incomplete_receipt(tmp_path, monkeypatch):
    pins = tmp_path / "resolved.txt"
    pins.write_text("torch==2.8.0")
    output = tmp_path / "result.json"
    def unavailable(_):
        raise URLError("temporarily unavailable")
    monkeypatch.setattr(audit, "query", unavailable)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert audit.main([str(pins), "--profile", "linux", "--baseline",
                       str(inventory(tmp_path, [])), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "incomplete"


def test_strict_mode_fails_on_recorded_risks_and_ci_warns(tmp_path, monkeypatch, capsys):
    pins = tmp_path / "resolved.txt"
    pins.write_text("torch==2.8.0")
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(audit, "query", lambda _: {("torch", "2.8.0"): {ID}})
    monkeypatch.setattr(audit, "date", type("Clock", (), {
        "today": staticmethod(lambda: TODAY), "fromisoformat": staticmethod(date.fromisoformat),
    }))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = [str(pins), "--profile", "linux", "--baseline", str(inventory(tmp_path, [baseline()]))]
    assert audit.main(args) == 0
    assert "::warning" in capsys.readouterr().out
    assert ID in summary.read_text()
    assert audit.main([*args, "--strict"]) == 1
