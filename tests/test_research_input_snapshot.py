from research_source_review_fixture import with_source_review
import copy
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
import zipfile
import pytest
from backend.web_v2 import research_report as report
from backend.web_v2.research_dossier import build_dossier
from backend.web_v2.research_input_snapshot import save_inputs, load_inputs


def context():
    return {"sequence": "ACDEFGHIKLMNPQRSTVWY", "structure_pdb": "HEADER TEST\n",
            "protein": {"name": "Supplied protein"}, "protein_label": "Supplied protein",
            "origin": "upload", "structure_chain": "A", "source_chain": "A",
            "sources": [], "entry": None, "accession": "", "pdb_id": ""}


def test_snapshot_is_immutable_content_addressed_and_concurrent(tmp_path):
    folder = tmp_path / "snapshots"
    value = context()
    with ThreadPoolExecutor(max_workers=2) as workers:
        refs = list(workers.map(lambda _: save_inputs(folder, value), range(2)))
    assert refs[0] == refs[1]
    ref = refs[0]
    stored = load_inputs(folder, ref)
    assert stored["sequence"] == value["sequence"] and stored["structure_pdb"] == value["structure_pdb"]
    assert ref["sequence_sha256"] == hashlib.sha256(value["sequence"].encode()).hexdigest()
    assert list(folder.iterdir()) == [folder / (ref["sha256"] + ".json")]
    path = folder / (ref["sha256"] + ".json")
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_inputs(folder, ref)
    with pytest.raises(ValueError, match="changed"):
        save_inputs(folder, value)
    assert path.read_bytes() == b"changed"


def test_snapshot_paths_and_links_cannot_escape_owner_storage(tmp_path):
    folder = tmp_path / "snapshots"
    ref = save_inputs(folder, context())
    with pytest.raises(ValueError):
        load_inputs(folder, {**ref, "sha256": "../private"})
    actual = folder / (ref["sha256"] + ".json")
    actual.unlink()
    private = tmp_path / "private"; private.write_text("private")
    actual.symlink_to(private)
    with pytest.raises(ValueError):
        load_inputs(folder, ref)
    other = tmp_path / "linked"; other.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError):
        save_inputs(other, context())
    assert private.read_text() == "private"


def uniprot_entry():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    return {"primaryAccession": "P00918",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Carbonic anhydrase 2"}}},
            "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
            "sequence": {"value": sequence, "length": len(sequence)},
            "uniProtKBCrossReferences": []}


@pytest.mark.asyncio
async def test_accession_inputs_are_exported_and_reused_without_refetch(monkeypatch, tmp_path):
    entry = uniprot_entry(); queries = []
    async def fetch(url, *args, **kwargs):
        queries.append(url)
        return copy.deepcopy(entry)
    monkeypatch.setattr(report, "get_json", fetch)
    payload = {"accession": "P00918", "model_id": "fixture", "analyses": ["properties"], "lang": "en"}
    folder = tmp_path / "inputs"
    first = await report.run_research(payload, lambda state: None, synthesize=False, input_store=folder)
    ref = first["input_snapshot"]
    assert ref["origin"] == "uniprot" and len(queries) == 1
    restored = load_inputs(folder, ref)
    assert restored["entry"] == entry
    async def forbidden(*args, **kwargs):
        raise AssertionError("A saved target must not be fetched again")
    monkeypatch.setattr(report, "get_json", forbidden)
    second = await report.run_research(payload, lambda state: None, synthesize=False,
                                       input_store=folder, input_snapshot=ref)
    assert second["input_snapshot"] == ref and second["sections"] == first["sections"]
    assert second["sources"][0] == first["sources"][0]
    with build_dossier({**first, "report": "# Measured properties", "id": "fixture"}, restored) as handle:
        with zipfile.ZipFile(handle) as archive:
            assert archive.read("inputs/protein.fasta").decode() == ">query\n" + entry["sequence"]["value"] + "\n"
            assert json.loads(archive.read("inputs/reference-uniprot.json")) == entry
            assert json.loads(archive.read("inputs/metadata.json"))["sha256"] == ref["sha256"]


@pytest.mark.asyncio
async def test_conflicting_targets_are_rejected_before_lookup(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Ambiguous input must not reach a database")
    monkeypatch.setattr(report, "get_json", forbidden)
    with pytest.raises(ValueError, match="Choose one"):
        await report.run_research({"accession": "P00918", "sequence": "MKT", "analyses": ["properties"]},
                                  lambda state: None, synthesize=False)


@pytest.mark.asyncio
async def test_collection_checkpoint_reuses_exact_target_before_writing(monkeypatch, tmp_path):
    from backend.web_v2 import research_longform as longform
    class WorkerStopped(BaseException):
        pass
    async def fetch(*args, **kwargs):
        return uniprot_entry()
    monkeypatch.setattr(report, "get_json", fetch)
    payload = {"accession": "P00918", "model_id": "fixture", "analyses": ["properties"], "lang": "en"}
    saved = {}
    folder = tmp_path / "inputs"
    def interrupt(state):
        if state.get("input_snapshot"):
            saved.update(copy.deepcopy(state))
            raise WorkerStopped()
    with pytest.raises(WorkerStopped):
        await longform.run_long_report(payload, interrupt, input_store=folder)
    assert saved["phase"] == "collecting" and saved["model_calls"] == 0
    original = copy.deepcopy(saved)
    async def forbidden(*args, **kwargs):
        raise AssertionError("Resuming a saved target must not refetch it")
    monkeypatch.setattr(report, "get_json", forbidden)
    calls = []
    @with_source_review
    async def writer(*args, **kwargs):
        calls.append(args)
        return ("Protein evidence " * 210 + "[S1]", False)
    monkeypatch.setattr(report, "complete", writer)
    result = await longform.run_long_report(payload, lambda _: None, checkpoint=saved, input_store=folder)
    assert result["input_snapshot"] == saved["input_snapshot"]
    assert result["completed_chapters"] == 8 and result["model_calls"] == 12
    assert len(calls) == 9 and saved == original
    assert result["sources"][0] == saved["sources"][0]
    path = folder / (saved["input_snapshot"]["sha256"] + ".json")
    path.unlink()
    calls.clear()
    with pytest.raises(ValueError):
        await longform.run_long_report(payload, lambda _: None, checkpoint=saved, input_store=folder)
    assert not calls and saved == original


def test_ambiguous_target_never_uses_quota_or_enters_queue(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from backend.web_v2 import research_jobs
    from backend.web_v2.job_queue import JobQueue
    monkeypatch.setattr(research_jobs, "model_option", lambda _: {"id": "fixture"})
    service = research_jobs.ResearchJobs(tmp_path / "reports", JobQueue(tmp_path / "queue"))
    start, admit = Mock(), Mock()
    monkeypatch.setattr(service, "_start", start)
    with pytest.raises(ValueError, match="Choose one"):
        service.start("owner", {"accession": "P00918", "sequence": "ACDEFGHIKLMNPQRSTVWY",
                                "model_id": "fixture", "analyses": ["properties"]}, admit=admit)
    start.assert_not_called()
    admit.assert_not_called()


@pytest.mark.asyncio
async def test_missing_reference_annotation_remains_partial_after_restore(monkeypatch, tmp_path):
    from mcp_tools.tools.database.rcsb import research_target
    card = {"name": "Hemoglobin", "organism": "Homo sapiens", "accession": "P69905",
            "pdb_id": "4HHB", "url": "https://www.rcsb.org/structure/4HHB"}
    async def target(*args):
        return {"protein": card, "sequence": "ACDEFGHIKLMNPQRSTVWY", "structure_pdb": "HEADER TEST",
                "structure_chain": "A", "source_chain": "A", "interpretation": "Observed chain"}
    async def unavailable(*args):
        raise ValueError("Reference annotation unavailable")
    monkeypatch.setattr(research_target, "load_pdb_target", target)
    monkeypatch.setattr(report, "get_json", unavailable)
    payload = {"pdb_id": "4HHB", "structure_chain": "A", "model_id": "fixture", "analyses": ["properties"]}
    folder = tmp_path / "inputs"
    first = await report.run_research(payload, lambda _: None, synthesize=False, input_store=folder)
    assert first["partial"] and first["input_snapshot"]["origin"] == "pdb"
    async def forbidden(*args):
        raise AssertionError("Restored coordinates cannot be replaced")
    monkeypatch.setattr(research_target, "load_pdb_target", forbidden)
    second = await report.run_research(payload, lambda _: None, synthesize=False,
                                      input_store=folder, input_snapshot=first["input_snapshot"])
    assert second["partial"] and second["input_snapshot"] == first["input_snapshot"]
    assert second["sections"] == first["sections"]
