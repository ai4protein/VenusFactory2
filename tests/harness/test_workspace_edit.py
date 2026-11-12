"""Version-checked edits share the same session boundary as other file tools."""
import hashlib
import multiprocessing
from contextlib import contextmanager

import pytest

from agent.harness.research_workspace import ResearchWorkspace
from utils import scoped_files


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.asyncio
async def test_edit_changes_only_selected_text_without_executing_source(tmp_path):
    workspace = tmp_path / "work"
    service = ResearchWorkspace(workspace, tmp_path / "control", {"steps": []})
    raw = "# 蛋白质分析\r\nvalues = [1, 2]]\r\nraise RuntimeError('must not execute')\r\n".encode()
    scoped_files.write_bytes(workspace, "src/analysis.py", raw)
    read = await service.read_research_file("src/analysis.py")
    result = await service.edit_research_file("src/analysis.py", "values = [1, 2]]", "values = [1, 2]", read["sha256"])
    changed = raw.replace(b"[1, 2]]", b"[1, 2]")
    assert (workspace / "src/analysis.py").read_bytes() == changed
    assert result == {"path": "src/analysis.py", "before_sha256": sha(raw),
                      "sha256": sha(changed), "bytes": len(changed), "replacements": 1}
    assert service.receipts[-1]["tool"] == "edit_research_file"
    assert "old_text" not in service.receipts[-1]  # Content stays in the scoped file, not duplicated in receipts.


@pytest.mark.asyncio
async def test_paged_read_supplies_file_hash_without_exposing_all_source(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    raw = ("# padding\n" * 4000 + "target = 1\n").encode()
    scoped_files.write_bytes(service.workspace, "long.py", raw)
    read = await service.read_research_file("long.py", offset_bytes=len(raw)-256, max_bytes=256)
    assert read["truncated"] and len(read["text"]) <= 256
    assert read["file_sha256"] == sha(raw) and read["slice_sha256"] != read["file_sha256"]
    with pytest.raises(ValueError, match="changed"):
        await service.edit_research_file("long.py", "target = 1", "target = 2", read["slice_sha256"])
    await service.edit_research_file("long.py", "target = 1", "target = 2", read["file_sha256"])
    assert (service.workspace / "long.py").read_bytes() == raw.replace(b"target = 1", b"target = 2")


@pytest.mark.parametrize("raw,old,new,error", [
    (b"aaa", "aa", "b", "ambiguous"),
    (b"target target", "target", "other", "ambiguous"),
    (b"target", "missing", "other", "not found"),
    (b"target", "target", "target", "unchanged"),
    (b"target", "", "other", "nonempty"),
    (b"\xfftarget", "target", "other", "UTF-8"),
    (b"\x00target", "target", "other", "UTF-8"),
    (b"target", "target", "x"*200001, "too large"),
])
def test_invalid_edits_leave_original_intact(tmp_path, raw, old, new, error):
    scoped_files.write_bytes(tmp_path, "model.py", raw)
    with pytest.raises(ValueError, match=error):
        scoped_files.replace_text(tmp_path, "model.py", old, new, sha(raw))
    assert (tmp_path / "model.py").read_bytes() == raw
    assert not list(tmp_path.glob(".write-*"))


@pytest.mark.parametrize("name", ["../outside", "/etc/passwd", ".env", "nested/.secret", "linked", "nested-link/file", "hardlink", "pipe"])
def test_edit_refuses_paths_links_and_special_files(tmp_path, name):
    import os
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("original")
    (root / "linked").symlink_to(outside)
    (root / "nested-link").symlink_to(tmp_path, target_is_directory=True)
    os.link(outside, root / "hardlink")
    os.mkfifo(root / "pipe")
    with pytest.raises((ValueError, OSError)):
        scoped_files.replace_text(root, name, "original", "modified", sha(b"original"))
    assert outside.read_text() == "original"


def test_interleaved_whole_file_write_is_not_overwritten(tmp_path, monkeypatch):
    scoped_files.write_bytes(tmp_path, "model.py", b"original")
    original_writer = scoped_files.atomic_writer

    @contextmanager
    def other_writer_first(root, relative, **kwargs):
        if kwargs.get("expected_sha256"):
            scoped_files.write_bytes(root, relative, b"other writer")
        with original_writer(root, relative, **kwargs) as stream:
            yield stream

    monkeypatch.setattr(scoped_files, "atomic_writer", other_writer_first)
    with pytest.raises(ValueError, match="changed"):
        scoped_files.replace_text(tmp_path, "model.py", "original", "edited", sha(b"original"))
    assert (tmp_path / "model.py").read_bytes() == b"other writer"


def concurrent_edit(root, barrier, results, replacement):
    writer = scoped_files.atomic_writer
    def ready_writer(*args, **kwargs):
        barrier.wait(timeout=20)  # Both processes have read the original version.
        return writer(*args, **kwargs)
    scoped_files.atomic_writer = ready_writer
    try:
        scoped_files.replace_text(root, "model.py", "original", replacement, sha(b"original"))
    except ValueError as error:
        results.put(("rejected", str(error)))
    else:
        results.put(("written", replacement))


def test_two_worker_edits_commit_only_one_version(tmp_path):
    scoped_files.write_bytes(tmp_path, "model.py", b"original")
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    workers = [context.Process(target=concurrent_edit, args=(str(tmp_path), barrier, results, text))
               for text in ("first", "second")]
    for worker in workers:
        worker.start()
    try:
        outcomes = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        results.close()
    assert sorted(row[0] for row in outcomes) == ["rejected", "written"]
    assert "changed" in next(row[1] for row in outcomes if row[0] == "rejected")
    assert (tmp_path / "model.py").read_text() == next(row[1] for row in outcomes if row[0] == "written")
    assert not list(tmp_path.glob(".write-*"))


@pytest.mark.asyncio
async def test_failed_training_can_be_edited_and_refitted_without_rewriting_history(tmp_path):
    from test_research_training import service_at
    service = await service_at(tmp_path)
    broken = "def fit(training, seed):\n    targets = [float(row['target']) for row in training['train']]]\n    return {'mean': sum(targets)/len(targets)}\ndef predict(model, inputs):\n    return {'value': model['mean']}\n"
    await service.write_research_file("src/mean.py", broken)
    with pytest.raises(ValueError, match="unmatched") as failed:
        await service.train_research_model("Mean fixture", "src/mean.py", "evaluation.json", "Software repair test only")
    assert "targets = " in str(failed.value) and "^" in str(failed.value)
    receipts = list((service.development.root / "training").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt_bytes = receipts[0].read_bytes()
    snapshot = receipts[0].parent / "algorithm.py"
    assert snapshot.read_text() == broken
    read = await service.read_research_file("src/mean.py")
    await service.edit_research_file("src/mean.py", "training['train']]]", "training['train']]", read["sha256"])
    await service.train_research_model("Mean fixture", "src/mean.py", "evaluation.json", "Software repair test only")
    record = service.development.records()[0]
    assert record["fresh_process_replay"] and record["test_used_for_feedback"] is False
    assert record["variant"] == "baseline"
    assert receipts[0].read_bytes() == receipt_bytes and snapshot.read_text() == broken
    assert not service.proposals
