import pytest
import torch

from agent.rsi import clean_training_inputs as c


def rows():
    return [
        {"Entry": "A", "EC number": "1.1.1.1", "Sequence": "ACDEF"},
        {"Entry": "B", "EC number": "2.2.2.2", "Sequence": "GGG"},
        {"Entry": "C", "EC number": "2.2.2.2", "Sequence": "GGH"},
    ]


def test_orphan_mask_coverage_and_token_length():
    augmented = {f"A_{i}": "AC<mask>EF" for i in range(10)}
    result = c.validate_augmentation(rows(), augmented)
    assert c.orphan_ids(rows()) == {"A"}
    assert len(result) == 10 and all(x["tokens"] == 5 and x["parent"] == "A" for x in result)
    augmented.pop("A_9")
    with pytest.raises(ValueError, match="cover"):
        c.validate_augmentation(rows(), augmented)


@pytest.mark.parametrize("sequence", ["<mask>CDEF", "ACDEG", "AC<mask>F", "AC<invalid>EF"])
def test_augmentation_cannot_change_author_scope(sequence):
    with pytest.raises(ValueError, match="outside"):
        c.validate_augmentation(rows(), {f"A_{i}": sequence for i in range(10)})


def test_repeated_fasta_identity_rejected(tmp_path):
    path = tmp_path / "x.fasta"
    path.write_text(">A_0\nACD\n>A_0\nACD\n")
    with pytest.raises(ValueError, match="Repeated"):
        c.read_fasta(path)


def test_receipt_binds_input_encoder_protocol_and_tensor(tmp_path):
    work, receipts = tmp_path / "work", tmp_path / "receipts"
    (work / "data/esm_data").mkdir(parents=True)
    receipts.mkdir()
    row = {"key": "A", "sequence": "ACD", "tokens": 3}
    path = work / "data/esm_data/A.pt"
    torch.save(torch.arange(1280, dtype=torch.float32), path)
    c.save(receipts / "A.json", c.receipt_payload(row, path, "encoder", "protocol"))
    assert c.verify_receipt(work, receipts, row, "encoder", "protocol").shape == (1280,)
    for altered, encoder, protocol in [
        (dict(row, sequence="AGD"), "encoder", "protocol"),
        (row, "other", "protocol"),
        (row, "encoder", "other"),
    ]:
        with pytest.raises(ValueError, match="binding"):
            c.verify_receipt(work, receipts, altered, encoder, protocol)
    torch.save(torch.zeros(1280), path)
    with pytest.raises(ValueError, match="binding"):
        c.verify_receipt(work, receipts, row, "encoder", "protocol")


@pytest.mark.parametrize(
    "tensor", [torch.zeros(1279), torch.zeros(1280, dtype=torch.float64), torch.full((1280,), float("nan"))]
)
def test_rehashed_invalid_tensors_still_fail(tmp_path, tensor):
    work, receipts = tmp_path / "work", tmp_path / "receipts"
    (work / "data/esm_data").mkdir(parents=True)
    receipts.mkdir()
    row = {"key": "A", "sequence": "ACD", "tokens": 3}
    path = work / "data/esm_data/A.pt"
    torch.save(tensor, path)
    c.save(receipts / "A.json", c.receipt_payload(row, path, "e", "p"))
    with pytest.raises(ValueError, match="Invalid embedding"):
        c.verify_receipt(work, receipts, row, "e", "p")


def test_symlink_and_parent_path_rejected(tmp_path):
    target = tmp_path / "actual"
    target.write_text("record")
    (tmp_path / "alias").symlink_to(target)
    for name in ["alias", "../actual"]:
        with pytest.raises(ValueError):
            c.ordinary(tmp_path, name)


def test_completed_flag_alone_cannot_prove_inputs(tmp_path):
    c.save(tmp_path / "report.json", {"status": "completed", "completed_records": c.TOTAL})
    c.save(tmp_path / "protocol.json", {})
    c.save(tmp_path / "plan.json", {})
    with pytest.raises(ValueError, match="admission"):
        c.verify_completed(tmp_path)


def test_full_plan_rejects_changed_residue_even_if_counts_match(monkeypatch):
    plan = {
        "source_commit": c.COMMIT,
        "asset_manifest_sha256": c.MANIFEST_SHA256,
        "counts": {"train": 1},
        "seed": 1234,
        "records": [{"key": "A", "cohort": "train", "sequence": "ACD", "tokens": 3}],
    }
    monkeypatch.setattr(c, "COUNTS", {"train": 1})
    monkeypatch.setattr(c, "TOTAL", 1)
    monkeypatch.setattr(c, "PLAN_SHA256", c.digest(plan))
    c.validate_plan(plan)
    plan["records"][0]["sequence"] = "AGD"
    with pytest.raises(ValueError, match="Full admitted"):
        c.validate_plan(plan)
