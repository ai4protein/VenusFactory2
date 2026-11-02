"""Scientific input units and independently checked CataPro cache/replay."""

import csv
import sqlite3

import numpy as np
import pytest

from agent.rsi.cases.catapro_case import FeatureCache, dataset_rows, log_targets, normalize_protein, numpy_forward


def test_units_follow_released_output_headers():
    values = log_targets({"kcat(s^-1)": "10", "Km(M)": "0.001"})
    assert values == {"kcat": 1.0, "km": 0.0, "activity": 1.0}
    with pytest.raises(ValueError, match="positive"):
        log_targets({"Km(M)": "0"})


def test_author_terminal_truncation_and_ambiguous_residues():
    sequence = "U" * 500 + "A" * 500 + "B" * 500
    assert normalize_protein(sequence) == "X" * 1000
    assert normalize_protein("ACUZOB") == "ACXXXX"


def test_feature_cache_binds_encoder_contract_and_detects_corruption(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = FeatureCache(path, {"encoder": "first"})
    cache.put("protein", "ABC", np.array([1.0, 2.0]))
    np.testing.assert_array_equal(cache.get("protein", "ABC", 2), [1.0, 2.0])
    assert cache.get("protein", "ABD", 2) is None
    cache.close()
    with pytest.raises(ValueError, match="contract"):
        FeatureCache(path, {"encoder": "other"})
    with sqlite3.connect(path) as db:
        db.execute("UPDATE vectors SET data=?", (np.array([3.0, 4.0], dtype="<f4").tobytes(),))
    cache = FeatureCache(path, {"encoder": "first"})
    with pytest.raises(ValueError, match="integrity"):
        cache.get("protein", "ABC", 2)
    cache.close()


def test_author_fold_is_not_silently_rounded(tmp_path):
    (tmp_path / "datasets").mkdir()
    path = tmp_path / "datasets/kcat-data_0.4simi-10fold.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "Sequence", "Smiles", "fold", "kcat(s^-1)"])
        writer.writeheader()
        writer.writerow({"": "one", "Sequence": "ACD", "Smiles": "CCO", "fold": "1.5", "kcat(s^-1)": "1"})
    with pytest.raises(ValueError, match="fold"):
        dataset_rows(tmp_path, "kcat")


@pytest.mark.parametrize("activity", [False, True])
def test_numpy_replay_matches_torch_eval_arithmetic(activity):
    torch = pytest.importorskip("torch")
    from torch.nn import functional as f

    rng = np.random.default_rng(77)
    state = {}

    def normal(prefix, n):
        state[prefix + ".running_mean"] = rng.normal(0, 0.1, n).astype("f4")
        state[prefix + ".running_var"] = rng.uniform(0.8, 1.2, n).astype("f4")
        state[prefix + ".weight"] = rng.uniform(0.8, 1.2, n).astype("f4")
        state[prefix + ".bias"] = rng.normal(0, 0.1, n).astype("f4")

    def linear(prefix, n, m):
        state[prefix + ".weight"] = rng.normal(0, 0.01, (m, n)).astype("f4")
        state[prefix + ".bias"] = rng.normal(0, 0.01, m).astype("f4")

    def component(prefix):
        normal(prefix + "prot_norm", 1024)
        normal(prefix + "molt5_norm", 768)
        linear(prefix + "decoder.0", 1959, 256)
        normal(prefix + "decoder.1", 256)

    component("")
    if activity:
        for prefix in ["kcat_model.", "Km_model."]:
            component(prefix)
            linear(prefix + "out.0", 256, 1)
        linear("attn.0", 256, 256)
        linear("out", 256, 1)
    else:
        linear("out.0", 256, 1)
    data = rng.normal(0, 1, (7, 1959)).astype("f4")
    ts = {k: torch.from_numpy(v) for k, v in state.items()}
    x = torch.from_numpy(data)

    def bn(value, prefix):
        return f.batch_norm(
            value,
            ts[prefix + ".running_mean"],
            ts[prefix + ".running_var"],
            ts[prefix + ".weight"],
            ts[prefix + ".bias"],
            training=False,
        )

    def lin(value, prefix):
        return f.linear(value, ts[prefix + ".weight"], ts[prefix + ".bias"])

    def hidden(prefix):
        features = torch.cat(
            [bn(x[:, :1024], prefix + "prot_norm"), bn(x[:, 1024:1792], prefix + "molt5_norm"), x[:, 1792:]], dim=1
        )
        return torch.relu(bn(lin(features, prefix + "decoder.0"), prefix + "decoder.1"))

    h = hidden("")
    if activity:
        expected = 0.6 * (
            lin(hidden("kcat_model."), "kcat_model.out.0") - lin(hidden("Km_model."), "Km_model.out.0")
        ) + 0.4 * lin(torch.softmax(lin(h, "attn.0"), dim=1) * h, "out")
    else:
        expected = lin(h, "out.0")
    np.testing.assert_allclose(
        numpy_forward(data, state, activity=activity), expected.numpy().ravel(), atol=2e-7, rtol=1e-5
    )
