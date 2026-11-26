import numpy as np
import pytest
import torch

from agent.rsi import clean_annotation as case


def test_numpy_network_replays_layernorm_and_projection():
    torch.manual_seed(42)
    layers = {}
    for i, (a, b) in enumerate([(1280, 512), (512, 512), (512, 128)], 1):
        layer = torch.nn.Linear(a, b)
        layers[f"fc{i}.weight"] = layer.weight.detach()
        layers[f"fc{i}.bias"] = layer.bias.detach()
        if i < 3:
            layers[f"ln{i}.weight"] = torch.randn(b)
            layers[f"ln{i}.bias"] = torch.randn(b)
    x = torch.randn(4, 1280)
    h = x
    for i in [1, 2, 3]:
        h = torch.nn.functional.linear(h, layers[f"fc{i}.weight"], layers[f"fc{i}.bias"])
        if i < 3:
            h = torch.nn.functional.layer_norm(h, (512,), layers[f"ln{i}.weight"], layers[f"ln{i}.bias"])
            h = torch.relu(h)
    replay = case.forward({k: v.numpy() for k, v in layers.items()}, x.numpy())
    np.testing.assert_allclose(replay, h.numpy(), atol=3e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="network state"):
        case.forward({}, x.numpy())


def test_ec_centers_preserve_csv_order_and_membership():
    rows = [{"Entry": "a", "EC number": "2.1.1.1;1.1.1.1"}, {"Entry": "b", "EC number": "1.1.1.1"}]
    index = case.class_index(rows)
    assert list(index) == ["2.1.1.1", "1.1.1.1"]
    x = np.tile(np.array([1.0, 3.0, 5.0])[:, None], (1, 128))
    np.testing.assert_array_equal(case.centers(x, index)[:, 0], [1.0, 4.0])
    with pytest.raises(ValueError, match="memberships"):
        case.centers(x[:2], index)


def test_label_selection_and_equal_distance_failure():
    labels = [str(i) for i in range(10)]
    d = np.array([[1.0, 1.2, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7]])
    assert case.select_labels(d, labels) == [["0", "1"]]
    with pytest.raises(ValueError, match="undefined"):
        case.select_labels(np.ones((1, 10)), labels)


def test_union_metrics_retain_out_of_cohort_false_positives():
    got = case.metrics([["A"], ["B"]], [["A", "X"], ["B"]])
    assert got["micro_precision"] == pytest.approx(2 / 3)
    assert got["micro_f1"] == pytest.approx(0.8)
    assert got["exact_match"] == 0.5
    assert got["predicted_labels_absent_from_truth"] == ["X"]
    assert got["weighted_f1"] == 1.0
    with pytest.raises(ValueError, match="incomplete"):
        case.metrics([["A"]], [])


def test_prediction_identity_and_artifact_paths(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("a,EC:1.1.1.1/2.0000\nb,EC:2.2.2.2/1.0000\n")
    assert case.read_predictions(p, ["a", "b"])[0] == [["1.1.1.1"], ["2.2.2.2"]]
    with pytest.raises(ValueError, match="identity"):
        case.read_predictions(p, ["b", "a"])
    with pytest.raises(ValueError, match="path"):
        case.ordinary(tmp_path, "../labels.csv")
    q = tmp_path / "link.csv"
    q.symlink_to(p)
    with pytest.raises(ValueError, match="Symlinked"):
        case.ordinary(tmp_path, "link.csv")


def test_cohort_checks_reject_reduced_or_changed_data(tmp_path):
    p = tmp_path / "source/app/data/new.csv"
    p.parent.mkdir(parents=True)
    p.write_text("Entry\tEC number\tSequence\na\t1.1.1.1\tACD\n")
    with pytest.raises(ValueError, match="fixed author"):
        case.dataset(tmp_path, "new")
