from pathlib import Path

import pytest

from agent.rsi.campaigns.case_plugins import invocation


def call(tmp_path, plugin, parameters, output=None):
    return invocation(
        plugin,
        parameters,
        root=tmp_path,
        output=output or tmp_path / "research_runs/rsi/run",
        controller_python=Path("/python"),
    )


def test_validated_shell_free_recipe(tmp_path):
    argv = call(tmp_path, "rfantibody.v1", {"backbones": 8, "sequences": 4})
    assert argv[-4:] == ["--backbones", "8", "--sequences", "4"]
    with pytest.raises(ValueError):
        call(tmp_path, "rfantibody.v1", {"backbones": "2; arbitrary"})
    with pytest.raises(ValueError):
        call(tmp_path, "rfantibody.v1", {"backbones": True})
    with pytest.raises(ValueError):
        call(tmp_path, "turbopetase.v1", {"models": ["untrusted.pt"]})
    with pytest.raises(ValueError):
        call(tmp_path, "rfantibody.v1", {}, tmp_path / "outside")


def test_symlink_output_escape_rejected(tmp_path):
    root = tmp_path / "repo"
    out = root / "research_runs/rsi/link"
    out.parent.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    out.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        call(root, "rfantibody.v1", {}, out / "case")


def test_new_protocol_pins_sequence_sampling_without_changing_legacy(tmp_path):
    assert "--deterministic-sequences" in call(tmp_path, "rfantibody.v2", {})
    assert "--deterministic-sequences" not in call(tmp_path, "rfantibody.v1", {})


def test_multitarget_case_requires_explicit_registered_antigen(tmp_path):
    argv = call(tmp_path, "rfantibody.targets.v1", {"target": "rsv_site3", "backbones": 8, "sequences": 4})
    assert "run_rfantibody_targets.py" in argv[2]
    assert argv[argv.index("--target") + 1] == "rsv_site3"
    for params in [{}, {"target": "unregistered"}, {"target": "ha", "backbones": True}]:
        with pytest.raises(ValueError):
            call(tmp_path, "rfantibody.targets.v1", params)


def test_alde_cpu_recipes_separate_existing_evidence_from_new_computation(tmp_path):
    from agent.rsi.campaigns.case_plugins import resource_for

    argv = call(
        tmp_path, "alde.onehot_dnn.v1", {"operation": "reproduce", "seeds": 70, "acquisitions": ["GREEDY", "UCB", "TS"]}
    )
    assert argv[-6:] == ["--seeds", "70", "--acquisitions", "GREEDY", "UCB", "TS"]
    assert resource_for("alde.onehot_dnn.v1", {}) == "cpu"
    assert resource_for("turbopetase.v1", {"device": "cpu"}) == "cpu"
    assert resource_for("rfantibody.targets.v1", {"target": "ha"}) == "gpu"
    assert "--campaign" in call(
        tmp_path, "alde.onehot_dnn.v1", {"operation": "verify_existing", "campaign": "author_ts70_20261011"}
    )
    for params in [
        {},
        {"operation": "reproduce", "seeds": True},
        {"operation": "reproduce", "seeds": 71},
        {"operation": "reproduce", "acquisitions": ["TS", "TS"]},
        {"operation": "reproduce", "campaign": "author_ts70_20261011"},
        {"operation": "verify_existing", "campaign": "../../unregistered"},
        {"operation": "verify_existing", "campaign": "author_ts70_20261011", "seeds": 70},
    ]:
        with pytest.raises(ValueError):
            call(tmp_path, "alde.onehot_dnn.v1", params)
