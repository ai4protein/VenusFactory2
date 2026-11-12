"""Training summaries and diagnostics do not expose held-out values."""
import json

import pytest

from agent.harness.research_input_profile import training_input_profile
from agent.harness.research_dataset import prepare_dataset, model_interface
from agent.harness.research_model_state import training_script


def test_profile_counts_stored_strings_without_labels_or_values():
    rows = [
        {"inputs": {"signal": value, "sequence": seq}, "target": "SECRET_TARGET"}
        for value, seq in [("1.2", "PRIVATE_SEQUENCE_A"), ("", "PRIVATE_SEQUENCE_B"),
                           ("nan", "PRIVATE_SEQUENCE_A"), ("-4", "PRIVATE_SEQUENCE_C")]
    ]
    result = training_input_profile(rows, ["signal", "sequence"])
    signal, sequence = result["features"]
    assert result["scope"] == "training_only" and result["rows"] == 4
    assert signal["storage"] == "string" and signal["empty_rows"] == 1
    assert signal["finite_numeric_rows"] == 2 and signal["distinct_values"] == 4
    assert sequence["distinct_values"] == 3 and sequence["finite_numeric_rows"] == 0
    text = json.dumps(result)
    assert "SECRET_TARGET" not in text and "PRIVATE_SEQUENCE" not in text
    assert "1.2" not in text and "-4" not in text


def test_profile_size_and_omissions_are_explicit():
    columns = [f"field_{i}" for i in range(128)]
    rows = [{"inputs": {name: "value" for name in columns}}]
    result = training_input_profile(rows, columns)
    assert len(result["features"]) == 16 and result["omitted_features"] == 112
    assert len(json.dumps(result).encode()) < 7000
    long_columns = ["x" * 1000 + str(i) for i in range(30)]
    result = training_input_profile([{"inputs": dict.fromkeys(long_columns, "x")}], long_columns)
    assert len(result["features"]) < 16
    assert result["omitted_features"] == len(long_columns) - len(result["features"])
    assert len(json.dumps(result).encode()) < 7000


def test_heldout_changes_do_not_change_training_profile(tmp_path):
    import csv
    path = tmp_path / "data.csv"
    rows = [{"id": str(i), "group": f"g{i}", "feature": str(i), "target": str(2*i+1)} for i in range(40)]
    def save():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "group", "feature", "target"])
            writer.writeheader()
            writer.writerows(rows)
    save()
    arguments = dict(data_file="data.csv", target_column="target", input_columns=["feature"],
                     group_column="group", id_column="id", provenance="Synthetic software fixture",
                     grouping="Independent groups", seed=3)
    first = prepare_dataset(tmp_path, **arguments)
    data = json.loads((tmp_path/first["evaluation_file"]).read_text())
    withheld = {row["id"] for split in ("validation", "test") for row in data["splits"][split]}
    for row in rows:
        if row["id"] in withheld:
            row["feature"] = "WITHHELD_" + row["id"] + "X"*300
            row["target"] = "991991"
    save()
    second = prepare_dataset(tmp_path, **arguments)
    assert first["evaluation_sha256"] != second["evaluation_sha256"]
    assert first["training_profile"] == second["training_profile"]
    assert "WITHHELD" not in json.dumps(second["training_profile"])


def test_shape_examples_are_explicitly_unmeasured_and_bounded():
    result = model_interface("classification", [f"field{i}" for i in range(19)])
    example = result["shape_example"]
    assert "not observations" in example["kind"]
    assert example["training_row"]["inputs"] == example["prediction_inputs"]
    assert len(example["prediction_inputs"]) == 4 and example["omitted_input_fields"] == 15
    assert result["predict"]["returns"]["value"] == "scalar class label"
    compile(training_script(0), "<training-entrypoint>", "exec")


KEY_ERROR = """def fit(training, seed):
    rows = training['train']
    x = [row['x'] for row in rows]
    return {'mean': sum(row['target'] for row in rows)/len(rows)}
def predict(model, inputs):
    return {'value': model['mean']}
"""
MATRIX_ERROR = """from sklearn.preprocessing import StandardScaler
def fit(training, seed):
    values = [row['inputs']['x'] for row in training['train']]
    scaler = StandardScaler().fit(values)
    return {'scaler': scaler}
def predict(model, inputs):
    return {'value': 0.0}
"""
CSV_BUFFER_ERROR = """import pandas as pd
def fit(training, seed):
    rows = pd.read_csv(training)
    return {'rows': rows}
def predict(model, inputs):
    return {'value': 0.0}
"""
PARAMETER_ERROR = """def fit(training, seed):
    rows = training['train']
    raise ValueError('Unsupported estimator option')
def predict(model, inputs):
    return {'value': 0.0}
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("source, expected, absent", [
    (KEY_ERROR, "row['inputs']['x']", "Feature contract:"),
    (MATRIX_ERROR, "two-dimensional feature matrix", "Iterating training itself"),
    (CSV_BUFFER_ERROR, "Do not pass training to pandas.read_csv", "Feature contract:"),
    (PARAMETER_ERROR, "Unsupported estimator option", "Input contract:"),
])
async def test_real_training_failure_reports_the_relevant_contract(tmp_path, source, expected, absent):
    from agent.harness.research_workspace import ResearchWorkspace
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps": []},
                                development_root=tmp_path/"development")
    table = "id,group,x,target\n" + "".join(f"{i},g{i},{i},{2*i+1}\n" for i in range(30))
    await service.write_research_file("data.csv", table)
    prepared = await service.prepare_research_dataset(
        "data.csv", "target", ["x"], "group", "Synthetic software fixture", "Independent groups", id_column="id")
    await service.write_research_file("algorithm.py", source)
    with pytest.raises(ValueError) as raised:
        await service.train_research_model("Fixture", "algorithm.py", prepared["evaluation_file"], "Test diagnostics")
    message = str(raised.value)
    assert expected in message and absent not in message
    assert str(tmp_path) not in message
    receipts = list(service.development.root.glob("training/*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "failed" and receipt["stage"] == "fitting"
    assert receipt["validation_and_test_mounted"] is False
    assert not service.development.records()


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [
    "[row['target']['localization'] for row in training['train']]",
    "LabelEncoder().fit_transform([row['target'] for row in training['train']])",
])
async def test_real_multilabel_failure_explains_parsed_target_mapping(tmp_path, expression):
    from agent.harness.research_workspace import ResearchWorkspace
    service = ResearchWorkspace(tmp_path/'work', tmp_path/'control', {'steps': []})
    labels = ['Cytoplasm', 'Nucleus']
    splits = {name: [{'id':name+str(i), 'group':name+str(i), 'inputs':{'sequence':'A'*i},
                     'target':{'Cytoplasm': i%2, 'Nucleus': int(i%3==0)}} for i in values]
              for name, values in [('train', range(1,9)), ('validation', range(10,13)), ('test', range(14,17))]}
    await service.write_research_file('evaluation.json', json.dumps({
        'task':'multilabel', 'labels':labels, 'provenance':'Synthetic fixture',
        'grouping':'Disjoint synthetic sequences', 'splits':splits}))
    source = ("from sklearn.preprocessing import LabelEncoder\n"
              "def fit(training, seed):\n    values = " + expression + "\n    return {'values': values}\n"
              "def predict(model, inputs):\n    return {'value':model['values']}\n")
    await service.write_research_file('algorithm.py', source)
    with pytest.raises(ValueError) as raised:
        await service.train_research_model('Synthetic mapping check', 'algorithm.py', 'evaluation.json', 'Check typed feedback')
    message = str(raised.value)
    assert "Multilabel target contract" in message
    assert "original CSV target-column name is not another nested key" in message
    assert "row['target'][label]" in message
    assert "LabelEncoder is for scalar classes" in message
    assert "At algorithm.py:" in message
    assert str(tmp_path) not in message
    assert not service.development.records()


def test_multilabel_interface_distinguishes_csv_column_from_parsed_labels():
    interface = model_interface('multilabel', ['sequence'], labels=['Nucleus', 'Cytoplasm'])
    assert "row['target'][label]" in interface['fit']['target_access']
    assert "CSV target-column name" in interface['fit']['target_access']
    assert "LabelEncoder" in interface['fit']['target_encoding']
    assert set(interface['shape_example']['training_row']['target']) == {'Nucleus', 'Cytoplasm'}
