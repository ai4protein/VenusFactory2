"""Static compiler/contract checks are not a replacement for scientific evaluation."""
import pytest

from agent.harness.research_algorithm_checks import (
    AlgorithmValidationError,
    algorithm_diagnostics,
    validate_algorithm,
)

VALID = """def fit(rows, random_seed):
    return {"mean": sum(row["target"] for row in rows["train"])/len(rows["train"])}
def predict(state, query):
    return {"value": state["mean"]}
"""


def test_inspection_never_imports_or_executes_untrusted_source(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nimport missing_fixture_package\n" + VALID
    assert algorithm_diagnostics(source)["valid"]
    assert not marker.exists()


def test_compiler_rejects_ast_parseable_but_unexecutable_source():
    result = algorithm_diagnostics("return 1\n" + VALID)
    assert not result["valid"]
    assert any(issue["code"] == "invalid_python" and issue["line"] == 1 for issue in result["issues"])


def test_model_checks_report_missing_comma_and_unused_training_together():
    source = """pipeline = [
    ('onehot', object())
    ('regressor', object())
]
def fit(training, seed):
    return pipeline
def predict(model, inputs):
    return {'value': 0}
"""
    result = algorithm_diagnostics(source)
    assert not result["valid"]
    assert any(i["code"] == "compiler_warning" and i["line"] == 2 for i in result["issues"])
    assert any(i["code"] == "unused_training" and i["line"] == 5 for i in result["issues"])
    with pytest.raises(AlgorithmValidationError) as error:
        validate_algorithm(source)
    assert "line 2:" in str(error.value) and "line 5:" in str(error.value)
    assert error.value.diagnostics == result


def test_helper_functions_and_renamed_training_argument_keep_same_ast_contract():
    source = """def estimate(rows):
    return {'mean':sum(row['target'] for row in rows)/len(rows)}
def fit(observations, random_seed):
    return estimate(observations['train'])
def predict(parameters, query):
    return {'value':parameters['mean']}
"""
    assert validate_algorithm(source) == validate_algorithm(source+"\n# only a comment\n")
    assert algorithm_diagnostics(source)["issues"] == []


def test_warnings_are_disclosed_without_becoming_an_algorithm_whitelist():
    source = VALID.replace('return {"mean":', 'unused = 1 is 1\n    return {"mean":')
    result = algorithm_diagnostics(source)
    assert result["valid"] and result["issues"]
    assert all(i["severity"] == "warning" for i in result["issues"])
    assert validate_algorithm(source)


@pytest.mark.parametrize("source", ["def fit(:", "\x00", "class Model:\n def fit(self,x):pass\n"])
def test_invalid_sources_have_structured_repair_diagnostics(source):
    result = algorithm_diagnostics(source)
    assert not result["valid"] and result["algorithm_ast_sha256"] is None
    assert result["issues"] and all(i["message"] for i in result["issues"])


def test_many_warnings_cannot_displace_contract_errors_or_expand_repair_context():
    source = "unused = 1 is 1\n"*100 + "def fit(training,seed):return {}\ndef predict(m,i):return {'value':0}\n"
    result = algorithm_diagnostics(source)
    assert not result["valid"]
    assert len(result["issues"]) == 12
    assert result["omitted_issue_count"] == result["issue_count"] - 12
    assert result["issues"][0]["code"] == "unused_training"
    with pytest.raises(ValueError, match="fit does not read"):
        validate_algorithm(source)


def test_undefined_prediction_object_is_a_nonblocking_repair_hint():
    source = VALID.replace('state["mean"]', 'onehot.transform(query)[0]')
    result = algorithm_diagnostics(source)
    assert result["valid"]
    unresolved = [issue for issue in result["issues"] if issue["code"] == "unresolved_global"]
    assert len(unresolved) == 1 and "onehot" in unresolved[0]["message"]
    assert unresolved[0]["severity"] == "warning" and unresolved[0]["line"] == 4


def test_dynamic_model_globals_remain_usable():
    source = "globals()['offset'] = 1\n" + VALID.replace('state["mean"]', 'state["mean"] + offset')
    assert validate_algorithm(source)


@pytest.mark.parametrize("name", ["fit", "predict"])
def test_class_method_diagnostic_names_the_exact_missing_module_entrypoint(name):
    source = "class BaselineModel:\n    def fit(self, training, seed):\n        return self\n    def predict(self, model, inputs):\n        return 0\n"
    result = algorithm_diagnostics(source)
    item = next(x for x in result["issues"] if f"module-level {name}(" in x["message"])
    assert f"BaselineModel.{name}" in item["message"]
    assert item["line"] == (2 if name == "fit" else 4)
    assert "class method" in item["message"]
    assert len({x["message"] for x in result["issues"]}) == 2


def test_three_argument_prediction_explains_single_fitted_state_transport():
    source = VALID.replace("def predict(state, query):", "def predict(encoder, state, query):")
    result = algorithm_diagnostics(source)
    item = next(x for x in result["issues"] if x["code"] == "invalid_entrypoint")
    assert "['encoder', 'state', 'query']" in item["message"]
    assert "single model argument" in item["message"] and "two positional" in item["message"]
    assert not result["valid"]


def test_training_row_hint_uses_names_not_observations_or_targets():
    source = "def fit(observations, seed):\n    values = [sample['sequence'] for sample in observations['train']]\n    return {'rows': values}\ndef predict(model, query):\n    return {'value': 0}\n"
    result = algorithm_diagnostics(source, input_fields=("sequence", "substrate_smiles"))
    hint = next(x for x in result["issues"] if x["code"] == "training_row_field")
    assert "sample['inputs']['sequence']" in hint["message"] and hint["line"] == 2
    assert hint["severity"] == "warning" and result["valid"]
    assert algorithm_diagnostics(source)["issues"] == []
    assert algorithm_diagnostics(source.replace("sample['sequence']", "sample['inputs']['sequence']"),
                                 input_fields=("sequence",))["issues"] == []


def test_prediction_nesting_is_a_hint_and_literal_inputs_column_remains_valid():
    source = VALID.replace('state["mean"]', 'query["inputs"]["x"]')
    result = algorithm_diagnostics(source, input_fields=("x",))
    assert result["valid"]
    assert any(x["code"] == "prediction_input_nesting" for x in result["issues"])
    assert algorithm_diagnostics(source, input_fields=("inputs",))["issues"] == []


def test_helpers_with_similarly_named_rows_do_not_acquire_dataset_contract():
    source = "def helper(rows):\n    return [row['sequence'] for row in rows]\n" + VALID
    assert algorithm_diagnostics(source, input_fields=("sequence",))["issues"] == []


def test_compiler_error_exposes_exact_bounded_source_location():
    source = "def fit(training, seed):\n    counts = [part.count(x) for x in 'ACDE']]\ndef predict(model, inputs):\n    return {}\n"
    result = algorithm_diagnostics(source)
    item = result["issues"][0]
    assert item["code"] == "invalid_python" and item["line"] == 2
    assert item["source_excerpt"] == source.splitlines()[1]
    assert item["source_excerpt"][item["excerpt_column"]-1] == "]"
    with pytest.raises(AlgorithmValidationError) as caught:
        validate_algorithm(source)
    assert item["source_excerpt"] in str(caught.value) and "^" in str(caught.value)


def test_long_syntax_error_excerpt_remains_near_the_error():
    source = "value = [" + "1, " * 1000 + "]]"
    item = algorithm_diagnostics(source)["issues"][0]
    assert len(item["source_excerpt"]) <= 200
    assert 1 <= item["excerpt_column"] <= len(item["source_excerpt"])
    assert item["source_excerpt"][item["excerpt_column"]-1] == "]"
