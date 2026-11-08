"""Static compiler/contract checks are not a replacement for scientific evaluation."""
import pytest

from agent.harness.research_algorithm_checks import (
    AlgorithmValidationError, algorithm_diagnostics, validate_algorithm,
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
