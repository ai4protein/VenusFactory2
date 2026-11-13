import json

import pytest

from backend.web_v2 import research_source_review as sr
from backend.web_v2.research_measurements import measured_values_anchored, measurements
from backend.web_v2.research_source_validation import validate_assessments

# Minimal public calculation fixture from 40267, report 863031ec4794604636731648e8700940,
# S27 source SHA 85580a98a5400fad09837bfd062d34a906387ce426bc88d0fe3868d7fdaab78d.
# Only the two explicitly dimensioned calculation fields are copied, not its draft.
PROTPARAM = {"length_aa": 148, "molecular_weight_Da": 16536.85}


def supports(claim, value):
    assert measurements(claim), "The case must exercise an explicit quantity"
    return measured_values_anchored(claim, [{"quote": json.dumps(value)}])


@pytest.mark.parametrize("claim", [
    "The sequence has 148 amino acids and its mass is 16536.85 Da.",
    "The computed mass is 16.53685 kDa.",
    "The calculated mass is 1.653685e4 Da for 148 residues.",
])
def test_actual_platform_fields_bind_exact_dimensions(claim):
    assert supports(claim, PROTPARAM)


@pytest.mark.parametrize("claim", [
    "The mass is 16536.86 Da.", "The mass is 16536.85 kDa.",
    "The sequence has 147 residues.", "The mass is 148 Da.",
    "The temperature is 16536.85 C.", "The length is 16536.85 residues.",
    "The mass is -16536.85 Da.",
])
def test_wrong_value_or_dimension_stays_rejected(claim):
    assert not supports(claim, PROTPARAM)


@pytest.mark.parametrize("key,claim", [
    ("molecular_weight_da", "The mass is 16536.85 Da."),
    ("molecularWeightDa", "The mass is 16536.85 Da."),
    ("molecular_weight_kDa", "The mass is 16536.85 Da."),
    ("estimated_molecular_weight_Da", "The mass is 16536.85 Da."),
    ("length_AA", "The length is 148 residues."),
    ("predicted_length_aa", "The length is 148 residues."),
])
def test_arbitrary_similar_keys_do_not_infer_units(key, claim):
    assert not supports(claim, {key: 148 if "length" in key else 16536.85})


@pytest.mark.parametrize("bad", [
    "16536.85", "16536.85 Da", True, None, [16536.85],
    {"value": 16536.85, "unit": "Da"}, 0, -16536.85,
    float("nan"), float("inf"), -float("inf"),
])
def test_platform_mass_requires_a_finite_positive_native_number(bad):
    assert not supports("The mass is 16536.85 Da.", {"molecular_weight_Da": bad})


@pytest.mark.parametrize("bad,claim", [
    ("148", "The sequence has 148 residues."),
    ("148 aa", "The sequence has 148 residues."),
    (True, "The sequence has 1 residue."),
    (0, "The sequence has 0 residues."),
    (-148, "The sequence has -148 residues."),
    (148.5, "The sequence has 148.5 residues."),
    (None, "The sequence has 148 residues."),
    (float("nan"), "The sequence has 148 residues."),
    (float("inf"), "The sequence has 148 residues."),
])
def test_platform_length_requires_a_positive_native_integer(bad, claim):
    assert not supports(claim, {"length_aa": bad})


@pytest.mark.parametrize("bad,display", [(0, "0"), (-16536.85, "-16536.85"), (True, "1")])
def test_nonpositive_mass_or_boolean_cannot_match_even_the_same_display(bad, display):
    assert not supports(f"The mass is {display} Da.", {"molecular_weight_Da": bad})


def test_integer_valued_json_number_and_nested_calculation_are_supported():
    assert supports("The sequence has 148 residues.", {"calculation": {"length_aa": 148.0}})
    assert supports("The mass is 16.53685 kDa.", {"calculation": PROTPARAM})


def test_complete_json_does_not_borrow_a_bad_typed_fields_text_value():
    quote = '{"length_aa":"148 aa","molecular_weight_Da":"16536.85 Da"}'
    assert not measured_values_anchored("The mass is 16536.85 Da.", [{"quote": quote}])
    assert not measured_values_anchored("The sequence has 148 residues.", [{"quote": quote}])


def test_free_text_and_explicit_quantity_records_keep_their_existing_scope():
    quote = 'The mass is 16.53685 kDa. {"length_aa":148} The yield is 5%.'
    assert measured_values_anchored("The mass is 16536.85 Da for 148 residues at 5% yield.", [{"quote": quote}])
    assert supports("The mass is 16536.85 Da.", {"description": "The mass is 16536.85 Da."})
    assert supports("The mass is 16536.85 Da.", {"value": 16536.85, "unit": "Da"})


def test_matching_quantities_do_not_override_independent_semantic_rejection():
    source = {"id": "S27", "kind": "calculation", "title": "Sequence properties",
              "text": json.dumps(PROTPARAM)}
    document = {"id": "biophysics", "title": "Biophysics",
                "text": "The mass is 16536.85 Da. These values establish cytoplasmic localization. [S27]"}
    request = sr.packet(sr.units(document, [source]), [source])
    passage = request["sources"][0]["passages"][0]
    issue = {"claim": "These values establish cytoplasmic localization.",
             "reason": "A matching molecular mass does not establish cellular localization."}
    response = {"assessments": [{
        "unit_id": request["units"][0]["id"], "classification": "factual",
        "verdict": "unsupported", "issues": [issue],
        "evidence": [{"source_id": "S27", "passage_id": passage["id"], "quote": passage["text"]}],
    }]}
    assert supports("The mass is 16536.85 Da.", PROTPARAM)
    checked = validate_assessments(json.dumps(response), request, [source])
    assert checked[0]["passed"] is False
    assert checked[0]["verdict"] == "unsupported"
    assert checked[0]["issues"] == [issue]
