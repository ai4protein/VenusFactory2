import json

import pytest

from backend.web_v2.research_measurements import measured_values_anchored
from backend.web_v2.research_measurements import measurements


@pytest.mark.parametrize("claim,quote,expected", [
    ("The constant is 2.9 nM.", "The constant is 2.9 mM.", False),
    ("The optimum temperature is 6-8 degrees Celsius.", "Optimum pH is 6-8.", False),
    ("The optimum pH is 8-10.", "Optimum pH is 6-8.", False),
    ("The optimum pH is 6-8.", "Optimum pH is 6-8.", True),
    ("The annotated optimum pH of carbonic anhydrase 2 is 8-10.", "Optimum pH is 6-8.", False),
    ("The optimum pH for this enzyme is 6-8.", "Optimum pH is 6-8.", True),
    ("The value is 1 uM.", "The value is 1000 nM.", True),
    ("The mass is 30 kDa.", "The mass is 30000 Da.", True),
    ("The resolution is 2 Å.", "The resolution is 0.2 nm.", True),
    ("The length is 147 residues.", '{"sequence_length":147}', True),
    ("The mass is 147 Da.", '{"sequence_length":147}', False),
    ("The optimum is 37 °C.", '{"reference":{"id":"37"}}', False),
    ("The constant is 2.9 mM.", '{"constant":2.9,"unit":"mM"}', True),
    ("The constant is 2.9 nM.", '{"constant":2.9,"unit":"mM"}', False),
    ("The value is 1 µM.", '{"value":0.001,"unit":"mM"}', True),
    ("The yield is 50%.", "The study reports 50 residues.", False),
    ("The value is 2.9 µM.", "The value is 2.9 μM.", True),
    ("The measurement is 1e-3 mM.", "The measurement is 1 uM.", True),
    ("The concentration range is 1–2 µM.", "The concentration range is 1000-2000 nM.", True),
    ("The concentration range is 1–2 µM.", "The concentration range is 1–3 µM.", False),
    ("The optimum is 80 C.", "The optimum is 80 °C.", True),
    ("The temperature range is -10 to -5 °C.", "The temperature range is -10--5 °C.", True),
    ("The mass concentration is 1 mg/mL.", "The mass concentration is 1 g/L.", True),
    ("The concentration is 1 mM.", "The mass concentration is 1 mg/mL.", False),
])
def test_measurement_support_requires_matching_dimensions_and_values(claim, quote, expected):
    assert measurements(claim), "The case must exercise a recognised quantity"
    assert measured_values_anchored(claim, [{"quote": quote}]) is expected


def test_nested_json_never_pairs_a_neighbours_number_and_unit():
    quote = json.dumps({"measurements": [
        {"constant": 2.9, "unit": "mM", "substrate": "A"},
        {"constant": 9.3, "unit": "nM", "substrate": "B"},
    ]})
    assert measured_values_anchored("The value is 2.9 mM.", [{"quote": quote}])
    assert not measured_values_anchored("The value is 2.9 nM.", [{"quote": quote}])


def test_complete_inner_record_can_be_read_from_a_short_json_quote():
    quote = '{"reaction":{"constants":[{"constant":2.9,"unit":"mM"}]'
    assert measured_values_anchored("The constant is 2.9 mM.", [{"quote": quote}])
    assert not measured_values_anchored("The constant is 2.9 nM.", [{"quote": quote}])


def test_unknown_unit_or_boolean_cannot_supply_a_measurement():
    assert not measured_values_anchored("The value is 1 mM.", [{"quote": '{"value":true,"unit":"mM"}'}])
    assert not measured_values_anchored("The value is 1 mM.", [{"quote": '{"value":1,"unit":"arbitrary"}'}])

def test_model_identifiers_are_not_treated_as_molar_measurements():
    from backend.web_v2.research_measurements import measurements
    assert not measurements("ESM2-650M and ESM2-8M generate predictions.")


def test_pathological_numeric_exponents_are_not_accepted():
    assert not measured_values_anchored(
        "The value is 1 mM.", [{"quote": '{"constant":"1e999999999","unit":"mM"}'}]
    )

def test_bare_model_capacity_cannot_supply_a_molar_measurement():
    from backend.web_v2.research_measurements import measurements
    assert not measurements("The ESM2 8M model was used.")
    assert not measured_values_anchored("The value is 8 M.", [{"quote": "The ESM2 8M model was used."}])
    assert measured_values_anchored("The value is 1 M.", [{"quote": "The value is 1000 mM."}])


@pytest.mark.parametrize("claim", [
    "The value is 1e999999999 mM.",
    "The range is 3-1 mM.",
    "The optimum pH is 8-6.",
])
def test_invalid_explicit_measurements_fail_closed(claim):
    assert not measured_values_anchored(claim, [{"quote": claim}])

def test_thousands_separators_preserve_the_complete_number():
    assert measured_values_anchored("The mass is 30 kDa.", [{"quote": "The mass is 30,000 Da."}])
    assert not measured_values_anchored("The mass is 0 Da.", [{"quote": "The mass is 30,000 Da."}])
    assert measured_values_anchored("The mass is 30,000 Da.", [{"quote": '{"value":"30,000","unit":"Da"}'}])


def test_exponent_fragments_are_not_standalone_measurements():
    assert not measured_values_anchored("The concentration is -3 M.", [{"quote": "The concentration is 1 x 10^-3 M."}])


@pytest.mark.parametrize("claim", [
    "The structure resolution is 1.7 A.", "Resolution: 1.7 A.",
    "The structure has 1.7 A resolution.", "The structure has 1.7 A spatial resolution.",
    "Resolution is 1.7-2.0 A.",
])
def test_ascii_resolution_units_do_not_bypass_numeric_review(claim):
    assert measurements(claim), "Explicit ASCII resolution must be checked"
    assert not measured_values_anchored(claim, [{"quote": "The structure resolution is 1.8 A."}])
    assert measured_values_anchored(claim, [{"quote": claim.replace(" A", " \u00c5")}])


def test_ascii_resolution_equates_to_declared_length_units():
    assert measured_values_anchored("The resolution is 1.7 A.", [{"quote": "The resolution is 0.17 nm."}])
    assert not measured_values_anchored("The resolution is 1.7 A.", [{"quote": "The temperature is 1.7 C."}])


@pytest.mark.parametrize("text", [
    "The current is 2 A.", "We used model 1A.", "The sequence has 2 A residues.",
    "The resolution of chain A is not reported; condition 2 A was tested.",
    "The resolution is unknown. 2 A chains are present.",
])
def test_ascii_a_without_resolution_context_is_not_a_length(text):
    assert not measurements(text)
