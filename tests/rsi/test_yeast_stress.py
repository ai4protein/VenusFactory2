"""Independent statistical and lifecycle checks for the yeast paper task."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agent.rsi.paper_step import PaperStepLifecycle, compare, validated_source
from agent.rsi.paper_step_catalog import catalog
from agent.rsi.storage import sha256_file
from agent.rsi.yeast_stress import HEADERS, admit_task, parse_sheet, read_supplement, reference

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / 'configs/rsi/paper_step_examples/yeast_abundance_contrasts.py'
SOURCE = SOURCE_PATH.read_text()


def candidate():
    namespace = {}
    exec(compile(SOURCE, str(SOURCE_PATH), 'exec'), namespace)
    return namespace['reproduce']


def inputs(seed, size=100):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(size):
        rows.append({'id': 'P' + str(i), 'control': rng.normal(20, 1, 3).tolist(),
                     'treatment': rng.normal(20 + i % 9, 0.2 + i % 3, 3).tolist()})
    rows.extend([
        {'id': 'missing', 'control': [None, 1, None], 'treatment': [2, 3, 4]},
        {'id': 'absent', 'control': [None] * 3, 'treatment': [2, 3, 4]},
        {'id': 'constant_different', 'control': [1, 1, 1], 'treatment': [4, 4, 4]},
        {'id': 'constant_same', 'control': [1, 1, 1], 'treatment': [1, 1, 1]},
        {'id': 'equal_means', 'control': [1, 2, 3], 'treatment': [0, 2, 4]},
        {'id': 'tail', 'control': [1, 1.0001, 0.9999], 'treatment': [100, 100.0001, 99.9999]},
        {'id': 'one_constant', 'control': [1, 1, 1], 'treatment': [1, 2, 3]},
    ])
    return {'contrast': 'synthetic_' + str(seed), 'rows': rows}


@pytest.mark.parametrize('seed', [7, 41, 314159])
def test_statistical_results_match_independent_scipy(seed):
    value = inputs(seed)
    expected, actual = reference(value), candidate()(value)
    assert compare(actual, expected, atol=1e-10, rtol=1e-8)['passed']
    rows = {row['id']: row for row in actual['rows']}
    assert rows['absent']['log2_fold_change'] is None
    assert rows['missing']['p_value'] is None
    assert rows['constant_different']['status'] == 'zero_variance'
    assert rows['equal_means']['p_value'] == pytest.approx(1)
    assert 0 <= rows['tail']['p_value'] < 1e-12
    assert rows['one_constant']['status'] == 'tested'
    sorted_rows = sorted([r for r in actual['rows'] if r['p_value'] is not None], key=lambda r: r['p_value'])
    assert all(a['q_value'] <= b['q_value'] for a, b in zip(sorted_rows, sorted_rows[1:]))


def test_sign_reversal_keeps_two_sided_p_and_fdr():
    value = inputs(99)
    reversed_value = copy.deepcopy(value)
    for row in reversed_value['rows']:
        row['control'], row['treatment'] = row['treatment'], row['control']
    a, b = candidate()(value), candidate()(reversed_value)
    for left, right in zip(a['rows'], b['rows']):
        assert left['p_value'] == right['p_value']
        assert left['q_value'] == right['q_value']
        if left['log2_fold_change'] is not None:
            assert left['log2_fold_change'] == -right['log2_fold_change']


def sheet_row(identity='P1'):
    return [float(i) for i in range(24)] + [0.001, identity, identity, 'GENE1', 'protein', identity, 'PROTEIN_YEAST']


def test_parser_retains_duplicate_header_positions_missingness_and_log2_values():
    row = sheet_row()
    row[1] = 'NaN'
    result = parse_sheet([HEADERS, row])[0]
    assert result['conditions']['ypd'] == [0.0, None, 2.0]
    assert result['conditions']['ethanol'] == [12.0, 13.0, 14.0]
    assert result['source_row'] == 2
    assert result['genes'] == ['GENE1']


@pytest.mark.parametrize('bad', [True, float('inf'), float('nan'), 'Filtered', '123'])
def test_bad_abundance_not_silently_converted(bad):
    row = sheet_row()
    row[0] = bad
    with pytest.raises(ValueError, match='Invalid numeric'):
        parse_sheet([HEADERS, row])


def test_source_layout_identity_and_hash_required(tmp_path):
    with pytest.raises(ValueError, match='source columns'):
        parse_sheet([list(reversed(HEADERS)), sheet_row()])
    with pytest.raises(ValueError, match='duplicate'):
        parse_sheet([HEADERS, sheet_row(), sheet_row()])
    bad = tmp_path / 'data.xlsx'
    bad.write_bytes(b'not the admitted source')
    with pytest.raises(ValueError, match='checksum'):
        read_supplement(bad)


def definition(tmp_path):
    ref = tmp_path / 'independent_reference.py'
    ref.write_text('Frozen synthetic test reference')
    return {
        'schema': 1, 'paper': {'title': 'Synthetic numerical conformance', 'doi': 'test:yeast'},
        'scope': {'covered_steps': ['Welch/BH arithmetic'], 'excluded_steps': ['scientific claims']},
        'interface': {'id': 'yeast_abundance_contrast.v1', 'description': 'Synthetic comparison fixture'},
        'references': {str(ref): sha256_file(ref)},
        'cases': {phase: [{'id': phase, 'inputs': inputs(seed, 12), 'expected': reference(inputs(seed, 12))}]
                  for phase, seed in [('development', 1), ('acceptance', 2), ('reuse', 3)]},
        'budget': {'candidates': 2, 'operation_seconds': 30}, 'tolerance': {'atol': 1e-9, 'rtol': 1e-8},
    }


def test_sandbox_acceptance_registration_and_fresh_reuse(tmp_path):
    validated_source(SOURCE)
    lifecycle = PaperStepLifecycle(tmp_path / 'host', definition(tmp_path))
    source_id = lifecycle.propose(SOURCE, declared_origin='controller_numerical_example_not_agent')
    assert lifecycle.assess(source_id)['passed']
    capability = lifecycle.register(source_id)
    assert lifecycle.reuse(capability, 'reuse')['passed']
    assert lifecycle.verify_reuse(capability, 'reuse')['fresh_execution']
    assert not lifecycle.capability(capability)['agent_authorship_verified']


def test_catalog_add_preserves_existing_and_rejects_replacement(tmp_path):
    config = tmp_path / 'configs/rsi'
    config.mkdir(parents=True)
    (config / 'paper_steps.json').write_text(json.dumps({'schema': 1, 'tasks': {}}))
    source = tmp_path / 'definition.json'
    data = definition(tmp_path)
    source.write_text(json.dumps(data))
    record = admit_task(tmp_path, source)
    assert admit_task(tmp_path, source) == record
    assert 'yeast_stress_abundance' in catalog(tmp_path)
    data['budget']['candidates'] = 3
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='differs'):
        admit_task(tmp_path, source)


def test_export_freezes_reference_paths_and_all_prespecified_cases(tmp_path, monkeypatch):
    from agent.rsi import yeast_stress
    from agent.rsi.paper_step import validate_definition

    proteins = parse_sheet([HEADERS, sheet_row()])
    monkeypatch.setattr(yeast_stress, 'read_supplement', lambda path: proteins)
    table = tmp_path / 'source.xlsx'
    table.write_bytes(b'synthetic admitted-source stand-in')
    destination = tmp_path / 'export'
    result = yeast_stress.export_task(table, destination)
    exported = validate_definition(json.loads(Path(result['definition']).read_text()))
    assert [len(exported['cases'][phase]) for phase in ('development', 'acceptance', 'reuse')] == [1, 2, 4]
    assert result['report']['observations'] == 24
    assert result['report']['context_model_admission']['admitted'] is False
    for filename, checksum in exported['references'].items():
        assert sha256_file(Path(filename)) == checksum
    with pytest.raises(FileExistsError):
        yeast_stress.export_task(table, destination)
