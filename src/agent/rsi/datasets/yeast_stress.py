"""Observed-data reanalysis of the public PNAS 2020 yeast stress supplement.

This supplies a numerical paper-step task, not a predictive model acquisition.
The original table is already log2 transformed and vendor cross-run normalized.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from agent.rsi.engine.storage import atomic_json, file_lock, sha256_file

DOI = '10.1073/pnas.2020197117'
SOURCE_SHA256 = '88adc80d279b79fa2d2554a79209ae8416021955b3a5a981772fe8a93f12dd54'
TITLE = 'DIA-based systems biology approach unveils E3 ubiquitin ligase-dependent responses to a metabolic shift'
CONDITIONS = ('ypd', 'heat37', 'heat42', 'osmotic', 'ethanol', 'sdc', 'aa_nt', 'aa')
HEADERS = (
    'YPD_1', 'YPD_2', 'YPD_3',
    'YPD_heat_37_1', 'YPD_heat_37_2', 'YPD_heat_37_3',
    'YPD_heat_42_1', 'YPD_heat_42_2', 'YPD_heat_42_3',
    'YPD_osmotic_1', 'YPD_osmotic_2', 'YPD_osmotic_3',
    'YP_EtOH_24h_!', 'YP_EtOH_24h_!', 'YP_EtOH_24h_!',
    'SDC_1', 'SDC_2', 'SDC_3',
    'SD-AA-Nt_1h_1', 'SD-AA-Nt_1h_2', 'SD-AA-Nt_1h_3',
    'SD-AA_1h_1', 'SD-AA_1h_2', 'SD-AA_1h_3',
    'T: PG.Qvalue', 'T: PG.ProteinGroups', 'T: PG.ProteinAccessions',
    'T: PG.Genes', 'T: PG.ProteinDescriptions', 'T: PG.UniProtIds', 'T: PG.ProteinNames',
)
CONTRASTS = {
    'development': [('heat37_vs_ypd', 'ypd', 'heat37')],
    'acceptance': [('heat42_vs_ypd', 'ypd', 'heat42'), ('sdc_vs_ypd', 'ypd', 'sdc')],
    'reuse': [('ethanol_vs_ypd', 'ypd', 'ethanol'), ('osmotic_vs_ypd', 'ypd', 'osmotic'),
              ('aa_nt_vs_sdc', 'sdc', 'aa_nt'), ('aa_vs_sdc', 'sdc', 'aa')],
}


def parse_sheet(rows):
    rows = iter(rows)
    if tuple(next(rows)) != HEADERS:
        raise ValueError('Unexpected source columns; duplicate ethanol headers must retain positional identity')
    result, seen = [], set()
    for number, row in enumerate(rows, 2):
        if len(row) != len(HEADERS):
            raise ValueError('Unexpected row width')
        identity = row[25]
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError('Invalid or duplicate protein-group identity')
        seen.add(identity)
        values = []
        for value in row[:24]:
            if value is None or (isinstance(value, str) and value.strip().lower() == 'nan'):
                values.append(None)
            elif type(value) in (float, int) and math.isfinite(value):
                values.append(float(value))
            else:
                raise ValueError('Invalid numeric abundance at spreadsheet row ' + str(number))
        result.append({'id': identity, 'source_row': number, 'genes': str(row[27] or '').split(';'),
                       'conditions': {name: values[i * 3:(i + 1) * 3] for i, name in enumerate(CONDITIONS)}})
    return result


def read_supplement(path):
    import openpyxl

    path = Path(path)
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError('Yeast stress source checksum differs from the admitted public supplement')
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames != ['Supp Table 1']:
            raise ValueError('Unexpected supplement sheets')
        result = parse_sheet(workbook['Supp Table 1'].iter_rows(values_only=True))
    finally:
        workbook.close()
    if len(result) != 3506:
        raise ValueError('Expected 3506 source protein groups')
    return result


def make_inputs(proteins, contrast, control, treatment):
    return {'contrast': contrast, 'rows': [
        {'id': row['id'], 'control': row['conditions'][control], 'treatment': row['conditions'][treatment]}
        for row in proteins
    ]}


def reference(inputs):
    """Independent host reference: SciPy Welch t-test and SciPy BH adjustment."""
    import numpy as np
    from scipy.stats import false_discovery_control, ttest_ind

    output, tested = [], []
    for row in inputs['rows']:
        control = np.asarray([x for x in row['control'] if x is not None], dtype=float)
        treatment = np.asarray([x for x in row['treatment'] if x is not None], dtype=float)
        ma = float(control.mean()) if len(control) else None
        mb = float(treatment.mean()) if len(treatment) else None
        effect = mb - ma if ma is not None and mb is not None else None
        p, status = None, 'insufficient_replicates'
        if len(control) == len(treatment) == 3:
            if float(control.var(ddof=1) + treatment.var(ddof=1)) > 0:
                p = float(ttest_ind(treatment, control, equal_var=False).pvalue)
                status = 'tested'
                tested.append(len(output))
            else:
                status = 'zero_variance'
        output.append({'id': row['id'], 'n_control': len(control), 'n_treatment': len(treatment),
                       'mean_control': ma, 'mean_treatment': mb, 'log2_fold_change': effect,
                       'status': status, 'p_value': p, 'q_value': None,
                       'fourfold_p005': False, 'bh_fdr005': False})
    if tested:
        adjusted = false_discovery_control([output[i]['p_value'] for i in tested], method='bh')
        for index, q in zip(tested, adjusted):
            row = output[index]
            row['q_value'] = float(q)
            row['fourfold_p005'] = row['p_value'] < 0.05 and abs(row['log2_fold_change']) >= 2
            row['bh_fdr005'] = q < 0.05
            row['bh_fdr005'] = bool(row['bh_fdr005'])
    return {'contrast': inputs['contrast'], 'rows': output, 'tested': len(tested)}


def export_task(table, destination):
    """Freeze one study, all protein groups and seven prespecified contrasts."""
    from agent.rsi.engine.paper_step import validate_definition

    table, destination = Path(table).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new output directory; existing evidence is immutable')
    proteins = read_supplement(table)
    destination.mkdir(parents=True)
    references = {str(table): sha256_file(table), str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    cases, summaries, results = {}, {}, {}
    for phase, contrasts in CONTRASTS.items():
        cases[phase] = []
        for identity, control, treatment in contrasts:
            inputs = make_inputs(proteins, identity, control, treatment)
            expected = reference(inputs)
            cases[phase].append({'id': identity, 'inputs': inputs, 'expected': expected})
            results[identity] = expected
            summaries[identity] = {
                'rows': len(proteins), 'complete_triplicates_tested': expected['tested'],
                'fourfold_p005': sum(row['fourfold_p005'] for row in expected['rows']),
                'bh_fdr005': sum(row['bh_fdr005'] for row in expected['rows']),
                'unavailable_tests': len(proteins) - expected['tested'],
            }
    metadata = {
        'schema': 1, 'doi': DOI, 'proteomexchange': 'PXD021559',
        'source_url': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7768684/supplementaryFiles',
        'source_file': table.name, 'source_sha256': SOURCE_SHA256,
        'unit': 'author_exported_log2_DIA_intensity', 'organism': 'Saccharomyces cerevisiae BY4741',
        'source_preprocessing': 'Spectronaut cross-run normalization; table values already log2',
        'protein_groups': len(proteins), 'conditions': len(CONDITIONS), 'biological_replicates_per_condition': 3,
        'observations': sum(v is not None for p in proteins for a in p['conditions'].values() for v in a),
        'missing_observations': sum(v is None for p in proteins for a in p['conditions'].values() for v in a),
        'column_mapping': {name: [{'index_zero_based': i * 3 + j, 'source_header': HEADERS[i * 3 + j]}
                                 for j in range(3)] for i, name in enumerate(CONDITIONS)},
        'ethanol_duration_conflict': 'The table labels 24h; main-text methods state 19h. No numeric duration inferred.',
        'protocol': 'Observed-only means; tests require all three observations in each condition; two-sided Welch t-test; BH across all tested protein groups per contrast; no imputation.',
        'scope': 'Computational reanalysis of published normalized measurements; contrasts share some source observations and are not predictive train/test splits.',
        'unresolved_samples': 'Extra raw export runs are not counted without author-bound condition mapping.',
        'context_model_admission': {'admitted': False, 'candidate_models': 0,
                                  'reason': '24 observations per protein; below 40/10/10 context model split floor, and physical culture grouping across conditions is unresolved.'},
        'contrasts': summaries,
    }
    atomic_json(destination / 'source_report.json', metadata)
    atomic_json(destination / 'protein_metadata.json', [{'id': p['id'], 'genes': p['genes'], 'source_row': p['source_row']} for p in proteins])
    for identity, expected in results.items():
        atomic_json(destination / (identity + '.reference.json'), expected)
    refs = [destination / 'source_report.json', destination / 'protein_metadata.json']
    references.update({str(p): sha256_file(p) for p in refs})
    definition = validate_definition({
        'schema': 1, 'paper': {'title': TITLE, 'doi': DOI},
        'scope': {
            'covered_steps': ['Seven observed-only protein-group abundance contrasts from Dataset S1',
                              'Log2 fold changes, two-sided Welch tests and BH correction',
                              'Independent numerical acceptance and fresh reuse on different contrasts'],
            'excluded_steps': ['Raw mass-spectrometry processing or spectral-library reconstruction',
                               'Exact Perseus imputation, ANOVA, GO enrichment or original figure reproduction',
                               'Independent predictive models, wet-lab replication or autonomous authorship']},
        'interface': {'id': 'yeast_abundance_contrast.v1',
                      'description': 'Input: contrast and ordered rows of id, control[3], treatment[3], with log2 values or null. Return contrast, rows, tested. Each output row has id, n_control, n_treatment, mean_control, mean_treatment, log2_fold_change (treatment minus control), status, p_value, q_value, fourfold_p005, bh_fdr005. Means use observed values; all three values in both groups are required for a two-sided Welch t-test. Both variances zero gives zero_variance; missing replicates gives insufficient_replicates; otherwise tested. Unavailable means/effects/tests are null, flags false. BH corrects over tested rows only per contrast. Flags: p<0.05 and absolute log2 change>=2; q<0.05. Preserve input order.'},
        'references': references, 'cases': cases,
        'budget': {'candidates': 4, 'operation_seconds': 60}, 'tolerance': {'atol': 1e-9, 'rtol': 1e-8},
    })
    atomic_json(destination / 'definition.json', definition)
    return {'definition': str(destination / 'definition.json'), 'report': metadata}


def admit_task(repository, definition_path):
    """Add a frozen task to the existing Scientist/Agent paper-step catalog."""
    from agent.rsi.engine.paper_step import validate_definition

    root, source = Path(repository).resolve(), Path(definition_path).resolve()
    payload = source.read_bytes()
    validate_definition(json.loads(payload))
    relative = 'research_runs/case_definitions/yeast_stress_abundance.json'
    target = root / relative
    catalog_path = root / 'configs/rsi/paper_steps.json'
    record = {'label': {'en': 'Yeast stress · protein abundance contrasts', 'zh': '酵母应激 · 蛋白丰度差异复现'},
              'definition': relative, 'sha256': hashlib.sha256(payload).hexdigest()}
    with file_lock(root / 'configs/rsi/.paper_steps.lock'):
        catalog = json.loads(catalog_path.read_text())
        existing = catalog['tasks'].get('yeast_stress_abundance')
        if existing is not None and existing != record:
            raise ValueError('Existing yeast paper task differs; do not overwrite admitted evidence')
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('Definition path already contains different evidence')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        catalog['tasks']['yeast_stress_abundance'] = record
        atomic_json(catalog_path, catalog)
    return record
