"""Build a small, runnable release from an independently admitted candidate."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import zipfile

from utils.scoped_files import read_bytes, write_bytes
from .contract import SCHEMA, family_id

RUNNER = '''"""JSON prediction/calculation entry point for this reviewed artifact."""
from pathlib import Path
import importlib.util
import json
import os
import sys
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent

def run(inputs):
    contract = json.loads((ROOT / 'capability.json').read_text())
    interface = json.loads((ROOT / 'interface.json').read_text())
    Draft202012Validator(interface['input_schema']).validate(inputs)
    runtime = ROOT / 'runtime'
    source = runtime / contract['entrypoint']['source']
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(runtime)
        sys.path[:0] = [str(source.parent), str(runtime)]
        spec = importlib.util.spec_from_file_location('scientist_artifact_runtime', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.run(inputs)
        Draft202012Validator(interface['output_schema']).validate(result)
        json.dumps(result, allow_nan=False)
        return result
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path

if __name__ == '__main__':
    inputs = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else json.load(sys.stdin)
    print(json.dumps(run(inputs), ensure_ascii=False, allow_nan=False))
'''

TESTS = '''"""Replay the declared and independent review interface and boundary cases."""
from pathlib import Path
import json
import math
import sys
import unittest
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import run

def equal(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected

class InterfaceTests(unittest.TestCase):
    def check_cases(self, filename):
        contract = json.loads((ROOT / 'capability.json').read_text())
        interface = json.loads((ROOT / 'interface.json').read_text())
        cases = json.loads((ROOT / 'tests' / filename).read_text())
        self.assertTrue(cases, 'The saved review cases are missing')
        for case in cases:
            with self.subTest(cases=filename, inputs=case['inputs']):
                self.assertEqual(sum(['expected' in case, 'invariants' in case,
                                      case.get('reject') is True]), 1)
                if case.get('reject'):
                    with self.assertRaises((ValidationError, ValueError, TypeError)):
                        run(case['inputs'])
                    continue
                actual = run(case['inputs'])
                if 'expected' in case:
                    self.assertTrue(equal(actual, case['expected']))
                    continue
                invariant = case['invariants']
                self.assertEqual(contract['kind'], 'model')
                self.assertIs(invariant.get('finite_prediction'), True)
                value = actual[interface['prediction_key']]
                self.assertIn(type(value), (int, float))
                self.assertTrue(math.isfinite(value))
                if 'min' in invariant:
                    self.assertGreaterEqual(value, invariant['min'])
                if 'max' in invariant:
                    self.assertLessEqual(value, invariant['max'])
                self.assertTrue(equal(actual, run(case['inputs'])), 'Predictions changed on replay')

    def test_examples_and_boundaries(self):
        self.check_cases('cases.json')

    def test_independent_review(self):
        self.check_cases('independent-cases.json')

if __name__ == '__main__':
    unittest.main()
'''


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def build_delivery(workspace, bundle, receipt, *, goal_id=''):
    from agent.harness.capability_review import admitted
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    verified = admitted(bundle, manifest)
    if verified != receipt:
        raise ValueError('Delivery must use the saved independent admission receipt')
    spec, development = manifest['review_spec'], manifest.get('development')
    is_model = spec['kind'] == 'model'
    if is_model and not development:
        raise ValueError('A reviewed model development record is required')
    identity = manifest['id']
    kind = 'model' if is_model else 'method'
    family = family_id(goal_id or identity, spec['kind'], spec['scope'])
    automatic = receipt['automatic']
    review_cases = receipt['independent'].get('tests')
    review_checks = automatic.get('independent_cases')
    from agent.harness.contracts import digest
    if (not isinstance(review_cases, list) or len(review_cases) < 2 or
            not isinstance(review_checks, list) or len(review_checks) != len(review_cases) or
            any(not isinstance(case, dict) or not isinstance(check, dict) or
                check.get('passed') is not True or
                check.get('input_sha256') != digest(case.get('inputs'))
                for case, check in zip(review_cases, review_checks))):
        raise ValueError('Independent review cases do not match the saved execution evidence')
    evaluation = automatic.get('evaluation')
    dependencies = dict(automatic['dependency_versions'])
    dependencies['jsonschema'] = importlib.metadata.version('jsonschema')
    runtime = {}
    expected_hashes = {**manifest.get('evidence_files', {}), **manifest['files']}
    source_names = set(manifest['files'])
    if is_model:
        source_names.add(spec['reproduction_script'])
    documentation = {}
    for name in sorted(source_names | {n for n in manifest.get('evidence_files', {}) if Path(n).suffix in {'.md', '.rst'}}):
        data = read_bytes(bundle, name, max_bytes=200 * 1024**2)
        if hashlib.sha256(data).hexdigest() != expected_hashes[name]:
            raise ValueError('Admitted source changed before packaging')
        if name in source_names:
            runtime['runtime/' + name] = data
        else:
            documentation['documentation/' + name] = data
    report = {'candidate_id': identity, 'kind': kind, 'review_status': receipt['status'],
              'validation_trajectory': development, 'final_test': evaluation,
              'interface_tests': automatic['interface_cases'], 'independent_tests': automatic['independent_cases'],
              'independent_review': receipt['independent'], 'dependencies': dependencies,
              'evidence_limit': 'Host checks of submitted evidence do not establish biological novelty, prospective validation or superiority.'}
    card_name = 'MODEL_CARD.md' if is_model else 'METHOD_CARD.md'
    card = '# ' + manifest['name'] + '\n\n' + manifest['description'] + '\n\n'
    card += '## Identity\n\n' + f"- Type: {kind}\n- Author: {spec['author']}\n- Version: {spec['version']}\n\n"
    for field in ('proteins', 'properties', 'scenarios', 'limitations'):
        card += '## ' + field.capitalize() + '\n\n' + ''.join('- ' + value + '\n' for value in spec['scope'][field]) + '\n'
    report_text = '# ' + ('Model' if is_model else 'Method') + ' test report\n\n' + manifest['name'] + '\n\n'
    if development:
        report_text += '## Validation trajectory\n\n| Generation | Strategy | Role | Score | Prediction behavior |\n|---|---|---|---|---|\n'
        for index, row in enumerate(development['generations']):
            diagnostics = row['validation'].get('diagnostics')
            behavior = 'Not recorded'
            if diagnostics:
                behavior = 'Constant' if diagnostics['constant_predictions'] else 'Varies across queries'
                if diagnostics['matches_training_reference']:
                    behavior += '; matches training reference'
            report_text += f"| {index} | {row['strategy']} | {row['variant']} | {row['validation']['score']:.6g} | {behavior} |\n"
        report_text += '\nSelected experiment: `' + development['selected_experiment_id'] + '`.\n'
    if evaluation:
        report_text += '\n## Final test\n\n```json\n' + json.dumps(evaluation, indent=2, ensure_ascii=False) + '\n```\n'
    report_text += '\n## Independent Review\n\n' + receipt['independent']['summary'] + '\n\n## Limitations\n\n' + report['evidence_limit'] + '\n'
    contract = {'schema': SCHEMA, 'id': identity, 'family_id': family, 'kind': kind,
                'name': manifest['name'], 'author': spec['author'], 'version': spec['version'], 'scope': spec['scope'],
                'entrypoint': {'source': manifest['source'], 'function': 'run', 'command': 'python predict.py examples/input.json'},
                'reproduction': {'script': spec.get('reproduction_script'), 'private_data_included': False},
                'selected_experiment_id': (development or {}).get('selected_experiment_id'),
                'review': {'status': 'approved', 'receipt_sha256': receipt['receipt_sha256']},
                'runtime_files': {name: hashlib.sha256(data).hexdigest() for name, data in runtime.items()}}
    example = next(case for case in spec['cases'] if 'expected' in case)
    readme = '# ' + manifest['name'] + '\n\n' + manifest['description'] + '\n\n'
    readme += '## Use\n\n```bash\npython -m pip install -r requirements.txt\npython predict.py examples/input.json\npython -m unittest discover -s tests -v\n```\n\n'
    readme += '## Contents\n\n- `runtime/`: selected implementation and fitted state, with original relative paths.\n- `capability.json` and `interface.json`: identity, applicability and input/output contract.\n- `' + card_name + '`: purpose and limitations.\n- `evaluation/`: host-measured results and independent review.\n- `examples/` and `tests/`: usage and interface checks.\n- `checksums.json`: SHA-256 for package members.\n\n'
    if is_model:
        readme += '## Reproduce training\n\nThe training entry point is `runtime/' + spec['reproduction_script'] + '`. Run it from `runtime/` with the documented private training inputs. Raw datasets and held-out labels are not shipped. See ' + card_name + ' and evaluation/test-report.md for the measured scope.\n\n'
    readme += 'Only the selected release is included. Intermediate runs and rejected candidates remain in the research record.\n'
    outputs = {
        'README.md': readme.encode(), 'capability.json': encode(contract), card_name: card.encode(),
        'interface.json': encode({k: spec[k] for k in ('input_schema', 'output_schema', 'prediction_key') if k in spec}),
        'predict.py': RUNNER.encode(), 'tests/test_interface.py': TESTS.encode(), 'tests/cases.json': encode(spec['cases']),
        'tests/independent-cases.json': encode(review_cases),
        'examples/input.json': encode(example['inputs']), 'examples/output.json': encode(example['expected']),
        'evaluation/test-report.json': encode(report), 'evaluation/test-report.md': report_text.encode(),
        'requirements.txt': ('\n'.join(k + '==' + v for k, v in sorted(dependencies.items())) + '\n').encode(),
    }
    outputs.update(documentation)
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in {**runtime, **outputs}.items()}
    package = io.BytesIO()
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted({**runtime, **outputs, 'checksums.json': encode(checksums)}.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    archive_bytes = package.getvalue()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    prefix = f'deliveries/{kind}-{identity[:16]}-{archive_sha256[:16]}'
    # Files exposes an uncluttered summary and one runnable release ZIP.
    visible = {'README.md': outputs['README.md'], card_name: outputs[card_name],
               'capability.json': outputs['capability.json'], 'test-report.md': outputs['evaluation/test-report.md'],
               'test-report.json': outputs['evaluation/test-report.json'], 'artifact.zip': archive_bytes}
    missing = {}
    for name, data in visible.items():
        path = prefix + '/' + name
        try:
            saved = read_bytes(workspace, path, max_bytes=max(len(data), 1))
        except FileNotFoundError:
            missing[path] = data
            continue
        except ValueError as exc:
            raise ValueError('Reviewed release file changed: ' + path) from exc
        if saved != data:
            raise ValueError('Reviewed release file changed: ' + path)
    for path, data in missing.items():
        write_bytes(workspace, path, data)
    return {'schema': SCHEMA, 'candidate_id': identity, 'family_id': family, 'kind': kind,
            'name': manifest['name'], 'version': spec['version'], 'package': prefix + '/artifact.zip',
            'model_package': prefix + '/artifact.zip', 'test_report': prefix + '/test-report.md',
            'model_card': prefix + '/' + card_name, 'readme': prefix + '/README.md',
            'manifest': prefix + '/capability.json', 'package_sha256': archive_sha256,
            'artifacts': [{'path': prefix + '/' + name, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in visible.items()]}
