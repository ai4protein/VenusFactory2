import copy
import json

import pytest

from agent.harness.capability_review import review_proposal
from agent.harness.research_workspace import ResearchWorkspace, run_python
from agent.harness.review_submission import ReviewSubmission
from agent.harness.contracts import file_hash


def proposal():
    return {
        'name': 'Length', 'description': 'Software interface fixture',
        'source': 'length.py', 'files': [], 'test_inputs': [{'sequence': 'A'}],
        'review_spec': {
            'kind': 'tool', 'author': 'Fixture team', 'version': '1.0',
            'scope': {'proteins': ['software fixture'], 'properties': ['length'],
                      'scenarios': ['testing'], 'limitations': ['not a biological model']},
            'input_schema': {'type': 'object', 'required': ['sequence'],
                             'properties': {'sequence': {'type': 'string', 'minLength': 1}},
                             'additionalProperties': False},
            'output_schema': {'type': 'object', 'required': ['value'],
                              'properties': {'value': {'type': 'number'}},
                              'additionalProperties': False},
            'dependencies': [], 'evidence_files': ['README.md'],
            'cases': [{'inputs': {'sequence': 'AC'}, 'expected': {'value': 2}},
                      {'inputs': {'sequence': ''}, 'reject': True}],
        },
    }


async def setup(tmp_path):
    service = ResearchWorkspace(tmp_path/'workspace', tmp_path/'iteration1', {'steps': []})
    await service.write_research_file('length.py', 'def run(inputs):\n return {"value": len(inputs["sequence"])}\n')
    await service.write_research_file('README.md', 'Software-only interface fixture.')
    return service


@pytest.mark.asyncio
async def test_saved_proposal_survives_incomplete_retry_and_new_iteration(tmp_path):
    service = await setup(tmp_path)
    original = proposal()
    first = await service.propose_research_capability(**original)
    assert first['name'] == 'Length'  # evidence filenames must not overwrite the name
    draft = service.workspace/first['proposal_file']
    assert json.loads(draft.read_text()) == original
    manifest = service.proposals[0]/'manifest.json'
    frozen_hash = file_hash(manifest)
    broken = copy.deepcopy(original)
    broken['review_spec'].pop('author')
    with pytest.raises(ValueError, match='Declare author.*proposal_file'):
        await service.propose_research_capability(**broken)
    assert json.loads(draft.read_text()) == original
    assert service.latest_submission['status'] == 'invalid'
    requests = [json.loads(p.read_text()) for p in (service.control/'submissions').glob('*/request.json')]
    assert len(requests) == 2
    assert any('author' not in row['review_spec'] for row in requests)
    next_service = ResearchWorkspace(service.workspace, tmp_path/'iteration2', {'steps': []})
    result = await next_service.propose_research_capability(proposal_file=first['proposal_file'])
    assert result['submission_status'] == 'awaiting_review'
    assert json.loads((next_service.proposals[0]/'manifest.json').read_text())['review_spec']['author'] == 'Fixture team'
    assert file_hash(manifest) == frozen_hash


@pytest.mark.asyncio
async def test_revision_file_does_not_overwrite_failed_admission(tmp_path):
    service = await setup(tmp_path)
    broken = proposal()
    broken['review_spec']['cases'][0]['expected']['value'] = 99
    first = await service.propose_research_capability(**broken)
    bundle = service.proposals[0]
    async def no_reviewer(_):
        raise AssertionError('Automatic failure must not reach the independent reviewer')
    receipt = await review_proposal(bundle, reviewer=no_reviewer,
                                   control=tmp_path/'admission1', runner=run_python)
    assert receipt['status'] == 'needs_revision' and receipt['independent'] is None
    assert 'case 1' in receipt['issues'][0]
    admission_hash = file_hash(bundle/'admission.json')
    corrected = json.loads((service.workspace/first['proposal_file']).read_text())
    corrected['review_spec']['cases'][0]['expected']['value'] = 2
    await service.write_research_file(first['proposal_file'], json.dumps(corrected))
    next_service = ResearchWorkspace(service.workspace, tmp_path/'iteration2', {'steps': []})
    result = await next_service.propose_research_capability(proposal_file=first['proposal_file'])
    assert result['id'] != first['id']
    assert file_hash(bundle/'admission.json') == admission_hash
    assert json.loads((next_service.proposals[0]/'manifest.json').read_text())['review_spec']['author'] == 'Fixture team'


@pytest.mark.parametrize('path', ['../outside.json', '/tmp/outside.json', '.hidden.json'])
def test_submission_paths_are_scoped(tmp_path, path):
    work = tmp_path/'work'; work.mkdir()
    with pytest.raises((ValueError, OSError)):
        ReviewSubmission(work, tmp_path/'control', {}, path)


def test_linked_draft_and_mixed_forms_are_rejected(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    outside = tmp_path/'outside.json'; outside.write_text(json.dumps(proposal()))
    (work/'draft.json').symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        ReviewSubmission(work, tmp_path/'control', {}, 'draft.json')
    (work/'plain.json').write_text(json.dumps(proposal()))
    with pytest.raises(ValueError, match='alone'):
        ReviewSubmission(work, tmp_path/'control', proposal(), 'plain.json')
    assert json.loads(outside.read_text()) == proposal()


def test_file_tool_schema_accepts_a_single_path(tmp_path):
    service = ResearchWorkspace(tmp_path/'work', tmp_path/'control', {'steps': []})
    tool = next(t for t in service.tools() if t.name == 'propose_research_capability')
    args = tool.args_schema.model_validate({'proposal_file': 'review/proposal.json'})
    assert args.proposal_file == 'review/proposal.json'
