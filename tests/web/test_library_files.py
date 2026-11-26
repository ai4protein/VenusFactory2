import json
from pathlib import Path
from backend.web_v2 import library_files as library


def test_lists_all_session_dates_and_excludes_control_files(tmp_path,monkeypatch):
    monkeypatch.setattr(library,'get_web_v2_root_dir',lambda:tmp_path)
    for name in ['sessions/older/2025/01/01/result.csv','sessions/newer/2026/10/31/input.fasta',
                 'sessions/newer/control/connection.json','sessions/newer/.env','sessions/newer/state.json']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    (tmp_path/'sessions/escape.txt').symlink_to(tmp_path/'sessions/newer/.env')
    files={str(p.relative_to(tmp_path)) for p in library.area_files('sessions')}
    assert files=={'sessions/older/2025/01/01/result.csv','sessions/newer/2026/10/31/input.fasta'}


def test_report_export_preserves_run_and_excludes_request_metadata(tmp_path,monkeypatch):
    from backend.web_v2 import research_jobs
    reports=tmp_path/'reports';identity='a'*32;folder=reports/identity;folder.mkdir(parents=True)
    state={'protein':{'name':'HBA1'},'report':'# HBA1\nExample report','sources':[{'title':'source'}]}
    state_file=folder/'state.json';state_file.write_text(json.dumps(state))
    (folder/'request.json').write_text(json.dumps({'owner':'private-owner','payload':{'sequence':'ACDE','focus':'private instructions'}}))
    monkeypatch.setattr(research_jobs,'ResearchJobs',lambda:type('Store',(),{'root':reports})())
    monkeypatch.setattr(library,'get_web_v2_root_dir',lambda:tmp_path/'web')
    before=state_file.read_bytes();library.report_exports()
    output=tmp_path/'web/results/researcher'/identity
    assert (output/'HBA1.md').read_text()=='# HBA1\nExample report'
    assert (output/'input.fasta').read_text()=='>input\nACDE\n'
    assert 'private-owner' not in (output/'evidence.json').read_text()
    assert 'private instructions' not in (output/'evidence.json').read_text()
    assert state_file.read_bytes()==before
    timestamp=(output/'HBA1.md').stat().st_mtime_ns
    library.report_exports();assert (output/'HBA1.md').stat().st_mtime_ns==timestamp


def test_agent_path_rejects_traversal_and_symlinks(tmp_path,monkeypatch):
    import pytest
    monkeypatch.setattr(library,'get_temp_outputs_base_dir',lambda:tmp_path)
    (tmp_path/'agent').mkdir();(tmp_path/'agent/result.csv').write_text('result')
    (tmp_path/'outside').write_text('secret');(tmp_path/'agent/link').symlink_to(tmp_path/'outside')
    assert library.agent_path('agent/result.csv')==tmp_path/'agent/result.csv'
    for path in ['agent/../outside','agent/link']:
        with pytest.raises(ValueError):library.agent_path(path)
def test_scientist_inputs_and_generated_workspace_are_in_library(tmp_path, monkeypatch):
    from backend.web_v2 import library_files
    root = tmp_path / 'science'
    monkeypatch.setattr(library_files, 'science_root', lambda: root)
    session = root / ('a' * 64)
    upload = session / 'uploads' / ('b' * 32) / 'observations.csv'
    output = session / ('c' * 32) / 'workspace' / 'model.py'
    config = session / ('c' * 32) / 'runtime.json'
    for path in (upload, output, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test')
    (upload.parent / '.metadata.json').write_text('{}')
    found = library_files.science_files()
    assert {p for p, _ in found} == {upload, output}
    for path, key in found:
        assert library_files.science_path(key) == path
    import pytest
    for forbidden in [config, upload.parent / '.metadata.json']:
        with pytest.raises((ValueError, OSError)):
            library_files.science_path('science/' + forbidden.relative_to(root).as_posix())
    linked = output.parent / 'linked'
    linked.symlink_to(upload.parent, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        library_files.science_path('science/' + (linked / upload.name).relative_to(root).as_posix())
