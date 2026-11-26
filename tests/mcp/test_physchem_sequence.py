import json
import pytest
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from mcp_tools.tools.predict.features.calculate_physchem import calculate_physchem_from_sequence, calculate_physchem_from_fasta

SEQUENCE = 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN'

def test_direct_sequence_matches_fasta_and_biopython(tmp_path):
    path = tmp_path / 'crambin.fasta'
    path.write_text('>sequence\n' + SEQUENCE)
    result = calculate_physchem_from_sequence(SEQUENCE)
    assert result == calculate_physchem_from_fasta(str(path))
    assert result['sequence_length'] == 46
    assert result['molecular_weight'] == round(ProteinAnalysis(SEQUENCE).molecular_weight(), 2)
    assert result['theoretical_pI'] == round(ProteinAnalysis(SEQUENCE).isoelectric_point(), 2)

def test_normalizes_single_fasta_lowercase_whitespace_and_terminal_stop():
    result = calculate_physchem_from_sequence('>crambin description\n' + SEQUENCE.lower()[:20] + '\n' + SEQUENCE.lower()[20:] + '*')
    assert result['sequence_id'] == 'crambin'
    assert result['sequence_length'] == 46

@pytest.mark.parametrize('sequence', ['', '   ', 'AXC', 'AA*CC', 'AC23DE', '>a\nAC\n>b\nDE', 'A'*100001])
def test_invalid_input_is_rejected_without_silent_residue_changes(sequence):
    with pytest.raises(ValueError): calculate_physchem_from_sequence(sequence)

def test_registered_sequence_tool_writes_a_result_without_model_or_network(tmp_path):
    from mcp_tools.tools.predict.tools_agent import calculate_physchem_from_sequence_tool
    from mcp_tools.scheduler.catalog import classify
    result = json.loads(calculate_physchem_from_sequence_tool.invoke({'sequence': SEQUENCE, 'out_dir': str(tmp_path)}))
    assert result['status'] == 'success'
    assert result['data']['sequence_length'] == 46
    assert result['units']['molecular_weight'] == 'Da'
    assert (tmp_path / 'sequence_physchem.json').is_file()
    assert classify(calculate_physchem_from_sequence_tool.name)['resource'] == 'cpu'

def test_invalid_input_does_not_write_result(tmp_path):
    from mcp_tools.tools.predict.tools_agent import calculate_physchem_from_sequence_tool
    result = json.loads(calculate_physchem_from_sequence_tool.invoke({'sequence': 'AXC', 'out_dir': str(tmp_path)}))
    assert result['status'] == 'error'
    assert list(tmp_path.iterdir()) == []
