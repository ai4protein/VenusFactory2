import pytest
from backend.web_v2.property_results import residue_rows

def test_sasa_preserves_chains_and_insertion_codes():
    result=residue_rows({"A":{"10A":{"sasa":12.5,"resname":"ALA"}},
                         "B":{"10":{"sasa":9.2,"resname":"GLY"}}},"sasa")
    assert result==[{"chain":"A","residue":"10A","sasa":12.5,"resname":"ALA"},
                    {"chain":"B","residue":"10","sasa":9.2,"resname":"GLY"}]

@pytest.mark.parametrize("kind,values",[("rsa",{"aa":"G","rsa":0.25}),
                                       ("secondary_structure",{"aa_seq":"G","ss3_seq":"C"})])
def test_single_chain_properties_keep_residue_alignment(kind,values):
    assert residue_rows({"12":values},kind,"H")==[{"chain":"H","residue":"12",**values}]

def test_malformed_results_do_not_claim_predictions():
    with pytest.raises(ValueError):residue_rows({"A":["bad"]},"sasa")
