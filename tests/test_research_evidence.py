import copy
import json

from backend.web_v2.research_evidence import evidence_excerpt
from backend.web_v2.research_longform import source_context
from backend.web_v2.research_editor import editorial_input


def source():
    record={"organism":{"scientificName":"Example organism"},
            "comments":[{"commentType":"FUNCTION","texts":[
                {"evidences":[{"source":"PubMed","id":str(i)} for i in range(300)],
                 "value":"Experimentally observed catalytic function."}]},
                {"commentType":"CATALYTIC ACTIVITY","reaction":{"name":"substrate + water = products"}},
                {"commentType":"SUBUNIT","texts":[{"value":"Experimentally observed monomer."}]}]}
    return {"id":"S1","kind":"database","text":json.dumps(record),"title":"Primary annotation"}


def test_large_reference_array_cannot_hide_annotation_or_reaction():
    row=source();original=copy.deepcopy(row)
    assert "Experimentally observed catalytic function." not in row["text"][:2400]
    text,excerpted=evidence_excerpt(row,8000)
    data=json.loads(text)
    assert data["comments"][0]["texts"][0]["value"]=="Experimentally observed catalytic function."
    assert "substrate + water = products" in text and "observed monomer" in text
    assert data["comments"][0]["texts"][0]["_omitted_items"]["evidences"]==297
    assert excerpted and row==original


def test_specialists_and_editor_receive_compacted_primary_facts():
    row=source()
    assert "observed catalytic function" in source_context([row])[0]["text"]
    state={"chapters":[],"sources":[row],"protein":{}}
    output=editorial_input(state,{})
    assert "observed catalytic function" in output["evidence_excerpts"][0]["text"]
    assert "substrate + water = products" in output["evidence_excerpts"][0]["text"]


def test_reference_only_parenthesis_is_shortened_without_deleting_claims():
    refs=", ".join("PubMed:"+str(i) for i in range(50))
    row={"kind":"database","text":json.dumps({"value":"First fact ("+refs+"). Second fact."})}
    text,_=evidence_excerpt(row,2000)
    assert "First fact" in text and "Second fact" in text and "47 additional references in original" in text


def test_materials_remain_verbatim_and_invalid_json_stays_explicitly_excerpted():
    text='{"not valid: '
    result,cut=evidence_excerpt({"kind":"database","text":text},6)
    assert result==text[:6] and cut
    material={"kind":"user_material","text":json.dumps({"evidences":list(range(100))})}
    result,cut=evidence_excerpt(material,8000)
    assert result==material["text"] and not cut


def test_original_reference_and_total_context_bounds_retained():
    first=source()
    rows=[first]+[{"id":f"S{i}","kind":"database","text":"x"*16000} for i in range(2,30)]
    result=editorial_input({"chapters":[],"sources":rows,"protein":{}},{})
    assert len(result["evidence_excerpts"])==20
    assert sum(len(row["text"]) for row in result["evidence_excerpts"])<=46000
    assert len(result["sources"])==29
