import hashlib
import json
from pathlib import Path

from agent.harness.research_input_index import input_descriptions, MAX_BYTES


def input_file(root, name, content):
    path=root/name
    path.write_text(content)
    return {"name":name,"path":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def test_planner_sees_types_and_assay_notes_without_measurements(tmp_path):
    secret_value="90210.12345"
    sequence="ACDEFGHIKLMNPQRSTVWY"*3
    item=input_file(tmp_path,"data.csv","id,sequence,score\nsecret_sample_id,"+sequence+","+secret_value+"\n")
    notes=input_file(tmp_path,"notes.md","The score is fluorescence, measured at 30 C; not percent solubility.")
    result=input_descriptions(tmp_path,[item,notes])
    encoded=json.dumps(result)
    assert secret_value not in encoded and sequence not in encoded and "secret_sample_id" not in encoded
    assert result["files"][0]["columns"][1]=={"name":"sequence","observed_types":["protein_sequence"]}
    assert "not percent solubility" in result["files"][1]["material_excerpt"]
    assert all(x["sha256"] for x in result["files"])


def test_changed_files_and_escaping_links_never_disclose_text(tmp_path):
    root=tmp_path/"work";root.mkdir()
    outside=tmp_path/"private.txt";outside.write_text("PRIVATE_MARKER")
    link=root/"link.txt";link.symlink_to(outside)
    base={"sha256":hashlib.sha256(outside.read_bytes()).hexdigest()}
    assert input_descriptions(root,[{**base,"path":"../private.txt"},{**base,"path":"link.txt"}])["files"]==[]
    original=input_file(root,"notes.md","Original")
    (root/"notes.md").write_text("CHANGED_SECRET")
    result=input_descriptions(root,[original])
    assert "CHANGED_SECRET" not in json.dumps(result)
    assert "differs" in result["files"][0]["inspection"]


def test_large_and_many_materials_are_bounded(tmp_path):
    items=[input_file(tmp_path,f"note{i}.txt",str(i)*4000) for i in range(32)]
    result=input_descriptions(tmp_path,items)
    assert result["total_inputs"]==32 and len(result["files"])<=8
    assert len(json.dumps(result,ensure_ascii=False))<6600
    assert all(x["excerpted"] for x in result["files"])
    item=input_file(tmp_path,"large.txt","x")
    with (tmp_path/"large.txt").open("r+b") as handle:handle.truncate(MAX_BYTES+1)
    result=input_descriptions(tmp_path,[item])
    assert "large input" in result["files"][0]["inspection"] and "material_excerpt" not in result["files"][0]


def test_json_values_are_not_implicitly_disclosed(tmp_path):
    item=input_file(tmp_path,"metadata.json",json.dumps({"measurement":123.456,"private_note":"VALUE_NOT_FOR_INDEX"}))
    result=input_descriptions(tmp_path,[item])
    assert result["files"][0]["top_level_fields"]==["measurement","private_note"]
    assert "123.456" not in json.dumps(result) and "VALUE_NOT_FOR_INDEX" not in json.dumps(result)
