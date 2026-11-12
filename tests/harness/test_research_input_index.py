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


def test_short_fasta_is_complete_and_large_fasta_is_never_a_sequence_prefix(tmp_path):
    original = ">control\nMKWVTFISLLFLFSSAYS\n"
    item = input_file(tmp_path, "control.fasta", original)
    row = input_descriptions(tmp_path, [item])["files"][0]
    assert row["fasta_content"] == original and row["content_complete"] is True
    assert row["sequences"][0]["length"] == 18
    large = input_file(tmp_path, "large.faa", ">long\n" + "ACDE" * 400 + "\n")
    row = input_descriptions(tmp_path, [large])["files"][0]
    assert row["sequences"][0]["length"] == 1600
    assert "fasta_content" not in row and "original file" in row["inspection"]


def test_fasta_preview_is_bound_to_original_hash_and_bounded_record_set(tmp_path):
    item = input_file(tmp_path, "input.fa", ">old\nACDE\n")
    (tmp_path / "input.fa").write_text(">changed\nWXYZ\n")
    result = input_descriptions(tmp_path, [item])
    assert "WXYZ" not in json.dumps(result) and "fasta_content" not in result["files"][0]
    many = input_file(tmp_path, "many.fa", "".join(f">record{i}\nACDE\n" for i in range(20)))
    row = input_descriptions(tmp_path, [many])["files"][0]
    assert len(row["sequences"]) == 4 and row["sequences_complete"] is False
    assert "fasta_content" not in row
    invalid = input_file(tmp_path, "invalid.fa", "ACDE-without-header")
    assert "could not be parsed" in input_descriptions(tmp_path, [invalid])["files"][0]["inspection"]
