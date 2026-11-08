import json

from mcp_tools.common.common_utils import redact_path_text


def test_json_receipt_preserves_escaped_preview_and_numeric_results(tmp_path, monkeypatch):
    from mcp_tools.common import common_utils
    monkeypatch.setattr(common_utils, "get_project_root", lambda: tmp_path)
    preview = json.dumps({"sequence":"ACDE", "value":2395.71}, indent=2)
    payload = {"status":"success",
               "file_info":{"file_path":str(tmp_path/"results/value.json")},
               "content_preview":preview, "data":{"pI":6.78}}
    result = json.loads(redact_path_text(json.dumps(payload)))
    assert result["content_preview"] == preview
    assert json.loads(result["content_preview"])["value"] == 2395.71
    assert result["file_info"]["file_path"] == "./results/value.json"
    assert result["data"] == {"pI":6.78}


def test_json_redaction_removes_host_path_without_swallowing_structure():
    original={"error":"/home/fixture/private/model.pkl", "value":7,
              "rows":[{"path":r"C:\Users\fixture\private\weights.bin"}]}
    result=json.loads(redact_path_text(json.dumps(original)))
    assert result=={"error":"model.pkl","value":7,"rows":[{"path":"weights.bin"}]}


def test_plain_scientific_content_is_not_path_normalized():
    text = r'Use r"\d+\.\d+"; write("\n"); \alpha + \beta'
    assert redact_path_text(text) == text
    url = "https://example.org/data/results.json"
    assert redact_path_text(url) == url
    assert redact_path_text('Cannot open "/home/fixture/private/data.csv". Retry.') == 'Cannot open "data.csv". Retry.'


def test_snapshot_keeps_nested_json_readable_and_redacts_secret_fields():
    from backend.web_v2.chat_api._shared import _redact_obj
    payload={"output":json.dumps({"text":'line one\n"line two"',"path":"/home/fixture/private/a.csv"}),
             "api_key":"fixture-secret-not-a-live-key"}
    result=_redact_obj(payload)
    assert json.loads(result["output"]) == {"text":'line one\n"line two"',"path":"a.csv"}
    assert result["api_key"] == "[REDACTED]"

def test_literal_double_backslashes_are_not_treated_as_network_paths():
    text = r'json escapes: \\n, \\t; regex \\d+'
    assert redact_path_text(text) == text
    assert redact_path_text(r'file \\server\share\private\weights.bin') == 'file weights.bin'
