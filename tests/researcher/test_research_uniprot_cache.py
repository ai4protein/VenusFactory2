import sqlite3
from unittest.mock import AsyncMock

import pytest
from backend.web_v2 import research_report as report


@pytest.mark.asyncio
async def test_selected_public_record_survives_a_worker_and_expires(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_REPORT_ROOT", str(tmp_path))
    entry = {"primaryAccession": "P42212", "sequence": {"value": "AG", "length": 2}}
    fetch = AsyncMock(return_value={"results": [entry]})
    monkeypatch.setattr(report, "get_json", fetch)
    cards = await report.resolve_protein("P42212")
    assert cards[0]["accession"] == "P42212"
    assert await report.selected_uniprot_entry("p42212") == entry
    assert fetch.await_count == 1
    with sqlite3.connect(tmp_path / ".uniprot.sqlite3") as conn:
        conn.execute("UPDATE entries SET fetched=0")
    fetch.return_value = entry
    assert await report.selected_uniprot_entry("P42212") == entry
    fetch.assert_awaited_with("https://rest.uniprot.org/uniprotkb/P42212.json")
