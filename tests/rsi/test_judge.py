import pandas as pd
import pytest

from agent.agent_eval import evaluation as ev


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        (" b\n", "B"),
        ("TIE", "TIE"),
        ("API unavailable", "INVALID"),
        ("A or B", "INVALID"),
        ("", "INVALID"),
        ("Winner: A", "INVALID"),
    ],
)
def test_strict_judgment(text, expected):
    assert ev.parse_judgment(text) == expected


@pytest.mark.asyncio
async def test_failed_analysis_never_selects_a(monkeypatch):
    async def fail(*args):
        return "Analysis failed: network"

    monkeypatch.setattr(ev, "analyze_pair_agent", fail)
    result = await ev.run_matchup(None, "q", "first", "x", "second", "y")
    assert result["status"] == "error" and result["winner_name"] is None


@pytest.mark.asyncio
async def test_ties_share_ranks_and_errors_suppress_rankings(monkeypatch):
    code = "TIE"

    async def matchup(llm, question, a, response_a, b, response_b):
        return {
            "pair_key": a + "_vs_" + b,
            "names": [a, b],
            "analysis": "",
            "winner_code": code,
            "winner_name": None,
            "status": "valid" if code == "TIE" else "invalid",
        }

    monkeypatch.setattr(ev, "run_matchup", matchup)
    row = pd.Series({"Question": "q", "one": "x", "two": "y"})
    result = await ev.process_single_question(row, ["one", "two"], None)
    assert result["Rank_one"] == result["Rank_two"] == 1
    code = "INVALID"
    result = await ev.process_single_question(row, ["one", "two"], None)
    assert result["Rank_one"] is None and not result["Evaluation_Complete"]
