import asyncio
import pandas as pd
from typing import Dict, List
from itertools import combinations

try:
    from .llm import LLM_Model
    from .prompts import PAIRWISE_ANALYST_PROMPT, JUDGE_PROMPT
except ImportError:  # Preserve the existing direct-script entry point.
    from llm import LLM_Model
    from prompts import PAIRWISE_ANALYST_PROMPT, JUDGE_PROMPT

async def analyze_pair_agent(llm: LLM_Model, question: str, 
                             response_a: str, response_b: str) -> str:
    """Agent 1: Generate response comparison analysis"""
    chain = PAIRWISE_ANALYST_PROMPT | llm
    try:
        result = await chain.ainvoke({
            "question": question,
            "response_a": response_a,
            "response_b": response_b
        })
        return result.content
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def parse_judgment(content: str) -> str:
    """Accept only an explicit decision; prose and malformed answers are invalid."""
    decision = str(content).strip().upper()
    return decision if decision in {"A", "B", "TIE"} else "INVALID"


async def judge_pair_agent(llm: LLM_Model, question: str, analysis: str) -> str:
    chain = JUDGE_PROMPT | llm
    try:
        result = await chain.ainvoke({"question": question, "analysis": analysis})
        return parse_judgment(result.content)
    except Exception:
        return "ERROR"


async def run_matchup(llm: LLM_Model, question: str, 
                      name_a: str, response_a: str, 
                      name_b: str, response_b: str) -> Dict:
    """
    Run single match: Agent 1 Analysis -> Agent 2 Judgment
    Returns: {
        "pair": "A vs B",
        "winner_name": "A" or "B",
        "analysis": "..."
    }
    """
    # 1. Run Analysis Agent
    analysis = await analyze_pair_agent(llm, question, response_a, response_b)
    
    # 2. Run Judge Agent
    winner_code = "ERROR" if analysis.startswith("Analysis failed:") else await judge_pair_agent(llm, question, analysis)
    
    winner_name = {"A": name_a, "B": name_b}.get(winner_code)
    
    return {
        "pair_key": f"{name_a}_vs_{name_b}",
        "winner_name": winner_name,
        "analysis": analysis,
        "winner_code": winner_code,
        "status": "valid" if winner_code in {"A", "B", "TIE"} else winner_code.lower(),
        "names": [name_a, name_b]
    }

async def process_single_question(row: pd.Series, model_columns: List[str], llm: LLM_Model):
    """
    Process single question:
    1. Extract data
    2. Generate all pairwise combinations
    3. Execute all matchups concurrently (Analysis + Judge)
    4. Compute rankings and record
    """
    question = row["Question"]
    
    # 1. Extract model response data
    model_data = {}
    for model_name in model_columns:
        model_data[model_name] = str(row.get(model_name, ""))
    
    model_names = list(model_data.keys())
    
    # 2. Generate matchup list
    pairs = list(combinations(model_names, 2))
    
    print(f"Processing Question ID {row.get('index', '?')}: Running {len(pairs)} pairwise matchups...")
    
    # 3. Execute all matchups concurrently
    matchup_tasks = []
    for name_a, name_b in pairs:
        matchup_tasks.append(
            run_matchup(llm, question, name_a, model_data[name_a], name_b, model_data[name_b])
        )
    
    matchup_results = await asyncio.gather(*matchup_tasks)
    
    # 4. Tally scores and record results
    scores = {name: 0 for name in model_names}
    result_row = row.to_dict()
    
    for res in matchup_results:
        # Tally scores
        if res["winner_code"] in {"A", "B"}:
            scores[res["winner_name"]] += 1
        elif res["winner_code"] == "TIE":
            for name in res["names"]:
                scores[name] += 0.5
        
        # Record detailed analysis for each match
        pair_key = res['pair_key']
        result_row[f'Analysis_{pair_key}'] = res['analysis']
        result_row[f'Winner_{pair_key}'] = res['winner_name']
        result_row[f'Status_{pair_key}'] = res['status']
        result_row[f'Decision_{pair_key}'] = res['winner_code']
    
    # Do not manufacture a complete ranking from failed matchups.
    complete = all(res["status"] == "valid" for res in matchup_results)
    result_row["Evaluation_Complete"] = complete
    result_row["Invalid_Matchups"] = sum(res["status"] != "valid" for res in matchup_results)
    for name, score in scores.items():
        result_row[f"Rank_{name}"] = 1 + sum(v > score for v in scores.values()) if complete else None

    result_row['Score_Matrix'] = str(scores)
    
    print(f"Finished Question ID {row.get('index', '?')}: Scores = {scores}")
    return result_row
