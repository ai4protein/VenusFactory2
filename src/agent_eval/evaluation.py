import re
import asyncio
import pandas as pd
from typing import Dict, List
from itertools import combinations

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

async def judge_pair_agent(llm: LLM_Model, question: str, analysis: str) -> str:
    """Agent 2: Determine winner"""
    chain = JUDGE_PROMPT | llm
    try:
        result = await chain.ainvoke({
            "question": question,
            "analysis": analysis
        })
        # Clean output
        content = result.content.strip().upper()
        # Extract A or B
        match = re.search(r'\b([AB])\b', content)
        if match:
            return match.group(1)
        if "A" in content: return "A"
        if "B" in content: return "B"
        return "A" # Fallback
    except Exception as e:
        print(f"Judge failed: {e}")
        return "A"

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
    winner_code = await judge_pair_agent(llm, question, analysis)
    
    winner_name = name_a if winner_code == 'A' else name_b
    
    return {
        "pair_key": f"{name_a}_vs_{name_b}",
        "winner_name": winner_name,
        "analysis": analysis,
        "winner_code": winner_code # A or B
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
        scores[res['winner_name']] += 1
        
        # Record detailed analysis for each match
        pair_key = res['pair_key']
        result_row[f'Analysis_{pair_key}'] = res['analysis']
        result_row[f'Winner_{pair_key}'] = res['winner_name']
    
    # 5. Calculate final rankings
    ranked_models = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    ranks = {}
    for i, (name, score) in enumerate(ranked_models):
        ranks[name] = i + 1 # Rank 1 is best
        
    # Record rankings and scores
    for name, rank in ranks.items():
        result_row[f'Rank_{name}'] = rank
    
    result_row['Score_Matrix'] = str(scores)
    
    print(f"Finished Question ID {row.get('index', '?')}: Scores = {scores}")
    return result_row
