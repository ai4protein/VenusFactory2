import os
import asyncio
import pandas as pd
from config import INPUT_FILE, OUTPUT_FILE, API_KEY, MODEL_COLUMNS
from llm import LLM_Model
from evaluation import process_single_question

async def main():
    
    # Instantiate LLM
    llm = LLM_Model(
        api_key=API_KEY,
        model_name="gpt-4o-mini", 
        temperature=0.0
    )
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, encoding='utf-8', encoding_errors='ignore')
    
    # Verify if columns exist
    missing_cols = [col for col in MODEL_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        return

    # --- Execute main loop ---
    tasks = []
    for index, row in df.iterrows():
        task = process_single_question(row, MODEL_COLUMNS, llm)
        tasks.append(task)
    
    # Limit concurrency (Prevent API rate limits)
    sem = asyncio.Semaphore(5)
    
    async def sem_task(task):
        async with sem:
            return await task
            
    results = await asyncio.gather(*(sem_task(t) for t in tasks))
    
    # --- Save results ---
    result_df = pd.DataFrame(results)
    
    cols = list(result_df.columns)
    rank_cols = sorted([c for c in cols if c.startswith("Rank_")])
    analysis_cols = sorted([c for c in cols if c.startswith("Analysis_") or c.startswith("Winner_")])
    base_cols = [c for c in cols if c not in rank_cols and c not in analysis_cols and c != "Score_Matrix"]
    
    final_df = result_df[base_cols + rank_cols + ["Score_Matrix"] + analysis_cols]
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n=== Completed! ===")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total questions processed: {len(results)}")

if __name__ == "__main__": 
    asyncio.run(main())