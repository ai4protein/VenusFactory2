import sys
import os
sys.path.append(os.getcwd())
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch
import gc
from typing import List, Dict, Tuple
from src.mutation.utils import generate_mutations_from_sequence
from src.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb

# Import all scoring functions
from src.mutation.models.saprot import saprot_score
from src.mutation.models.protssn import protssn_score
from src.mutation.models.prosst import prosst_score
from src.mutation.models.esmif1 import esmif1_score


def clear_gpu_memory():
    """
    Clear GPU memory to prevent out-of-memory errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory cleared")


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores using z-score normalization.
    
    Args:
        scores: List of raw scores
        
    Returns:
        List of normalized scores
    """
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    if std == 0:
        return scores  # Return original scores if std is 0
    normalized = (scores_array - mean) / std
    return normalized.tolist()


def select_recommended_mutations(df: pd.DataFrame, num_recommendations: int = 30) -> List[str]:
    """
    Select recommended mutations based on frequency analysis of top mutations from each model.
    
    Args:
        df: DataFrame with all model scores
        num_recommendations: Number of recommended mutations
        
    Returns:
        List of recommended mutations
    """
    model_columns = ['saprot_score', 'protssn_score', 'prosst_score', 'esmif1_score']
    selection_reasons = {}  # Track selection reasons for each mutation
    
    def get_position_from_mutant(mutant: str) -> str:
        """Extract position from mutant string, e.g., M1A -> 1"""
        return ''.join(filter(str.isdigit, mutant))
    
    def count_mutation_frequency(top_n: int) -> Dict[str, int]:
        """Count frequency of mutations in top N from each model"""
        mutation_counts = {}
        mutation_sources = {}  # Track which models contribute to each mutation
        
        for model_col in model_columns:
            top_mutations = df.nlargest(top_n, model_col)['mutant'].tolist()
            print(f"  {model_col} top {top_n}: {top_mutations[:10]}...")  # Show first 10 for debugging
            for mut in top_mutations:
                mutation_counts[mut] = mutation_counts.get(mut, 0) + 1
                if mut not in mutation_sources:
                    mutation_sources[mut] = []
                mutation_sources[mut].append(model_col)
        
        return mutation_counts, mutation_sources
    
    def get_mutation_rankings(mutant: str) -> Dict[str, int]:
        """Get ranking of mutation in each model"""
        rankings = {}
        for model_col in model_columns:
            model_df = df.sort_values(model_col, ascending=False).reset_index(drop=True)
            try:
                rank = model_df[model_df['mutant'] == mutant].index[0] + 1
                rankings[model_col] = rank
            except IndexError:
                rankings[model_col] = None
        return rankings
    
    def format_rankings(rankings: Dict[str, int]) -> str:
        """Format rankings for display"""
        rank_strs = []
        for model, rank in rankings.items():
            model_short = model.replace('_score', '')
            if rank is not None:
                rank_strs.append(f"{model_short}:#{rank}")
            else:
                rank_strs.append(f"{model_short}:N/A")
        return ', '.join(rank_strs)
    
    def filter_by_position(mutation_counts: Dict[str, int], mutation_sources: Dict[str, List[str]], step_name: str, top_n: int) -> List[str]:
        """Filter mutations to keep only one per position, preferring higher frequency and better ranking"""
        position_best = {}
        
        for mutant, count in mutation_counts.items():
            position = get_position_from_mutant(mutant)
            
            if position not in position_best:
                position_best[position] = (mutant, count)
            else:
                current_mutant, current_count = position_best[position]
                if count > current_count:
                    position_best[position] = (mutant, count)
                elif count == current_count:
                    # If same frequency, compare rankings and keep the better one
                    current_rankings = get_mutation_rankings(current_mutant)
                    new_rankings = get_mutation_rankings(mutant)
                    
                    # Calculate average rank (lower is better)
                    current_ranks = [r for r in current_rankings.values() if r is not None]
                    new_ranks = [r for r in new_rankings.values() if r is not None]
                    
                    if not current_ranks or not new_ranks:
                        continue  # Skip if we can't compare rankings
                        
                    current_avg_rank = np.mean(current_ranks)
                    new_avg_rank = np.mean(new_ranks)
                    
                    if new_avg_rank < current_avg_rank:  # New mutation has better average rank
                        position_best[position] = (mutant, count)
        
        # Extract mutations
        selected_mutations = []
        
        for position, (mutant, count) in position_best.items():
            rankings = get_mutation_rankings(mutant)
            ranking_str = format_rankings(rankings)
            overlap_models = mutation_sources.get(mutant, [])
            overlap_models_short = [m.replace('_score', '') for m in overlap_models]
            
            selected_mutations.append(mutant)
            selection_reasons[mutant] = f"{step_name}: Best at position {position} ({count}/4 models). Rankings: [{ranking_str}]. Overlap in top{top_n}: {', '.join(overlap_models_short)}"
        
        return selected_mutations
    
    # Step 1: Count frequency in top N from each model (N = num_recommendations)
    initial_top_n = num_recommendations
    print(f"Step 1: Analyzing frequency in top {initial_top_n} mutations from each model...")
    initial_counts, initial_sources = count_mutation_frequency(initial_top_n)
    
    # Filter to keep only mutations that appear in multiple models (frequency >= 2)
    frequent_mutations = {mut: count for mut, count in initial_counts.items() if count >= 2}
    print(f"Found {len(frequent_mutations)} mutations appearing in multiple models' top {initial_top_n}")
    
    # Filter by position preference
    selected_mutations = filter_by_position(frequent_mutations, initial_sources, "Step1", initial_top_n)
    selected_positions_count = len(set(get_position_from_mutant(mut) for mut in selected_mutations))
    print(f"After position filtering: {len(selected_mutations)} mutations, {selected_positions_count} unique positions")
    
    # Sort by frequency (descending) and then by ensemble normalized score
    selected_mutations.sort(key=lambda x: (frequent_mutations.get(x, 0), 
                                         df[df['mutant'] == x]['ensemble_norm_score'].iloc[0]), 
                          reverse=True)
    
    # Track positions already selected in Step 1
    selected_positions_step1 = set(get_position_from_mutant(mut) for mut in selected_mutations)
    
    # Check if we have enough unique positions
    selected_positions_count = len(selected_positions_step1)
    if selected_positions_count >= num_recommendations:
        # Take only the first num_recommendations positions
        final_selections = []
        seen_positions = set()
        for mut in selected_mutations:
            if len(final_selections) >= num_recommendations:
                break
            pos = get_position_from_mutant(mut)
            if pos not in seen_positions:
                final_selections.append(mut)
                seen_positions.add(pos)
        
        print(f"\n=== Selection Reasons ===")
        for i, mut in enumerate(final_selections, 1):
            print(f"{i:2d}. {mut}: {selection_reasons.get(mut, 'Unknown reason')}")
        return final_selections
    
    # Step 2: If not enough positions, expand to top N*3 from each model
    remaining_needed = num_recommendations - selected_positions_count
    expand_top_n = max(num_recommendations * 3, 30)  # At least 30, but scale with num_recommendations
    print(f"Step 2: Need {remaining_needed} more positions, expanding to top {expand_top_n}...")
    
    expanded_counts, expanded_sources = count_mutation_frequency(expand_top_n)
    
    # Filter to keep only mutations that appear in multiple models, not already selected, and not at already selected positions
    additional_frequent = {}
    for mut, count in expanded_counts.items():
        if count >= 2 and mut not in selected_mutations:
            pos = get_position_from_mutant(mut)
            if pos not in selected_positions_step1:  # Exclude positions already selected in Step 1
                additional_frequent[mut] = count
    
    print(f"Found {len(additional_frequent)} additional mutations in top {expand_top_n} (excluding already selected positions)")
    
    # Filter by position preference for additional mutations
    additional_selected = filter_by_position(additional_frequent, expanded_sources, "Step2", expand_top_n)
    
    # Remove any that are already selected or at already selected positions
    additional_selected = [mut for mut in additional_selected 
                          if mut not in selected_mutations and get_position_from_mutant(mut) not in selected_positions_step1]
    
    # Sort additional mutations by frequency and ensemble normalized score
    additional_selected.sort(key=lambda x: (additional_frequent.get(x, 0), 
                                          df[df['mutant'] == x]['ensemble_norm_score'].iloc[0]), 
                           reverse=True)
    
    # Add additional mutations, but limit by remaining needed positions
    positions_added = 0
    for mut in additional_selected:
        if positions_added >= remaining_needed:
            break
        pos = get_position_from_mutant(mut)
        # Double check - this position should not be covered (but should be redundant now)
        if pos not in selected_positions_step1:
            selected_mutations.append(mut)
            selected_positions_step1.add(pos)  # Update the set of selected positions
            positions_added += 1
    
    # Update selected positions count after Step 2
    selected_positions_count = len(selected_positions_step1)
    
    # Step 3: If still not enough positions, select from individual models with specific ratios
    if selected_positions_count < num_recommendations:
        remaining_needed = num_recommendations - selected_positions_count
        print(f"Step 3: Need {remaining_needed} more positions, selecting from individual models...")
        print(f"Current selected mutations: {len(selected_mutations)}")
        print(f"Current unique positions: {selected_positions_count}")
        print(f"Target: {num_recommendations}")
        
        # Define model ratios: prosst:protssn:saprot:esmif1 = 3:2:2:2
        model_ratios = {
            'prosst_score': 3,
            'protssn_score': 2, 
            'saprot_score': 2,
            'esmif1_score': 2
        }
        
        # Calculate how many to select from each model
        total_ratio = sum(model_ratios.values())  # 9
        model_selections = {}
        for model_col, ratio in model_ratios.items():
            model_selections[model_col] = max(1, int(remaining_needed * ratio / total_ratio))
        
        # Adjust to ensure we get exactly remaining_needed mutations
        current_total = sum(model_selections.values())
        if current_total < remaining_needed:
            # Add extra to prosst (highest priority)
            model_selections['prosst_score'] += remaining_needed - current_total
        elif current_total > remaining_needed:
            # Remove excess from lowest priority models
            excess = current_total - remaining_needed
            for model_col in ['esmif1_score', 'saprot_score', 'protssn_score', 'prosst_score']:
                if excess <= 0:
                    break
                reduce_by = min(excess, model_selections[model_col])
                model_selections[model_col] -= reduce_by
                excess -= reduce_by
        
        print(f"Model selection targets: {model_selections}")
        
        print(f"Positions already covered from previous steps: {len(selected_positions_step1)}")
        print(f"Available positions for Step 3: {len(set(get_position_from_mutant(mut) for mut in df['mutant'])) - len(selected_positions_step1)}")
        
        # Select mutations from each model
        additional_mutations = []
        
        # Priority order: prosst > protssn > saprot > esmif1
        priority_order = ['prosst_score', 'protssn_score', 'saprot_score', 'esmif1_score']
        
        for model_col in priority_order:
            target_count = model_selections[model_col]
            if target_count <= 0:
                continue
                
            print(f"  Selecting {target_count} mutations from {model_col}...")
            
            # Get top mutations from this model, excluding already selected positions
            model_df = df.sort_values(model_col, ascending=False)
            model_candidates = []
            model_positions_covered = set()  # Track positions covered within this model
            
            for _, row in model_df.iterrows():
                mut = row['mutant']
                pos = get_position_from_mutant(mut)
                
                # Skip if position already covered globally (from Step 1 & 2)
                if pos in selected_positions_step1:
                    continue
                    
                # Skip if position already covered within this model
                if pos in model_positions_covered:
                    continue
                    
                # Skip if mutation already selected in previous steps
                if mut in selected_mutations:
                    continue
                
                model_candidates.append(mut)
                model_positions_covered.add(pos)  # Mark this position as covered within this model
                
                if len(model_candidates) >= target_count:
                    break
            
            # Add selected mutations from this model
            for mut in model_candidates:
                additional_mutations.append(mut)
                selected_positions_step1.add(get_position_from_mutant(mut))  # Update global position set
                rankings = get_mutation_rankings(mut)
                ranking_str = format_rankings(rankings)
                model_short = model_col.replace('_score', '')
                selection_reasons[mut] = f"Step3: Top {model_short} selection. Rankings: [{ranking_str}]"
            
            print(f"    Selected {len(model_candidates)} mutations from {model_col}")
            if len(model_candidates) == 0:
                print(f"    Warning: No mutations selected from {model_col} - all positions may be covered")
        
        # Add step 3 mutations
        selected_mutations.extend(additional_mutations[:remaining_needed])
        print(f"Step 3 added {len(additional_mutations[:remaining_needed])} mutations")
    else:
        print(f"Step 3: Not needed - already have {selected_positions_count} unique positions (target: {num_recommendations})")
    
    # Final position deduplication: ensure no duplicate positions
    print(f"Final selection before deduplication: {len(selected_mutations)} mutations")
    
    # Remove duplicates by position, keeping the highest ranked mutation for each position
    final_selections = []
    seen_positions = set()
    
    for mut in selected_mutations:
        pos = get_position_from_mutant(mut)
        if pos not in seen_positions:
            final_selections.append(mut)
            seen_positions.add(pos)
        else:
            # Find the existing mutation at this position
            existing_mut = None
            for existing in final_selections:
                if get_position_from_mutant(existing) == pos:
                    existing_mut = existing
                    break
            
            if existing_mut:
                # Compare rankings and keep the better one
                existing_rankings = get_mutation_rankings(existing_mut)
                new_rankings = get_mutation_rankings(mut)
                
                # Calculate average rank (lower is better)
                existing_ranks = [r for r in existing_rankings.values() if r is not None]
                new_ranks = [r for r in new_rankings.values() if r is not None]
                
                if not existing_ranks or not new_ranks:
                    continue  # Skip if we can't compare rankings
                    
                existing_avg_rank = np.mean(existing_ranks)
                new_avg_rank = np.mean(new_ranks)
                
                if new_avg_rank < existing_avg_rank:  # New mutation has better average rank
                    # Replace the existing mutation
                    final_selections.remove(existing_mut)
                    final_selections.append(mut)
                    print(f"Replaced {existing_mut} with {mut} at position {pos} (better rank)")
    
    print(f"Final selection after deduplication: {len(final_selections)} mutations")
    
    # Ensure we return exactly num_recommendations mutations
    if len(final_selections) > num_recommendations:
        # Sort by ensemble normalized score to keep the best ones
        final_selections.sort(key=lambda x: df[df['mutant'] == x]['ensemble_norm_score'].iloc[0], reverse=True)
        final_selections = final_selections[:num_recommendations]
        print(f"Trimmed to target number: {len(final_selections)} mutations")
    
    print(f"Final recommended mutations: {len(final_selections)}")
    
    # Print selection reasons
    print(f"\n=== Selection Reasons ===")
    for i, mut in enumerate(final_selections, 1):
        print(f"{i:2d}. {mut}: {selection_reasons.get(mut, 'Unknown reason')}")
    
    return final_selections


def easy_mutation_prediction(pdb_file: str, num_recommendations: int = 30, 
                           mutations_csv: str = None, output_dir: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Perform ensemble mutation prediction using multiple models.
    
    Args:
        pdb_file: Path to the PDB file
        num_recommendations: Number of recommended mutations
        mutations_csv: Path to mutations CSV file (optional)
        output_dir: Output directory for results
        
    Returns:
        Tuple of (DataFrame with all scores, list of recommended mutations)
    """
    # Clear GPU memory at the start
    clear_gpu_memory()
    
    print(f"Starting ensemble mutation prediction for {pdb_file}")
    print(f"Target recommendations: {num_recommendations}")
    
    # Extract sequence from PDB
    sequence = extract_seq_from_pdb(pdb_file)
    print(f"Sequence length: {len(sequence)}")
    
    # Generate or load mutations
    if mutations_csv is not None:
        print(f"Loading mutations from {mutations_csv}")
        df = pd.read_csv(mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        print("Generating all possible single mutations...")
        mutants = generate_mutations_from_sequence(sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])
    
    # Save results
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    
    # Save all scores
    scores_file = os.path.join(output_dir, f"{pdb_name}_all_scores.csv")
    
    if not os.path.exists(scores_file):
        print(f"Processing {len(mutants)} mutations...")
        
        # Calculate ProSST scores
        print("Calculating ProSST scores...")
        prosst_scores = prosst_score(pdb_file, mutants)
        df['prosst_score'] = prosst_scores
        # Clear GPU memory after ProSST
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  ProSST memory cleared")
        
        # Calculate SaProt scores
        print("Calculating SaProt scores...")
        saprot_scores = saprot_score(pdb_file, mutants)
        df['saprot_score'] = saprot_scores
        # Clear GPU memory after SaProt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  SaProt memory cleared")
        
        # Calculate ProtSSN scores
        print("Calculating ProtSSN scores...")
        protssn_scores = protssn_score(pdb_file, mutants)
        df['protssn_score'] = protssn_scores
        # Clear GPU memory after ProtSSN
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  ProtSSN memory cleared")
        
        # Calculate ESM-IF1 scores
        print("Calculating ESM-IF1 scores...")
        esmif1_scores = esmif1_score(pdb_file, mutants)
        df['esmif1_score'] = esmif1_scores
        # Clear GPU memory after ESM-IF1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  ESM-IF1 memory cleared")
        
        # Normalize scores for each model
        print("Normalizing scores...")
        df['saprot_score_norm'] = normalize_scores(saprot_scores)
        df['protssn_score_norm'] = normalize_scores(protssn_scores)
        df['prosst_score_norm'] = normalize_scores(prosst_scores)
        df['esmif1_score_norm'] = normalize_scores(esmif1_scores)
        
        # Calculate ensemble normalized score (average of normalized scores)
        norm_columns = ['saprot_score_norm', 'protssn_score_norm', 'prosst_score_norm', 'esmif1_score_norm']
        df['ensemble_norm_score'] = df[norm_columns].mean(axis=1)
        
        # Sort by ensemble normalized score
        df = df.sort_values('ensemble_norm_score', ascending=False)
        df.to_csv(scores_file, index=False)
        print(f"All scores saved to: {scores_file}")
    else:
        print(f"Loading existing scores from: {scores_file}")
        df = pd.read_csv(scores_file)
        
    # Select recommended mutations
    recommended_mutations = select_recommended_mutations(df, num_recommendations)
    recommended_file = os.path.join(output_dir, f"{pdb_name}_recommended.csv")
    # Save recommended mutations
    recommended_df = df[df['mutant'].isin(recommended_mutations)].copy()
    recommended_df = recommended_df.sort_values('ensemble_norm_score', ascending=False)
    
    recommended_df.to_csv(recommended_file, index=False)
    print(f"Recommended mutations saved to: {recommended_file}")
    
    # Final memory cleanup
    clear_gpu_memory()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total mutations processed: {len(df)}")
    print(f"Recommended mutations: {len(recommended_mutations)}")
    print(f"Top 10 recommended mutations:")
    for i, mut in enumerate(recommended_mutations[:10], 1):
        print(f"  {i}. {mut}")
    
    return df, recommended_mutations


def main():
    parser = argparse.ArgumentParser(description='Easy Mutation Prediction - Ensemble Strategy')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--num_recommendations', type=int, default=30, help='Number of recommended mutations')
    args = parser.parse_args()
    
    # Perform ensemble prediction
    df, recommended_mutations = easy_mutation_prediction(
        pdb_file=args.pdb_file,
        num_recommendations=args.num_recommendations,
        mutations_csv=args.mutations_csv,
        output_dir=args.output_dir
    )
    
    print(f"\nPrediction completed successfully!")
    print(f"Recommended mutations: {recommended_mutations}")


if __name__ == "__main__":
    main()
