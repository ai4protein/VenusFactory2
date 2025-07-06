import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.mutation.utils import generate_mutations_from_sequence
from src.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb

# Import all scoring functions
from src.mutation.models.saprot import saprot_score
from src.mutation.models.protssn import protssn_score
from src.mutation.models.prosst import prosst_score
from src.mutation.models.esmif1 import esmif1_score


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
        """Filter mutations to keep only one per position, preferring higher frequency"""
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
                    # If same frequency, keep both if frequency is high (>=3)
                    if count >= 3:
                        # Keep both, we'll handle this later
                        continue
        
        # Extract mutations, handling high-frequency duplicates
        selected_mutations = []
        
        for position, (mutant, count) in position_best.items():
            rankings = get_mutation_rankings(mutant)
            ranking_str = format_rankings(rankings)
            overlap_models = mutation_sources.get(mutant, [])
            overlap_models_short = [m.replace('_score', '') for m in overlap_models]
            
            if count >= 3:
                # Check if there are other high-frequency mutations at this position
                same_pos_mutations = [(m, c) for m, c in mutation_counts.items() 
                                    if get_position_from_mutant(m) == position and c >= 3]
                if len(same_pos_mutations) > 1:
                    # Keep all high-frequency mutations at this position
                    for mut, freq in same_pos_mutations:
                        selected_mutations.append(mut)
                        mut_rankings = get_mutation_rankings(mut)
                        mut_ranking_str = format_rankings(mut_rankings)
                        mut_overlap_models = mutation_sources.get(mut, [])
                        mut_overlap_models_short = [m.replace('_score', '') for m in mut_overlap_models]
                        selection_reasons[mut] = f"{step_name}: High frequency ({freq}/4 models), multiple high-freq at position {position}. Rankings: [{mut_ranking_str}]. Overlap in top{top_n}: {', '.join(mut_overlap_models_short)}"
                else:
                    selected_mutations.append(mutant)
                    selection_reasons[mutant] = f"{step_name}: High frequency ({count}/4 models). Rankings: [{ranking_str}]. Overlap in top{top_n}: {', '.join(overlap_models_short)}"
            else:
                selected_mutations.append(mutant)
                selection_reasons[mutant] = f"{step_name}: Best at position {position} ({count}/4 models). Rankings: [{ranking_str}]. Overlap in top{top_n}: {', '.join(overlap_models_short)}"
        
        return selected_mutations
    
    # Step 1: Count frequency in top 30 from each model
    print("Step 1: Analyzing frequency in top 30 mutations from each model...")
    top30_counts, top30_sources = count_mutation_frequency(30)
    
    # Filter to keep only mutations that appear in multiple models (frequency >= 2)
    frequent_mutations = {mut: count for mut, count in top30_counts.items() if count >= 2}
    print(f"Found {len(frequent_mutations)} mutations appearing in multiple models' top 30")
    
    # Filter by position preference
    selected_mutations = filter_by_position(frequent_mutations, top30_sources, "Step1", 30)
    print(f"After position filtering: {len(selected_mutations)} mutations")
    
    # Sort by frequency (descending) and then by ensemble score
    selected_mutations.sort(key=lambda x: (frequent_mutations.get(x, 0), 
                                         df[df['mutant'] == x]['ensemble_score'].iloc[0]), 
                          reverse=True)
    
    if len(selected_mutations) >= num_recommendations:
        final_selections = selected_mutations[:num_recommendations]
        print(f"\n=== Selection Reasons ===")
        for i, mut in enumerate(final_selections, 1):
            print(f"{i:2d}. {mut}: {selection_reasons.get(mut, 'Unknown reason')}")
        return final_selections
    
    # Step 2: If not enough, expand to top N*3 from each model
    remaining_needed = num_recommendations - len(selected_mutations)
    expand_top_n = num_recommendations * 3
    print(f"Step 2: Need {remaining_needed} more mutations, expanding to top {expand_top_n}...")
    
    expanded_counts, expanded_sources = count_mutation_frequency(expand_top_n)
    
    # Filter to keep only mutations that appear in multiple models and not already selected
    additional_frequent = {mut: count for mut, count in expanded_counts.items() 
                          if count >= 2 and mut not in selected_mutations}
    
    print(f"Found {len(additional_frequent)} additional mutations in top {expand_top_n}")
    
    # Filter by position preference for additional mutations
    additional_selected = filter_by_position(additional_frequent, expanded_sources, "Step2", expand_top_n)
    
    # Remove any that are already selected
    additional_selected = [mut for mut in additional_selected if mut not in selected_mutations]
    
    # Sort additional mutations by frequency and ensemble score
    additional_selected.sort(key=lambda x: (additional_frequent.get(x, 0), 
                                          df[df['mutant'] == x]['ensemble_score'].iloc[0]), 
                           reverse=True)
    
    # Add additional mutations
    selected_mutations.extend(additional_selected[:remaining_needed])
    
    print(f"Final selection: {len(selected_mutations)} mutations")
    
    # Print selection reasons
    print(f"\n=== Selection Reasons ===")
    for i, mut in enumerate(selected_mutations, 1):
        print(f"{i:2d}. {mut}: {selection_reasons.get(mut, 'Unknown reason')}")
    
    return selected_mutations[:num_recommendations]


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
        print("Calculating ProSST scores...")
        prosst_scores = prosst_score(pdb_file, mutants)
        df['prosst_score'] = prosst_scores
        
        # Calculate scores from all models
        print("Calculating SaProt scores...")
        saprot_scores = saprot_score(pdb_file, mutants)
        df['saprot_score'] = saprot_scores
        
        print("Calculating ProtSSN scores...")
        protssn_scores = protssn_score(pdb_file, mutants)
        df['protssn_score'] = protssn_scores
        
        print("Calculating ESM-IF1 scores...")
        esmif1_scores = esmif1_score(pdb_file, mutants)
        df['esmif1_score'] = esmif1_scores
        
        # Normalize scores for each model
        print("Normalizing scores...")
        df['saprot_score_norm'] = normalize_scores(saprot_scores)
        df['protssn_score_norm'] = normalize_scores(protssn_scores)
        df['prosst_score_norm'] = normalize_scores(prosst_scores)
        df['esmif1_score_norm'] = normalize_scores(esmif1_scores)
        
        # Calculate ensemble score (average of normalized scores)
        norm_columns = ['saprot_score_norm', 'protssn_score_norm', 'prosst_score_norm', 'esmif1_score_norm']
        df['ensemble_score'] = df[norm_columns].mean(axis=1)
        
        # Sort by ensemble score
        df = df.sort_values('ensemble_score', ascending=False)
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
    recommended_df = recommended_df.sort_values('ensemble_score', ascending=False)
    
    recommended_df.to_csv(recommended_file, index=False)
    print(f"Recommended mutations saved to: {recommended_file}")
    
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
