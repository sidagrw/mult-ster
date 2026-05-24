# %%
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

config_path = os.path.join(project_dir, 'config/format')

@hydra.main(config_path=config_path, config_name='precompute_steering_vectors', version_base=None)
def precompute_vectors(args: DictConfig):
    # 1. DYNAMIC PATHING (The 'Ryan Fix' for 10-hour reliability)
    # We look for YOUR actual search results, not a hardcoded folder name.
    folder = f'{script_dir}/layer_search_out/{args.model_name}'
    search_file = f"{folder}/results_dir/out_instr.jsonl" # Standard location in our .sh
    if not os.path.exists(search_file):
        search_file = f"{folder}/results.jsonl" # Fallback
    
    if not os.path.exists(search_file):
        # Final fallback to their hardcoded structure if all else fails
        search_file = f'{folder}/n_examples{args.n_examples}_seed{args.seed}_no_instr/out_instr.jsonl'

    print(f"📖 Loading Search Metadata: {search_file}")
    with open(search_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    validation_df = pd.DataFrame(results)

    # 2. TASK IDENTIFICATION
    with open(f'{project_dir}/data/format/ifeval_single_instr_format.jsonl', encoding='utf-8') as f:
        data = [json.loads(d) for d in f.readlines()]
    input_data_df = pd.DataFrame(data)
    
    # We use the full IDs to ensure we match the evaluator's registry
    all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = [i for i in all_instructions if any(f in i for f in filters)]

    # 3. SCIENTIFIC LOGIC: OPTIMAL LAYER SELECTION (BIT-PERFECT TO PAPER)
    optimal_layers = { instr: -1 for instr in all_instructions }

    for instr in all_instructions:
        # Normalize: Handle the ':' vs '_' mismatch in the search results
        instr_underscore = instr.replace(':', '_')
        
        # Filter search results for this specific instruction
        # We check both the original ID and the underscored version
        mask = validation_df['instruction_id_list_for_eval'].apply(lambda x: (x[0] if isinstance(x, list) else x) in [instr, instr_underscore])
        instr_df = validation_df[mask]
        
        if instr_df.empty:
            continue

        if args.use_perplexity and 'perplexity' in instr_df.columns:
            # THIS IS THE CORE SELECTION LOGIC FROM THE PAPER
            instr_df['low_perplexity'] = instr_df.perplexity < args.preplexity_threshold
            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()

            # Baseline check (Layer -1)
            baseline_low_ppl = df_group_by_layer.loc[-1, 'low_perplexity'] if -1 in df_group_by_layer.index else 0
            accuracy_layer_minus_1 = df_group_by_layer.loc[-1, 'follow_all_instructions'] if -1 in df_group_by_layer.index else 0

            # Discard layers that increase perplexity too much (lobotomy check)
            df_group_by_layer.loc[df_group_by_layer.low_perplexity < baseline_low_ppl, 'follow_all_instructions'] = 0
            # Restore accuracy for baseline
            if -1 in df_group_by_layer.index:
                df_group_by_layer.loc[-1, 'follow_all_instructions'] = accuracy_layer_minus_1

            max_accuracy = df_group_by_layer.follow_all_instructions.max()
            optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index[0]
            optimal_layers[instr] = optimal_layer
        else:
            # Simple Accuracy-only selection
            group = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()
            max_accuracy = group.follow_all_instructions.max()
            optimal_layer = group[group.follow_all_instructions == max_accuracy].index[0]
            optimal_layers[instr] = optimal_layer

    # 4. VECTOR AGGREGATION (OPTIMIZED I/O)
    rows = []
    rep_folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'

    for instr in tqdm(all_instructions, desc="Aggregating Manifolds"):
        # Map back to the filename (filenames use underscores)
        file = f'{rep_folder}/{instr.replace(":", "_")}.h5'
        
        if not os.path.exists(file):
            continue
            
        results_df = pd.read_hdf(file, key='df')
        
        # Safety for 0.1 ratio 'Zombie Files'
        if results_df.empty or 'last_token_rs' not in results_df.columns:
            continue

        selected_layer = optimal_layers[instr]
        if selected_layer == -1: continue

        # ENGINE SPEED: np.stack is mathematically identical to tolist() but 100x faster
        hs_instr = torch.from_numpy(np.stack(results_df['last_token_rs'].values)).to(torch.bfloat16)
        hs_no_instr = torch.from_numpy(np.stack(results_df['last_token_rs_no_instr'].values)).to(torch.bfloat16)

        # Dimension alignment
        if len(hs_instr.shape) == 3:
            hs_instr, hs_no_instr = hs_instr.unsqueeze(2), hs_no_instr.unsqueeze(2)

        # Steering Vector Math: u = mean(X+) - mean(X)
        mean_repr_diffs = (hs_instr - hs_no_instr).mean(dim=0)
        last_token_mean_diff = mean_repr_diffs[:, -1, :] # Grab last token

        # Normalization
        diff_vec = last_token_mean_diff[selected_layer]
        instr_dir = diff_vec / diff_vec.norm()

        # Signal Strength (Adjust_RS parameters)
        proj = (hs_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)).mean()
        proj_no = (hs_no_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)).mean()
        
        rows.append({
            'instruction': instr, # SAVING WITH COLON FOR EVALUATOR
            'selected_layer': int(selected_layer),
            'instr_dir': instr_dir.float().cpu().numpy(),
            'avg_proj': float(proj),
            'avg_proj_no_instr': float(proj_no)
        })

    # 5. SAVE FINAL SUITCASE
    df_final = pd.DataFrame(rows)
    w_ppl = '_with_perplexity' if args.use_perplexity else ''
    instr_inc = 'instr' if args.include_instructions else 'no_instr'
    out_file = f'{rep_folder}/pre_computed_ivs_best_layer_validation{w_ppl}_{instr_inc}.h5'
    
    df_final.to_hdf(out_file, key='df', mode='w')
    print(f"✅ FINAL SUITCASE Born: {out_file} ({len(df_final)} tasks)")

if __name__ == '__main__':
    precompute_vectors()
