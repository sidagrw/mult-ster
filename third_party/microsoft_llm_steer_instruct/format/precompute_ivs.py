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

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

config_path = os.path.join(project_dir, 'config/format')

@hydra.main(config_path=config_path, config_name='precompute_steering_vectors', version_base=None)
def precompute_vectors(args: DictConfig):
    # 1. DYNAMIC PATH RESOLUTION
    # This finds YOUR results regardless of if they are in 'results_dir' or the hardcoded 'n_examples8'
    model_results_path = f"{script_dir}/layer_search_out/{args.model_name}"
    
    # Priority list of files to check
    possible_files = [
        f"{model_results_path}/results_dir/out_instr.jsonl",
        f"{model_results_path}/results.jsonl",
        f"{model_results_path}/n_examples{args.n_examples}_seed{args.seed}/out_instr.jsonl"
    ]
    
    search_file = None
    for f_path in possible_files:
        if os.path.exists(f_path):
            search_file = f_path
            break
            
    if not search_file:
        raise FileNotFoundError(f"❌ Could not find search results for {args.model_name} in any expected location.")

    print(f"📖 Reading Results from: {search_file}")
    with open(search_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    validation_df = pd.DataFrame(results)

    # 2. IDENTIFY INSTRUCTIONS
    with open(f'{project_dir}/data/format/ifeval_single_instr_format.jsonl', encoding='utf-8') as f:
        data = [json.loads(d) for d in f.readlines()]
    input_data_df = pd.DataFrame(data)
    
    # Original Filtering Logic
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique()
    all_instructions = [i for i in all_instructions if any(f in i for f in filters)]

    # 3. OPTIMAL LAYER SELECTION (Preserving Original Functionality)
    optimal_layers = {instr: -1 for instr in all_instructions}
    
    for instr in all_instructions:
        # ID Normalization: Handles the 'language:' prefix bug
        clean_instr = instr.split(':')[-1] if ':' in instr else instr
        
        # Check if we have data for this instruction
        if instr not in validation_df.instruction_id_list_for_eval.apply(lambda x: x[0] if isinstance(x, list) else x).values:
            continue

        instr_df = validation_df[validation_df.instruction_id_list_for_eval.apply(lambda x: (x[0] if isinstance(x, list) else x) == instr)]
        
        if args.use_perplexity and 'perplexity' in instr_df.columns:
            instr_df['low_perplexity'] = instr_df.perplexity < args.preplexity_threshold
            df_group = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()
            
            # Baseline (No steering)
            accuracy_no_steer = df_group.loc[-1, 'follow_all_instructions'] if -1 in df_group.index else 0
            baseline_low_ppl = df_group.loc[-1, 'low_perplexity'] if -1 in df_group.index else 0
            
            # Penalty for high perplexity
            df_group.loc[df_group.low_perplexity < baseline_low_ppl, 'follow_all_instructions'] = 0
            df_group.loc[-1, 'follow_all_instructions'] = accuracy_no_steer # Restore baseline
            
            max_acc = df_group.follow_all_instructions.max()
            optimal_layers[instr] = df_group[df_group.follow_all_instructions == max_acc].index[0]
        else:
            df_group = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()
            max_acc = df_group.follow_all_instructions.max()
            optimal_layers[instr] = df_group[df_group.follow_all_instructions == max_acc].index[0]

    # 4. VECTOR AGGREGATION (High-Speed Engine)
    rows = []
    rep_base = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'

    for instr in tqdm(all_instructions, desc="Aggregating Vectors"):
        file = f'{rep_base}/{"".join(instr).replace(":", "_")}.h5'
        if not os.path.exists(file): continue

        results_df = pd.read_hdf(file, key='df')
        if results_df.empty or 'last_token_rs' not in results_df.columns: continue

        selected_layer = optimal_layers[instr]
        if selected_layer == -1: continue

        # ALPHA SPEED: Stacking vs .tolist()
        # We use bfloat16 to match A100/T4 optimized extraction
        hs_instr = torch.from_numpy(np.stack(results_df['last_token_rs'].values)).to(torch.bfloat16)
        hs_no_instr = torch.from_numpy(np.stack(results_df['last_token_rs_no_instr'].values)).to(torch.bfloat16)

        if len(hs_instr.shape) == 3:
            hs_instr, hs_no_instr = hs_instr.unsqueeze(2), hs_no_instr.unsqueeze(2)

        # Compute Direction
        last_token_mean_diff = (hs_instr - hs_no_instr).mean(dim=0)[:, -1, :]
        instr_dir = last_token_mean_diff[selected_layer] / last_token_mean_diff[selected_layer].norm()

        # Compute Average Projection (Signal Strength)
        proj = (hs_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)).mean()
        proj_no = (hs_no_instr[:, selected_layer, -1, :].to(args.device) @ instr_dir.to(args.device)).mean()

        rows.append({
            'instruction': instr,
            'selected_layer': selected_layer,
            'instr_dir': instr_dir.cpu().float().numpy(),
            'avg_proj': proj.item(),
            'avg_proj_no_instr': proj_no.item()
        })

    # 5. SAVE FINAL PRODUCT
    if rows:
        df_final = pd.DataFrame(rows)
        w_perplexity = '_with_perplexity' if args.use_perplexity else ''
        instr_inc = 'instr' if args.include_instructions else 'no_instr'
        out_name = f'pre_computed_ivs_best_layer_validation{w_perplexity}_{instr_inc}.h5'
        out_path = f'{rep_base}/{out_name}'
        df_final.to_hdf(out_path, key='df', mode='w')
        print(f"✅ IVS SUCCESS: Saved to {out_path}")
    else:
        print("❌ FAILED: No valid vectors were precomputed. Check your input data.")

if __name__ == '__main__':
    precompute_vectors()