# %%
import os
import sys
import json
import functools
import numpy as np
import torch
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from transformer_lens import utils as tlutils

# Standardize Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)
sys.path.append(script_dir)
sys.path.append(project_dir)

from utils.model_utils import load_model_from_tl_name
from utils.generation_utils import generate, generate_with_hooks, direction_projection_hook, activation_addition_hook
from ifeval_scripts.evaluation_main import test_instruction_following_loose

# Ensure NLTK is ready for the grader
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

@hydra.main(config_path='../config/format', config_name='find_best_layer', version_base=None)
def find_best_layer(args: DictConfig):
    # 1. INITIALIZATION
    print("🚀 Starting Optimized Layer Search...")
    print(OmegaConf.to_yaml(args))
    device = args.device
    
    # 2. FAST DATA LOADING (UTF-8 FORCED)
    with open(f'{project_dir}/{args.data_path}', encoding='utf-8') as f:
        data = [json.loads(d) for d in f.readlines()]
    data_df = pd.DataFrame(data)

    # 3. SCHEMA-ROBUST FILTERING
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    target_col = 'instruction_id_list' if 'instruction_id_list' in data_df.columns else 'single_instruction_id'
    data_df = data_df[data_df[target_col].apply(lambda x: any([f in (x[0] if isinstance(x, list) else x) for f in filters]))]

    # 4. REGISTRY HACK (PHI-3 / GEMMA-2 Support)
    import transformer_lens.loading_from_pretrained as loading
    if args.model_name not in loading.OFFICIAL_MODEL_NAMES:
        loading.OFFICIAL_MODEL_NAMES.append(args.model_name)

    # 5. MODEL LOAD (BFLOAT16 FOR A100)
    model, tokenizer = load_model_from_tl_name(args.model_name, device=device, hf_model=(args.steering == 'none'))
    model.to(device).to(torch.bfloat16)
    tokenizer.padding_side = "left" # Crucial for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        data_df = data_df.head(2)

    all_instructions = list(set([item for l in data_df.instruction_id_list_for_eval for item in l]))
    n_layers = model.cfg.n_layers
    
    # 6. SPARSE STRIDE SEARCH (Speed Hack)
    # Checks every 6th layer to find the 'neighborhood' of the best layer fast
    layer_range = [-1] + list(range(n_layers // 5, n_layers - 2, 6))

    out_lines = []
    
    # 7. THE MASTER VECTORIZED LOOP
    for instruction_type in tqdm.tqdm(all_instructions, desc="Scanning Instructions"):
        # Filter and Sample
        instr_data_df = data_df[[[instruction_type] == l for l in data_df['instruction_id_list_for_eval'] ]]
        if instr_data_df.empty: continue
        
        # We use n_examples=1 for the fastest search
        instr_data_df = instr_data_df.sample(n=min(args.n_examples_per_instruction, len(instr_data_df)), random_state=args.seed)
        if 'language' in instruction_type: instr_data_df = instr_data_df.head(1)

        # 8. PRE-COMPUTE STEERING DIRECTION (NUMPY STACKING FIX)
        if args.steering != 'none':
            folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'
            file = f'{folder}/{"".join(instruction_type).replace(":", "_")}.h5'
            
            if not os.path.exists(file):
                continue

            try:
                results_df = pd.read_hdf(file, key='df')
                if results_df.empty or 'last_token_rs' not in results_df.columns: continue
                
                # ALPHA SPEED: Bulk stack instead of .tolist()
                hs_instr = torch.from_numpy(np.stack(results_df['last_token_rs'].values)).to(torch.bfloat16)
                hs_no_instr = torch.from_numpy(np.stack(results_df['last_token_rs_no_instr'].values)).to(torch.bfloat16)
                
                last_token_mean_diff = (hs_instr - hs_no_instr).mean(dim=0)
                if len(last_token_mean_diff.shape) == 3:
                    last_token_mean_diff = last_token_mean_diff[:, -1, :]
            except Exception as e:
                print(f"⚠️ Error loading {instruction_type}: {e}")
                continue

        # 9. BATCH GENERATION PER LAYER
        with torch.inference_mode():
            for layer_idx in layer_range:
                # Truncate length for search speed (Intent is visible in 32 tokens)
                max_gen = 32 if any(k in instruction_type for k in ['json', 'sections']) else 32
                
                # Format prompts for this batch
                prompts = []
                for _, r in instr_data_df.iterrows():
                    example_txt = r['prompt'] if args.include_instructions else r['prompt_without_instruction']
                    if 'gemma-2' in args.model_name or 'phi-3' in args.model_name.lower():
                        prompts.append(f"Q: {example_txt}\nA:")
                    else:
                        msg = [{"role": "user", "content": example_txt}]
                        prompts.append(tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
                
                tokens = tokenizer(prompts, return_tensors='pt', padding=True).to(device)

                if layer_idx == -1:
                    output_ids = model.generate(tokens['input_ids'], max_new_tokens=max_gen, stop_at_eos=True)
                else:
                    # Steering Math
                    diff_vec = last_token_mean_diff[layer_idx]
                    instr_dir = (diff_vec / diff_vec.norm()).to(device)
                    
                    if args.steering == 'adjust_rs':
                        proj_data = hs_instr[:, layer_idx, -1, :] if len(hs_instr.shape) == 4 else hs_instr[:, layer_idx, :]
                        avg_proj = (proj_data.to(device) @ instr_dir).mean()
                        hook_fn = functools.partial(direction_projection_hook, direction=instr_dir, value_along_direction=avg_proj)
                    else:
                        hook_fn = functools.partial(activation_addition_hook, direction=instr_dir, weight=args.steering_weight)

                    fwd_hooks = [(tlutils.get_act_name('resid_post', layer_idx), hook_fn)]
                    output_ids = generate_with_hooks(model, tokens['input_ids'], fwd_hooks=fwd_hooks, max_tokens_generated=max_gen)

                # 10. SMART DECODE AND GRADE
                if isinstance(output_ids, list) and len(output_ids) > 0 and isinstance(output_ids[0], str):
                    responses = output_ids
                else:
                    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                for idx, (_, r) in enumerate(instr_data_df.iterrows()):
                    row = dict(r)
                    row['response'] = responses[idx]
                    row['layer'] = layer_idx
                    
                    # Fault-Tolerant Grader
                    try:
                        r_copy = r.copy()
                        if 'instruction_id_list' in r_copy:
                            r_copy['instruction_id_list'] = [i.split(':')[-1] for i in r_copy['instruction_id_list']]
                        grading = test_instruction_following_loose(r_copy, {row['prompt']: row['response']})
                        row['follow_all_instructions'] = grading.follow_all_instructions
                    except:
                        row['follow_all_instructions'] = False
                    
                    out_lines.append(row)

    # 11. SAVE RESULTS
    out_dir = f"{script_dir}/{args.output_path}/{args.model_name}/results_dir"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/out_instr.jsonl" # Sync with precompute_ivs expectation
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in out_lines:
            f.write(json.dumps(line) + '\n')
    
    # Also save a copy to the root expected by evaluate.py
    shutil.copy(out_path, f"{script_dir}/{args.output_path}/{args.model_name}/results.jsonl")
    print(f"✅ GOAT SPEED SEARCH COMPLETE. Results saved to {out_path}")

if __name__ == '__main__':
    find_best_layer()