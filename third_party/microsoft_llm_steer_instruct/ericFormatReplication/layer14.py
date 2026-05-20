import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 🎯 CONFIGURATION
model_name = "google/gemma-2-2b-it"
subset = "subset_0.1"
layer = 14
root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"
rep_folder = f"{root}/format/representations/{model_name}/{subset}"

print(f"🛠️  RYAN'S BULLETPROOF FIX: Filtering Zombies for Layer {layer}...")

# 1. Find all raw representation files
all_files = [f for f in os.listdir(rep_folder) if f.endswith('.h5') and not f.startswith('pre_computed')]

rows = []
for f_name in tqdm(all_files, desc="Processing Tasks"):
    try:
        # Load the raw math
        df = pd.read_hdf(os.path.join(rep_folder, f_name))
        
        # 🚨 THE ZOMBIE CHECK: Skip if the file is empty or missing columns
        if df.empty or 'last_token_rs' not in df.columns:
            # print(f" Skipping {f_name} (Empty)")
            continue
        
        # ALPHA SPEED: Stacking logic
        hs_instr = torch.from_numpy(np.stack(df['last_token_rs'].values))
        hs_no_instr = torch.from_numpy(np.stack(df['last_token_rs_no_instr'].values))

        # Calculate the Mean Difference
        diff = (hs_instr - hs_no_instr).mean(dim=0)
        
        # Handle cases where shape might be slightly different
        if len(diff.shape) == 2: # [layers, dim]
            vec = diff[layer]
        else: # [layers, 1, dim]
            vec = diff[layer, 0]

        instr_dir = vec / vec.norm()

        # Calculate average signal strength
        avg_proj = (hs_instr[:, layer, :] @ instr_dir).mean().item()
        avg_proj_no = (hs_no_instr[:, layer, :] @ instr_dir).mean().item()

        rows.append({
            'instruction': f_name.replace('.h5', ''),
            'selected_layer': layer,
            'instr_dir': instr_dir.numpy(),
            'avg_proj': avg_proj,
            'avg_proj_no_instr': avg_proj_no
        })
    except Exception as e:
        continue

# 2. Save the final "Suitcase"
if rows:
    out_df = pd.DataFrame(rows)
    save_path1 = f"{rep_folder}/pre_computed_ivs_layer14.h5"
    save_path2 = f"{rep_folder}/pre_computed_ivs_layer_14.h5"
    out_df.to_hdf(save_path1, key='df', mode='w')
    out_df.to_hdf(save_path2, key='df', mode='w')
    print(f"✅ SUCCESS: Created Suitcase with {len(rows)} tasks. {len(all_files)-len(rows)} empty tasks skipped.")
else:
    print("❌ FATAL: No valid data found in ANY file. Check your representations folder.")