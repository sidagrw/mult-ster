import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 🎯 CONFIGURATION
MODELS = [
    "google/gemma-2-2b-it", 
    "microsoft/Phi-3-mini-4k-instruct", 
    "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen/Qwen1.5-1.8B-Chat"
]
SUBSET = "subset_0.1"
TARGET_LAYER = 14

# Determine Root
root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"
if not os.path.exists(root): root = "/content/mult-ster/third_party/microsoft_llm_steer_instruct"

print(f"🛠️  RYAN'S UNIVERSAL FIX: Building Suitcases for Layer {TARGET_LAYER}...")

for model_name in MODELS:
    rep_folder = f"{root}/format/representations/{model_name}/{SUBSET}"
    
    if not os.path.exists(rep_folder):
        print(f"⏩ Skipping {model_name}: No representations found.")
        continue

    print(f"\nProcessing {model_name}...")
    all_files = [f for f in os.listdir(rep_folder) if f.endswith('.h5') and not f.startswith('pre_computed')]
    
    rows = []
    for f_name in tqdm(all_files, desc=f"Averaging {model_name}"):
        try:
            df = pd.read_hdf(os.path.join(rep_folder, f_name))
            
            # Zombie Check
            if df.empty or 'last_token_rs' not in df.columns:
                continue
            
            # Fast Stacking
            hs_instr = torch.from_numpy(np.stack(df['last_token_rs'].values))
            hs_no_instr = torch.from_numpy(np.stack(df['last_token_rs_no_instr'].values))

            # Calculate Mean Diff for the whole model
            diff = (hs_instr - hs_no_instr).mean(dim=0)
            
            # Handle 3D vs 4D shapes (TransformerLens variance)
            if len(diff.shape) == 2: # [layers, dim]
                vec = diff[TARGET_LAYER]
            else: # [layers, tokens, dim]
                vec = diff[TARGET_LAYER, -1]

            instr_dir = vec / vec.norm()

            # Calculate average signal strength (Projections)
            # We must project the high-dim data onto our new normalized direction
            avg_proj = (hs_instr[:, TARGET_LAYER, :] @ instr_dir).mean().item()
            avg_proj_no = (hs_no_instr[:, TARGET_LAYER, :] @ instr_dir).mean().item()

            rows.append({
                'instruction': f_name.replace('.h5', ''),
                'selected_layer': TARGET_LAYER,
                'instr_dir': instr_dir.numpy(),
                'avg_proj': avg_proj,
                'avg_proj_no_instr': avg_proj_no
            })
        except Exception as e:
            continue

    if rows:
        out_df = pd.DataFrame(rows)
        # We save BOTH naming conventions to prevent FileNotFound errors
        out_df.to_hdf(f"{rep_folder}/pre_computed_ivs_layer{TARGET_LAYER}.h5", key='df', mode='w')
        out_df.to_hdf(f"{rep_folder}/pre_computed_ivs_layer_{TARGET_LAYER}.h5", key='df', mode='w')
        print(f"✅ Created Suitcase for {model_name} ({len(rows)} tasks).")
    else:
        print(f"❌ No valid data for {model_name}.")

print("\n🚀 ALL SUITCASES READY. You can now run the evaluate loop.")