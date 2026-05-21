import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 🎯 CONFIGURATION
model_name = "google/gemma-2-2b-it"
subset = "all"
layer = 14
root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"
rep_folder = f"{root}/format/representations/{model_name}/{subset}"

# 1. Load the original data to build a Name Map
# This allows us to map "no_comma" -> "punctuation:no_comma"
with open(f"{root}/data/format/ifeval_augmented_filtered.jsonl", 'r') as f:
    import json
    data = [json.loads(line) for line in f]
df_master = pd.DataFrame(data)

# Create the map: { 'no_comma': 'punctuation:no_comma' }
name_map = {}
for _, row in df_master.iterrows():
    short_id = row['single_instruction_id']
    full_id = row['instruction_id_list_for_eval'][0]
    name_map[short_id] = full_id

print(f"🛠️  RYAN'S NAMESPACE BRIDGE: Building Suitcase for Layer {layer}...")

files = [f for f in os.listdir(rep_folder) if f.endswith('.h5') and not f.startswith('pre_computed')]

rows = []
for f_name in tqdm(files):
    df = pd.read_hdf(os.path.join(rep_folder, f_name))
    if df.empty or 'last_token_rs' not in df.columns: continue
    
    short_id = f_name.replace('.h5', '')
    # 🚨 THE FIX: Get the FULL ID (e.g., "punctuation:no_comma")
    full_id = name_map.get(short_id, short_id)

    hs_instr = torch.from_numpy(np.stack(df['last_token_rs'].values))
    hs_no_instr = torch.from_numpy(np.stack(df['last_token_rs_no_instr'].values))
    diff = (hs_instr - hs_no_instr).mean(dim=0)
    
    vec = diff[layer] if len(diff.shape) == 2 else diff[layer, -1]
    instr_dir = vec / vec.norm()
    avg_proj = (hs_instr[:, layer, :] @ instr_dir).mean().item()
    avg_proj_no = (hs_no_instr[:, layer, :] @ instr_dir).mean().item()

    rows.append({
        'instruction': full_id, # <--- Evaluator will now find this!
        'selected_layer': layer,
        'instr_dir': instr_dir.numpy(),
        'avg_proj': avg_proj,
        'avg_proj_no_instr': avg_proj_no
    })

out_df = pd.DataFrame(rows)
out_df.to_hdf(f"{rep_folder}/pre_computed_ivs_layer_14.h5", key='df', mode='w')
print(f"✅ SUCCESS: Mapped {len(rows)} instructions with full category prefixes.")