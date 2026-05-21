import pandas as pd
import os

# 🎯 CONFIGURATION
model_name = "google/gemma-2-2b-it"
subset = "all"
root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"
if not os.path.exists(root): root = "/content/mult-ster/third_party/microsoft_llm_steer_instruct"

path = f"{root}/format/representations/{model_name}/{subset}/pre_computed_ivs_layer14.h5"

if os.path.exists(path):
    print(f"🛠️  Repairing Namespace in: {path}")
    df = pd.read_hdf(path)

    # List of IFEval categories that use colons
    categories = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']

    def restore_colons(name):
        for cat in categories:
            # If the name starts with the category and an underscore, swap it for a colon
            # e.g., 'detectable_format_json' -> 'detectable_format:json'
            if name.startswith(cat + "_"):
                return name.replace(cat + "_", cat + ":", 1)
        return name

    df['instruction'] = df['instruction'].apply(restore_colons)
    
    # Save back (Overwriting BOTH versions to be safe)
    df.to_hdf(path, key='df', mode='w')
    df.to_hdf(path.replace('layer14', 'layer_14'), key='df', mode='w')
    
    print("✅ REPAIR COMPLETE.")
    print("New names in suitcase:")
    print(df['instruction'].unique()[:5])
else:
    print("❌ Could not find the suitcase file!")