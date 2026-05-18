import pandas as pd
import json
import os
import shutil

def shuffle_ifeval():
    # 1. Get the path relative to THIS file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Standardize the path to the data file
    data_path = os.path.abspath(os.path.join(current_dir, "..", "data", "format", "ifeval_augmented_filtered.jsonl"))
    backup_path = data_path + ".bak"

    if not os.path.exists(data_path):
        print(f"❌ ERROR: Could not find data at {data_path}")
        return

    # 2. Create backup if it doesn't exist
    if not os.path.exists(backup_path):
        # FIX: Changed 'path' to 'data_path'
        shutil.copy(data_path, backup_path)
        print(f"✅ Created backup at: {backup_path}")

    # 3. Load from backup, shuffle, and save to original
    print(f"🎲 Shuffling data from backup...")
    with open(backup_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    # Shuffle 100% of the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save back to the original filename
    df_shuffled.to_json(data_path, orient='records', lines=True, force_ascii=False)
    print(f"🚀 SUCCESS: Shuffled data saved to {data_path}")

if __name__ == "__main__":
    shuffle_ifeval()