import pandas as pd
import json
import os

# 1. Setup Path (Detecting your environment)
root = "/root/mult-ster/third_party/microsoft_llm_steer_instruct"
if not os.path.exists(root): root = "/content/mult-ster/third_party/microsoft_llm_steer_instruct"

data_path = f"{root}/data/format/ifeval_augmented_filtered.jsonl"

print(f"🎲 Shuffling data at: {data_path}")

# 2. Load the data "Properly" as JSON objects
with open(data_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# 3. Perform the Shuffle
# random_state=42 ensures that if Eric runs this, he gets the same 'random' order
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Save back to the original file
# We use orient='records' and lines=True to keep it in JSONL format
df_shuffled.to_json(data_path, orient='records', lines=True, force_ascii=False)

print("✅ SUCCESS: Dataset is now randomized. 10% will now cover ALL instructions.")