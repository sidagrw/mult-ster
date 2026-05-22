import pandas as pd
import json

# Path to your latest evaluate output
file_path = "format/out/google/gemma-2-2b-it/no_instr/out.jsonl"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# 1. Identify which prompts actually received steering
# Layer 14 = Steered, Layer -1 = Not Found (Defaulted to Baseline)
steered_df = df[df['steering_layer'] != -1]
unsteered_df = df[df['steering_layer'] == -1]

# 2. Calculate the "Genuine" Accuracies
total_acc = df['follow_all_instructions'].mean() * 100
steered_acc = steered_df['follow_all_instructions'].mean() * 100 if not steered_df.empty else 0
unsteered_acc = unsteered_df['follow_all_instructions'].mean() * 100 if not unsteered_df.empty else 0

print(f"--- 🔍 STEERING AUDIT: {len(df)} total samples ---")
print(f"✅ Steered Samples: {len(steered_df)}")
print(f"❌ Missed Samples (Baseline): {len(unsteered_df)}")
print("-" * 40)
print(f"📊 Global Accuracy: {total_acc:.2f}%")
print(f"🔥 REAL Steering Accuracy: {steered_acc:.2f}%")
print(f"🧊 Baseline (Misses) Accuracy: {unsteered_acc:.2f}%")