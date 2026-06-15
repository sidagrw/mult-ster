import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def get_stats(path):
    if not os.path.exists(path): 
        print(f'Warning: {path} not found'); return None
    with open(path) as f:
        data = [json.loads(l) for l in f]
    df = pd.DataFrame(data)
    # Handle both list and string formats for task names
    col = 'instruction_id_list' if 'instruction_id_list' in df.columns else 'instruction'
    df['task'] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df.groupby('task')['follow_all_instructions'].mean()

# --- DEFINE PATHS ---
base_path = 'out/google/gemma-2-2b-it'
paths = {
    'Baseline': f'{base_path}/no_instr/out.jsonl',
    'Adjustive': f'{base_path}/adjust_rs_20_perplexity/out.jsonl',
    'Regular': f'{base_path}/add_vector_20_perplexity_1/out.jsonl',
    'Multiplicative': f'{base_path}/mult_rs_20_perplexity/out.jsonl'
}

# Collect all stats
all_stats = {name: get_stats(p) for name, p in paths.items()}
all_stats = {k: v for k, v in all_stats.items() if v is not None}

# Create a combined DataFrame of all tasks found
df_final = pd.DataFrame(all_stats).fillna(0)

# Filter for 'Interesting' tasks (where at least one method got > 0)
df_final = df_final[(df_final.T != 0).any()]
# Shorten task names for the graph
df_final.index = [i.split(':')[-1].replace('_', ' ').title() for i in df_final.index]

# --- PLOTTING ---
ax = df_final.plot(kind='bar', figsize=(14, 7), width=0.8, color=['#95a5a6', '#3498db', '#e74c3c', '#1e792c'], edgecolor='black')

plt.title('Dynamic Replication Results: Gemma-2-2B Steering', fontsize=16, fontweight='bold')
plt.ylabel('IFEval Accuracy', fontsize=12)
plt.xlabel('Instruction Task', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=10)

# Add value labels on top of bars
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('dynamic_replication_resultsMultSteer.png', dpi=300)
print('Success! Graph saved as dynamic_replication_results.png')
