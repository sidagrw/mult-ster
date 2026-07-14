import pandas as pd
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. Parse "N" (number of training instances) per instruction from the
#    training log. Robust to tqdm progress-bar text getting concatenated
#    onto the same line as the next print statement (no \r/\n anchoring;
#    just relies on "for:" / "(N):" pairs appearing in the same order).
# =====================================================================
def parse_sample_counts(log_path):
    with open(log_path, encoding='utf-8') as f:
        text = f.read()

    tasks = re.findall(r"Number of training instances for:\s*(\S+)", text)
    counts = re.findall(r"Number of training instances \(N\):\s*(\d+)", text)

    if len(tasks) != len(counts):
        print(f"Warning: found {len(tasks)} task names but {len(counts)} "
              f"N values — pairing the first min(len) of each in order.")

    n = min(len(tasks), len(counts))
    return {tasks[i]: int(counts[i]) for i in range(n)}


# =====================================================================
# 2. Load accuracy per task, per method (same logic as your bar chart)
# =====================================================================
def get_stats(path):
    if not os.path.exists(path):
        print(f'Warning: {path} not found')
        return None
    with open(path) as f:
        data = [json.loads(l) for l in f]
    df = pd.DataFrame(data)
    col = 'instruction_id_list' if 'instruction_id_list' in df.columns else 'instruction'
    df['task'] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df.groupby('task')['follow_all_instructions'].mean()


# --- EDIT THESE PATHS ---
LOG_PATH = 'training_log.txt'   # <-- point this at the file containing the printed N stats
paths = {
    'Baseline': 'NOSTEER_backup_out.jsonl',
    'Adjustive': 'ADJUSTRS_backup_out.jsonl',
    'Regular': 'ADDVECTOR_backup_out.jsonl',
    'Multiplicative': 'MULTRS_backup_out.jsonl',
}

all_stats = {name: get_stats(p) for name, p in paths.items()}
all_stats = {k: v for k, v in all_stats.items() if v is not None}
df_acc = pd.DataFrame(all_stats).fillna(0)          # index = raw task id, columns = method

sample_counts = parse_sample_counts(LOG_PATH)
n_series = pd.Series(sample_counts, name='N')
print(n_series)
# =====================================================================
# 3. Merge on raw task id (BEFORE any display-name shortening, since
#    the log's task names use the same colon-separated id as
#    instruction_id_list, e.g. "punctuation:no_comma")
# =====================================================================
df_merged = df_acc.join(n_series, how='inner')
print(f"Matched {len(df_merged)} of {len(df_acc)} tasks to a sample count "
      f"(unmatched tasks are dropped from this analysis — check naming "
      f"if this number looks low).")

methods = [c for c in df_merged.columns if c != 'N']
display_names = [i.split(':')[-1].replace('_', ' ').title() for i in df_merged.index]

# =====================================================================
# 3b. Original grouped bar chart: accuracy by instruction task, per
#     method (kept as-is, just using df_acc before it gets merged/
#     filtered for the N-correlation plots below).
# =====================================================================
df_bar = df_acc[(df_acc.T != 0).any()].copy()  # drop tasks where every method scored 0
df_bar.index = [i.split(':')[-1].replace('_', ' ').title() for i in df_bar.index]

ax = df_bar.plot(kind='bar', figsize=(14, 7), width=0.8,
                  color=['#95a5a6', '#3498db', '#e74c3c', '#1e792c'], edgecolor='black')

plt.title('Dynamic Replication Results: Gemma-2-2B Steering', fontsize=16, fontweight='bold')
plt.ylabel('IFEval Accuracy', fontsize=12)
plt.xlabel('Instruction Task', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=10)

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('dynamic_replication_resultsMultSteerv2.png', dpi=300)
plt.show()

# =====================================================================
# 4a. N vs. raw accuracy, one panel per method, with trend line + r
# =====================================================================
fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
if len(methods) == 1:
    axes = [axes]

for ax, method in zip(axes, methods):
    x = df_merged['N'].values
    y = df_merged[method].values
    ax.scatter(x, y, alpha=0.75, edgecolor='black', s=60)

    r = np.corrcoef(x, y)[0, 1] if np.unique(x).size > 1 else float('nan')
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(z, xs), '--', color='gray', linewidth=1.5)

    ax.set_title(f'{method}  (r = {r:.2f})', fontsize=11)
    ax.set_xlabel('# Training Samples (N)')
    ax.grid(alpha=0.3)

axes[0].set_ylabel('IFEval Accuracy')
plt.suptitle('Training Sample Count vs. Accuracy, by Method', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('n_vs_accuracy_by_method.png', dpi=300)
plt.show()

# =====================================================================
# 4b. N vs. improvement over baseline (the key correlation question)
# =====================================================================
if 'Baseline' in df_merged.columns:
    steer_methods = [m for m in methods if m != 'Baseline']
    colors = ['#3498db', '#e74c3c', '#1e792c']

    fig, ax = plt.subplots(figsize=(8, 6))
    for color, method in zip(colors, steer_methods):
        delta = df_merged[method] - df_merged['Baseline']
        ax.scatter(df_merged['N'], delta, label=method, alpha=0.75,
                   color=color, edgecolor='black', s=60)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('# Training Samples (N)')
    ax.set_ylabel('Accuracy Improvement over Baseline')
    ax.set_title('Does More Contrastive Data Predict a Bigger Steering Gain?',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('n_vs_improvement.png', dpi=300)
    plt.show()

# =====================================================================
# 5. Print correlation table
# =====================================================================
print("\nPearson correlation with N:")
for method in methods:
    r_acc = np.corrcoef(df_merged['N'], df_merged[method])[0, 1]
    line = f"  {method:15s} r(N, accuracy)    = {r_acc:+.3f}"
    if method != 'Baseline' and 'Baseline' in df_merged.columns:
        delta = df_merged[method] - df_merged['Baseline']
        r_delta = np.corrcoef(df_merged['N'], delta)[0, 1]
        line += f"   r(N, improvement) = {r_delta:+.3f}"
    print(line)

print("\nDone. Saved: n_vs_accuracy_by_method.png, n_vs_improvement.png")