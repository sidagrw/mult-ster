import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_all_results_fixed(folder_path):
    all_data = []
    # Search for all out.jsonl files
    files = glob.glob(os.path.join(folder_path, "**/out.jsonl"), recursive=True)
    
    if not files:
        print(f"❌ No files found in {folder_path}")
        return pd.DataFrame()

    for f in files:
        # 1. Identify Model from Path
        # We look for keywords in the full file path
        path_lower = f.lower()
        model_label = "Other"
        if "gemma" in path_lower: model_label = "Gemma-2-2B"
        elif "phi" in path_lower: model_label = "Phi-3-Mini"
        elif "qwen" in path_lower: model_label = "Qwen-1.8B"
        elif "mistral" in path_lower: model_label = "Mistral-7B"

        # 2. Identify Method from Path
        method_label = "Baseline (None)"
        if "adjust_rs_14" in path_lower: method_label = "Adjust_RS (Projection)"
        elif "add_vector_14_1.0" in path_lower: method_label = "Add_Vector (Simple)"

        print(f"📂 Found: {model_label} | {method_label}")

        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    item = json.loads(line)
                    item['model_label'] = model_label
                    item['steering_method'] = method_label
                    all_data.append(item)
                except: continue
    
    return pd.DataFrame(all_data)

def create_final_viz(df, output_name="replication_success.png"):
    if df.empty: return

    # Calculate Accuracy per Model per Method
    summary = df.groupby(['model_label', 'steering_method'])['follow_all_instructions'].mean().reset_index()
    summary['Accuracy (%)'] = summary['follow_all_instructions'] * 100

    # Sort so Baseline is always first in the legend
    summary['steering_method'] = pd.Categorical(summary['steering_method'], 
                                    ["Baseline (None)", "Add_Vector (Simple)", "Adjust_RS (Projection)"])
    summary = summary.sort_values('steering_method')

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Professional ICLR Colors
    palette = {"Baseline (None)": "#95a5a6", "Add_Vector (Simple)": "#3498db", "Adjust_RS (Projection)": "#e67e22"}

    ax = sns.barplot(
        data=summary, 
        x='model_label', 
        y='Accuracy (%)', 
        hue='steering_method',
        palette=palette
    )

    # Title & Labels
    plt.title("Steering Efficacy Across Architectures (Layer 14)\nComparing IFEval Adherence (No-Instruction Prompts)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("IFEval Adherence Score (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Model Family", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    
    # Legend outside
    plt.legend(title="Intervention", bbox_to_anchor=(1, 1), loc='upper left')

    # Value Labels
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}%', 
                        (p.get_x() + p.get_width() / 2., h), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"✅ FINAL CHART SAVED: {output_name}")
    plt.show()

# --- EXECUTION ---
# Use your local folder name
df = load_all_results_fixed("out")
create_final_viz(df)