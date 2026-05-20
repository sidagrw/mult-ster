import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def generate_research_viz(file_path):
    # 1. Extract model name from the path for the title
    # Logic: Splits path and looks for the folder before 'no_instr'
    path_parts = file_path.replace('\\', '/').split('/')
    model_label = path_parts[-3] if len(path_parts) > 3 else "Model"

    # 2. Load the data
    data = []
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except: continue
    
    df = pd.DataFrame(data)
    if df.empty:
        print(f"⚠️ {model_label} data is empty.")
        return

    # 3. Extract primary category (e.g., 'punctuation:no_comma' -> 'punctuation')
    df['category'] = df['instruction_id_list_for_eval'].apply(
        lambda x: x[0].split(':')[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
    )
    
    # 4. Calculate Accuracy per Category
    category_stats = df.groupby('category')['follow_all_instructions'].mean().reset_index()
    category_stats['accuracy'] = category_stats['follow_all_instructions'] * 100
    category_stats = category_stats.sort_values('accuracy', ascending=False)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Professional color palette for ICLR
    ax = sns.barplot(x='category', y='accuracy', data=category_stats, palette='magma')
    
    plt.title(f"Baseline Instruction Adherence: {model_label}\n(Zero-Shot: No Steering, No Prompt Instructions)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.xlabel("Constraint Category", fontsize=10)
    plt.ylim(0, 100)
    
    # Add Mean Accuracy Line
    avg_acc = df['follow_all_instructions'].mean() * 100
    plt.axhline(avg_acc, color='red', linestyle='--', label=f'Mean: {avg_acc:.2f}%')
    plt.legend()

    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    output_filename = f"viz_{model_label.replace('.', '_')}.png"
    plt.savefig(output_filename, dpi=300) # High DPI for paper quality
    print(f"✅ Visualization saved: {output_filename}")
    plt.show()

# --- RUN ON YOUR LOCAL FILES ---
# Use raw strings (r"") to avoid Windows path errors
# Adjust these paths to match your local 'out' folder exactly
base_out = r"C:\Users\ryany\Algoverse25-26Spring\mult-ster\third_party\microsoft_llm_steer_instruct\format"

targets = [
    os.path.join(base_out, "Qwen1.5-1.8B-Chat", "no_instr", "out.jsonl"),
    os.path.join(base_out, "Phi-3-mini-4k-instruct", "no_instr", "out.jsonl"),
    os.path.join(base_out, "Mistral-7B-Instruct-v0.1", "no_instr", "out.jsonl"),
    os.path.join(base_out, "gemma-2-2b-it", "no_instr", "out.jsonl")
]

for target in targets:
    generate_research_viz(target)