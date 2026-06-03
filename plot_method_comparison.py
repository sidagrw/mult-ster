import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="results/graphs/gemma_method_comparison.png")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    labels = []
    tokens = []
    lines = []
    pass10 = []
    syntax = []

    for _, row in df.iterrows():
        if row["method"] == "No steering":
            label = "No\nsteering"
        elif row["method"] == "Additive vector":
            label = f"Add\n{row['scale_alpha']}"
        else:
            label = f"Matrix\n{row['scale_alpha']}"

        labels.append(label)
        tokens.append(row["avg_total_tokens"])
        lines.append(row["avg_code_lines"])
        pass10.append(row["pass_rate_10_lines"])
        syntax.append(row["syntax_valid_rate"])

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    axes[0].bar(x, tokens)
    axes[0].set_title("Avg Total Tokens")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")

    axes[1].bar(x, lines)
    axes[1].axhline(10, linestyle="--")
    axes[1].set_title("Avg Code/Text Lines")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    axes[2].bar(x, pass10)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Pass Rate <= 10 Lines")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")

    axes[3].bar(x, syntax)
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Syntax Validity Rate")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
