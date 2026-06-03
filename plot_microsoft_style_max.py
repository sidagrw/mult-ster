import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def hist_line(values, bins):
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results/graphs/gemma_microsoft_style_max.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # (a) Length vs. Steering Weights
    ax = axes[0]
    matrix = df[df["method"] == "Multiplicative matrix"].copy()
    weights = [0.25, 0.5, 1.0, 2.0]

    max_len = max(matrix["word_count"].max(), 1)
    bins = np.linspace(0, max_len + 5, 25)

    for w in weights:
        sub = matrix[matrix["scale_alpha"] == w]
        centers, density = hist_line(sub["word_count"], bins)

        ax.bar(centers, density, width=(bins[1] - bins[0]) * 0.7, alpha=0.18)
        ax.plot(centers, density, linewidth=2, label=f"{w}")

        marker_y = max(density) + 0.005 if len(density) else 0.005
        ax.scatter(sub["word_count"], [marker_y] * len(sub), s=12)

    ax.set_title("(a) Length vs. Steering Weights", fontsize=13)
    ax.set_xlabel("Length (# of words)")
    ax.set_ylabel("Probability Density")
    ax.legend(title="Weight", fontsize=8)

    # (b) Accuracy per code-length constraint
    ax = axes[1]
    constraints = [
        ("5*", "pass_5_lines"),
        ("8*", "pass_8_lines"),
        ("10*", "pass_10_lines"),
        ("12*", "pass_12_lines"),
        ("15*", "pass_15_lines"),
    ]

    no = df[df["method"] == "No steering"]
    add = df[(df["method"] == "Additive vector") & (df["scale_alpha"] == 1.0)]
    mat = df[(df["method"] == "Multiplicative matrix") & (df["scale_alpha"] == 1.0)]

    no_rates = [no[col].mean() for _, col in constraints]
    add_rates = [add[col].mean() for _, col in constraints]
    mat_rates = [mat[col].mean() for _, col in constraints]

    x = np.arange(len(constraints))
    width = 0.25

    ax.bar(x - width, no_rates, width, label="Std. Inference")
    ax.bar(x, add_rates, width, label="Additive")
    ax.bar(x + width, mat_rates, width, label="Matrix", hatch="//")

    ax.set_title("(b) Accuracy Per Code-Length Constraint", fontsize=13)
    ax.set_xlabel("Max. # of Code/Text Lines")
    ax.set_ylabel("Pass Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in constraints])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)

    # (c) pre/post steering length distribution
    ax = axes[2]

    mat1 = df[(df["method"] == "Multiplicative matrix") & (df["scale_alpha"] == 1.0)]

    no_lengths = no["code_lines"]
    mat_lengths = mat1["code_lines"]

    min_len = int(min(no_lengths.min(), mat_lengths.min()))
    max_len = int(max(no_lengths.max(), mat_lengths.max()))
    bins = np.arange(min_len, max_len + 2)

    ax.hist(no_lengths, bins=bins, density=True, alpha=0.25, label="Std. Inference")
    ax.hist(mat_lengths, bins=bins, density=True, alpha=0.25, label="Matrix Steering")

    no_centers, no_density = hist_line(no_lengths, bins)
    mat_centers, mat_density = hist_line(mat_lengths, bins)

    ax.plot(no_centers, no_density, linewidth=2)
    ax.plot(mat_centers, mat_density, linewidth=2)

    ax.axvline(10, linestyle="--", linewidth=2)
    ax.text(10.2, ax.get_ylim()[1] * 0.85, "Length constraint: 10")

    ax.set_title("(c) Code Length: Pre- vs. Post-Steering", fontsize=13)
    ax.set_xlabel("Length (# of non-empty lines)")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
