import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results/graphs/gemma2_2b_microsoft_style_length.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # -----------------------------
    # (a) Length vs. Steering Alpha
    # -----------------------------
    ax = axes[0]

    steered = df[df["method"] == "Matrix steering"].copy()

    alpha_values = sorted(steered["alpha"].dropna().unique())

    for alpha in alpha_values:
        sub = steered[steered["alpha"] == alpha]
        ax.hist(
            sub["total_tokens"],
            bins=10,
            alpha=0.35,
            density=True,
            label=f"α={alpha}",
        )

        mean_len = sub["total_tokens"].mean()
        ax.scatter(
            [mean_len],
            [0.0],
            marker="o",
            s=45,
        )

    ax.set_title("(a) Length vs. Steering Alpha", fontsize=13)
    ax.set_xlabel("Length (# of tokens)")
    ax.set_ylabel("Probability Density")
    ax.legend(title="Alpha", fontsize=8)

    # -----------------------------------
    # (b) Accuracy Per Length Constraint
    # -----------------------------------
    ax = axes[1]

    constraints = [
        ("5", "pass_5_lines"),
        ("8", "pass_8_lines"),
        ("10", "pass_10_lines"),
        ("12", "pass_12_lines"),
        ("15", "pass_15_lines"),
    ]

    no = df[df["method"] == "No steering"]
    st = df[(df["method"] == "Matrix steering") & (df["alpha"] == 1.0)]

    no_rates = [no[col].mean() for _, col in constraints]
    st_rates = [st[col].mean() for _, col in constraints]

    x = np.arange(len(constraints))
    width = 0.36

    ax.bar(x - width / 2, no_rates, width, label="Std. Inference")
    ax.bar(x + width / 2, st_rates, width, label="Matrix Steering", hatch="//")

    ax.set_title("(b) Accuracy Per Code-Length Constraint", fontsize=13)
    ax.set_xlabel("Max. # of Code/Text Lines")
    ax.set_ylabel("Pass Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}*" for c, _ in constraints])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)

    # ------------------------------------
    # (c) Length: Pre- vs. Post-Steering
    # ------------------------------------
    ax = axes[2]

    no_lengths = no["code_lines"]
    st_lengths = st["code_lines"]

    bins = range(
        int(min(no_lengths.min(), st_lengths.min())),
        int(max(no_lengths.max(), st_lengths.max())) + 2,
    )

    ax.hist(
        no_lengths,
        bins=bins,
        alpha=0.35,
        density=True,
        label="Std. Inference",
    )

    ax.hist(
        st_lengths,
        bins=bins,
        alpha=0.35,
        density=True,
        label="Matrix Steering",
    )

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
