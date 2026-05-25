import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def label_row(row):
    if row["method"] == "no_steering":
        return "No steering"
    return f"Matrix α={row['alpha']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="results/graphs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    df["label"] = df.apply(label_row, axis=1)

    # Graph 1: total tokens
    plt.figure(figsize=(9, 5))
    plt.bar(df["label"], df["total_tokens"])
    plt.ylabel("Total output tokens")
    plt.xlabel("Method")
    plt.title("Gemma 2 2B Length Task: Total Output Tokens")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "gemma2_2b_total_tokens.png")
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

    # Graph 2: code lines
    plt.figure(figsize=(9, 5))
    plt.bar(df["label"], df["code_lines"])
    plt.axhline(y=10, linestyle="--")
    plt.ylabel("Non-empty code/text lines")
    plt.xlabel("Method")
    plt.title("Gemma 2 2B Length Task: Code-Line Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "gemma2_2b_code_lines.png")
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

    # Graph 3: token reduction percentage
    steered = df[df["method"] != "no_steering"].copy()
    if "token_reduction_pct" in steered.columns and len(steered) > 0:
        plt.figure(figsize=(9, 5))
        plt.bar(steered["label"], steered["token_reduction_pct"])
        plt.axhline(y=0, linestyle="--")
        plt.ylabel("Token reduction vs no steering (%)")
        plt.xlabel("Method")
        plt.title("Gemma 2 2B Length Task: Token Reduction")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, "gemma2_2b_token_reduction.png")
        plt.savefig(out_path, dpi=200)
        print("Saved:", out_path)

    # Graph 4: alpha sweep
    alpha_df = df[df["method"] != "no_steering"].copy()
    if len(alpha_df) > 0:
        alpha_df = alpha_df.sort_values("alpha")

        plt.figure(figsize=(8, 5))
        plt.plot(alpha_df["alpha"], alpha_df["total_tokens"], marker="o")
        plt.ylabel("Total output tokens")
        plt.xlabel("Steering strength alpha")
        plt.title("Gemma 2 2B Length Task: Alpha vs Tokens")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, "gemma2_2b_alpha_vs_tokens.png")
        plt.savefig(out_path, dpi=200)
        print("Saved:", out_path)

        plt.figure(figsize=(8, 5))
        plt.plot(alpha_df["alpha"], alpha_df["code_lines"], marker="o")
        plt.axhline(y=10, linestyle="--")
        plt.ylabel("Non-empty code/text lines")
        plt.xlabel("Steering strength alpha")
        plt.title("Gemma 2 2B Length Task: Alpha vs Code Lines")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, "gemma2_2b_alpha_vs_code_lines.png")
        plt.savefig(out_path, dpi=200)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
