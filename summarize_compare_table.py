import argparse
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results/length_task/gemma_compare_summary_max.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    summary = (
        df.groupby(["method", "scale_alpha"])
        .agg(
            avg_total_tokens=("total_tokens", "mean"),
            avg_code_tokens=("code_tokens", "mean"),
            avg_word_count=("word_count", "mean"),
            avg_code_lines=("code_lines", "mean"),
            pass_rate_5_lines=("pass_5_lines", "mean"),
            pass_rate_8_lines=("pass_8_lines", "mean"),
            pass_rate_10_lines=("pass_10_lines", "mean"),
            pass_rate_12_lines=("pass_12_lines", "mean"),
            pass_rate_15_lines=("pass_15_lines", "mean"),
            syntax_valid_rate=("syntax_valid", "mean"),
            n=("prompt_id", "count"),
        )
        .reset_index()
    )

    for col in summary.columns:
        if col.startswith("avg") or col.startswith("pass") or col.endswith("rate"):
            summary[col] = summary[col].round(3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)

    print(summary.to_string(index=False))
    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
