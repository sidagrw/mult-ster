import argparse
import os
import re
import pandas as pd
from transformers import AutoTokenizer


MODEL_ALIASES = {
    "gpt2-small": "gpt2",
}


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def extract_code_block(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def count_code_lines(code):
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def infer_method_alpha(path):
    name = os.path.basename(path)

    if "no_steering" in name:
        return "no_steering", 0.0

    if "alpha025" in name:
        return "low_rank_matrix", 0.25

    if "alpha05" in name:
        return "low_rank_matrix", 0.5

    if "alpha10" in name:
        return "low_rank_matrix", 1.0

    if "alpha20" in name:
        return "low_rank_matrix", 2.0

    return "unknown", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--max_lines", type=int, default=10)
    parser.add_argument("--out", default="results/length_task/gemma2_2b_length_metrics_summary.csv")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    rows = []

    for path in args.files:
        if not os.path.exists(path):
            print("Skipping missing file:", path)
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        code = extract_code_block(text)
        method, alpha = infer_method_alpha(path)

        total_tokens = len(tokenizer.encode(text))
        code_tokens = len(tokenizer.encode(code))
        words = count_words(text)
        code_lines = count_code_lines(code)
        line_pass = code_lines <= args.max_lines
        line_excess = max(0, code_lines - args.max_lines)

        rows.append({
            "model": model_name,
            "method": method,
            "alpha": alpha,
            "file": path,
            "total_tokens": total_tokens,
            "code_tokens": code_tokens,
            "word_count": words,
            "code_lines": code_lines,
            "max_lines": args.max_lines,
            "line_pass": line_pass,
            "line_excess": line_excess,
        })

    df = pd.DataFrame(rows)

    baseline = df[df["method"] == "no_steering"]["total_tokens"]
    if len(baseline) > 0:
        base_tokens = baseline.iloc[0]
        df["token_reduction_pct"] = ((base_tokens - df["total_tokens"]) / base_tokens) * 100
    else:
        df["token_reduction_pct"] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
