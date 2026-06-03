import argparse
import ast
import os
import re
import pandas as pd
from transformers import AutoTokenizer


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


def syntax_valid(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def parse_filename(filename):
    name = filename.replace(".txt", "")
    parts = name.split("_")
    prompt_id = int(parts[1])

    if "no_steering" in name:
        return prompt_id, "No steering", 0.0

    if "additive_scale" in name:
        raw = name.split("additive_scale")[-1]
        mapping = {"025": 0.25, "05": 0.5, "10": 1.0, "20": 2.0}
        return prompt_id, "Additive vector", mapping[raw]

    if "matrix_alpha" in name:
        raw = name.split("matrix_alpha")[-1]
        mapping = {"025": 0.25, "05": 0.5, "10": 1.0, "20": 2.0}
        return prompt_id, "Multiplicative matrix", mapping[raw]

    return prompt_id, "Unknown", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--out", default="results/length_task/gemma_compare_metrics_max.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(args.input_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        code = extract_code_block(text)
        prompt_id, method, scale_alpha = parse_filename(filename)

        total_tokens = len(tokenizer.encode(text))
        code_tokens = len(tokenizer.encode(code))
        word_count = count_words(text)
        code_lines = count_code_lines(code)

        rows.append({
            "prompt_id": prompt_id,
            "method": method,
            "scale_alpha": scale_alpha,
            "file": path,
            "total_tokens": total_tokens,
            "code_tokens": code_tokens,
            "word_count": word_count,
            "code_lines": code_lines,
            "pass_5_lines": code_lines <= 5,
            "pass_8_lines": code_lines <= 8,
            "pass_10_lines": code_lines <= 10,
            "pass_12_lines": code_lines <= 12,
            "pass_15_lines": code_lines <= 15,
            "syntax_valid": syntax_valid(code),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.groupby(["method", "scale_alpha"]).size())
    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
