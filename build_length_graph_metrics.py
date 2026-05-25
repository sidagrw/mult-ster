import argparse
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


def parse_file_name(path):
    name = os.path.basename(path)
    parts = name.replace(".txt", "").split("_")

    prompt_id = int(parts[1])

    setting = "_".join(parts[2:])

    if setting == "no_steering":
        method = "No steering"
        alpha = 0.0
    elif setting == "alpha025":
        method = "Matrix steering"
        alpha = 0.25
    elif setting == "alpha05":
        method = "Matrix steering"
        alpha = 0.5
    elif setting == "alpha10":
        method = "Matrix steering"
        alpha = 1.0
    elif setting == "alpha20":
        method = "Matrix steering"
        alpha = 2.0
    else:
        method = "Unknown"
        alpha = None

    return prompt_id, method, alpha, setting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--out", default="results/length_task/gemma2_2b_graph_metrics.csv")
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
        prompt_id, method, alpha, setting = parse_file_name(path)

        total_tokens = len(tokenizer.encode(text))
        code_tokens = len(tokenizer.encode(code))
        code_lines = count_code_lines(code)

        rows.append({
            "model": args.model,
            "prompt_id": prompt_id,
            "method": method,
            "setting": setting,
            "alpha": alpha,
            "file": path,
            "total_tokens": total_tokens,
            "code_tokens": code_tokens,
            "code_lines": code_lines,
            "pass_5_lines": code_lines <= 5,
            "pass_8_lines": code_lines <= 8,
            "pass_10_lines": code_lines <= 10,
            "pass_12_lines": code_lines <= 12,
            "pass_15_lines": code_lines <= 15,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
