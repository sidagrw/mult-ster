import argparse
import re
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--max_lines", type=int, default=10)
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    code = extract_code_block(text)

    total_tokens = len(tokenizer.encode(text))
    code_tokens = len(tokenizer.encode(code))
    words = count_words(text)
    code_lines = count_code_lines(code)
    line_pass = code_lines <= args.max_lines
    line_excess = max(0, code_lines - args.max_lines)

    print("File:", args.file)
    print("Model tokenizer:", model_name)
    print("Total output tokens:", total_tokens)
    print("Code-only tokens:", code_tokens)
    print("Word count:", words)
    print("Non-empty code lines:", code_lines)
    print("Max allowed code lines:", args.max_lines)
    print("Line constraint pass:", line_pass)
    print("Line excess:", line_excess)
    print()
    print("Extracted code/text:")
    print(code)


if __name__ == "__main__":
    main()
