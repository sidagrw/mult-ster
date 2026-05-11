import argparse
import re


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--max_lines", type=int, default=10)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    code = extract_code_block(text)
    lines = count_code_lines(code)
    passed = lines <= args.max_lines

    print("Code lines:", lines)
    print("Max allowed:", args.max_lines)
    print("Pass:", passed)
    print("\nExtracted code/text:")
    print(code)


if __name__ == "__main__":
    main()
