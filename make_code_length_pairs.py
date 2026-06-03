import argparse
import json
import os


CONSTRAINTS = [5, 8, 10, 12, 15]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.base, "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

    with open(args.out, "w", encoding="utf-8") as f:
        for prompt in prompts:
            for max_lines in CONSTRAINTS:
                row = {
                    "base_prompt": prompt,
                    "instructed_prompt": f"{prompt} Use at most {max_lines} lines of Python code.",
                    "max_lines": max_lines,
                }
                f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(prompts) * len(CONSTRAINTS)} paired examples to {args.out}")


if __name__ == "__main__":
    main()
