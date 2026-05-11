import argparse

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


MODEL_ALIASES = {
    "gpt2-small": "gpt2",
}


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def load_tokenizer_compat(model_name):
    model_name = normalize_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        add_bos_token=False,
    )

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/generations/no_steering.txt")
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)
    tokenizer = load_tokenizer_compat(model_name)

    model = HookedTransformer.from_pretrained(
        model_name,
        tokenizer=tokenizer,
        device=args.device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    tokens = model.to_tokens(args.prompt, prepend_bos=False).to(args.device)

    out = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        verbose=False,
        prepend_bos=False,
    )

    text = model.to_string(out[0])

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print("\nSaved generation to:", args.out)


if __name__ == "__main__":
    main()
