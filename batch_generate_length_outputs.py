import argparse
import json
import os
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


def get_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    return prompts


def load_tokenizer_compat(model_name):
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


def residual_matrix_hook(activation, hook, matrix, alpha=1.0):
    matrix = matrix.to(device=activation.device, dtype=activation.dtype)
    return activation + alpha * (activation @ matrix.T)


@torch.no_grad()
def generate_text(model, prompt, device, max_new_tokens):
    tokens = model.to_tokens(prompt, prepend_bos=False).to(device)

    out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        verbose=False,
        prepend_bos=False,
    )

    return model.to_string(out[0])


@torch.no_grad()
def generate_text_steered(model, prompt, device, max_new_tokens, layer, matrix, alpha):
    tokens = model.to_tokens(prompt, prepend_bos=False).to(device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    hook_fn = lambda act, hook: residual_matrix_hook(
        act,
        hook,
        matrix=matrix,
        alpha=alpha,
    )

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            prepend_bos=False,
        )

    return model.to_string(out[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--steering_file", required=True)
    parser.add_argument("--out_dir", default="outputs/generations/batch_gemma")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = get_dtype(args.dtype)
    tokenizer = load_tokenizer_compat(args.model)

    model = HookedTransformer.from_pretrained(
        args.model,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()

    steering_data = torch.load(args.steering_file, map_location="cpu")
    layer = steering_data["layer"]
    matrix = steering_data["steering"]

    prompts = load_prompts(args.prompts)

    settings = [
        ("no_steering", None),
        ("alpha025", 0.25),
        ("alpha05", 0.5),
        ("alpha10", 1.0),
        ("alpha20", 2.0),
    ]

    for i, prompt in enumerate(prompts):
        for label, alpha in settings:
            out_path = os.path.join(args.out_dir, f"prompt_{i:02d}_{label}.txt")

            print(f"Generating prompt {i}, setting {label}")

            if alpha is None:
                text = generate_text(
                    model=model,
                    prompt=prompt,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                text = generate_text_steered(
                    model=model,
                    prompt=prompt,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    layer=layer,
                    matrix=matrix,
                    alpha=alpha,
                )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print("Saved:", out_path)

            if args.device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
