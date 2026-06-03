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


def load_prompts(path, limit=None):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
                if limit is not None and len(prompts) >= limit:
                    break
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


def additive_hook(activation, hook, vector, scale):
    vector = vector.to(device=activation.device, dtype=activation.dtype)
    return activation + scale * vector


def matrix_hook(activation, hook, matrix, alpha):
    matrix = matrix.to(device=activation.device, dtype=activation.dtype)
    return activation + alpha * (activation @ matrix.T)


@torch.no_grad()
def generate_plain(model, prompt, device, max_new_tokens):
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
def generate_with_hook(model, prompt, device, max_new_tokens, hook_name, hook_fn):
    tokens = model.to_tokens(prompt, prepend_bos=False).to(device)
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
    parser.add_argument("--additive_file", required=True)
    parser.add_argument("--matrix_file", required=True)
    parser.add_argument("--out_dir", default="outputs/generations/compare_gemma_max")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=None)
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

    additive_data = torch.load(args.additive_file, map_location="cpu")
    matrix_data = torch.load(args.matrix_file, map_location="cpu")

    additive_layer = additive_data["layer"]
    additive_vector = additive_data["steering"]
    matrix_layer = matrix_data["layer"]
    matrix = matrix_data["steering"]

    additive_hook_name = f"blocks.{additive_layer}.hook_resid_post"
    matrix_hook_name = f"blocks.{matrix_layer}.hook_resid_post"

    prompts = load_prompts(args.prompts, limit=args.limit)

    additive_scales = [0.25, 0.5, 1.0, 2.0]
    matrix_alphas = [0.25, 0.5, 1.0, 2.0]

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i}: {prompt}")

        out_path = os.path.join(args.out_dir, f"prompt_{i:02d}_no_steering.txt")
        if not os.path.exists(out_path):
            text = generate_plain(model, prompt, args.device, args.max_new_tokens)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print("Saved:", out_path)
        else:
            print("Skipping existing:", out_path)

        for scale in additive_scales:
            label = str(scale).replace(".", "")
            out_path = os.path.join(args.out_dir, f"prompt_{i:02d}_additive_scale{label}.txt")

            if os.path.exists(out_path):
                print("Skipping existing:", out_path)
                continue

            hook_fn = lambda act, hook, s=scale: additive_hook(
                act,
                hook,
                vector=additive_vector,
                scale=s,
            )

            text = generate_with_hook(
                model,
                prompt,
                args.device,
                args.max_new_tokens,
                additive_hook_name,
                hook_fn,
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print("Saved:", out_path)

        for alpha in matrix_alphas:
            label = str(alpha).replace(".", "")
            out_path = os.path.join(args.out_dir, f"prompt_{i:02d}_matrix_alpha{label}.txt")

            if os.path.exists(out_path):
                print("Skipping existing:", out_path)
                continue

            hook_fn = lambda act, hook, a=alpha: matrix_hook(
                act,
                hook,
                matrix=matrix,
                alpha=a,
            )

            text = generate_with_hook(
                model,
                prompt,
                args.device,
                args.max_new_tokens,
                matrix_hook_name,
                hook_fn,
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            print("Saved:", out_path)

        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
