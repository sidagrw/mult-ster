import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer


MODEL_ALIASES = {
    "gpt2-small": "gpt2",
}


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def get_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


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


@torch.no_grad()
def get_final_token_resid_post_low_memory(model, prompt, layer, device):
    tokens = model.to_tokens(prompt, prepend_bos=False).to(device)

    saved = {}

    def save_hook(act, hook):
        saved["h"] = act[0, -1, :].detach().float().cpu()
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"

    with model.hooks(fwd_hooks=[(hook_name, save_hook)]):
        _ = model(tokens)

    return saved["h"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)
    dtype = get_dtype(args.dtype)

    print("Loading tokenizer:", model_name)
    tokenizer = load_tokenizer_compat(model_name)

    print("Loading model:", model_name)
    print("Device:", args.device)
    print("Dtype:", args.dtype)

    model = HookedTransformer.from_pretrained(
        model_name,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    model.eval()

    pairs = load_pairs(args.pairs)

    base_acts = []
    instr_acts = []

    for idx, pair in enumerate(pairs):
        h_base = get_final_token_resid_post_low_memory(
            model=model,
            prompt=pair["base_prompt"],
            layer=args.layer,
            device=args.device,
        )

        h_instr = get_final_token_resid_post_low_memory(
            model=model,
            prompt=pair["instructed_prompt"],
            layer=args.layer,
            device=args.device,
        )

        base_acts.append(h_base)
        instr_acts.append(h_instr)

        print(f"Processed {idx + 1}/{len(pairs)}")

        if args.device == "cuda":
            torch.cuda.empty_cache()

    base_acts = torch.stack(base_acts, dim=0)
    instr_acts = torch.stack(instr_acts, dim=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model_name,
            "layer": args.layer,
            "dtype": args.dtype,
            "h_base": base_acts,
            "h_instr": instr_acts,
        },
        out_path,
    )

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
