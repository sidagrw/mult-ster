import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = torch.load(args.activations, map_location="cpu")

    h_base = data["h_base"].float()
    h_instr = data["h_instr"].float()

    vector = (h_instr - h_base).mean(dim=0)

    if args.normalize:
        vector = vector / (vector.norm() + 1e-8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "method": "additive_vector",
            "normalize": args.normalize,
            "model": data["model"],
            "layer": data["layer"],
            "steering": vector,
        },
        out_path,
    )

    print("Saved additive vector:", out_path)


if __name__ == "__main__":
    main()
