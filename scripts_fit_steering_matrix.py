import argparse
from pathlib import Path

import torch


def fit_ridge_matrix(h_base, h_instr, lam=1.0, residual_form=True):
    if h_base.shape != h_instr.shape:
        raise ValueError(f"Shape mismatch: {h_base.shape} vs {h_instr.shape}")

    x = h_base.float()
    y = h_instr.float() - h_base.float() if residual_form else h_instr.float()

    n, d = x.shape
    eye = torch.eye(d, device=x.device, dtype=x.dtype)

    xtx = x.T @ x
    xty = x.T @ y

    w_t = torch.linalg.solve(xtx + lam * eye, xty)
    return w_t.T


def fit_diagonal_matrix(h_base, h_instr, lam=1.0, residual_form=True):
    x = h_base.float()
    y = h_instr.float() - h_base.float() if residual_form else h_instr.float()

    numerator = (x * y).sum(dim=0)
    denominator = (x * x).sum(dim=0) + lam

    return numerator / denominator


def fit_low_rank_matrix_svd(h_base, h_instr, rank=8, lam=1.0, residual_form=True):
    w = fit_ridge_matrix(h_base, h_instr, lam=lam, residual_form=residual_form)

    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = min(rank, s.shape[0])

    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--method", choices=["ridge", "diagonal", "low_rank"], default="low_rank")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--residual_form", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = torch.load(args.activations, map_location="cpu")
    h_base = data["h_base"]
    h_instr = data["h_instr"]

    if args.method == "ridge":
        steering = fit_ridge_matrix(h_base, h_instr, args.lambda_reg, args.residual_form)
    elif args.method == "diagonal":
        steering = fit_diagonal_matrix(h_base, h_instr, args.lambda_reg, args.residual_form)
    else:
        steering = fit_low_rank_matrix_svd(h_base, h_instr, args.rank, args.lambda_reg, args.residual_form)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "method": args.method,
            "rank": args.rank,
            "lambda_reg": args.lambda_reg,
            "residual_form": args.residual_form,
            "model": data["model"],
            "layer": data["layer"],
            "steering": steering,
        },
        out_path,
    )

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
