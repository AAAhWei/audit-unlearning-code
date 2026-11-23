#!/usr/bin/env python3
"""
This script calculates the difficulty score of each training point (spec.md section 3).

The score is the per-instance privacy loss of Sepahvand et al., arXiv 2505.18786,
Definition 4.2:

    P(x, alpha) = sum_{t=1..T} C_t * ln E_w f_t(||g(x, w)||_2)
    C_t         = (1 / (alpha - 1)) * ((p - 1) / p)^(t + 1)
    ln f_t(g)   = p * ln( sum_{k=0..o_t} binom(o_t, k) * q^k * (1 - q)^(o_t - k)
                          * exp(g^2 * (k^2 - k) / (2 * sigma^2)) )
    o_t         = o_p applied t times to alpha,   o_p(a) = p / (p - 1) * a - 1 / p
    p           = 3 * T
    q           = batch size / dataset size

g(x, w) is the gradient of the loss of the point x for all parameters.
The script uses one model for each step: the weights after step t.

R2D is full-batch gradient descent, thus q = 1. Only the term k = o_t stays:

    P_hat(x, alpha) = p / (2 * sigma^2) * sum_t C_t * (o_t^2 - o_t) * ||g(x, w_t)||^2

Only --alg r2d is available.

Output files in --out-dir (spec.md sections 3.4 and 3.5):
    grad_norms_r2d.npy   float32 [T, n]. Row t - 1 is the gradient norm after step t.
    scores_r2d.csv       columns: global_idx, label, P_hat_a<alpha>, avg_gradnorm, lastlayer_score
    scores_r2d.json      the checks: rank correlations, weights, accuracy for each step
The columns of the .npy file and the rows of the .csv file have the same order:
ascending global index.

Example:
    CUDA_VISIBLE_DEVICES=4 python3 forget_set_design/r2d/compute_scores.py \\
        --ckpt-dir /tmp2/$USER/audit-unlearning/reference/r2d/checkpoints \\
        --data-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/cifar100_uniform_bs_750_seed1 \\
        --out-dir /tmp2/$USER/audit-unlearning/scores
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R2D_DIR = REPO / "r2d_audit"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from make_custom_split import DEFAULT_DATA_DIR, INDEX_FILE, load_batch_dir, paper_holdout  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the difficulty score of each training point.")
    parser.add_argument("--alg", default="r2d", choices=["r2d", "modelclip"],
                        help="The unlearning algorithm. Only r2d is available.")
    parser.add_argument("--ckpt-dir", required=True,
                        help="The directory of the step_<t>.pt files from train_reference.py.")
    parser.add_argument("--data-dir", required=True,
                        help="The split directory that the reference model trained on.")
    parser.add_argument("--raw-train-dir", default=str(DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train"),
                        help="The CIFAR-100 train files in the original row order. "
                             "The script uses them to make sure that the global indices are correct.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    parser.add_argument("--model", default="tinynetcifar100")
    parser.add_argument("--alphas", default="2,4,8,16", help="The Renyi orders, with commas between them.")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="The noise level of the score. It does not change the order of the points for R2D.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="The validation fraction of the split. The script uses it if the split "
                             "directory has no index file.")
    parser.add_argument("--split-seed", type=int, default=1,
                        help="The seed of the split. The script uses it if the split directory has no index file.")
    parser.add_argument("--chunk", type=int, default=500,
                        help="The number of points for one per-example gradient calculation.")
    parser.add_argument("--device", default="cuda:0",
                        help="The torch device. Set CUDA_VISIBLE_DEVICES to select the GPU.")
    parser.add_argument("--threads", type=int, default=4, help="The number of CPU threads.")
    return parser.parse_args()


# ---------------------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------------------
def renyi_orders(alpha: float, n_steps: int) -> np.ndarray:
    """Return o_0 to o_T. o_0 = alpha and o_t = o_p(o_{t-1}) with p = 3 * T."""
    p = 3.0 * n_steps
    orders = [float(alpha)]
    for _ in range(n_steps):
        orders.append(p / (p - 1.0) * orders[-1] - 1.0 / p)
    return np.array(orders)


def step_weights(alpha: float, n_steps: int, sigma: float, order_shift: int = 0) -> np.ndarray:
    """
    Return the weight of ||g(x, w_t)||^2 for t = 1 to T in the full-batch score.

    order_shift = 0 uses o_t at step t. This is Definition 4.2.
    order_shift = 1 uses o_{t-1} at step t. Use it only to examine the effect of the index.
    """
    p = 3.0 * n_steps
    t = np.arange(1, n_steps + 1)
    order = renyi_orders(alpha, n_steps)[t - order_shift]
    c_t = (1.0 / (alpha - 1.0)) * ((p - 1.0) / p) ** (t + 1)
    return p / (2.0 * sigma ** 2) * c_t * (order ** 2 - order)


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def load_train_points(args: argparse.Namespace):
    """
    Read the training points of the split directory.
    Return (images uint8 NHWC, labels, global indices), in ascending global index order.
    """
    data_dir = Path(args.data_dir)
    images, labels = load_batch_dir(data_dir / "train")

    index_file = data_dir / INDEX_FILE
    if index_file.exists():
        global_idx = np.load(index_file)["train"].astype(np.int64)
        source = INDEX_FILE
    else:
        # The split script of the paper writes no index file. Calculate the row order again.
        raw_n = len(images) + int(round(args.val_split * len(images) / (1.0 - args.val_split)))
        global_idx = paper_holdout(raw_n, args.val_split, args.split_seed)["train_order"].astype(np.int64)
        source = "the random draws of split_cifar100_train.py"
    if len(global_idx) != len(images):
        raise SystemExit(f"ERROR: {len(images)} training rows but {len(global_idx)} global indices.")

    # Make sure that each row is the point with that global index.
    raw_images, raw_labels = load_batch_dir(Path(args.raw_train_dir))
    if not (np.array_equal(raw_images[global_idx], images) and np.array_equal(raw_labels[global_idx], labels)):
        raise SystemExit("ERROR: The training rows do not agree with the global indices. "
                         "Examine --data-dir, --raw-train-dir, --val-split and --split-seed.")
    print(f"[scores] {len(images)} training points. Global indices from {source}. The check passed.")

    order = np.argsort(global_idx)
    return images[order], labels[order].astype(np.int64), global_idx[order]


# ---------------------------------------------------------------------------------------
# Gradient norms
# ---------------------------------------------------------------------------------------
def checkpoint_statistics(model, images, labels, device, chunk):
    """
    Calculate the statistics of all points for the weights that `model` has.

    Return a dictionary of arrays with one value for each point:
        grad_norm   ||gradient of the loss for all parameters||_2
        lastlayer   ||softmax - one-hot label||_2 * ||input of the last layer||_2
        correct     1 if the prediction is correct
        loss        the cross-entropy loss
    """
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call, grad, vmap

    model.eval()
    params = {name: value.detach() for name, value in model.named_parameters()}

    def loss_of_one_point(p, x, y):
        logits = functional_call(model, p, (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))

    per_example_grad = vmap(grad(loss_of_one_point), in_dims=(None, 0, 0))

    # The hook keeps the input of the last layer.
    kept = {}
    hook = model.fc.register_forward_pre_hook(lambda module, inputs: kept.__setitem__("h", inputs[0].detach()))

    out = {key: [] for key in ("grad_norm", "lastlayer", "correct", "loss")}
    n_classes = model.fc.out_features
    try:
        for start in range(0, len(images), chunk):
            # The input is the same as in the audit: float32, channels first, values 0 to 255.
            x = torch.from_numpy(images[start:start + chunk]).to(device).permute(0, 3, 1, 2).float()
            y = torch.from_numpy(labels[start:start + chunk]).to(device)

            grads = per_example_grad(params, x, y)
            sq_norm = sum(g.flatten(1).double().pow(2).sum(1) for g in grads.values())
            out["grad_norm"].append(sq_norm.sqrt().cpu().numpy())

            with torch.no_grad():
                logits = model(x)
                residual = F.softmax(logits.double(), dim=1) - F.one_hot(y, n_classes).double()
                out["lastlayer"].append((residual.norm(dim=1) * kept["h"].double().norm(dim=1)).cpu().numpy())
                out["correct"].append((logits.argmax(1) == y).double().cpu().numpy())
                out["loss"].append(F.cross_entropy(logits, y, reduction="none").double().cpu().numpy())
    finally:
        hook.remove()
    return {key: np.concatenate(value) for key, value in out.items()}


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if args.alg != "r2d":
        print("ERROR: --alg modelclip is not available. Only r2d is available.", file=sys.stderr)
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import torch
    from scipy.stats import spearmanr

    sys.path.insert(0, str(R2D_DIR))
    import models

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    alphas = [float(a) for a in args.alphas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, global_idx = load_train_points(args)
    n = len(images)

    # step_000.pt contains the weights before training. The score does not use it.
    ckpts = sorted(p for p in Path(args.ckpt_dir).glob("step_*.pt") if p.stem != "step_000")
    n_steps = len(ckpts)
    if n_steps == 0:
        raise SystemExit(f"ERROR: {args.ckpt_dir} contains no step_<t>.pt file.")
    expected = [f"step_{t:03d}" for t in range(1, n_steps + 1)]
    if [p.stem for p in ckpts] != expected:
        raise SystemExit(f"ERROR: The files in {args.ckpt_dir} are not step_001 to step_{n_steps:03d}.")

    model = models.get_model(args.model, num_classes=int(labels.max()) + 1, filters_percentage=1.0).to(device)
    grad_norms = np.zeros((n_steps, n), dtype=np.float32)
    lastlayer = np.zeros((n_steps, n), dtype=np.float64)
    per_step = []
    for row, path in enumerate(ckpts):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        stats = checkpoint_statistics(model, images, labels, device, args.chunk)
        grad_norms[row] = stats["grad_norm"]
        lastlayer[row] = stats["lastlayer"]
        per_step.append({
            "step": row + 1,
            "train_accuracy": float(stats["correct"].mean()),
            "train_loss": float(stats["loss"].mean()),
            "grad_norm_median": float(np.median(stats["grad_norm"])),
            "grad_norm_max": float(stats["grad_norm"].max()),
        })
        print(f"[scores] step {row + 1:3d}/{n_steps}: train accuracy {100 * per_step[-1]['train_accuracy']:.2f}%, "
              f"median gradient norm {per_step[-1]['grad_norm_median']:.4g}")
    np.save(out_dir / f"grad_norms_{args.alg}.npy", grad_norms)

    # The scores. Use float64: the factor p / (2 * sigma^2) is large.
    sq = grad_norms.astype(np.float64) ** 2
    scores = {a: step_weights(a, n_steps, args.sigma) @ sq for a in alphas}
    scores_shifted = {a: step_weights(a, n_steps, args.sigma, order_shift=1) @ sq for a in alphas}
    avg_gradnorm = grad_norms.astype(np.float64).mean(axis=0)
    # Spec.md section 3.5: combine the last-layer norms with the weights of the score.
    lastlayer_score = step_weights(alphas[0], n_steps, args.sigma) @ lastlayer ** 2

    def column(alpha: float) -> str:
        return f"P_hat_a{alpha:g}"

    with open(out_dir / f"scores_{args.alg}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["global_idx", "label"] + [column(a) for a in alphas] + ["avg_gradnorm", "lastlayer_score"])
        for i in range(n):
            writer.writerow([int(global_idx[i]), int(labels[i])] + [repr(float(scores[a][i])) for a in alphas]
                            + [repr(float(avg_gradnorm[i])), repr(float(lastlayer_score[i]))])

    def rho(a, b) -> float:
        return float(spearmanr(a, b).statistic)

    base = scores[alphas[0]]
    report = {
        "alg": args.alg, "n_points": n, "n_steps": n_steps, "p": 3 * n_steps, "sigma": args.sigma,
        "alphas": alphas, "ckpt_dir": str(args.ckpt_dir), "data_dir": str(args.data_dir),
        "weight_last_over_first": {column(a): float(np.divide(*step_weights(a, n_steps, args.sigma)[[-1, 0]]))
                                   for a in alphas},
        "spearman_with_" + column(alphas[0]): {
            **{column(a): rho(base, scores[a]) for a in alphas[1:]},
            "avg_gradnorm": rho(base, avg_gradnorm),
            "lastlayer_score": rho(base, lastlayer_score),
        },
        "spearman_order_index_t_and_t_minus_1": {column(a): rho(scores[a], scores_shifted[a]) for a in alphas},
        "score_summary": {column(a): {"min": float(scores[a].min()), "median": float(np.median(scores[a])),
                                      "max": float(scores[a].max())} for a in alphas},
        "per_step": per_step,
    }
    with open(out_dir / f"scores_{args.alg}.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[scores] weight of the last step / weight of the first step: {report['weight_last_over_first']}")
    print(f"[scores] Spearman correlation with {column(alphas[0])}: {report['spearman_with_' + column(alphas[0])]}")
    print(f"[scores] Spearman correlation, order index t and t - 1: {report['spearman_order_index_t_and_t_minus_1']}")
    print(f"[scores] The script wrote grad_norms_{args.alg}.npy, scores_{args.alg}.csv and "
          f"scores_{args.alg}.json to {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
