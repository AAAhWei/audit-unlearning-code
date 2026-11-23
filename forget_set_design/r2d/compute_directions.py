#!/usr/bin/env python3
"""
This script calculates the gradient direction of each training point (spec.md section 4.2).

For the cross-entropy loss, the gradient of the weights of the last linear layer is
    G_x = (p_x - y_x) h_x^T        a matrix with C rows and d columns
    p_x    the softmax output
    y_x    the one-hot label
    h_x    the input of the last linear layer
A forward pass gives G_x. The script does not include the bias of the last layer.

For each point x:
    1. At each step t, flatten G_x and divide it by its length.
    2. Add the vectors of all steps.
    3. Divide the sum by its length. The result is the direction u_x.

The script reads the weight files of train_reference.py. It uses the steps 1 to T, the
same steps as compute_scores.py.

Output files in --out-dir:
    directions_<alg>.npy     u_x for all training points, shape [points, C * d], float16.
                             The rows are in ascending global index order, the same order
                             as scores_<alg>.csv.
    directions_<alg>.json    the settings and the global index of each row. It also
                             contains the cosine between u_x and the direction at the
                             last step only.

The script can use the CPU. For 45,000 points and 35 steps, it needs approximately
35 minutes with 4 CPU threads.

Example:
    CUDA_VISIBLE_DEVICES="" python3 forget_set_design/r2d/compute_directions.py --device cpu \\
        --ckpt-dir /tmp2/$USER/audit-unlearning/reference/r2d/checkpoints \\
        --data-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/cifar100_uniform_bs_750_seed1 \\
        --out-dir /tmp2/$USER/audit-unlearning/scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R2D_DIR = REPO / "r2d_audit"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from compute_scores import load_train_points  # noqa: E402
from make_custom_split import DEFAULT_DATA_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the last-layer gradient direction of each training point.")
    parser.add_argument("--alg", default="r2d", choices=["r2d", "modelclip"],
                        help="The unlearning algorithm. Only r2d is available.")
    parser.add_argument("--ckpt-dir", required=True, help="The checkpoints directory of train_reference.py.")
    parser.add_argument("--data-dir", required=True, help="The split directory that train_reference.py used.")
    parser.add_argument("--raw-train-dir", default=str(DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train"),
                        help="The CIFAR-100 train files in the original row order.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    parser.add_argument("--model", default="tinynetcifar100")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="The validation fraction of the split. Only for a split with no index file.")
    parser.add_argument("--split-seed", type=int, default=1,
                        help="The seed of the split. Only for a split with no index file.")
    parser.add_argument("--chunk", type=int, default=500, help="The number of points for one forward pass.")
    parser.add_argument("--device", default="cpu",
                        help="The torch device. For a GPU, set CUDA_VISIBLE_DEVICES to select it.")
    parser.add_argument("--threads", type=int, default=4, help="The number of CPU threads.")
    return parser.parse_args()


def unit_gradients(model, images, labels, device, chunk):
    """
    Calculate G_x / ||G_x|| for all points, for the weights that `model` has.
    This is a generator. Each item is (first row, array [rows, C * d]).
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    kept = {}
    hook = model.fc.register_forward_pre_hook(lambda module, inputs: kept.__setitem__("h", inputs[0].detach()))
    n_classes = model.fc.out_features
    try:
        for start in range(0, len(images), chunk):
            # The input is the same as in the audit: float32, channels first, values 0 to 255.
            x = torch.from_numpy(images[start:start + chunk]).to(device).permute(0, 3, 1, 2).float()
            y = torch.from_numpy(labels[start:start + chunk]).to(device)
            with torch.no_grad():
                residual = F.softmax(model(x).double(), dim=1) - F.one_hot(y, n_classes).double()
                h = kept["h"].double()
                # ||G_x|| = ||p_x - y_x|| * ||h_x||. Divide each factor by its length.
                residual = residual / residual.norm(dim=1, keepdim=True).clamp_min(1e-300)
                h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-300)
                g = (residual.unsqueeze(2) * h.unsqueeze(1)).flatten(1)
            yield start, g.float().cpu().numpy()
    finally:
        hook.remove()


def main() -> int:
    args = parse_args()
    if args.alg != "r2d":
        print("ERROR: --alg modelclip is not available. Only r2d is available.", file=sys.stderr)
        return 2
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import torch

    sys.path.insert(0, str(R2D_DIR))
    import models

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, global_idx = load_train_points(args)
    n = len(images)

    # step_000.pt contains the weights before training. The direction does not use it.
    ckpts = sorted(p for p in Path(args.ckpt_dir).glob("step_*.pt") if p.stem != "step_000")
    n_steps = len(ckpts)
    if n_steps == 0:
        raise SystemExit(f"ERROR: {args.ckpt_dir} contains no step_<t>.pt file.")

    model = models.get_model(args.model, num_classes=int(labels.max()) + 1, filters_percentage=1.0).to(device)
    dim = model.fc.out_features * model.fc.in_features
    total = np.zeros((n, dim), dtype=np.float32)
    last = None
    for row, path in enumerate(ckpts):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        is_last = row == n_steps - 1
        if is_last:
            last = np.zeros(n, dtype=np.float64)
        for start, g in unit_gradients(model, images, labels, device, args.chunk):
            total[start:start + len(g)] += g
            if is_last:
                # Keep the cosine between the sum of all steps and the last step.
                rows = total[start:start + len(g)].astype(np.float64)
                last[start:start + len(g)] = (rows * g).sum(1) / np.linalg.norm(rows, axis=1)
        print(f"[directions] step {row + 1:3d}/{n_steps} is complete.")

    # The length of the sum is n_steps if the direction is the same at all steps.
    length = np.linalg.norm(total.astype(np.float64), axis=1)
    total /= np.maximum(length, 1e-300)[:, None].astype(np.float32)
    np.save(out_dir / f"directions_{args.alg}.npy", total.astype(np.float16))

    report = {
        "alg": args.alg, "ckpt_dir": str(args.ckpt_dir), "data_dir": str(args.data_dir),
        "model": args.model, "n_points": n, "n_steps": n_steps,
        "n_classes": int(model.fc.out_features), "last_layer_inputs": int(model.fc.in_features), "dim": dim,
        # 1 = the direction of the point is the same at all steps.
        "sum_length_over_steps_median": float(np.median(length) / n_steps),
        "sum_length_over_steps_min": float(length.min() / n_steps),
        "cosine_with_last_step_median": float(np.median(last)),
        "cosine_with_last_step_min": float(last.min()),
        "global_idx": global_idx.tolist(),
    }
    with open(out_dir / f"directions_{args.alg}.json", "w") as fh:
        json.dump(report, fh)
    print(f"[directions] {n} points, {dim} dimensions. Median of (length of the sum / steps): "
          f"{report['sum_length_over_steps_median']:.3f}. Median cosine with the last step: "
          f"{report['cosine_with_last_step_median']:.3f}")
    print(f"[directions] The script wrote {out_dir / f'directions_{args.alg}.npy'} and directions_{args.alg}.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
