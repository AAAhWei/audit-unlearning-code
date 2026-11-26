#!/usr/bin/env python3
"""
This script calculates the gradient direction of each training point for model clipping
(spec.md section 4.2).

For the cross-entropy loss, the gradient of the weights of the last linear layer is
    G_x = h_x (p_x - y_x)^T        a matrix with d rows and C columns
    p_x    the softmax output
    y_x    the one-hot label
    h_x    the input of the last linear layer
The script does not include the bias of the last layer.

For each point x:
    1. At each checkpoint, flatten G_x and divide it by its length.
    2. Add the vectors of all checkpoints.
    3. Divide the sum by its length. The result is the direction u_x.

The script reads the checkpoints of the reference model of run_scores.sh. It uses the
same checkpoint epochs as compute_scores.py.

Output files in --out-dir:
    directions_modelclip.npy     u_x for all training points, shape [points, d * C],
                                 float16. The rows are in ascending global index order,
                                 the same order as scores_modelclip.csv.
    directions_modelclip.json    the settings and the global index of each row. It also
                                 contains the cosine between u_x and the direction at the
                                 last checkpoint only.

Example:
    CUDA_VISIBLE_DEVICES=4 python3 forget_set_design/modelclip/compute_directions.py \\
        --run-dir /tmp2/$USER/audit-unlearning/reference/modelclip/runs/<id> \\
        --data-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/cifar100_uniform_bs_750_seed1 \\
        --out-dir /tmp2/$USER/audit-unlearning/scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# JAX then allocates GPU memory when it needs the memory, not all at the start.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from compute_scores import PIPELINE_DIR, restore_params, select_epochs  # noqa: E402
from make_custom_split import DEFAULT_DATA_DIR  # noqa: E402
from score_utils import load_train_points  # noqa: E402

ALG = "modelclip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the last-layer gradient direction of each training point for model clipping.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="The run directory of the reference model. It contains config.yaml and "
                             "ckpt/checkpoint_<epoch>.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The split directory that the reference model trained on.")
    parser.add_argument("--raw-train-dir", type=Path,
                        default=DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train",
                        help="The CIFAR-100 train files in the original row order.")
    parser.add_argument("--out-dir", type=Path, required=True, help="The output directory.")
    parser.add_argument("--num-checkpoints", type=int, default=35,
                        help="The number of checkpoints N. The script selects the epochs "
                             "round(i * epochs / N) for i = 1 to N.")
    parser.add_argument("--epochs", default="",
                        help="A list of checkpoint epochs with commas. It replaces --num-checkpoints.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="The validation fraction of the split. Only for a split with no index file.")
    parser.add_argument("--split-seed", type=int, default=1,
                        help="The seed of the split. Only for a split with no index file.")
    parser.add_argument("--chunk", type=int, default=500, help="The number of points for one calculation.")
    return parser.parse_args()


def make_unit_gradients(model, last_layer: str):
    """
    Return a function (params, images, labels) -> G_x / ||G_x|| for each point, with the
    shape [points, d * C].

    The function calculates the gradient for the weights of the last layer only. Thus the
    cost is approximately that of a forward pass.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def loss_of_one_point(kernel, params, x, y):
        last = {**params[last_layer], "kernel": kernel}
        logits = model.apply({"params": {**params, last_layer: last}}, x[None], train=False)[0]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y)

    per_example = jax.vmap(jax.grad(loss_of_one_point), in_axes=(None, None, 0, 0))

    @jax.jit
    def unit_gradients(params, images, labels):
        g = per_example(params[last_layer]["kernel"], params, images, labels)
        g = g.reshape(g.shape[0], -1)
        return g / jnp.maximum(jnp.linalg.norm(g, axis=1, keepdims=True), 1e-30)

    return unit_gradients


def main() -> int:
    args = parse_args()

    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import yaml

    sys.path.insert(0, str(PIPELINE_DIR))
    from src.models.model import ModelFactory

    with open(args.run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, global_idx = load_train_points(args.data_dir, args.raw_train_dir,
                                                   args.val_split, args.split_seed, tag="directions")
    n = len(images)
    epochs = select_epochs(args, int(config["training"]["epochs"]))
    paths = [args.run_dir / "ckpt" / f"checkpoint_{e}" for e in epochs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} checkpoints do not exist. The first one is {missing[0]}.")

    model = ModelFactory.create_model(model_name=config["model"]["name"],
                                      num_classes=config["model"]["n_classes"])
    checkpointer = ocp.PyTreeCheckpointer()
    unit_gradients = None
    total = None
    last = np.zeros(n, dtype=np.float64)
    shape = None
    for row, path in enumerate(paths):
        params = restore_params(checkpointer, path)
        if unit_gradients is None:
            # The last layer is the only layer whose weights have 2 dimensions.
            dense = [name for name, leaf in params.items() if np.ndim(leaf.get("kernel")) == 2]
            if len(dense) != 1:
                raise SystemExit(f"ERROR: The model must have one dense layer. It has {len(dense)}.")
            shape = params[dense[0]]["kernel"].shape
            unit_gradients = make_unit_gradients(model, dense[0])
            total = np.zeros((n, int(np.prod(shape))), dtype=np.float32)
        is_last = row == len(paths) - 1
        for start in range(0, n, args.chunk):
            # The input is the same as in the training: the uint8 values of the split files.
            g = np.asarray(unit_gradients(params, jnp.asarray(images[start:start + args.chunk]),
                                          jnp.asarray(labels[start:start + args.chunk])))
            total[start:start + len(g)] += g
            if is_last:
                # Keep the cosine between the sum of all checkpoints and the last checkpoint.
                rows = total[start:start + len(g)].astype(np.float64)
                last[start:start + len(g)] = (rows * g).sum(1) / np.linalg.norm(rows, axis=1)
        print(f"[directions] checkpoint {row + 1:3d}/{len(paths)} (epoch {epochs[row]}) is complete.", flush=True)
    if not np.isfinite(total).all():
        raise SystemExit("ERROR: The directions contain values that are not finite.")

    # The length of the sum is the number of checkpoints if the direction does not change.
    length = np.linalg.norm(total.astype(np.float64), axis=1)
    total /= np.maximum(length, 1e-300)[:, None].astype(np.float32)
    np.save(args.out_dir / f"directions_{ALG}.npy", total.astype(np.float16))

    report = {
        "alg": ALG, "run_dir": str(args.run_dir), "data_dir": str(args.data_dir),
        "model": config["model"]["name"], "n_points": n, "n_steps": len(paths),
        "checkpoint_epochs": [int(e) for e in epochs],
        "n_classes": int(shape[1]), "last_layer_inputs": int(shape[0]), "dim": int(np.prod(shape)),
        # 1 = the direction of the point is the same at all checkpoints.
        "sum_length_over_steps_median": float(np.median(length) / len(paths)),
        "sum_length_over_steps_min": float(length.min() / len(paths)),
        "cosine_with_last_step_median": float(np.median(last)),
        "cosine_with_last_step_min": float(last.min()),
        "global_idx": global_idx.tolist(),
    }
    with open(args.out_dir / f"directions_{ALG}.json", "w") as fh:
        json.dump(report, fh)
    print(f"[directions] {n} points, {report['dim']} dimensions. Median of (length of the sum / checkpoints): "
          f"{report['sum_length_over_steps_median']:.3f}. Median cosine with the last checkpoint: "
          f"{report['cosine_with_last_step_median']:.3f}")
    print(f"[directions] The script wrote {args.out_dir / f'directions_{ALG}.npy'} and directions_{ALG}.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
