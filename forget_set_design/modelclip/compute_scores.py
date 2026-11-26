#!/usr/bin/env python3
"""
This script calculates the difficulty score of each training point for model clipping
(spec.md section 3.3).

The score is the per-instance privacy loss of Sepahvand et al., arXiv 2505.18786,
Definition 4.2:

    P(x, alpha) = sum_{t=1..T} C_t * ln f_t(||g(x, w_t)||_2)
    C_t         = (1 / (alpha - 1)) * ((p - 1) / p)^(t + 1)
    ln f_t(g)   = p * ln( sum_{k=0..o_t} binom(o_t, k) * q^k * (1 - q)^(o_t - k)
                          * exp(g^2 * (k^2 - k) / (2 * sigma^2)) )
    o_t         = o_p applied t times to alpha,   o_p(a) = p / (p - 1) * a - 1 / p
    p           = 3 * T
    q           = batch size / dataset size

g(x, w) is the gradient of the cross-entropy loss of the point x for all parameters.
T is the number of training steps of the reference model.

The reference model has too many steps to keep the weights of each step. The script uses
N checkpoints at the steps s_1 < ... < s_N, with s_0 = 0:

    P_hat(x, alpha) = sum_{i=1..N} C_{s_i} * ln f_{s_i}(||g(x, w_{s_i})||_2) * (s_i - s_{i-1})

Approximations:
    - The order o_t is not an integer. The script uses the next integer above o_t in the
      binomial sum.
    - One model for each checkpoint replaces the mean over the models.

The reference model is a run of experiment_unlearning_exhaustive_combinations.py with
--forget_fraction 1.0 and --num_unlearn_per_combo 0. That run trains on all training
points and saves the weights after each epoch. run_scores.sh starts it.

Output files in --out-dir:
    grad_norms_modelclip.npy   float32 [N, n]. Row i - 1 is the gradient norm at s_i.
    scores_modelclip.csv       columns: global_idx, label, P_hat_a<alpha>, avg_gradnorm,
                               lastlayer_score
    scores_modelclip.json      the checks: rank correlations, accuracy for each checkpoint
The columns of the .npy file and the rows of the .csv file have the same order:
ascending global index.

Example:
    CUDA_VISIBLE_DEVICES=4 python3 forget_set_design/modelclip/compute_scores.py \\
        --run-dir /tmp2/$USER/audit-unlearning/reference/modelclip/runs/<id> \\
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

# JAX then allocates GPU memory when it needs the memory, not all at the start.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO / "certified_unlearning_plus_grad_based_methods"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from make_custom_split import DEFAULT_DATA_DIR, load_batch_dir  # noqa: E402
from score_utils import load_train_points  # noqa: E402

ALG = "modelclip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the difficulty score of each training point for model clipping.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="The run directory of the reference model. It contains config.yaml and "
                             "ckpt/checkpoint_<epoch>.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The split directory that the reference model trained on.")
    parser.add_argument("--raw-train-dir", type=Path,
                        default=DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train",
                        help="The CIFAR-100 train files in the original row order. "
                             "The script uses them to make sure that the global indices are correct.")
    parser.add_argument("--out-dir", type=Path, required=True, help="The output directory.")
    parser.add_argument("--num-checkpoints", type=int, default=35,
                        help="The number of checkpoints N. The script selects the epochs "
                             "round(i * epochs / N) for i = 1 to N.")
    parser.add_argument("--epochs", default="",
                        help="A list of checkpoint epochs with commas. It replaces --num-checkpoints.")
    parser.add_argument("--alphas", default="2,4,8,16", help="The Renyi orders, with commas between them.")
    parser.add_argument("--sigma", type=float, default=0.1, help="The noise level of the score.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="The validation fraction of the split. The script uses it if the split "
                             "directory has no index file.")
    parser.add_argument("--split-seed", type=int, default=1,
                        help="The seed of the split. The script uses it if the split directory has no index file.")
    parser.add_argument("--chunk", type=int, default=500,
                        help="The number of points for one per-example gradient calculation.")
    return parser.parse_args()


# ---------------------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------------------
def order_at(alpha: float, total_steps: int, step: np.ndarray) -> np.ndarray:
    """
    Return o_t for each step t in `step`. o_0 = alpha and o_t = o_p(o_{t-1}) with p = 3 * T.

    The recursion has the fixed point (p - 1) / p, thus
        o_t = fixed + (alpha - fixed) * (p / (p - 1))^t
    """
    p = 3.0 * total_steps
    fixed = (p - 1.0) / p
    return fixed + (alpha - fixed) * np.exp(np.asarray(step, dtype=np.float64) * np.log(p / (p - 1.0)))


def coefficient_at(alpha: float, total_steps: int, step: np.ndarray) -> np.ndarray:
    """Return C_t for each step t in `step`."""
    p = 3.0 * total_steps
    return (1.0 / (alpha - 1.0)) * np.exp((np.asarray(step, dtype=np.float64) + 1.0) * np.log((p - 1.0) / p))


def log_f(grad_norm: np.ndarray, order: float, q: float, sigma: float, total_steps: int) -> np.ndarray:
    """
    Return ln f_t(g) for each gradient norm g. The script uses the next integer above
    `order`. All terms stay as logarithms, because the exponent is very large.
    """
    from scipy.special import gammaln, logsumexp

    p = 3.0 * total_steps
    top = int(np.ceil(order - 1e-9))
    k = np.arange(top + 1, dtype=np.float64)
    log_binomial = gammaln(top + 1.0) - gammaln(k + 1.0) - gammaln(top - k + 1.0)
    log_weight = log_binomial + k * np.log(q) + (top - k) * np.log1p(-q)
    exponent = np.square(grad_norm.astype(np.float64))[:, None] * (k * k - k)[None, :] / (2.0 * sigma ** 2)
    return p * logsumexp(log_weight[None, :] + exponent, axis=1)


def riemann_score(norms: np.ndarray, steps: np.ndarray, alpha: float, q: float, sigma: float,
                  total_steps: int) -> np.ndarray:
    """Return P_hat for each point. `norms` has the shape [checkpoints, points]."""
    widths = np.diff(np.concatenate([[0], steps])).astype(np.float64)
    orders = order_at(alpha, total_steps, steps)
    coefficients = coefficient_at(alpha, total_steps, steps)
    score = np.zeros(norms.shape[1], dtype=np.float64)
    for i in range(len(steps)):
        score += coefficients[i] * widths[i] * log_f(norms[i], orders[i], q, sigma, total_steps)
    return score


def full_batch_form(norms: np.ndarray, steps: np.ndarray, alpha: float, sigma: float,
                    total_steps: int) -> np.ndarray:
    """
    Return the score in the form for q = 1: a weighted sum of the squared gradient norms.
    The script uses it only to compare the order of the points (spec.md section 3.3).
    """
    p = 3.0 * total_steps
    widths = np.diff(np.concatenate([[0], steps])).astype(np.float64)
    orders = order_at(alpha, total_steps, steps)
    weights = p / (2.0 * sigma ** 2) * coefficient_at(alpha, total_steps, steps) * (orders ** 2 - orders) * widths
    return weights @ np.square(norms.astype(np.float64))


# ---------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------
def select_epochs(args: argparse.Namespace, train_epochs: int) -> list:
    """Return the list of checkpoint epochs, in ascending order."""
    if args.epochs:
        epochs = sorted({int(e) for e in args.epochs.split(",") if e.strip()})
    else:
        n = args.num_checkpoints
        epochs = [int(round(i * train_epochs / n)) for i in range(1, n + 1)]
        if len(set(epochs)) != n or epochs[0] < 1:
            raise SystemExit(f"ERROR: --num-checkpoints {n} is too large for {train_epochs} epochs.")
    if not epochs or epochs[0] < 1 or epochs[-1] > train_epochs:
        raise SystemExit(f"ERROR: The checkpoint epochs must be between 1 and {train_epochs}.")
    return epochs


def restore_params(checkpointer, path: Path) -> dict:
    """
    Read the weights of a checkpoint as numpy arrays.

    A checkpoint records the device of each array. The default restore fails if that
    device is not available, for example for a GPU checkpoint on a CPU. The numpy type
    has no device.
    """
    import jax
    import orbax.checkpoint as ocp

    path = str(path.resolve())
    tree = checkpointer.metadata(path).item_metadata.tree
    as_numpy = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
    return checkpointer.restore(path, restore_args=as_numpy)["params"]


def make_statistics(model, last_layer: str):
    """
    Return a function (params, images, labels) -> (grad_norm, lastlayer, loss, correct),
    with one value for each point:
        grad_norm   ||gradient of the loss for all parameters||_2
        lastlayer   ||gradient of the loss for the weights of the last layer||_2. This is
                    ||softmax - one-hot label||_2 * ||input of the last layer||_2.
        loss        the cross-entropy loss
        correct     1 if the prediction is correct
    """
    import jax
    import jax.numpy as jnp
    import optax

    def loss_of_one_point(params, x, y):
        logits = model.apply({"params": params}, x[None], train=False)[0]
        return optax.softmax_cross_entropy_with_integer_labels(logits, y), logits

    per_example = jax.vmap(jax.value_and_grad(loss_of_one_point, has_aux=True), in_axes=(None, 0, 0))

    def squared_norm(leaf):
        return jnp.sum(jnp.square(leaf).reshape(leaf.shape[0], -1), axis=1)

    @jax.jit
    def statistics(params, images, labels):
        (loss, logits), grads = per_example(params, images, labels)
        total = sum(squared_norm(leaf) for leaf in jax.tree_util.tree_leaves(grads))
        last = squared_norm(grads[last_layer]["kernel"])
        correct = (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
        return jnp.sqrt(total), jnp.sqrt(last), loss, correct

    return statistics


def all_points(statistics, params, images: np.ndarray, labels: np.ndarray, chunk: int) -> dict:
    """Apply `statistics` to all points in parts of `chunk` points."""
    import jax.numpy as jnp

    out = {key: [] for key in ("grad_norm", "lastlayer", "loss", "correct")}
    for start in range(0, len(images), chunk):
        # The input is the same as in the training: the uint8 values of the split files.
        values = statistics(params, jnp.asarray(images[start:start + chunk]),
                            jnp.asarray(labels[start:start + chunk]))
        for key, value in zip(out, values):
            out[key].append(np.asarray(value, dtype=np.float64))
    return {key: np.concatenate(value) for key, value in out.items()}


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    import orbax.checkpoint as ocp
    import yaml
    from scipy.stats import spearmanr

    from src.models.model import ModelFactory

    with open(args.run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    train_epochs = int(config["training"]["epochs"])
    batch_size = int(config["dataset"]["batch_size"])
    alphas = [float(a) for a in args.alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, global_idx = load_train_points(args.data_dir, args.raw_train_dir,
                                                   args.val_split, args.split_seed)
    n = len(images)
    # The training loader does not use the last points that do not fill a batch.
    steps_per_epoch = n // batch_size
    total_steps = train_epochs * steps_per_epoch
    q = batch_size / n

    epochs = select_epochs(args, train_epochs)
    paths = [args.run_dir / "ckpt" / f"checkpoint_{e}" for e in epochs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} checkpoints do not exist. The first one is {missing[0]}.")
    steps = np.array(epochs, dtype=np.int64) * steps_per_epoch
    print(f"[scores] {len(epochs)} checkpoints, epochs {epochs[0]} to {epochs[-1]}. "
          f"T = {total_steps} steps, q = {q:.6f}, sigma = {args.sigma}.")

    model = ModelFactory.create_model(model_name=config["model"]["name"],
                                      num_classes=config["model"]["n_classes"])
    checkpointer = ocp.PyTreeCheckpointer()
    statistics = None
    grad_norms = np.zeros((len(epochs), n), dtype=np.float32)
    lastlayer = np.zeros((len(epochs), n), dtype=np.float64)
    per_checkpoint = []
    params = None
    for row, path in enumerate(paths):
        params = restore_params(checkpointer, path)
        if statistics is None:
            # The last layer is the only layer whose weights have 2 dimensions.
            dense = [name for name, leaf in params.items() if np.ndim(leaf.get("kernel")) == 2]
            if len(dense) != 1:
                raise SystemExit(f"ERROR: The model must have one dense layer. It has {len(dense)}.")
            statistics = make_statistics(model, dense[0])
        stats = all_points(statistics, params, images, labels, args.chunk)
        if not all(np.isfinite(v).all() for v in stats.values()):
            raise SystemExit(f"ERROR: The values of {path} are not finite.")
        grad_norms[row] = stats["grad_norm"]
        lastlayer[row] = stats["lastlayer"]
        per_checkpoint.append({
            "epoch": int(epochs[row]), "step": int(steps[row]),
            "train_accuracy": float(stats["correct"].mean()),
            "train_loss": float(stats["loss"].mean()),
            "grad_norm_median": float(np.median(stats["grad_norm"])),
            "grad_norm_max": float(stats["grad_norm"].max()),
        })
        print(f"[scores] checkpoint {row + 1:3d}/{len(epochs)} (epoch {epochs[row]}): "
              f"train accuracy {100 * per_checkpoint[-1]['train_accuracy']:.2f}%, "
              f"median gradient norm {per_checkpoint[-1]['grad_norm_median']:.4g}", flush=True)
    np.save(args.out_dir / f"grad_norms_{ALG}.npy", grad_norms)

    # The test accuracy of the last checkpoint (spec.md section 2.3).
    test_images, test_labels = load_batch_dir(args.data_dir / "test")
    test_stats = all_points(statistics, params, test_images, test_labels.astype(np.int64), args.chunk)
    test_accuracy = float(test_stats["correct"].mean())
    print(f"[scores] last checkpoint: train accuracy {100 * per_checkpoint[-1]['train_accuracy']:.2f}%, "
          f"test accuracy {100 * test_accuracy:.2f}%")

    scores = {a: riemann_score(grad_norms, steps, a, q, args.sigma, total_steps) for a in alphas}
    full_batch = {a: full_batch_form(grad_norms, steps, a, args.sigma, total_steps) for a in alphas}
    avg_gradnorm = grad_norms.astype(np.float64).mean(axis=0)
    # Spec.md section 3.5: combine the last-layer norms with the formula of the score.
    lastlayer_score = riemann_score(lastlayer, steps, alphas[0], q, args.sigma, total_steps)

    def column(alpha: float) -> str:
        return f"P_hat_a{alpha:g}"

    with open(args.out_dir / f"scores_{ALG}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["global_idx", "label"] + [column(a) for a in alphas] + ["avg_gradnorm", "lastlayer_score"])
        for i in range(n):
            writer.writerow([int(global_idx[i]), int(labels[i])] + [repr(float(scores[a][i])) for a in alphas]
                            + [repr(float(avg_gradnorm[i])), repr(float(lastlayer_score[i]))])

    def rho(a, b) -> float:
        return float(spearmanr(a, b).statistic)

    base = scores[alphas[0]]
    report = {
        "alg": ALG, "n_points": n, "total_steps": total_steps, "p": 3 * total_steps, "q": q,
        "sigma": args.sigma, "alphas": alphas, "checkpoint_epochs": [int(e) for e in epochs],
        "run_dir": str(args.run_dir), "data_dir": str(args.data_dir),
        "last_checkpoint": {"train_accuracy": per_checkpoint[-1]["train_accuracy"],
                            "test_accuracy": test_accuracy,
                            "gap": per_checkpoint[-1]["train_accuracy"] - test_accuracy},
        "order_first_and_last": {column(a): [float(v) for v in order_at(a, total_steps, steps[[0, -1]])]
                                 for a in alphas},
        "spearman_with_" + column(alphas[0]): {
            **{column(a): rho(base, scores[a]) for a in alphas[1:]},
            "avg_gradnorm": rho(base, avg_gradnorm),
            "lastlayer_score": rho(base, lastlayer_score),
        },
        "spearman_with_full_batch_form": {column(a): rho(scores[a], full_batch[a]) for a in alphas},
        # A score of 0 has no order. Many such points make the easy pool arbitrary.
        "fraction_of_points_with_score_zero": {column(a): float(np.mean(scores[a] <= 0.0)) for a in alphas},
        "score_summary": {column(a): {"min": float(scores[a].min()), "median": float(np.median(scores[a])),
                                      "max": float(scores[a].max())} for a in alphas},
        "per_checkpoint": per_checkpoint,
    }
    with open(args.out_dir / f"scores_{ALG}.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[scores] Spearman correlation with {column(alphas[0])}: {report['spearman_with_' + column(alphas[0])]}")
    print(f"[scores] Spearman correlation with the full-batch form: {report['spearman_with_full_batch_form']}")
    print(f"[scores] fraction of points with the score 0: {report['fraction_of_points_with_score_zero']}")
    print(f"[scores] The script wrote grad_norms_{ALG}.npy, scores_{ALG}.csv and "
          f"scores_{ALG}.json to {args.out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
