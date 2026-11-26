#!/usr/bin/env python3
"""
This module contains the parts of the difficulty score that R2D and model clipping share
(spec.md section 3).

    load_train_points   reads the training points of a split directory and their global
                        indices
    renyi_orders        gives the order o_t of the score formula for each step
    load_forget_points  reads the global indices, the labels and the batches of the
                        forget points of a split directory
    read_scores         reads the scores file for a list of points
    separation          gives the audit signal d_x of each forget point
    correlations        gives the rank correlation between d_x and a score

The compute_scores.py and validate_score.py scripts of r2d/ and modelclip/ use this module.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from make_custom_split import INDEX_FILE, load_batch_dir, paper_holdout


def load_train_points(data_dir: Path, raw_train_dir: Path, val_split: float, split_seed: int,
                      tag: str = "scores"):
    """
    Read the training points of the split directory.
    Return (images uint8 NHWC, labels, global indices), in ascending global index order.

    The global index of a point is its row in the original CIFAR-100 train file.
    val_split and split_seed are necessary only if the split directory has no index file.
    """
    data_dir = Path(data_dir)
    images, labels = load_batch_dir(data_dir / "train")

    index_file = data_dir / INDEX_FILE
    if index_file.exists():
        global_idx = np.load(index_file)["train"].astype(np.int64)
        source = INDEX_FILE
    else:
        # The split script of the paper writes no index file. Calculate the row order again.
        raw_n = len(images) + int(round(val_split * len(images) / (1.0 - val_split)))
        global_idx = paper_holdout(raw_n, val_split, split_seed)["train_order"].astype(np.int64)
        source = "the random draws of split_cifar100_train.py"
    if len(global_idx) != len(images):
        raise SystemExit(f"ERROR: {len(images)} training rows but {len(global_idx)} global indices.")

    # Make sure that each row is the point with that global index.
    raw_images, raw_labels = load_batch_dir(Path(raw_train_dir))
    if not (np.array_equal(raw_images[global_idx], images) and np.array_equal(raw_labels[global_idx], labels)):
        raise SystemExit("ERROR: The training rows do not agree with the global indices. "
                         "Examine --data-dir, --raw-train-dir, --val-split and --split-seed.")
    print(f"[{tag}] {len(images)} training points. Global indices from {source}. The check passed.")

    order = np.argsort(global_idx)
    return images[order], labels[order].astype(np.int64), global_idx[order]


def renyi_orders(alpha: float, n_steps: int) -> np.ndarray:
    """Return o_0 to o_T. o_0 = alpha and o_t = o_p(o_{t-1}) with p = 3 * T."""
    p = 3.0 * n_steps
    orders = [float(alpha)]
    for _ in range(n_steps):
        orders.append(p / (p - 1.0) * orders[-1] - 1.0 / p)
    return np.array(orders)


def load_forget_points(split_dir: Path, raw_train_dir: Path):
    """
    Return (global indices, labels, batch ids) of the forget points.
    The order is the order of the audit scores: batch 0 first, rows in file order.
    """
    index_file = split_dir / INDEX_FILE
    if not index_file.exists():
        raise SystemExit(f"ERROR: {index_file} does not exist.")
    index = np.load(index_file)
    global_idx = index["forget"].astype(np.int64)
    batch_id = index["forget_batch_id"].astype(np.int64)

    images, labels = load_batch_dir(split_dir / "forget")
    raw_images, raw_labels = load_batch_dir(raw_train_dir)
    if len(images) != len(global_idx) or not (
            np.array_equal(raw_images[global_idx], images) and np.array_equal(raw_labels[global_idx], labels)):
        raise SystemExit("ERROR: The forget rows do not agree with the global indices.")
    return global_idx, labels.astype(np.int64), batch_id


def read_scores(path: Path, global_idx: np.ndarray) -> dict:
    """Read the scores file. Return a dictionary: column name -> values for `global_idx`."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    row_of = {int(r["global_idx"]): i for i, r in enumerate(rows)}
    missing = [g for g in global_idx if int(g) not in row_of]
    if missing:
        raise SystemExit(f"ERROR: {path} has no score for {len(missing)} forget points.")
    select = [row_of[int(g)] for g in global_idx]
    columns = [c for c in rows[0] if c not in ("global_idx", "label")]
    return {c: np.array([float(rows[i][c]) for i in select]) for c in columns}


def separation(cal_mean: np.ndarray, cal_var: np.ndarray, batch_id: np.ndarray, m: int) -> np.ndarray:
    """Return d_x for each point. cal_mean and cal_var have the shape [configurations, points]."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pooled_linear_auditor"))
    import pooled_linear as pl

    _, signs = pl.sign_configs(m)
    if cal_mean.shape[0] != len(signs):
        raise SystemExit(f"ERROR: {cal_mean.shape[0]} configurations in the file, but m = {m} gives {len(signs)}.")
    d = np.zeros(cal_mean.shape[1])
    for b in range(m):
        cols = batch_id == b
        stats = []
        for sign in (1, -1):
            rows = signs[:, b] == sign
            mean = cal_mean[rows][:, cols]
            stats.append((mean.mean(0), cal_var[rows][:, cols].mean(0) + mean.var(0)))
        (mu_plus, v_plus), (mu_minus, v_minus) = stats
        d[cols] = np.abs(mu_plus - mu_minus) / np.sqrt(np.maximum((v_plus + v_minus) / 2.0, 1e-300))
    return d


def correlations(d: np.ndarray, score: np.ndarray, labels: np.ndarray) -> dict:
    """Return the Spearman correlation for all points, and the mean of the per-class values."""
    from scipy.stats import spearmanr

    per_class = [spearmanr(d[labels == k], score[labels == k]).statistic
                 for k in np.unique(labels) if (labels == k).sum() >= 5]
    per_class = np.array([r for r in per_class if np.isfinite(r)])
    return {
        "all": float(spearmanr(d, score).statistic),
        "within_class": float(per_class.mean()),
        # The standard error shows if the mean of the per-class values is different from 0.
        "within_class_stderr": float(per_class.std(ddof=1) / np.sqrt(len(per_class))),
        "n_classes": int(len(per_class)),
    }
