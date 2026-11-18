#!/usr/bin/env python3
"""
This script examines a split directory from make_custom_split.py.
It compares the split directory with a reference split directory from
scripts/split_cifar100_train.py (spec.md section 1, "First task").

The script does these checks:
  1. The two directories have the same subdirectories and file names. Each file has
     the same tuple type, dtypes and shapes.
  2. train/, val/ and test/ contain the same data as the reference, row for row.
  3. forget and retain have no common point. Together they are the training set.
     They contain no validation point.
  4. The rows agree with split_indices.npz. The global index of a point is its row in
     the original CIFAR-100 file.
  5. With --expect_identical_forget: each file in forget/ is the same as in the
     reference, and retain/ contains the same points.
  6. The readers of the two pipelines can load the split directory. These are the JAX
     data_cache module and the R2D function build_train_pickle.

Do this check on the CPU. If a GPU is visible, the JAX loader allocates GPU memory.
    CUDA_VISIBLE_DEVICES="" python forget_set_design/check_custom_split.py \
        --split <split directory> --reference <reference split directory>
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test", "retain", "forget")
FAILED: list[str] = []


def check(ok: bool, msg: str) -> None:
    """Print the result of one check. Record the message if the check fails."""
    print(f"[{'ok' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILED.append(msg)


def load_files(d: Path) -> list[tuple[str, object]]:
    """Read all batch files in a directory. Return a list of (file name, content)."""
    out = []
    for p in sorted(d.glob("batch_*.pkl")):
        with p.open("rb") as f:
            out.append((p.name, pickle.load(f)))
    return out


def cat(files) -> tuple[np.ndarray, np.ndarray]:
    """Join the images and the labels of a list of batch files."""
    return (np.concatenate([b[0] for _, b in files]), np.concatenate([b[1] for _, b in files]))


def row_keys(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Return one byte key for each row (image and label), in sorted order.
    Two sets of rows are the same if their sorted keys are the same.
    """
    flat = np.concatenate(
        [images.reshape(images.shape[0], -1), labels.astype(np.uint8)[:, None]], axis=1)
    keys = np.ascontiguousarray(flat).view([("k", f"V{flat.shape[1]}")]).ravel()
    return np.sort(keys)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", type=Path, required=True,
                   help="Split directory from make_custom_split.py.")
    p.add_argument("--reference", type=Path, required=True,
                   help="Split directory from split_cifar100_train.py.")
    p.add_argument("--expect_identical_forget", action="store_true",
                   help="Use this option if --paper_uniform made the split directory "
                        "with the seed of the reference.")
    p.add_argument("--raw_train", type=Path, default=None,
                   help="The original CIFAR-100 train file. "
                        "The default is the file in data/cifar100/extracted.")
    args = p.parse_args()
    split, ref = args.split.resolve(), args.reference.resolve()

    new = {s: load_files(split / s) for s in SPLITS}
    old = {s: load_files(ref / s) for s in SPLITS}

    # Check 1: layout.
    for s in SPLITS:
        same_names = [n for n, _ in new[s]] == [n for n, _ in old[s]]
        sig = lambda fs: [(type(b).__name__, len(b), type(b[0]).__name__, b[0].dtype.str,
                           b[0].shape, b[1].dtype.str, b[1].shape) for _, b in fs]
        check(same_names and sig(new[s]) == sig(old[s]),
              f"{s}: {len(new[s])} files. Names, types, dtypes and shapes agree with the reference.")

    # Check 2: train/, val/ and test/ are the same as the reference.
    for s in ("train", "val", "test"):
        a, b = cat(new[s]), cat(old[s])
        check(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]),
              f"{s}: each row is the same as in the reference.")

    # Checks 3 and 4: the global indices.
    idx = np.load(split / "split_indices.npz")
    tr, va, re, fo = (idx[k] for k in ("train", "val", "retain", "forget"))
    check(np.intersect1d(fo, re).size == 0, "forget and retain have no common point.")
    check(np.array_equal(np.sort(np.concatenate([fo, re])), np.sort(tr)),
          "forget and retain together are the training set.")
    check(np.intersect1d(np.concatenate([fo, re]), va).size == 0,
          "forget and retain contain no validation point.")
    bid = idx["forget_batch_id"]
    bs = int(idx["batch_size"])
    check(np.array_equal(bid, np.arange(fo.shape[0]) // bs),
          f"forget file k contains the {bs} points of batch k.")

    raw = args.raw_train or (ROOT / "certified_unlearning_plus_grad_based_methods" / "data"
                             / "cifar100" / "extracted" / "cifar-100-python" / "train")
    with Path(raw).open("rb") as f:
        d = pickle.load(f, encoding="bytes")
    raw_x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    raw_y = np.asarray(d[b"fine_labels"], dtype=np.int64)
    for s, gi in (("train", tr), ("val", va), ("retain", re), ("forget", fo)):
        x, y = cat(new[s])
        check(np.array_equal(x, raw_x[gi]) and np.array_equal(y, raw_y[gi]),
              f"{s}: each row is the original CIFAR-100 image at its global index.")
    coarse = np.asarray(d[b"coarse_labels"])
    check(all(np.unique(coarse[raw_y == c]).size == 1 for c in range(100)),
          "original file: each fine class belongs to one superclass.")

    # Check 5: forget/ and retain/ are the same as in the paper split.
    if args.expect_identical_forget:
        same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                   for (_, a), (_, b) in zip(new["forget"], old["forget"]))
        check(same, "forget: each batch file is the same as in the reference.")
        check(np.array_equal(row_keys(*cat(new["retain"])), row_keys(*cat(old["retain"]))),
              "retain: the points are the same as in the reference. The order is different.")

    # Check 6: the readers of the two pipelines.
    sys.path.insert(0, str(ROOT / "certified_unlearning_plus_grad_based_methods"))
    try:
        from src.utils.data_cache import load_cifar_splits_with_batch_subset
        tl, vl, te, fl, rl, chosen = load_cifar_splits_with_batch_subset(
            str(split), forget_fraction=0.5, rng_seed=0, batch_size=128,
            chosen_idx_override=np.array([0, 2, 4]))
        n_f = sum(int(b[0].shape[0]) for b in fl)
        check(len(tl) == (re.shape[0] + 3 * bs) // 128 and n_f == 3 * bs
              and list(chosen) == [0, 2, 4],
              f"model-clipping loader: {len(tl)} train batches of 128 and {n_f} forget points.")
    except Exception as e:  # noqa: BLE001
        check(False, f"model-clipping loader stopped with {type(e).__name__}: {e}")

    sys.path.insert(0, str(ROOT / "r2d_audit"))
    try:
        from random_forget_runner_all_combinations import build_train_pickle
        with tempfile.TemporaryDirectory() as tmp:
            out = build_train_pickle(str(split / "retain"), str(split / "forget"), [1, 3, 5], tmp)
            with open(out, "rb") as f:
                x, y, ids = pickle.load(f)
        exp_x, exp_y = cat(new["retain"] + [new["forget"][k] for k in (1, 3, 5)])
        check(np.array_equal(x, exp_x) and np.array_equal(y, exp_y)
              and x.shape[0] == re.shape[0] + 3 * bs,
              f"R2D build_train_pickle: {x.shape[0]} training points for the combination (1,3,5).")
    except Exception as e:  # noqa: BLE001
        check(False, f"R2D build_train_pickle stopped with {type(e).__name__}: {e}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else f'FAILED CHECKS: {len(FAILED)}'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
