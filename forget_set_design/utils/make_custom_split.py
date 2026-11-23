#!/usr/bin/env python3
"""
This script writes a split directory from a forget pool and a batch assignment.

The split directory has the same layout as the output of
scripts/split_cifar100_train.py (spec.md section 1):

    <out_dir>/
        train/ val/ test/ retain/     Each batch_XXXXX.pkl file is a tuple
                                      (images uint8 NHWC, labels int64).
        forget/batch_0000k.pkl        This file contains the points of batch k.
        split_indices.npz             This file contains the global index of each row.
                                      The pipelines do not read this file.

The global index of a point is its row in the original CIFAR-100 train file (0 to 49999).
The files in data_split/cifar100_no_forget/train have the same row order.

The script makes the same random selections as split_cifar100_train.py for --seed and
--val_split. Thus the validation set, train/ and val/ are the same in all split
directories. Only forget/ and retain/ change with the assignment.
The rows of forget batch k have the same order as in the assignment.
The script shuffles the rows of retain/ with --seed.

Select one source for the assignment:
    --assignment FILE.csv   The file has a header and the columns global_idx, batch_id.
    --paper_uniform         The script calculates the forget pool and the batches that
                            split_cifar100_train.py --split_mode uniform gives for --seed.
    --random                The script selects a random pool and a random assignment
                            with --assign_seed.
                            Add --balanced to get the same number of points for each class.

Usage (from the repository root):
    python forget_set_design/utils/make_custom_split.py --assignment a.csv --out_dir <split dir>
    python forget_set_design/utils/make_custom_split.py --paper_uniform --out_dir <split dir>
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "certified_unlearning_plus_grad_based_methods" / "data" / "cifar100"
)
INDEX_FILE = "split_indices.npz"


def load_batch_dir(batch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read all batch files in a directory. Return (images, labels) in file order."""
    files = sorted(batch_dir.glob("batch_*.pkl"))
    if not files:
        raise FileNotFoundError(f"The directory {batch_dir} contains no batch_*.pkl file.")
    images, labels = [], []
    for p in files:
        with p.open("rb") as f:
            batch = pickle.load(f)
        images.append(batch[0])
        labels.append(batch[1])
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def save_batches(images: np.ndarray, labels: np.ndarray, out_dir: Path, batch_size: int) -> int:
    """
    Write the points to batch files. Return the number of files.
    The file format and the file names are the same as in split_cifar100_train.py.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for start in range(0, images.shape[0], batch_size):
        path = out_dir / f"batch_{n_files:05d}.pkl"
        with path.open("wb") as f:
            pickle.dump(
                (images[start:start + batch_size], labels[start:start + batch_size]),
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
        n_files += 1
    return n_files


def paper_holdout(n_full: int, val_split: float, seed: int) -> dict[str, np.ndarray]:
    """
    Make the same random selections as split_cifar100_train.py in uniform mode.

    The function returns a dictionary. All index arrays contain global indices.
        train_idx    The training points after the script removes the validation points.
                     The order is the order before the script shuffles the rows.
        val_idx      The validation points before the script shuffles the rows.
        train_order  The training points in the row order that the script writes to train/.
        val_order    The validation points in the row order that the script writes to val/.
        uniform_perm The permutation of range(n_train) that the script uses to select
                     the forget pool.
        rng          The random generator after the shuffle of the validation rows.
                     The script uses it next to shuffle retain, then to shuffle forget.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_full)
    n_val = int(round(val_split * n_full))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    n_train = n_full - n_val

    # The script creates a new generator from the same seed before it selects the forget pool.
    rng = np.random.default_rng(seed)
    uniform_perm = rng.permutation(np.arange(n_train))
    train_order = train_idx[rng.permutation(n_train)]
    val_order = val_idx[rng.permutation(n_val)]
    return {
        "train_idx": train_idx, "val_idx": val_idx,
        "train_order": train_order, "val_order": val_order,
        "uniform_perm": uniform_perm, "rng": rng,
    }


def paper_uniform_assignment(
    n_full: int, val_split: float, forget_fraction: float, seed: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (global_idx, batch_id) of the uniform split of the paper.
    The rows have the same order as in the forget files of that split.
    """
    h = paper_holdout(n_full, val_split, seed)
    n_train = h["train_idx"].shape[0]
    target = int(round(forget_fraction * n_train))
    forget = h["train_idx"][h["uniform_perm"][:target]]
    rng = h["rng"]
    # The script shuffles retain before forget. Use the generator one time for retain.
    rng.permutation(n_train - target)
    forget = forget[rng.permutation(target)]
    return forget, np.arange(target) // batch_size


def random_assignment(
    train_idx: np.ndarray, n_batches: int, batch_size: int, assign_seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a random forget pool and put the points into batches at random.

    If the caller gives `labels`, the pool has the same number of points for each class.
    `labels` contains the label of each global index.
    """
    rng = np.random.default_rng(assign_seed)
    n = n_batches * batch_size
    candidates = np.sort(train_idx)
    if labels is None:
        pool = rng.choice(candidates, size=n, replace=False)
    else:
        classes = np.unique(labels)
        if n % classes.shape[0]:
            raise ValueError(
                f"The pool size {n} is not a multiple of the number of classes {classes.shape[0]}.")
        per_class = n // classes.shape[0]
        cand_labels = labels[candidates]
        pool = np.concatenate([
            rng.choice(candidates[cand_labels == c], size=per_class, replace=False)
            for c in classes
        ])
        pool = pool[rng.permutation(n)]
    return pool, np.arange(n) // batch_size


def read_assignment_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an assignment file. Return (global_idx, batch_id)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"global_idx", "batch_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"The file {path} does not contain the columns {sorted(missing)}.")
        rows = [(int(r["global_idx"]), int(r["batch_id"])) for r in reader]
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def validate_assignment(
    global_idx: np.ndarray, batch_id: np.ndarray, train_idx: np.ndarray,
    n_batches: int, batch_size: int,
) -> None:
    """Stop with an error if the assignment is not a correct set of forget batches."""
    n = n_batches * batch_size
    if global_idx.shape[0] != n:
        raise ValueError(f"The assignment has {global_idx.shape[0]} rows. It must have {n} rows.")
    if np.unique(global_idx).shape[0] != n:
        raise ValueError("The assignment contains the same global index more than one time.")
    outside = ~np.isin(global_idx, train_idx)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} global indices are not in the training set. "
            f"They are validation points or they are out of range. "
            f"The first such indices are {global_idx[outside][:5].tolist()}."
        )
    counts = np.bincount(batch_id, minlength=n_batches) if batch_id.min() >= 0 else None
    if counts is None or counts.shape[0] != n_batches or (counts != batch_size).any():
        raise ValueError(
            f"Each batch id from 0 to {n_batches - 1} must have {batch_size} points. "
            f"The counts are {None if counts is None else counts.tolist()}."
        )


def write_split(
    out_dir: Path, input_root: Path, global_idx: np.ndarray, batch_id: np.ndarray,
    *, n_batches: int, batch_size: int, val_split: float, seed: int,
) -> dict[str, np.ndarray]:
    """Write the split directory. Return the global indices of each subdirectory."""
    images, labels = load_batch_dir(input_root / "train")
    h = paper_holdout(images.shape[0], val_split, seed)
    validate_assignment(global_idx, batch_id, h["train_idx"], n_batches, batch_size)

    # A stable sort keeps the assignment order of the rows in each batch.
    forget_order = global_idx[np.argsort(batch_id, kind="stable")]
    retain = h["train_idx"][~np.isin(h["train_idx"], global_idx)]
    retain_order = retain[np.random.default_rng(seed).permutation(retain.shape[0])]

    order = {
        "train": h["train_order"], "val": h["val_order"],
        "retain": retain_order, "forget": forget_order,
    }
    for split, idx in order.items():
        n_files = save_batches(images[idx], labels[idx], out_dir / split, batch_size)
        print(f"{split:>6}: {idx.shape[0]:>6} points in {n_files} files")

    test_src = input_root / "test"
    if test_src.is_dir():
        shutil.copytree(test_src, out_dir / "test")
        print(f"  test: copy of {test_src}")

    np.savez(
        out_dir / INDEX_FILE,
        train=order["train"], val=order["val"], retain=order["retain"],
        forget=order["forget"], forget_batch_id=np.sort(batch_id, kind="stable"),
        seed=seed, val_split=val_split, batch_size=batch_size,
    )
    return order


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--assignment", type=Path, help="CSV file with the columns global_idx, batch_id.")
    src.add_argument("--paper_uniform", action="store_true",
                     help="Use the forget pool and batches of split_cifar100_train.py --split_mode uniform.")
    src.add_argument("--random", action="store_true",
                     help="Use a random pool and a random assignment. --assign_seed sets the seed.")
    p.add_argument("--out_dir", type=Path, required=True, help="Split directory that the script creates.")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory that contains data_split/<input_subfolder>/train and test.")
    p.add_argument("--input_subfolder", type=str, default="cifar100_no_forget",
                   help="Subdirectory of data_split/ with the input batch files.")
    p.add_argument("--batch_size", type=int, default=750, help="Number of points in each forget batch.")
    p.add_argument("--n_batches", type=int, default=6, help="Number of forget batches.")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of the 50,000 train points that become validation points.")
    p.add_argument("--seed", type=int, default=1,
                   help="Seed of the paper split. It sets the validation set and the row order "
                        "of train/ and val/.")
    p.add_argument("--assign_seed", type=int, default=0, help="Seed for --random.")
    p.add_argument("--balanced", action="store_true",
                   help="With --random, select the same number of points for each class.")
    p.add_argument("--save_assignment", type=Path, default=None,
                   help="Write the assignment to this CSV file also.")
    p.add_argument("--overwrite", action="store_true", help="Delete --out_dir first if it exists.")
    args = p.parse_args()

    input_root = args.data_dir.resolve() / "data_split" / args.input_subfolder
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            print(f"The directory {out_dir} is not empty. Add --overwrite to replace it.",
                  file=sys.stderr)
            return 1
        shutil.rmtree(out_dir)

    all_labels = load_batch_dir(input_root / "train")[1]
    n_full = all_labels.shape[0]

    if args.assignment is not None:
        global_idx, batch_id = read_assignment_csv(args.assignment)
    elif args.paper_uniform:
        n_train = n_full - int(round(args.val_split * n_full))
        fraction = args.n_batches * args.batch_size / n_train
        global_idx, batch_id = paper_uniform_assignment(
            n_full, args.val_split, fraction, args.seed, args.batch_size)
    else:
        train_idx = paper_holdout(n_full, args.val_split, args.seed)["train_idx"]
        global_idx, batch_id = random_assignment(
            train_idx, args.n_batches, args.batch_size, args.assign_seed,
            labels=all_labels if args.balanced else None)

    write_split(out_dir, input_root, global_idx, batch_id, n_batches=args.n_batches,
                batch_size=args.batch_size, val_split=args.val_split, seed=args.seed)

    if args.save_assignment is not None:
        args.save_assignment.parent.mkdir(parents=True, exist_ok=True)
        with args.save_assignment.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["global_idx", "batch_id"])
            w.writerows(zip(global_idx.tolist(), batch_id.tolist()))
        print(f"Assignment file: {args.save_assignment}")
    print(f"Split directory: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
