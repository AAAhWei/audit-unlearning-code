#!/usr/bin/env python3
"""
Write a forget-pool split directory from an explicit forget pool and batch assignment.

The output has the same layout as scripts/split_cifar100_train.py (spec.md section 1):

    <out_dir>/
        train/ val/ test/ retain/     batch_XXXXX.pkl = (images uint8 NHWC, labels int64)
        forget/batch_0000k.pkl        exactly the points assigned to batch k
        split_indices.npz             global indices of every row (ours; loaders ignore it)

Global index = row in the original CIFAR-100 train file (0..49999), which is also the row
order of data_split/cifar100_no_forget/train written by download_cifar100_batches.py.

The validation hold-out, train/ and val/ reproduce split_cifar100_train.py for the same
--seed and --val_split, so every custom split shares them with the paper's uniform split.
Only forget/ and retain/ depend on the assignment. Rows inside forget batch k keep the
order they have in the assignment; retain/ rows are shuffled with --seed.

Assignment sources (exactly one):
    --assignment FILE.csv   columns global_idx,batch_id (header required)
    --paper_uniform         the forget pool and batches split_cifar100_train.py
                            --split_mode uniform would produce for --seed
    --random                a uniformly random pool and assignment (--assign_seed);
                            add --balanced for the same number of points per class

Usage (from the repository root):
    python forget_set_design/make_custom_split.py --assignment a.csv --out_dir <split dir>
    python forget_set_design/make_custom_split.py --paper_uniform --out_dir <split dir>
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
    Path(__file__).resolve().parent.parent
    / "certified_unlearning_plus_grad_based_methods" / "data" / "cifar100"
)
INDEX_FILE = "split_indices.npz"


def load_batch_dir(batch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(batch_dir.glob("batch_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pkl in {batch_dir}")
    images, labels = [], []
    for p in files:
        with p.open("rb") as f:
            batch = pickle.load(f)
        images.append(batch[0])
        labels.append(batch[1])
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def save_batches(images: np.ndarray, labels: np.ndarray, out_dir: Path, batch_size: int) -> int:
    """Same file format and naming as split_cifar100_train.save_batches."""
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
    Replay the random draws of split_cifar100_train.py (uniform mode).

    Returns global indices:
        train_idx    the post-validation training points, in the script's pre-shuffle order
        val_idx      the validation points, pre-shuffle
        train_order  train_idx in the row order the script writes to train/
        val_order    val_idx in the row order the script writes to val/
        uniform_perm permutation of range(n_train) the script uses to pick the forget set
        rng          generator positioned right after the val shuffle (the script next
                     draws the retain shuffle, then the forget shuffle)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_full)
    n_val = int(round(val_split * n_full))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    n_train = n_full - n_val

    rng = np.random.default_rng(seed)  # the script reseeds before the forget selection
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
    """(global_idx, batch_id) of the paper's uniform split, rows in its file order."""
    h = paper_holdout(n_full, val_split, seed)
    n_train = h["train_idx"].shape[0]
    target = int(round(forget_fraction * n_train))
    forget = h["train_idx"][h["uniform_perm"][:target]]
    rng = h["rng"]
    rng.permutation(n_train - target)          # retain shuffle, drawn before forget's
    forget = forget[rng.permutation(target)]
    return forget, np.arange(target) // batch_size


def random_assignment(
    train_idx: np.ndarray, n_batches: int, batch_size: int, assign_seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random forget pool, randomly dealt into batches. With `labels` (indexed by global
    index) the pool is class-balanced: the same number of points from every class.
    """
    rng = np.random.default_rng(assign_seed)
    n = n_batches * batch_size
    candidates = np.sort(train_idx)
    if labels is None:
        pool = rng.choice(candidates, size=n, replace=False)
    else:
        classes = np.unique(labels)
        if n % classes.shape[0]:
            raise ValueError(f"{n} points cannot be split evenly over {classes.shape[0]} classes")
        per_class = n // classes.shape[0]
        cand_labels = labels[candidates]
        pool = np.concatenate([
            rng.choice(candidates[cand_labels == c], size=per_class, replace=False)
            for c in classes
        ])
        pool = pool[rng.permutation(n)]
    return pool, np.arange(n) // batch_size


def read_assignment_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"global_idx", "batch_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks column(s) {sorted(missing)}")
        rows = [(int(r["global_idx"]), int(r["batch_id"])) for r in reader]
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def validate_assignment(
    global_idx: np.ndarray, batch_id: np.ndarray, train_idx: np.ndarray,
    n_batches: int, batch_size: int,
) -> None:
    n = n_batches * batch_size
    if global_idx.shape[0] != n:
        raise ValueError(f"Assignment has {global_idx.shape[0]} rows, expected {n}")
    if np.unique(global_idx).shape[0] != n:
        raise ValueError("Assignment contains duplicate global indices")
    outside = ~np.isin(global_idx, train_idx)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} forget indices are not in the post-validation training "
            f"set (validation or out of range), e.g. {global_idx[outside][:5].tolist()}"
        )
    counts = np.bincount(batch_id, minlength=n_batches) if batch_id.min() >= 0 else None
    if counts is None or counts.shape[0] != n_batches or (counts != batch_size).any():
        raise ValueError(
            f"Every batch id 0..{n_batches - 1} must hold exactly {batch_size} points; "
            f"got counts {None if counts is None else counts.tolist()}"
        )


def write_split(
    out_dir: Path, input_root: Path, global_idx: np.ndarray, batch_id: np.ndarray,
    *, n_batches: int, batch_size: int, val_split: float, seed: int,
) -> dict[str, np.ndarray]:
    images, labels = load_batch_dir(input_root / "train")
    h = paper_holdout(images.shape[0], val_split, seed)
    validate_assignment(global_idx, batch_id, h["train_idx"], n_batches, batch_size)

    # Stable sort: rows of one batch keep their order from the assignment.
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
        print(f"  test: copied from {test_src}")

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
    src.add_argument("--assignment", type=Path, help="CSV with columns global_idx,batch_id")
    src.add_argument("--paper_uniform", action="store_true",
                     help="Reproduce split_cifar100_train.py --split_mode uniform")
    src.add_argument("--random", action="store_true",
                     help="Uniformly random pool and assignment (see --assign_seed)")
    p.add_argument("--out_dir", type=Path, required=True, help="Split directory to create")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Root holding data_split/<input_subfolder>/{train,test}")
    p.add_argument("--input_subfolder", type=str, default="cifar100_no_forget")
    p.add_argument("--batch_size", type=int, default=750, help="Points per forget batch")
    p.add_argument("--n_batches", type=int, default=6, help="Number of forget batches")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1,
                   help="Seed of the paper split: fixes val hold-out and train/val order")
    p.add_argument("--assign_seed", type=int, default=0, help="Seed for --random")
    p.add_argument("--balanced", action="store_true",
                   help="With --random: class-balanced pool (same count per class)")
    p.add_argument("--save_assignment", type=Path, default=None,
                   help="Also write the assignment used as a global_idx,batch_id CSV")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --out_dir")
    args = p.parse_args()

    input_root = args.data_dir.resolve() / "data_split" / args.input_subfolder
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            print(f"{out_dir} exists and is not empty; pass --overwrite to replace it",
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
        print(f"Assignment written to {args.save_assignment}")
    print(f"Split written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
