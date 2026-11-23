#!/usr/bin/env python3
"""
This script calculates the direction statistics of batch assignments before an audit
(spec.md section 4.4, items 1 to 3).

The script reads the assignment files of build_pools.py and the directions u_x of
compute_directions.py. It does not read audit results.

For each assignment, with the batch mean m_b = (sum of u_x for x in batch b) / |b|:
    alignment        ||m_b|| for each batch. The value is 1 if all points of the batch
                     have the same direction. The value is approximately 0.037
                     (1 / sqrt(750)) for random directions.
    batch_cosine     the cosine between m_a and m_b for each pair of batches.
                     offdiag_abs_mean is the mean of the absolute values for a != b.
    retain_cosine    for each batch, the mean cosine between m_b and the --neighbors
                     nearest retain directions. The retain points are a random sample
                     of the training points that are not in the pool. A high value
                     shows that similar retain points can hide the signal of the batch.

build_pools.py writes the score summary and the class counts of each batch (item 4).
compute_scores.py writes the correlations between the scores (item 5).

Output: diagnostics_<alg>.json in --out-dir.

Example:
    python3 forget_set_design/utils/diagnostics.py \\
        --scores-csv /tmp2/$USER/audit-unlearning/scores/scores_r2d.csv \\
        --directions /tmp2/$USER/audit-unlearning/scores/directions_r2d.npy \\
        --out-dir /tmp2/$USER/audit-unlearning/scores \\
        /tmp2/$USER/audit-unlearning/data/cifar100/data_split/assignment_r2d_hard_*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the direction statistics of batch assignments.")
    parser.add_argument("assignments", nargs="+", help="The assignment files from build_pools.py (.csv).")
    parser.add_argument("--scores-csv", required=True, help="The scores file from compute_scores.py.")
    parser.add_argument("--directions", required=True, help="The directions file from compute_directions.py (.npy).")
    parser.add_argument("--alg", default="r2d", help="The name of the algorithm in the name of the output file.")
    parser.add_argument("--retain-sample", type=int, default=10000, help="The number of retain points to use.")
    parser.add_argument("--neighbors", type=int, default=50, help="The number of nearest retain directions.")
    parser.add_argument("--seed", type=int, default=0, help="The seed of the retain sample.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    return parser.parse_args()


def read_rows(directions: Path, positions: np.ndarray) -> np.ndarray:
    """Return the unit directions of the rows `positions`, float64."""
    order = np.argsort(positions)
    u = np.load(directions, mmap_mode="r")[positions[order]].astype(np.float64)
    u = u[np.argsort(order)]
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def main() -> int:
    args = parse_args()
    directions = Path(args.directions)
    with open(args.scores_csv, newline="") as fh:
        global_idx = np.array([int(r["global_idx"]) for r in csv.DictReader(fh)], dtype=np.int64)
    with open(directions.with_suffix(".json")) as fh:
        if not np.array_equal(np.array(json.load(fh)["global_idx"], dtype=np.int64), global_idx):
            raise SystemExit(f"ERROR: The rows of {directions} and the rows of the scores file are not the same points.")
    row_of = {int(g): i for i, g in enumerate(global_idx)}

    report = {"scores_csv": str(args.scores_csv), "directions": str(directions),
              "retain_sample": args.retain_sample, "neighbors": args.neighbors, "seed": args.seed,
              "assignments": {}}
    retain_cache = {}
    for path in map(Path, args.assignments):
        with open(path, newline="") as fh:
            rows = [(row_of[int(r["global_idx"])], int(r["batch_id"])) for r in csv.DictReader(fh)]
        positions = np.array([r[0] for r in rows], dtype=np.int64)
        batch = np.array([r[1] for r in rows], dtype=np.int64)
        m = int(batch.max()) + 1
        u = read_rows(directions, positions)

        means = np.stack([u[batch == b].mean(0) for b in range(m)])
        alignment = np.linalg.norm(means, axis=1)
        unit_means = means / alignment[:, None]
        cosine = unit_means @ unit_means.T
        off_diagonal = np.abs(cosine[~np.eye(m, dtype=bool)])

        # Assignments of the same pool use the same retain sample.
        key = tuple(np.sort(positions).tolist())
        if key not in retain_cache:
            others = np.setdiff1d(np.arange(len(global_idx)), positions)
            sample = np.random.default_rng(args.seed).choice(
                others, size=min(args.retain_sample, len(others)), replace=False)
            retain_cache = {key: read_rows(directions, np.sort(sample))}
        similarity = unit_means @ retain_cache[key].T
        nearest = np.sort(similarity, axis=1)[:, -args.neighbors:].mean(1)

        report["assignments"][path.stem] = {
            "file": str(path), "n_points": int(len(positions)), "n_batches": m,
            "alignment": alignment.tolist(), "alignment_mean": float(alignment.mean()),
            "alignment_random_directions": float(1.0 / np.sqrt(len(positions) / m)),
            "batch_cosine": cosine.tolist(), "batch_cosine_offdiag_abs_mean": float(off_diagonal.mean()),
            "retain_cosine": nearest.tolist(), "retain_cosine_mean": float(nearest.mean()),
        }
        print(f"[diagnostics] {path.stem}: alignment {alignment.min():.3f} to {alignment.max():.3f}, "
              f"mean |cosine| between batches {off_diagonal.mean():.3f}, "
              f"retain cosine {nearest.min():.3f} to {nearest.max():.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diagnostics_{args.alg}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[diagnostics] The script wrote {out_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
