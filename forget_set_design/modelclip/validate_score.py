#!/usr/bin/env python3
"""
This script compares the difficulty score with the audit signal of each forget point,
for model clipping (spec.md section 5.5).

Use it only as a check. Do not select pool points from audit runs (spec.md section 6).

The input is the output of calibration_sweep.py for a configuration with a random pool, for
example MC1. For each unlearning step, the file scores_step<s>.npz contains the phi score
of each forget point, for each of the 20 configurations and each trial.

For a point x in batch b:
    "+"  the configurations that delete batch b
    "-"  the configurations that keep batch b
    mu   the mean of phi(x) in the group
    v    the variance of phi(x) in the group. It is the mean of the variances between the
         trials of a configuration, plus the variance of the configuration means.

    d_x = |mu+ - mu-| / sqrt((v+ + v-) / 2)

The script reports the Spearman correlation between d_x and each score column:
    all           for all forget points
    within_class  the mean of the correlations of each class. The pools of spec.md
                  section 4.1 are class-balanced, thus this value shows if the score
                  can select points in one class.

Output: validate_score_modelclip_<name>.json in --out-dir. <name> is the name of the
split directory.

Example:
    python3 forget_set_design/modelclip/validate_score.py \\
        --audit-dir /tmp2/$USER/audit-unlearning/modelclip/MC1/audit \\
        --split-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/c1_random_rand_seed0 \\
        --scores-csv /tmp2/$USER/audit-unlearning/scores/scores_modelclip.csv \\
        --out-dir /tmp2/$USER/audit-unlearning/scores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from make_custom_split import DEFAULT_DATA_DIR  # noqa: E402
from score_utils import correlations, load_forget_points, read_scores, separation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the difficulty score with the audit signal of each point, for model clipping.")
    parser.add_argument("--audit-dir", type=Path, required=True,
                        help="The output directory of calibration_sweep.py for one configuration.")
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="The split directory of that configuration. It must contain split_indices.npz.")
    parser.add_argument("--scores-csv", type=Path, required=True, help="The scores file from compute_scores.py.")
    parser.add_argument("--raw-train-dir", type=Path,
                        default=DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train",
                        help="The CIFAR-100 train files in the original row order. "
                             "The script uses them to make sure that the global indices are correct.")
    parser.add_argument("--steps", default="10,20,30",
                        help="The unlearning steps, with commas between them.")
    parser.add_argument("--out-dir", type=Path, required=True, help="The output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    global_idx, labels, batch_id = load_forget_points(args.split_dir, args.raw_train_dir)
    m = int(batch_id.max()) + 1
    scores = read_scores(args.scores_csv, global_idx)
    print(f"[check] {len(global_idx)} forget points, {m} batches, {len(np.unique(labels))} classes")

    report = {"audit_dir": str(args.audit_dir), "split_dir": str(args.split_dir),
              "scores_csv": str(args.scores_csv), "n_points": int(len(global_idx)), "m": m, "steps": {}}
    for step in args.steps.split(","):
        path = args.audit_dir / f"scores_step{step}.npz"
        if not path.exists():
            print(f"[check] step {step}: {path.name} does not exist. The script skips this value.")
            continue
        data = np.load(path)
        z = data["z"].astype(np.float64)
        if z.shape[2] != len(global_idx) or not np.array_equal(data["batch_of_point"], batch_id):
            raise SystemExit(f"ERROR: The points of {path} do not agree with the split {args.split_dir}.")
        d = separation(z.mean(axis=1), z.var(axis=1, ddof=1), batch_id, m)
        entry = {"n_trials": int(z.shape[1]),
                 "d_median": float(np.median(d)), "d_p90": float(np.percentile(d, 90)), "d_max": float(d.max()),
                 "spearman": {name: correlations(d, values, labels) for name, values in scores.items()}}
        report["steps"][step] = entry
        print(f"[check] step {step}: d median {entry['d_median']:.3g}, 90% {entry['d_p90']:.3g}, "
              f"max {entry['d_max']:.3g}")
        for name, rho in entry["spearman"].items():
            print(f"[check]   {name:16s} all {rho['all']:+.3f}   within class {rho['within_class']:+.3f} "
                  f"(standard error {rho['within_class_stderr']:.3f})")

    out_path = args.out_dir / f"validate_score_modelclip_{args.split_dir.name}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[check] The script wrote {out_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
