#!/usr/bin/env python3
"""
This script compares the difficulty score with the audit signal of each forget point
(spec.md section 5.5).

Use it only as a check. Do not select pool points from audit runs (spec.md section 6).

The input is the output of calibration_sweep.py for a C1 configuration. For each claimed
epsilon, the file scores_eps<value>.npz contains the mean and the variance of the phi
score of each forget point, for each of the 20 configurations.

For a point x in batch b:
    "+"  the configurations that delete batch b
    "-"  the configurations that keep batch b
    mu   the mean of phi(x) in the group
    v    the variance of phi(x) in the group. It is the mean of the noise variances
         plus the variance of the configuration means.

    d_x = |mu+ - mu-| / sqrt((v+ + v-) / 2)

The script reports the Spearman correlation between d_x and each score column:
    all           for all forget points
    within_class  the mean of the correlations of each class. The pools of spec.md
                  section 4.1 are class-balanced, thus this value shows if the score
                  can select points in one class.

Output: validate_score_<name>.json in --out-dir. <name> is the name of the split directory.

Example:
    python3 forget_set_design/r2d/validate_score.py \\
        --audit-dir /tmp2/$USER/audit-unlearning/r2d/C1_seed0/sweep \\
        --split-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/c1_random_rand_seed0 \\
        --scores-csv /tmp2/$USER/audit-unlearning/scores/scores_r2d.csv \\
        --out-dir /tmp2/$USER/audit-unlearning/scores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pooled_linear_auditor"))

from make_custom_split import DEFAULT_DATA_DIR  # noqa: E402
from score_utils import correlations, load_forget_points, read_scores, separation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the difficulty score with the audit signal of each point.")
    parser.add_argument("--audit-dir", required=True,
                        help="The output directory of calibration_sweep.py for one configuration: the directory "
                             "sweep/, or the directory joint/ of results from before the merge of the audit scripts.")
    parser.add_argument("--split-dir", required=True,
                        help="The split directory of that configuration. It must contain split_indices.npz.")
    parser.add_argument("--scores-csv", required=True, help="The scores file from compute_scores.py.")
    parser.add_argument("--raw-train-dir", default=str(DEFAULT_DATA_DIR / "data_split" / "cifar100_no_forget" / "train"),
                        help="The CIFAR-100 train files in the original row order. "
                             "The script uses them to make sure that the global indices are correct.")
    parser.add_argument("--epsilons", default="inf,2000,1000,100",
                        help="The claimed epsilon values, with commas between them.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_dir, split_dir = Path(args.audit_dir), Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_idx, labels, batch_id = load_forget_points(split_dir, Path(args.raw_train_dir))
    m = int(batch_id.max()) + 1
    scores = read_scores(Path(args.scores_csv), global_idx)
    print(f"[check] {len(global_idx)} forget points, {m} batches, {len(np.unique(labels))} classes")

    report = {"audit_dir": str(audit_dir), "split_dir": str(split_dir), "scores_csv": str(args.scores_csv),
              "n_points": int(len(global_idx)), "m": m, "epsilons": {}}
    for eps in args.epsilons.split(","):
        path = audit_dir / f"scores_eps{eps}.npz"
        if not path.exists():
            print(f"[check] epsilon {eps}: {path.name} does not exist. The script skips this value.")
            continue
        data = np.load(path)
        if data["cal_mean"].shape[1] != len(global_idx):
            raise SystemExit(f"ERROR: {path} has {data['cal_mean'].shape[1]} points, the split has {len(global_idx)}.")
        d = separation(data["cal_mean"].astype(np.float64), data["cal_var"].astype(np.float64), batch_id, m)
        entry = {"d_median": float(np.median(d)), "d_p90": float(np.percentile(d, 90)), "d_max": float(d.max()),
                 "spearman": {name: correlations(d, values, labels) for name, values in scores.items()}}
        report["epsilons"][eps] = entry
        print(f"[check] epsilon {eps}: d median {entry['d_median']:.3g}, 90% {entry['d_p90']:.3g}, "
              f"max {entry['d_max']:.3g}")
        for name, rho in entry["spearman"].items():
            print(f"[check]   {name:16s} all {rho['all']:+.3f}   within class {rho['within_class']:+.3f} "
                  f"(standard error {rho['within_class_stderr']:.3f})")

    out_path = out_dir / f"validate_score_{split_dir.name}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[check] The script wrote {out_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
