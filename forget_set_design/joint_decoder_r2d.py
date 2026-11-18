#!/usr/bin/env python3
"""
This script audits R2D runs with the joint sign-vector decoder (decoder B2 of spec.md).

The script is an adapter. It does not contain a decoder or an epsilon test.
    - The decoder is pooled_linear_auditor/pooled_linear.py. The method "b2" is the
      joint Gaussian decoder of the paper (Algorithm 3).
    - The epsilon test is r2d_audit/cum_runs_eps_lab.py.
    - The model, the noise scale and the phi score come from r2d_audit/.

Definitions:
    configuration   One set of deleted forget batches. For 6 batches there are 20
                    configurations. Each run of the pipeline trains and unlearns one
                    configuration.
    score           The phi value of one forget point for one noisy model.
    noise draw      One sample of Gaussian noise that the script adds to the weights
                    of an unlearned model. The noise scale comes from the claimed
                    epsilon.

The script does these steps for each claimed epsilon:
    1. Calibration. For each configuration, the script makes --num-calib noise draws
       and records the scores of all forget points.
    2. Evaluation. The script selects --num-eval configurations at random. Each
       configuration has the same probability. For each selection, the script makes
       one new noise draw and records the scores.
    3. The decoder fits the calibration scores and predicts the configuration of
       each evaluation trial.
    4. The epsilon test converts the number of correct batch predictions to a lower
       bound.

The calibration and the evaluation use different random generators (--calib-seed,
--eval-seed). Each evaluation trial has its own noise draw. Thus the trials are
independent, as the epsilon test requires.

If the claimed epsilon is inf, the script adds no noise. The scores of a
configuration are then always the same, and the decoder is always correct. The
result is the maximum lower bound for --num-eval trials.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=<gpu> python forget_set_design/joint_decoder_r2d.py \\
        --results-dir <directory that contains run_001, run_002, ...> \\
        --data-dir <split directory> --output-dir <output directory> \\
        --epsilon inf,100,1000,2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
R2D_DIR = REPO / "r2d_audit"
sys.path.insert(0, str(R2D_DIR))
sys.path.insert(0, str(R2D_DIR / "evaluation_scripts"))
sys.path.insert(0, str(REPO / "pooled_linear_auditor"))

import models  # noqa: E402
import pooled_linear as pl  # noqa: E402
from evaluate_combination_models import (  # noqa: E402
    compute_phi_and_loss_batch,
    find_checkpoints,
    find_run_directories,
    infer_model_from_checkpoint,
    load_all_forget_points,
    load_forget_batch_indices,
)
from forget_phi_noisy_loader import compute_sigma_from_log_new_structure  # noqa: E402

SUMMARY_COLUMNS = [
    "algorithm", "config_id", "decoder", "claimed_eps", "eps_lb_avg", "eps_lb_median",
    "rho_lb", "exact_match_rate", "mean_v", "eps_ceiling", "n_calib", "n_eval",
    "calib_seed", "eval_seed", "sigma_mean",
]


def map_runs_to_configurations(results_dir: Path, combos: list) -> dict:
    """
    Find the run of each configuration.

    Return a dictionary: configuration index -> run directory. The configuration
    index is the position in `combos`, which comes from pooled_linear.sign_configs.
    """
    index_of = {c: i for i, c in enumerate(combos)}
    run_of = {}
    for run_dir, _ in find_run_directories(results_dir):
        batches = load_forget_batch_indices(run_dir)
        if batches is None:
            raise ValueError(f"The run {run_dir} has no forget_ids.json file with batch indices.")
        key = tuple(sorted(batches))
        if key not in index_of:
            raise ValueError(f"The run {run_dir} deletes the batches {key}. This set is not a configuration.")
        if index_of[key] in run_of:
            raise ValueError(f"Two runs delete the batches {key}: {run_of[index_of[key]]} and {run_dir}.")
        run_of[index_of[key]] = run_dir
    missing = [combos[i] for i in range(len(combos)) if i not in run_of]
    if missing:
        raise ValueError(
            f"{len(missing)} configurations have no run in {results_dir}. "
            f"The first such configurations are {missing[:3]}."
        )
    return run_of


class NoisyScorer:
    """
    This class calculates the scores of all forget points for one unlearned model.

    It keeps the weights of the unlearned model. For each call of `scores`, it adds
    one new noise draw to these weights and calculates phi for each forget point.
    """

    def __init__(self, run_dir: Path, images: torch.Tensor, labels: torch.Tensor,
                 device: torch.device, eval_batch_size: int):
        self.ckpt_path, self.log_path = find_checkpoints(run_dir, "unlearnt")
        arch, num_classes, filters = infer_model_from_checkpoint(self.ckpt_path)
        self.model = models.get_model(
            arch, num_classes=num_classes, filters_percentage=filters).to(device)
        state = torch.load(self.ckpt_path, map_location=device)
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state)
        self.model.eval()
        # The base weights are the weights of the unlearned model before noise.
        self.base = [p.detach().clone() for p in self.model.parameters()]
        self.images, self.labels = images, labels
        self.eval_batch_size = eval_batch_size

    def sigma(self, epsilon: float, delta: float, data_dir: Path) -> float:
        """Return the noise scale for the claimed epsilon. The pipeline function calculates it."""
        value = compute_sigma_from_log_new_structure(
            self.log_path, self.ckpt_path.name, epsilon, delta, dataroot=data_dir, verbose=False)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"The noise scale {value} is not correct for {self.ckpt_path.name}.")
        return float(value)

    @torch.no_grad()
    def scores(self, sigma: float | None, generator: torch.Generator | None) -> np.ndarray:
        """
        Return phi for each forget point as a float64 array of shape [n].

        If `sigma` is None, the function adds no noise. If `sigma` is a number, the
        function adds one noise draw from `generator` to each trainable weight.
        r2d.add_gaussian_noise_to_weights adds the same noise to the same weights.
        """
        for p, base in zip(self.model.parameters(), self.base):
            p.copy_(base)
            if sigma is not None and p.requires_grad:
                noise = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
                p.add_(noise, alpha=sigma)
        out = []
        for start in range(0, self.images.shape[0], self.eval_batch_size):
            stop = start + self.eval_batch_size
            phi, _ = compute_phi_and_loss_batch(
                self.model, self.images[start:stop], self.labels[start:stop])
            out.append(phi)
        return np.concatenate(out).astype(np.float64)


def collect_scores(run_of: dict, images, labels, device, args, epsilon: float):
    """
    Collect the calibration scores and the evaluation scores for one claimed epsilon.

    Return (cal, evz, trues, sigmas):
        cal     configuration index -> calibration scores [--num-calib, n]
        evz     evaluation scores [--num-eval, n]
        trues   true configuration index of each evaluation trial [--num-eval]
        sigmas  configuration index -> noise scale. The value is None without noise.
    """
    add_noise = not math.isinf(epsilon)
    n_configs = len(run_of)

    # The two generators have different seeds. Thus the calibration and the
    # evaluation do not use the same noise draws.
    calib_gen = torch.Generator(device=device).manual_seed(args.calib_seed)
    eval_gen = torch.Generator(device=device).manual_seed(args.eval_seed)
    trues = np.random.default_rng(args.eval_seed).integers(0, n_configs, size=args.num_eval)

    # Without noise, all draws of a configuration are the same. Two rows are
    # sufficient, because the decoder needs a minimum of two rows.
    n_calib = args.num_calib if add_noise else 2

    cal, sigmas = {}, {}
    evz = np.empty((args.num_eval, images.shape[0]), dtype=np.float64)
    for c in range(n_configs):
        scorer = NoisyScorer(run_of[c], images, labels, device, args.eval_batch_size)
        sigma = scorer.sigma(epsilon, args.delta, args.data_dir) if add_noise else None
        sigmas[c] = sigma
        cal[c] = np.stack([scorer.scores(sigma, calib_gen) for _ in range(n_calib)])
        for trial in np.flatnonzero(trues == c):
            evz[trial] = scorer.scores(sigma, eval_gen)
        print(f"  configuration {c + 1}/{n_configs}: {run_of[c].name}, "
              f"sigma={'none' if sigma is None else f'{sigma:.4e}'}, "
              f"{int((trues == c).sum())} evaluation trials", flush=True)
        del scorer
    return cal, evz, trues, sigmas


def variance_report(cal: dict) -> dict:
    """
    Summarize the calibration variances of each forget point (spec.md section 7).

    For each point, the ratio is the largest variance of a configuration divided
    by the smallest variance of a configuration.
    """
    var = np.stack([cal[c].var(axis=0) for c in sorted(cal)])
    floor = pl.VAR_FLOOR
    ratio = np.maximum(var.max(axis=0), floor) / np.maximum(var.min(axis=0), floor)
    return {
        "variance_median": float(np.median(var)),
        "variance_min": float(var.min()),
        "fraction_at_floor": float((var <= floor).mean()),
        "max_min_ratio_median": float(np.median(ratio)),
        "max_min_ratio_p95": float(np.quantile(ratio, 0.95)),
        "max_min_ratio_max": float(ratio.max()),
    }


def audit_one_epsilon(eps_raw: str, run_of, images, labels, device, args, m: int, evaluator):
    """Audit one claimed epsilon. Return one result dictionary for each decoder method."""
    epsilon = float(eps_raw)
    print(f"\nClaimed epsilon {eps_raw}", flush=True)
    cal, evz, trues, sigmas = collect_scores(run_of, images, labels, device, args, epsilon)
    n_eval = int(args.num_eval)
    sigma_values = [s for s in sigmas.values() if s is not None]

    np.savez_compressed(
        args.output_dir / f"scores_eps{eps_raw}.npz",
        cal_mean=np.stack([cal[c].mean(axis=0) for c in sorted(cal)]).astype(np.float32),
        cal_var=np.stack([cal[c].var(axis=0) for c in sorted(cal)]).astype(np.float32),
        eval_scores=evz.astype(np.float32), eval_true=trues,
        sigma=np.array([np.nan if sigmas[c] is None else sigmas[c] for c in sorted(sigmas)]),
    )

    results = []
    for method in args.methods:
        auditor = pl.fit(cal, m, method=method)
        pred, _ = auditor.predict(evz)
        # v is the number of batches with a correct prediction, from 0 to m.
        v = [int(x) for x in pl.overlap(trues, pred, m)]
        avg = evaluator.compute_avg_v_test_epsilon_lb(
            m=m, r=m, T=n_eval, v_list=v, ci_delta=args.ci_delta, direction="ge")
        med = evaluator.compute_median_v_test_epsilon_lb(
            m=m, r=m, T=n_eval, v_list=v, ci_delta=args.ci_delta)
        rho = evaluator.compute_avg_v_test_rho_lb(
            m=m, r=m, T=n_eval, v_list=v, ci_delta=args.ci_delta)
        row = {
            "algorithm": "r2d",
            "config_id": args.config_id,
            "decoder": f"joint_{method}",
            "claimed_eps": eps_raw,
            "eps_lb_avg": avg.get("epsilon_lb"),
            "eps_lb_median": med.get("epsilon_lb"),
            "rho_lb": rho.get("rho_lb"),
            "exact_match_rate": float(np.mean(pred == trues)),
            "mean_v": float(np.mean(v)),
            "eps_ceiling": pl.epsilon_ceiling(m, n_eval, args.ci_delta),
            "n_calib": int(cal[0].shape[0]),
            "n_eval": n_eval,
            "calib_seed": args.calib_seed,
            "eval_seed": args.eval_seed,
            "sigma_mean": float(np.mean(sigma_values)) if sigma_values else None,
        }
        results.append(row)
        print(f"  {row['decoder']}: eps_lb_avg={row['eps_lb_avg']}, "
              f"eps_lb_median={row['eps_lb_median']}, "
              f"exact={row['exact_match_rate']:.3f}, mean_v={row['mean_v']:.3f}", flush=True)

    detail = {
        "results": results,
        "variance_report": variance_report(cal),
        "m": m, "n_points": int(images.shape[0]), "delta": args.delta,
        "ci_delta": args.ci_delta,
        "results_dir": str(args.results_dir), "data_dir": str(args.data_dir),
    }
    with open(args.output_dir / f"joint_eps{eps_raw}.json", "w") as f:
        json.dump(detail, f, indent=2)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory that contains run_001, run_002, and the other runs.")
    p.add_argument("--data-dir", type=Path, required=True, help="Split directory of the runs.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for the result files.")
    p.add_argument("--epsilon", type=str, default="inf",
                   help="List of claimed epsilon values with commas. inf adds no noise.")
    p.add_argument("--delta", type=float, default=1e-3, help="Delta for the noise calibration.")
    p.add_argument("--num-calib", type=int, default=100,
                   help="Number of calibration noise draws for each configuration.")
    p.add_argument("--num-eval", type=int, default=500, help="Number of evaluation trials.")
    p.add_argument("--methods", type=str, default="b2",
                   help="List of decoder methods with commas: b2, ours_a, ours_b. "
                        "b2 is the decoder of the paper.")
    p.add_argument("--calib-seed", type=int, default=0, help="Seed of the calibration noise.")
    p.add_argument("--eval-seed", type=int, default=1,
                   help="Seed of the evaluation noise and of the selected configurations.")
    p.add_argument("--ci-delta", type=float, default=0.05,
                   help="Confidence parameter of the lower bounds.")
    p.add_argument("--config-id", type=str, default="",
                   help="Name of the audit configuration. The script writes it to the summary.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for the models.")
    p.add_argument("--eval-batch-size", type=int, default=1500,
                   help="Number of forget points in one forward pass.")
    args = p.parse_args()

    args.results_dir = args.results_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    eps_list = [s.strip() for s in args.epsilon.split(",") if s.strip()]
    if args.calib_seed == args.eval_seed:
        print("--calib-seed and --eval-seed must be different.", file=sys.stderr)
        return 1
    unknown = [s for s in args.methods if s not in pl.METHODS]
    if unknown:
        print(f"The methods {unknown} are not known. Use {list(pl.METHODS)}.", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    points = load_all_forget_points(args.data_dir / "forget")
    images = torch.stack([pt[0] for pt in points]).to(device)
    labels = torch.tensor([pt[1] for pt in points], device=device)
    m = len({pt[2] for pt in points})
    combos, _ = pl.sign_configs(m)
    run_of = map_runs_to_configurations(args.results_dir, combos)
    evaluator = pl.load_official_evaluator(R2D_DIR / "cum_runs_eps_lab.py")
    print(f"{images.shape[0]} forget points, {m} forget batches, {len(combos)} configurations.")

    rows = []
    for eps_raw in eps_list:
        rows.extend(audit_one_epsilon(eps_raw, run_of, images, labels, device, args, m, evaluator))

    summary = args.output_dir / "joint_summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary file: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
