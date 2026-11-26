#!/usr/bin/env python3
"""
This script audits R2D runs with the joint sign-vector decoders. It compares the decoder
methods for different calibration budgets.

The script uses the unlearned models of run_audits.sh. It does not train a model.
The script is an adapter. It does not contain a decoder or an epsilon test.
    - The decoders b2, ours_a and ours_b are in pooled_linear_auditor/pooled_linear.py.
      The method "b2" is the joint Gaussian decoder of the paper (Algorithm 3).
    - The epsilon test is r2d_audit/cum_runs_eps_lab.py.
    - The model, the noise scale and the phi score come from r2d_audit/.

Definitions:
    configuration   One set of deleted forget batches (one sign vector). For 6 batches
                    there are 20 configurations. Each run of the pipeline trains and
                    unlearns one configuration.
    score           The phi value of one forget point for one noisy model.
    noise draw      One sample of Gaussian noise that the script adds to the weights
                    of an unlearned model. The noise scale comes from the claimed
                    epsilon.
    budget          The number of calibration noise draws for each configuration.
    pass            One collection of --num-calib calibration draws for each
                    configuration, for one calibration seed.
    group           A set of `budget` draws of a pass. The groups of a pass have no
                    common draw. Thus one pass of 100 draws gives 50 groups for the
                    budget 2 and 10 groups for the budget 10.
    design          The configurations that the calibration uses. The full design uses
                    all configurations. A sparse design uses N complement pairs
                    (2 * N configurations). Only ours_b can use a sparse design.

The script does these steps for each claimed epsilon:
    1. Calibration. For each calibration seed, it does one pass: for each
       configuration, it makes --num-calib noise draws and records the scores of all
       forget points.
    2. Evaluation. The script selects --num-eval configurations at random. Each
       configuration has the same probability. For each selection, the script makes
       one new noise draw and records the scores. The evaluation trials are the same
       for all passes, because they use --eval-seed only.
    3. For each budget, each group, each design and each method, the decoder fits the
       group and predicts the configuration of each evaluation trial.
    4. The epsilon test converts the number of correct batch predictions to a lower
       bound. If the test gives no positive bound, the script records the bound 0.

The calibration and the evaluation use different random generators (--calib-seeds,
--eval-seed). Each evaluation trial has its own noise draw. Thus the trials are
independent, as the epsilon test requires.

The audit of a configuration is the row of the method b2 with the full design and the
budget --num-calib.

If the claimed epsilon is inf, the script adds no noise. The scores of a configuration
are then always the same, and the budget has no effect. The script does one fit for
each method and each design, with 2 calibration rows and the first calibration seed.
The decoder is always correct, and the result is the maximum lower bound for
--num-eval trials.

Output files in --output-dir:
    sweep_eps<value>.csv      one row for each fit
    sweep_eps<value>.json     the settings of that file
    scores_eps<value>.npz     the mean and the variance of the calibration scores of each
                              point for each configuration, and the evaluation scores,
                              for the first calibration seed. validate_score.py reads
                              this file.
    variance_eps<value>.json  a summary of the calibration variances
    sweep_table.csv           the mean, the standard deviation, the minimum and the maximum
                              of the bound for each (epsilon, method, design, budget).
                              The script makes it from all sweep_eps*.csv files in the
                              directory.
The script skips a claimed epsilon if its sweep files exist with the same settings.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=<gpu> python forget_set_design/r2d/calibration_sweep.py \\
        --results-dir <directory that contains run_001, run_002, ...> \\
        --data-dir <split directory> --output-dir <output directory> \\
        --epsilon inf,100,1000,2000 --budgets 2,10,100
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

REPO = Path(__file__).resolve().parents[2]
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

ROW_COLUMNS = [
    "algorithm", "config_id", "claimed_eps", "decoder", "n_cal_configs", "n_calib",
    "calib_seed", "group", "eps_lb_avg", "exact_match_rate", "mean_v", "eps_ceiling",
    "n_eval", "eval_seed", "sigma_mean",
]
TABLE_COLUMNS = [
    "config_id", "claimed_eps", "decoder", "n_cal_configs", "n_calib", "n_fits",
    "eps_lb_mean", "eps_lb_std", "eps_lb_min", "eps_lb_max", "exact_match_mean", "eps_ceiling",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the decoder methods for different calibration budgets.")
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory that contains run_001, run_002, and the other runs.")
    p.add_argument("--data-dir", type=Path, required=True, help="Split directory of the runs.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for the result files.")
    p.add_argument("--epsilon", type=str, default="inf,100,1000,2000",
                   help="List of claimed epsilon values with commas. inf adds no noise.")
    p.add_argument("--delta", type=float, default=1e-3, help="Delta for the noise calibration.")
    p.add_argument("--budgets", type=str, default="2,10,100",
                   help="List of calibration budgets with commas. The minimum is 2.")
    p.add_argument("--num-calib", type=int, default=100,
                   help="Number of calibration noise draws for each configuration in one pass. "
                        "It must not be less than the largest budget.")
    p.add_argument("--max-groups", type=int, default=10,
                   help="Maximum number of groups that the script uses from one pass for one budget.")
    p.add_argument("--calib-seeds", type=str, default="0",
                   help="List of calibration seeds with commas. Each seed gives one pass.")
    p.add_argument("--eval-seed", type=int, default=1,
                   help="Seed of the evaluation noise and of the selected configurations.")
    p.add_argument("--num-eval", type=int, default=500, help="Number of evaluation trials.")
    p.add_argument("--methods", type=str, default="b2,ours_a,ours_b",
                   help="List of decoder methods with commas: b2, ours_a, ours_b.")
    p.add_argument("--design-pairs", type=str, default="0,5",
                   help="List of designs with commas. 0 is the full design. N is a sparse design with "
                        "N complement pairs, for ours_b only. The minimum N is the number of batches minus 1.")
    p.add_argument("--design-seed", type=int, default=0, help="Seed of the selection of a sparse design.")
    p.add_argument("--ci-delta", type=float, default=0.05, help="Confidence parameter of the lower bounds.")
    p.add_argument("--config-id", type=str, default="", help="Name of the audit configuration.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for the models.")
    p.add_argument("--eval-batch-size", type=int, default=1500,
                   help="Number of forget points in one forward pass.")
    p.add_argument("--force", action="store_true", help="Calculate all values again.")
    return p.parse_args()


def int_list(text: str) -> list:
    return [int(s) for s in text.split(",") if s.strip()]


def fit_rows(cal: dict, evz, trues, m: int, designs: dict, args, evaluator, base: dict) -> list:
    """Fit each method and each design on the calibration scores `cal`. Return one row for each fit."""
    rows = []
    for n_pairs, configs in designs.items():
        subset = {c: cal[c] for c in configs}
        # A sparse design does not contain all configurations. Only ours_b can use it.
        methods = args.methods if n_pairs == 0 else [s for s in args.methods if s == "ours_b"]
        for method in methods:
            pred, _ = pl.fit(subset, m, method=method).predict(evz)
            v = [int(x) for x in pl.overlap(trues, pred, m)]
            bound = evaluator.compute_avg_v_test_epsilon_lb(
                m=m, r=m, T=len(v), v_list=v, ci_delta=args.ci_delta, direction="ge").get("epsilon_lb")
            rows.append(dict(base, decoder=f"joint_{method}", n_cal_configs=len(configs),
                             eps_lb_avg=float(bound) if bound is not None and bound > 0 else 0.0,
                             exact_match_rate=float(np.mean(pred == trues)), mean_v=float(np.mean(v))))
    return rows


def save_scores(output_dir: Path, eps_raw: str, cal: dict, evz, trues, sigmas: dict) -> None:
    """Write the scores file and the variance summary of one claimed epsilon."""
    np.savez_compressed(
        output_dir / f"scores_eps{eps_raw}.npz",
        cal_mean=np.stack([cal[c].mean(axis=0) for c in sorted(cal)]).astype(np.float32),
        cal_var=np.stack([cal[c].var(axis=0) for c in sorted(cal)]).astype(np.float32),
        eval_scores=evz.astype(np.float32), eval_true=trues,
        sigma=np.array([np.nan if sigmas[c] is None else sigmas[c] for c in sorted(sigmas)]),
    )
    with open(output_dir / f"variance_eps{eps_raw}.json", "w") as f:
        json.dump(variance_report(cal), f, indent=2)


def sweep_one_epsilon(eps_raw: str, run_of, images, labels, device, args, m: int, designs: dict, evaluator) -> list:
    """Do the passes and the fits for one claimed epsilon. Return the rows."""
    add_noise = not math.isinf(float(eps_raw))
    # Without noise, all passes are the same. Thus one pass is sufficient.
    seeds = args.calib_seeds if add_noise else args.calib_seeds[:1]
    rows = []
    for index, seed in enumerate(seeds):
        print(f"\nClaimed epsilon {eps_raw}, calibration seed {seed}", flush=True)
        pass_args = argparse.Namespace(**vars(args))
        pass_args.calib_seed = seed
        cal, evz, trues, sigmas = collect_scores(run_of, images, labels, device, pass_args, float(eps_raw))
        if index == 0:
            save_scores(args.output_dir, eps_raw, cal, evz, trues, sigmas)
        base = {
            "algorithm": "r2d", "config_id": args.config_id, "claimed_eps": eps_raw, "calib_seed": seed,
            "eps_ceiling": pl.epsilon_ceiling(m, args.num_eval, args.ci_delta), "n_eval": args.num_eval,
            "eval_seed": args.eval_seed,
            "sigma_mean": float(np.mean(list(sigmas.values()))) if add_noise else "",
        }
        if not add_noise:
            # The budget has no effect. The calibration has 2 rows for each configuration.
            rows.extend(fit_rows(cal, evz, trues, m, designs, args, evaluator,
                                 dict(base, n_calib=int(cal[0].shape[0]), group=0)))
            print("  no noise: 1 fit for each method and each design", flush=True)
            continue
        for budget in args.budgets:
            n_groups = min(args.num_calib // budget, args.max_groups)
            for group in range(n_groups):
                part = {c: z[group * budget:(group + 1) * budget] for c, z in cal.items()}
                rows.extend(fit_rows(part, evz, trues, m, designs, args, evaluator,
                                     dict(base, n_calib=budget, group=group)))
            print(f"  budget {budget}: {n_groups} groups", flush=True)
    return rows


def write_table(output_dir: Path) -> list:
    """Read all sweep_eps*.csv files in the directory. Write and return the table rows."""
    cells = {}
    for path in sorted(output_dir.glob("sweep_eps*.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["config_id"], r["claimed_eps"], r["decoder"], int(r["n_cal_configs"]), int(r["n_calib"]))
                cells.setdefault(key, []).append((float(r["eps_lb_avg"]), float(r["exact_match_rate"]),
                                                  float(r["eps_ceiling"])))
    table = []
    for key in sorted(cells, key=lambda k: (k[0], float(k[1]), k[2], -k[3], k[4])):
        bound = np.array([x[0] for x in cells[key]])
        table.append(dict(zip(TABLE_COLUMNS[:5], key), n_fits=len(bound),
                          eps_lb_mean=float(bound.mean()),
                          eps_lb_std=float(bound.std(ddof=1)) if len(bound) > 1 else 0.0,
                          eps_lb_min=float(bound.min()), eps_lb_max=float(bound.max()),
                          exact_match_mean=float(np.mean([x[1] for x in cells[key]])),
                          eps_ceiling=cells[key][0][2]))
    with open(output_dir / "sweep_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    return table


def main() -> int:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    args.budgets = int_list(args.budgets)
    args.calib_seeds = int_list(args.calib_seeds)
    design_pairs = int_list(args.design_pairs)
    eps_list = [s.strip() for s in args.epsilon.split(",") if s.strip()]

    errors = []
    if [s for s in args.methods if s not in pl.METHODS]:
        errors.append(f"--methods must contain only {list(pl.METHODS)}.")
    if not args.budgets or min(args.budgets) < 2 or max(args.budgets) > args.num_calib:
        errors.append("Each budget must be from 2 to --num-calib.")
    if args.eval_seed in args.calib_seeds:
        errors.append("--eval-seed must not be one of --calib-seeds.")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
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

    # Design: number of complement pairs -> configuration indices. 0 is the full design.
    designs = {n: list(range(len(combos))) if n == 0
               else pl.design_complement_pairs(m, n, seed=args.design_seed) for n in design_pairs}
    print(f"{images.shape[0]} forget points, {m} forget batches, {len(combos)} configurations. "
          f"Designs: {[len(c) for c in designs.values()]} configurations.")

    settings = {
        "results_dir": str(args.results_dir), "data_dir": str(args.data_dir), "config_id": args.config_id,
        "delta": args.delta, "budgets": args.budgets, "num_calib": args.num_calib, "max_groups": args.max_groups,
        "calib_seeds": args.calib_seeds, "eval_seed": args.eval_seed, "num_eval": args.num_eval,
        "methods": args.methods, "designs": {str(n): c for n, c in designs.items()}, "ci_delta": args.ci_delta,
    }
    for eps_raw in eps_list:
        csv_path = args.output_dir / f"sweep_eps{eps_raw}.csv"
        json_path = args.output_dir / f"sweep_eps{eps_raw}.json"
        if not args.force and csv_path.exists() and json_path.exists():
            with open(json_path) as f:
                if json.load(f) == settings:
                    print(f"\nClaimed epsilon {eps_raw}: the files exist with the same settings. The script skips it.")
                    continue
        rows = sweep_one_epsilon(eps_raw, run_of, images, labels, device, args, m, designs, evaluator)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(settings, f, indent=2)

    table = write_table(args.output_dir)
    print("\n[sweep] epsilon  decoder       configs  budget  fits   mean    std    min    max   exact")
    for r in table:
        print(f"[sweep] {r['claimed_eps']:>7s}  {r['decoder']:12s}  {r['n_cal_configs']:7d}  {r['n_calib']:6d}  "
              f"{r['n_fits']:4d}  {r['eps_lb_mean']:5.3f}  {r['eps_lb_std']:5.3f}  {r['eps_lb_min']:5.3f}  "
              f"{r['eps_lb_max']:5.3f}  {r['exact_match_mean']:5.3f}")
    print(f"[sweep] Table file: {args.output_dir / 'sweep_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
