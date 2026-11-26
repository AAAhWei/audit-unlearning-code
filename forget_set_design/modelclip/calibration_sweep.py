#!/usr/bin/env python3
"""
This script audits the model-clipping runs of run_audits.sh with the joint decoders.

The script is an adapter, as r2d/calibration_sweep.py is for R2D:
    - The decoders b2, ours_a and ours_b are in pooled_linear_auditor/pooled_linear.py.
    - The model, the data loader and the epsilon test are those of
      certified_unlearning_plus_grad_based_methods/.
    - The score of a forget point is phi = log(p / (1 - p)), where p is the probability
      of the correct label. This is the score of the repository audits.

Definitions:
    configuration   One set of deleted forget batches (one sign vector).
    trial           One unlearning run of a configuration. Each trial has its own noise
                    seed. All trials of a configuration start from the same trained model.
    step            The number of noisy unlearning steps before the audited checkpoint.
    budget          The number of calibration trials for each configuration.
    group           A set of `budget` calibration trials. The groups have no common trial.
    design          The configurations that the calibration uses. The full design uses
                    all configurations. A sparse design uses N complement pairs
                    (2 * N configurations). Only ours_b can use a sparse design.

The calibration trials and the evaluation trials are different trials:
    trials 0 to --num-calib - 1                          calibration
    trials --num-calib to --num-calib + --num-eval - 1   evaluation

The script does these steps for each step in --steps:
    1. It restores the checkpoint of each trial and calculates the scores of all forget
       points. It writes them to scores_step<step>.npz. If that file exists for the
       same runs, the script reads it and restores no checkpoint.
    2. For each budget, each group, each design and each method, it fits the decoder
       on the group and predicts the configuration of each evaluation trial.
    3. The epsilon test converts the predictions to a lower bound.

If the epsilon test gives no positive bound, the script records the bound 0.

Output files in --output-dir:
    scores_step<step>.npz   z [configurations, trials, points], float32, and the labels,
                            the batch of each point and the run directories
    sweep_step<step>.csv    one row for each fit
    sweep_step<step>.json   the settings of that file
    sweep_table.csv         the mean, the standard deviation, the minimum and the maximum
                            of the bound for each (step, method, design, budget). The
                            script makes it from all sweep_step*.csv files in the directory.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=<gpu> python forget_set_design/modelclip/calibration_sweep.py \\
        --results-dir <directory that contains the run directories> \\
        --output-dir <output directory> --steps 10,20,30
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
sys.path.insert(0, str(REPO / "pooled_linear_auditor"))

import pooled_linear as pl  # noqa: E402

ROW_COLUMNS = [
    "algorithm", "config_id", "unlearn_step", "decoder", "n_cal_configs", "n_calib", "group",
    "eps_lb_avg", "exact_match_rate", "mean_v", "eps_ceiling", "n_eval",
]
TABLE_COLUMNS = [
    "config_id", "unlearn_step", "decoder", "n_cal_configs", "n_calib", "n_fits",
    "eps_lb_mean", "eps_lb_std", "eps_lb_min", "eps_lb_max", "exact_match_mean", "eps_ceiling",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit model-clipping runs with the joint decoders.")
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory that contains one run directory for each configuration.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for the output files.")
    p.add_argument("--steps", type=str, default="10,20,30",
                   help="List of unlearning steps with commas. Each trial must have the checkpoint "
                        "unlearn_step_<step>.")
    p.add_argument("--num-calib", type=int, default=50,
                   help="Number of calibration trials for each configuration.")
    p.add_argument("--num-eval", type=int, default=100,
                   help="Number of evaluation trials for each configuration.")
    p.add_argument("--budgets", type=str, default="2,5,10,20,50",
                   help="List of budgets with commas. The maximum is --num-calib.")
    p.add_argument("--max-groups", type=int, default=10, help="Maximum number of groups for one budget.")
    p.add_argument("--methods", type=str, default="b2,ours_a,ours_b",
                   help="List of decoder methods with commas: b2, ours_a, ours_b.")
    p.add_argument("--design-pairs", type=str, default="0",
                   help="List of designs with commas. 0 is the full design. N is a sparse design with "
                        "N complement pairs, for ours_b only. The minimum N is the number of batches minus 1.")
    p.add_argument("--design-seed", type=int, default=0, help="Seed of the selection of a sparse design.")
    p.add_argument("--ci-delta", type=float, default=0.05, help="Confidence parameter of the lower bounds.")
    p.add_argument("--config-id", type=str, default="", help="Name of the audit configuration.")
    p.add_argument("--chunk", type=int, default=1500, help="Number of forget points in one forward pass.")
    p.add_argument("--force", action="store_true", help="Calculate all values again.")
    return p.parse_args()


def int_list(text: str) -> list:
    return [int(s) for s in text.split(",") if s.strip()]


def map_runs_to_configurations(results_dir: Path, combos: list, n_trials: int) -> list:
    """
    Return the run directory of each configuration, in the order of `combos`.
    The function stops if a configuration has no complete run or has two runs.
    """
    run_of = {}
    for vars_path in sorted(results_dir.glob("*/run_vars.json")):
        with open(vars_path) as f:
            run_vars = json.load(f)
        index = int(run_vars["combo_idx"])
        if (results_dir / f"_run_{index}.dir").exists():
            raise SystemExit(f"ERROR: The run of configuration {index} is not complete "
                             f"({results_dir / f'_run_{index}.dir'} exists).")
        if index in run_of:
            raise SystemExit(f"ERROR: Configuration {index} has two run directories: "
                             f"{run_of[index].name} and {vars_path.parent.name}.")
        if tuple(run_vars["chosen_idx"]) != tuple(combos[index]):
            raise SystemExit(f"ERROR: {vars_path} has the forget batches {run_vars['chosen_idx']}. "
                             f"Configuration {index} must have {list(combos[index])}.")
        if int(run_vars["num_unlearn_per_combo"]) < n_trials:
            raise SystemExit(f"ERROR: {vars_path.parent} has {run_vars['num_unlearn_per_combo']} trials. "
                             f"The audit needs {n_trials}.")
        run_of[index] = vars_path.parent
    missing = [i for i in range(len(combos)) if i not in run_of]
    if missing:
        raise SystemExit(f"ERROR: {results_dir} has no run for the configurations {missing}.")
    return [run_of[i] for i in range(len(combos))]


def collect_scores(runs: list, step: int, n_trials: int, chunk: int):
    """
    Calculate the scores of all forget points for each configuration and each trial.
    Return (z [configurations, trials, points], labels, batch of each point, data directory).
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import yaml

    from src.models.model import ModelFactory
    from src.utils.data_cache import load_split

    with open(runs[0] / "run_vars.json") as f:
        data_dir = json.load(f)["data_dir"]
    with open(runs[0] / "config.yaml") as f:
        config = yaml.safe_load(f)
    batches = load_split(data_dir, "forget")
    images = jnp.concatenate([b[0] for b in batches], axis=0)
    labels = jnp.concatenate([b[1] for b in batches], axis=0)
    batch_of_point = np.concatenate([np.full(len(b[1]), i) for i, b in enumerate(batches)])
    model = ModelFactory.create_model(model_name=config["model"]["name"],
                                      num_classes=config["model"]["n_classes"])

    @jax.jit
    def phi_of(params, x, y):
        log_probs = jax.nn.log_softmax(model.apply({"params": params}, x, train=False), axis=-1)
        log_p = jnp.take_along_axis(log_probs, y[:, None], axis=-1).squeeze(-1)
        # The limits are the same as in audit_utils._compute_phi_per_sample.
        return log_p - jnp.log(1.0 - jnp.clip(jnp.exp(log_p), 1e-9, 1.0 - 1e-9))

    checkpointer = ocp.PyTreeCheckpointer()
    n_points = int(images.shape[0])
    z = np.zeros((len(runs), n_trials, n_points), dtype=np.float32)
    for index, run_dir in enumerate(runs):
        for trial in range(n_trials):
            path = run_dir / f"ckpt_trial_{trial}" / f"unlearn_step_{step}"
            if not path.exists():
                raise SystemExit(f"ERROR: The checkpoint {path} does not exist.")
            params = checkpointer.restore(str(path.resolve()))["params"]
            for start in range(0, n_points, chunk):
                z[index, trial, start:start + chunk] = np.asarray(
                    phi_of(params, images[start:start + chunk], labels[start:start + chunk]))
        print(f"  step {step}: configuration {index + 1}/{len(runs)} is complete.", flush=True)
    if not np.isfinite(z).all():
        raise SystemExit(f"ERROR: The scores of step {step} contain values that are not finite.")
    return z, np.asarray(labels), batch_of_point, data_dir


def load_or_collect(args, runs: list, step: int, n_trials: int):
    """Read scores_step<step>.npz if it is for the same runs. If not, calculate and write it."""
    path = args.output_dir / f"scores_step{step}.npz"
    run_names = np.array([r.name for r in runs])
    if not args.force and path.exists():
        with np.load(path) as old:
            if np.array_equal(old["runs"], run_names) and old["z"].shape[1] >= n_trials:
                print(f"\nStep {step}: the script reads {path}.")
                return old["z"][:, :n_trials], old["batch_of_point"]
    print(f"\nStep {step}: the script restores {len(runs) * n_trials} checkpoints.", flush=True)
    z, labels, batch_of_point, data_dir = collect_scores(runs, step, n_trials, args.chunk)
    np.savez(path, z=z, labels=labels, batch_of_point=batch_of_point, runs=run_names,
             data_dir=data_dir, unlearn_step=step)
    return z, batch_of_point


def sweep_one_step(args, z: np.ndarray, step: int, m: int, designs: dict, evaluator) -> list:
    """Do the fits for one step. Return one row for each fit."""
    n_configs = z.shape[0]
    cal = {c: z[c, :args.num_calib] for c in range(n_configs)}
    evz = z[:, args.num_calib:args.num_calib + args.num_eval].reshape(n_configs * args.num_eval, -1)
    trues = np.repeat(np.arange(n_configs), args.num_eval)
    base = {"algorithm": "modelclip", "config_id": args.config_id, "unlearn_step": step,
            "eps_ceiling": pl.epsilon_ceiling(m, len(trues), args.ci_delta), "n_eval": len(trues)}
    rows = []
    for budget in args.budgets:
        n_groups = min(args.num_calib // budget, args.max_groups)
        for group in range(n_groups):
            part = {c: s[group * budget:(group + 1) * budget] for c, s in cal.items()}
            for n_pairs, configs in designs.items():
                subset = {c: part[c] for c in configs}
                # A sparse design does not contain all configurations. Only ours_b can use it.
                methods = args.methods if n_pairs == 0 else [s for s in args.methods if s == "ours_b"]
                for method in methods:
                    pred, _ = pl.fit(subset, m, method=method).predict(evz)
                    v = [int(x) for x in pl.overlap(trues, pred, m)]
                    bound = evaluator.compute_avg_v_test_epsilon_lb(
                        m=m, r=m, T=len(v), v_list=v, ci_delta=args.ci_delta, direction="ge").get("epsilon_lb")
                    rows.append(dict(base, decoder=f"joint_{method}", n_cal_configs=len(configs),
                                     n_calib=budget, group=group,
                                     eps_lb_avg=float(bound) if bound is not None and bound > 0 else 0.0,
                                     exact_match_rate=float(np.mean(pred == trues)), mean_v=float(np.mean(v))))
        print(f"  step {step}, budget {budget}: {n_groups} groups", flush=True)
    return rows


def write_table(output_dir: Path) -> list:
    """Read all sweep_step*.csv files in the directory. Write and return the table rows."""
    cells = {}
    for path in sorted(output_dir.glob("sweep_step*.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["config_id"], int(r["unlearn_step"]), r["decoder"], int(r["n_cal_configs"]),
                       int(r["n_calib"]))
                cells.setdefault(key, []).append((float(r["eps_lb_avg"]), float(r["exact_match_rate"]),
                                                  float(r["eps_ceiling"])))
    table = []
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], -k[3], k[4])):
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
    args.output_dir = args.output_dir.resolve()
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    args.budgets = int_list(args.budgets)
    steps = int_list(args.steps)
    design_pairs = int_list(args.design_pairs)

    errors = []
    if [s for s in args.methods if s not in pl.METHODS]:
        errors.append(f"--methods must contain only {list(pl.METHODS)}.")
    if not args.budgets or min(args.budgets) < 2 or max(args.budgets) > args.num_calib:
        errors.append("Each budget must be from 2 to --num-calib.")
    if not steps:
        errors.append("--steps must contain a minimum of one step.")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_trials = args.num_calib + args.num_eval
    first = sorted(args.results_dir.glob("*/run_vars.json"))
    if not first:
        raise SystemExit(f"ERROR: {args.results_dir} contains no run directory.")
    with open(first[0]) as f:
        m = int(json.load(f)["n_forget"])
    combos, _ = pl.sign_configs(m)
    runs = map_runs_to_configurations(args.results_dir, combos, n_trials)
    evaluator = pl.load_official_evaluator(PIPELINE_DIR / "cum_runs_eps_lab.py")

    # Design: number of complement pairs -> configuration indices. 0 is the full design.
    designs = {n: list(range(len(combos))) if n == 0
               else pl.design_complement_pairs(m, n, seed=args.design_seed) for n in design_pairs}
    print(f"{m} forget batches, {len(combos)} configurations, {args.num_calib} calibration trials and "
          f"{args.num_eval} evaluation trials for each configuration. "
          f"Designs: {[len(c) for c in designs.values()]} configurations.")

    settings = {
        "results_dir": str(args.results_dir), "config_id": args.config_id, "runs": [r.name for r in runs],
        "budgets": args.budgets, "num_calib": args.num_calib, "num_eval": args.num_eval,
        "max_groups": args.max_groups, "methods": args.methods,
        "designs": {str(n): c for n, c in designs.items()}, "ci_delta": args.ci_delta,
    }
    for step in steps:
        csv_path = args.output_dir / f"sweep_step{step}.csv"
        json_path = args.output_dir / f"sweep_step{step}.json"
        if not args.force and csv_path.exists() and json_path.exists():
            with open(json_path) as f:
                if json.load(f) == settings:
                    print(f"\nStep {step}: the files exist with the same settings. The script skips it.")
                    continue
        z, batch_of_point = load_or_collect(args, runs, step, n_trials)
        if len(set(batch_of_point.tolist())) != m:
            raise SystemExit(f"ERROR: The forget points are in {len(set(batch_of_point.tolist()))} batches, not {m}.")
        rows = sweep_one_step(args, z, step, m, designs, evaluator)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(settings, f, indent=2)

    table = write_table(args.output_dir)
    print("\n[sweep]  step  decoder       configs  budget  fits   mean    std    min    max   exact")
    for r in table:
        print(f"[sweep] {r['unlearn_step']:5d}  {r['decoder']:12s}  {r['n_cal_configs']:7d}  {r['n_calib']:6d}  "
              f"{r['n_fits']:4d}  {r['eps_lb_mean']:5.3f}  {r['eps_lb_std']:5.3f}  {r['eps_lb_min']:5.3f}  "
              f"{r['eps_lb_max']:5.3f}  {r['exact_match_mean']:5.3f}")
    print(f"[sweep] Table file: {args.output_dir / 'sweep_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
