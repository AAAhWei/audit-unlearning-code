#!/usr/bin/env python3
"""
SYNTHETIC demonstration of the pooled-variance / linear-mean auditor (not evidence about any unlearning algorithm).

Scores follow the auditor's own model with an additive mean: z_x | S = c_x + sum_j b_{xj} S_j + sd_x * noise, where b_{xj}
is a small effect of every batch j on example x plus a larger effect of the example's own batch. Calibration: Gamma reloads
for every configuration (full coverage) or for a sparse complement-pair design (ours_b only). Evaluation: L configurations
drawn uniformly with replacement, one fresh draw each, decoded over all C(m, m/2) configurations.

    python demo_synthetic.py                 # m = 6
    python demo_synthetic.py --m 10 --pairs 10
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import pooled_linear as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n_per_batch", type=int, default=150)
    ap.add_argument("--own", type=float, default=0.15, help="effect of the example's own batch")
    ap.add_argument("--cross", type=float, default=0.01, help="SD of the effects of the other batches")
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--pairs", type=int, default=None, help="complement pairs in the sparse ours_b design (default m - 1, the minimum for rank m)")
    ap.add_argument("--L", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = args.m; combos, S_all = pl.sign_configs(m); C = len(combos); rng = np.random.default_rng(args.seed)
    n = m * args.n_per_batch; batch = np.repeat(np.arange(m), args.n_per_batch)
    c0 = rng.normal(0.0, 1.0, n); sd = rng.uniform(0.5, 1.5, n)
    b = rng.normal(0.0, args.cross, (m, n)); b[batch, np.arange(n)] += args.own
    mean = c0[None, :] + S_all @ b                                                      # [C, n]

    def draw(cfg, k):
        return mean[cfg][None, :] + sd[None, :] * rng.standard_normal((k, n))

    gmax = max(args.gammas)
    cal = {cfg: draw(cfg, gmax) for cfg in range(C)}
    sparse = pl.design_complement_pairs(m, args.pairs or m - 1, seed=args.seed)
    trues = rng.integers(0, C, args.L); evz = np.stack([draw(t, 1)[0] for t in trues])
    try:
        ev = pl.load_official_evaluator()
    except (FileNotFoundError, ImportError):
        ev = None
    print(f"SYNTHETIC m = {m}: {C} configurations, n = {n} examples, L = {args.L}; chance exact {1 / C:.4f}; "
          f"perfect-prediction eps {pl.epsilon_ceiling(m, args.L):.3f}")
    print(f"{'design':<22}{'method':<8}{'Gamma':>6}{'K':>7}{'exact':>8}{'overlap':>9}{'eps_lb':>9}")
    rows = [("full", meth, g, {k: v[:g] for k, v in cal.items()}) for g in args.gammas for meth in pl.METHODS]
    rows += [(f"sparse ({len(sparse)} configs)", "ours_b", g, {k: cal[k][:g] for k in sparse}) for g in args.gammas]
    for design, meth, g, sub in rows:
        pred, _ = pl.fit(sub, m, meth).predict(evz)
        v = pl.overlap(trues, pred, m)
        e = pl.epsilon_lower_bound(v, m, evaluator=ev) if ev is not None else math.nan
        print(f"{design:<22}{meth:<8}{g:>6}{len(sub) * g:>7}{np.mean(pred == trues):>8.3f}{v.mean():>9.2f}"
              f"{(e if e is not None else 0.0):>9.3f}")


if __name__ == "__main__":
    main()
