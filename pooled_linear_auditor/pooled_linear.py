"""
Pooled-variance / linear-mean Gaussian auditor for joint sign-vector auditing of unlearning (MVP).

Setting: m audit batches; a configuration is a balanced sign vector S in {-1, +1}^m with m/2 entries +1 (+1 = the batch is
in the forget set), so there are C(m, m/2) legal configurations. For every candidate example x the auditor observes one
scalar score z_x per released model (the official phi = log-odds of the correct label). Calibration: Gamma_S independent
models (or noise reloads) for each observed configuration S. Model: z | S ~ N(mu_S, diag(var)), examples independent.

  b2      per-S mean, per-S population variance (ddof 0)       the original joint Gaussian (official Algorithm 3)
  ours_a  per-S mean, pooled within-S variance                  needs every legal configuration observed
  ours_b  linear mean c_x + b_x^T S, pooled within-S variance   needs a calibration design [1, S] of rank m only

Pooled within-S variance: sum_S sum_r (z_{S,r} - mean_S)^2 / sum_S (Gamma_S - 1), per example, over the observed
configurations. Linear mean: min-norm least squares of z on [1, S] over all calibration rows (rank m is full rank, because
the entries of a balanced S sum to 0). Decoder: Gaussian log-likelihood summed over examples, variance floored at 1e-10,
argmax over ALL legal configurations (first maximum on ties, as the official code).

numpy only; epsilon_lower_bound() loads the repository's official evaluator (cum_runs_eps_lab.py, needs scipy).
"""
from __future__ import annotations

import importlib.util
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VAR_FLOOR = 1e-10
METHODS = ("b2", "ours_a", "ours_b")
REPO = Path(__file__).resolve().parents[1]
EVALUATOR_PATHS = (REPO / "certified_unlearning_plus_grad_based_methods" / "cum_runs_eps_lab.py", REPO / "r2d_audit" / "cum_runs_eps_lab.py")


def sign_configs(m: int):
    """All balanced configurations: (combos, S) with combos[i] = indices of the +1 batches, S [C(m, m/2), m] in {-1, +1}."""
    if m < 2 or m % 2:
        raise ValueError(f"m must be even and >= 2, got {m}")
    combos = [tuple(c) for c in itertools.combinations(range(m), m // 2)]
    S = np.array([[1 if j in c else -1 for j in range(m)] for c in combos], dtype=np.float64)
    return combos, S


def _check_cal(cal: dict) -> dict:
    """cal: configuration index -> scores [Gamma_S, n]; returns float64 arrays, one common n."""
    if not cal:
        raise ValueError("empty calibration set")
    out = {int(c): np.asarray(z, dtype=np.float64) for c, z in cal.items()}
    if any(z.ndim != 2 or z.shape[0] < 1 for z in out.values()) or len({z.shape[1] for z in out.values()}) != 1:
        raise ValueError("every calibration entry must be a [Gamma_S >= 1, n] array with the same n")
    return out


# ------------------------------------------------------------------------------------------------ estimators
def pooled_within_variance(cal: dict) -> np.ndarray:
    """Per-example pooled within-configuration variance [n] (ddof = sum_S (Gamma_S - 1))."""
    cal = _check_cal(cal); keys = sorted(cal)
    dof = sum(cal[c].shape[0] - 1 for c in keys)
    if dof < 1:
        raise ValueError("pooled within-S variance undefined: no configuration has Gamma_S >= 2")
    return sum(((cal[c] - cal[c].mean(0)[None, :]) ** 2).sum(0) for c in keys) / dof


def linear_mean_fit(cal: dict, S_all: np.ndarray):
    """Min-norm least squares of the scores on [1, S] -> (coef [m + 1, n], design rank)."""
    cal = _check_cal(cal); keys = sorted(cal)
    X = np.concatenate([np.repeat(np.concatenate([[1.0], S_all[c]])[None, :], cal[c].shape[0], 0) for c in keys])
    Y = np.concatenate([cal[c] for c in keys])
    return np.linalg.lstsq(X, Y, rcond=None)[0], int(np.linalg.matrix_rank(X))


@dataclass
class GaussianAuditor:
    """A fitted diagonal-Gaussian auditor: mu [C, n] for every legal configuration, var [n] (pooled) or [C, n] (per S)."""
    method: str
    m: int
    mu: np.ndarray
    var: np.ndarray
    info: dict = field(default_factory=dict)

    def log_likelihood(self, evz: np.ndarray) -> np.ndarray:
        """evz [L, n] -> log-likelihood [L, C] of every legal configuration."""
        evz = np.atleast_2d(np.asarray(evz, dtype=np.float64))
        var = np.maximum(np.broadcast_to(self.var, self.mu.shape), VAR_FLOOR)
        inv = 1.0 / var
        const = -0.5 * np.log(2 * math.pi * var).sum(1)
        a = evz ** 2 @ inv.T; b = evz @ (self.mu * inv).T; c = (self.mu ** 2 * inv).sum(1)
        return const[None, :] - 0.5 * (a - 2 * b + c[None, :])

    def predict(self, evz: np.ndarray):
        """evz [L, n] -> (predicted configuration indices [L], log-likelihood margin MAP - runner-up [L])."""
        ll = self.log_likelihood(evz)
        top2 = -np.sort(-ll, axis=1)[:, :2]
        return np.argmax(ll, axis=1), top2[:, 0] - top2[:, 1]


def fit(cal: dict, m: int, method: str = "ours_b") -> GaussianAuditor:
    """Fit b2 / ours_a / ours_b on calibration scores cal: configuration index (into sign_configs(m)) -> [Gamma_S, n]."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cal = _check_cal(cal); combos, S_all = sign_configs(m)
    if not set(cal) <= set(range(len(combos))):
        raise ValueError(f"configuration indices must lie in [0, {len(combos)})")
    if method == "ours_b":
        coef, rank = linear_mean_fit(cal, S_all)
        if rank < m:
            raise ValueError(f"calibration design [1, S] has rank {rank} < m = {m}: the linear mean is not identified for "
                             f"every configuration (use e.g. design_complement_pairs)")
        mu = np.concatenate([np.ones((len(combos), 1)), S_all], 1) @ coef
        return GaussianAuditor(method, m, mu, pooled_within_variance(cal), {"design_rank": rank, "n_cal_configs": len(cal)})
    if sorted(cal) != list(range(len(combos))):
        raise ValueError(f"{method} needs calibration scores for all {len(combos)} configurations (got {len(cal)}); use ours_b")
    mu = np.stack([cal[c].mean(0) for c in range(len(combos))])
    if method == "ours_a":
        var = pooled_within_variance(cal)
    else:
        if min(z.shape[0] for z in cal.values()) < 2:
            raise ValueError("b2 needs Gamma_S >= 2 for every configuration")
        var = np.stack([((cal[c] - mu[c][None, :]) ** 2).mean(0) for c in range(len(combos))])
    return GaussianAuditor(method, m, mu, var, {"n_cal_configs": len(cal)})


# ------------------------------------------------------------------------------------------------ design and evaluation
def design_complement_pairs(m: int, n_pairs: int, n_draws: int = 5000, seed: int = 0) -> list:
    """Sparse calibration design for ours_b: n_pairs complement pairs (S, -S), the best-conditioned draw of [1, S]
    (largest smallest non-zero singular value among the rank-m draws). Chosen on S only."""
    combos, S_all = sign_configs(m); idx = {c: i for i, c in enumerate(combos)}
    pairs = sorted({tuple(sorted((i, idx[tuple(j for j in range(m) if j not in c)]))) for i, c in enumerate(combos)})
    if n_pairs > len(pairs):
        raise ValueError(f"only {len(pairs)} complement pairs exist for m = {m}")
    rng = np.random.default_rng(seed); best = None
    for _ in range(n_draws):
        cfg = sorted(c for p in rng.choice(len(pairs), size=n_pairs, replace=False) for c in pairs[p])
        sv = np.linalg.svd(np.concatenate([np.ones((len(cfg), 1)), S_all[cfg]], 1), compute_uv=False)
        rank = int((sv > 1e-9 * sv[0]).sum())
        if rank == m and (best is None or sv[rank - 1] > best[0] + 1e-12):
            best = (float(sv[rank - 1]), cfg)
    if best is None:
        raise ValueError(f"no rank-{m} design found with {n_pairs} pairs")
    return best[1]


def overlap(true_idx, pred_idx, m: int) -> np.ndarray:
    """Matching coordinates of true and predicted sign vectors (0..m) = the official statistic v for r = m."""
    _, S_all = sign_configs(m)
    return (S_all[np.asarray(true_idx)] == S_all[np.asarray(pred_idx)]).sum(1)


def epsilon_ceiling(m: int, L: int, ci_delta: float = 0.05) -> float:
    """eps lower bound of perfect predictions in L trials (closed form of the official mean test)."""
    q = ci_delta ** (1.0 / L)
    return 0.5 * math.log((math.comb(m, m // 2) - 1) * q / (1 - q))


def load_official_evaluator(path=None):
    """Import the repository's official cum_runs_eps_lab.py (first existing of EVALUATOR_PATHS unless path is given)."""
    cands = [Path(path)] if path else [p for p in EVALUATOR_PATHS if p.exists()]
    if not cands or not cands[0].exists():
        raise FileNotFoundError("official evaluator cum_runs_eps_lab.py not found")
    spec = importlib.util.spec_from_file_location("cum_runs_eps_lab", cands[0])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def epsilon_lower_bound(v, m: int, ci_delta: float = 0.05, evaluator=None):
    """Official mean-test eps lower bound (delta = 0, r = m) from the overlaps v of L independent trials; None when the
    test gives no positive bound."""
    ev = evaluator or load_official_evaluator()
    v = [int(x) for x in v]
    return ev.compute_avg_v_test_epsilon_lb(m, m, len(v), v, ci_delta).get("epsilon_lb")
