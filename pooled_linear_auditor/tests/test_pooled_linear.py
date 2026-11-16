"""Unit tests for pooled_linear.py (numpy only; the evaluator test is skipped when the official module cannot be loaded)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pooled_linear as pl  # noqa: E402


def additive_problem(m, n_per_batch=60, own=0.3, cross=0.02, seed=0):
    rng = np.random.default_rng(seed); combos, S_all = pl.sign_configs(m)
    n = m * n_per_batch; batch = np.repeat(np.arange(m), n_per_batch)
    b = rng.normal(0, cross, (m, n)); b[batch, np.arange(n)] += own
    mean = rng.normal(0, 1, n)[None, :] + S_all @ b
    sd = rng.uniform(0.5, 1.5, n)
    return rng, mean, sd


def test_sign_configs():
    combos, S = pl.sign_configs(6)
    assert len(combos) == 20 and S.shape == (20, 6)
    assert np.all(S.sum(1) == 0) and len({tuple(r) for r in S}) == 20
    assert all(np.all(S[i, list(c)] == 1) for i, c in enumerate(combos))
    with pytest.raises(ValueError):
        pl.sign_configs(5)


def test_pooled_variance_formula():
    rng = np.random.default_rng(1)
    cal = {0: rng.normal(size=(2, 7)), 3: rng.normal(size=(5, 7)), 9: rng.normal(size=(1, 7))}
    ss = sum(((z - z.mean(0)) ** 2).sum(0) for z in cal.values())
    np.testing.assert_allclose(pl.pooled_within_variance(cal), ss / (1 + 4 + 0))
    eq = {c: rng.normal(size=(4, 7)) for c in range(5)}
    np.testing.assert_allclose(pl.pooled_within_variance(eq), np.mean([z.var(0, ddof=1) for z in eq.values()], 0))
    with pytest.raises(ValueError):
        pl.pooled_within_variance({0: rng.normal(size=(1, 7)), 1: rng.normal(size=(1, 7))})


def test_linear_mean_exact_on_sparse_additive_design():
    m = 6; _, mean, _ = additive_problem(m)
    design = pl.design_complement_pairs(m, m - 1, n_draws=500)
    coef, rank = pl.linear_mean_fit({c: mean[c][None, :] for c in design}, pl.sign_configs(m)[1])
    assert rank == m
    _, S_all = pl.sign_configs(m)
    np.testing.assert_allclose(np.concatenate([np.ones((len(S_all), 1)), S_all], 1) @ coef, mean, atol=1e-10)


def test_design_complement_pairs():
    m = 10; design = pl.design_complement_pairs(m, 10, n_draws=300, seed=3)
    combos, S_all = pl.sign_configs(m)
    assert len(design) == 20 and len(set(design)) == 20
    assert {tuple(-S_all[c]) for c in design} == {tuple(S_all[c]) for c in design}
    assert np.linalg.matrix_rank(np.concatenate([np.ones((20, 1)), S_all[design]], 1)) == m


def test_fit_requirements():
    m = 6; rng = np.random.default_rng(2)
    one_pair = {0: rng.normal(size=(3, 5)), 19: rng.normal(size=(3, 5))}
    with pytest.raises(ValueError, match="rank"):
        pl.fit(one_pair, m, "ours_b")
    for meth in ("ours_a", "b2"):
        with pytest.raises(ValueError, match="all 20"):
            pl.fit(one_pair, m, meth)
    with pytest.raises(ValueError):
        pl.fit({c: rng.normal(size=(1, 5)) for c in range(20)}, m, "b2")
    with pytest.raises(ValueError):
        pl.fit({25: rng.normal(size=(2, 5))}, m, "ours_b")


def test_estimators_of_each_method():
    m = 6; rng = np.random.default_rng(4)
    cal = {c: rng.normal(size=(3, 8)) for c in range(20)}
    a, b2 = pl.fit(cal, m, "ours_a"), pl.fit(cal, m, "b2")
    np.testing.assert_array_equal(a.mu, np.stack([cal[c].mean(0) for c in range(20)]))
    np.testing.assert_allclose(a.var, pl.pooled_within_variance(cal))
    np.testing.assert_allclose(b2.var, np.stack([cal[c].var(0, ddof=0) for c in range(20)]))
    ob = pl.fit(cal, m, "ours_b")
    assert ob.mu.shape == (20, 8) and ob.info["design_rank"] == m
    np.testing.assert_allclose(ob.var, a.var)


def test_decoder_matches_bruteforce():
    rng = np.random.default_rng(5); C, n, L = 20, 9, 4
    mu, var = rng.normal(size=(C, n)), rng.uniform(0.2, 2.0, (C, n)); var[0, 0] = 0.0
    aud = pl.GaussianAuditor("b2", 6, mu, var)
    evz = rng.normal(size=(L, n)); ll = aud.log_likelihood(evz)
    v = np.maximum(var, pl.VAR_FLOOR)
    brute = np.array([[np.sum(-0.5 * np.log(2 * math.pi * v[c]) - 0.5 * (z - mu[c]) ** 2 / v[c]) for c in range(C)] for z in evz])
    np.testing.assert_allclose(ll, brute, rtol=1e-9, atol=1e-6)
    pred, margin = aud.predict(evz)
    assert np.array_equal(pred, brute.argmax(1)) and np.all(margin >= 0)


def test_recovery_on_additive_data():
    m = 6; rng, mean, sd = additive_problem(m)
    draw = lambda c, k: mean[c][None, :] + sd[None, :] * rng.standard_normal((k, mean.shape[1]))  # noqa: E731
    cal = {c: draw(c, 4) for c in range(20)}
    trues = rng.integers(0, 20, 200); evz = np.stack([draw(t, 1)[0] for t in trues])
    acc = {meth: np.mean(pl.fit(cal, m, meth).predict(evz)[0] == trues) for meth in pl.METHODS}
    assert acc["ours_b"] >= 0.9 and acc["ours_b"] >= acc["ours_a"] >= acc["b2"]
    # minimal sparse design (m - 1 complement pairs): the extrapolated means are noisier than the fitted ones, so it needs
    # more reloads per configuration than full coverage to decode the configurations it never calibrated on
    sparse = pl.design_complement_pairs(m, m - 1, n_draws=500); unseen = ~np.isin(trues, sparse)
    p = pl.fit({c: draw(c, 20) for c in sparse}, m, "ours_b").predict(evz)[0]
    assert unseen.sum() > 50 and np.mean(p[unseen] == trues[unseen]) > 0.8


def test_overlap_and_epsilon():
    m, L = 6, 300; rng = np.random.default_rng(6); trues = rng.integers(0, 20, L)
    assert np.all(pl.overlap(trues, trues, m) == m)
    _, S_all = pl.sign_configs(m)
    comp = [int(np.where((S_all == -S_all[t]).all(1))[0][0]) for t in trues]
    assert np.all(pl.overlap(trues, comp, m) == 0)
    try:
        ev = pl.load_official_evaluator()
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"official evaluator unavailable: {e}")
    assert abs(pl.epsilon_lower_bound(pl.overlap(trues, trues, m), m, evaluator=ev) - pl.epsilon_ceiling(m, L)) < 1e-6
    e = pl.epsilon_lower_bound(pl.overlap(trues, rng.integers(0, 20, L), m), m, evaluator=ev)
    assert e is None or e < 0.2
