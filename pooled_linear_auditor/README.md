# Pooled-variance / linear-mean joint auditor (MVP)

A minimal, numpy-only implementation of two calibration-efficient variants of the **joint (sign-vector) Gaussian auditor**
used in this repository (the paper's Algorithm 3). Everything else stays as in the official audit:
* the same per-example scores;
* the same Gaussian log-likelihood decoder over all legal configurations;
* the same official ε lower bound (`cum_runs_eps_lab.compute_avg_v_test_epsilon_lb`).

Only the way the per-configuration Gaussian is *estimated* from calibration models changes:

| method | mean of the score given configuration S | variance | calibration it needs |
|---|---|---|---|
| `b2` (original joint Gaussian) | per-S sample mean | per-S population variance (ddof 0) | every configuration, Γ ≥ 2 each |
| `ours_a` | per-S sample mean | **pooled within-configuration variance** | every configuration, Γ ≥ 2 for at least one |
| `ours_b` | **linear in S**: c_x + b_xᵀS (least squares) | pooled within-configuration variance | any set of configurations whose design [1, S] has rank m |

## Files

| file | content |
|---|---|
| `pooled_linear.py` | estimators, decoder, sparse-design helper, overlap statistic, wrapper around the official ε evaluator |
| `demo_synthetic.py` | synthetic end-to-end demo (b2 vs ours_a vs ours_b, full and sparse calibration) |
| `tests/test_pooled_linear.py` | unit tests (`python -m pytest -q tests`) |

Requirements: Python ≥ 3.9 and numpy. The ε wrapper loads the repository's
`certified_unlearning_plus_grad_based_methods/cum_runs_eps_lab.py` (needs scipy), or `r2d_audit/cum_runs_eps_lab.py` if the
first is missing.

## Setting

* **Configurations.** There are m audit batches of candidate examples. A configuration is a balanced sign vector
  S ∈ {−1, +1}^m with m/2 entries +1, where +1 means the batch is in the forget set. There are C(m, m/2) legal configurations
  (20 for m = 6, 252 for m = 10). `sign_configs(m)` enumerates them in `itertools.combinations` order. A configuration index
  always refers to that order.
* **Scores.** For every released (unlearnt) model, the auditor observes one score per candidate example: z ∈ ℝⁿ, n = number of
  candidates. In this repository the score is the official φ, the log-odds of the correct label
  (`r2d_audit/evaluation_scripts/evaluate_combination_models.py::compute_phi_and_loss_batch`).
* **Calibration.** For each calibrated configuration S, Γ_S independent models (or noise reloads of a deterministic base model)
  with known S give scores Z_S ∈ ℝ^{Γ_S × n}. K = Σ_S Γ_S is the total number of calibration models.
* **Model.** z | S ~ N(μ_S, diag(σ²_S)), examples independent. The audit decodes S from one fresh z.

## Method

**Pooled within-configuration variance** (`pooled_within_variance`, used by `ours_a` and `ours_b`). For every example x:

    σ̂²_x = Σ_S Σ_r (z_{S,r,x} − z̄_{S,x})² / Σ_S (Γ_S − 1)

Every configuration shares this one variance. With small Γ, a per-configuration variance is very noisy: at Γ = 2 it rests on a
single degree of freedom, and E[1/σ̂²] is infinite for Γ ≤ 3. The decoder divides by it, so a few tiny estimates dominate the
likelihood. Pooling uses all K − (number of calibrated configurations) degrees of freedom instead.

**Linear mean** (`linear_mean_fit`, used by `ours_b`). For every example x, fit μ_{S,x} = c_x + b_xᵀS by least squares over all
calibration rows, then predict the mean of *every* legal configuration, including configurations that were never calibrated.
* Each example has m + 1 coefficients, of which m are identified (the entries of a balanced S sum to 0). Compare C(m, m/2)
  free means per example for the per-S estimators.
* Only a calibration design of rank m is required. `design_complement_pairs(m, n_pairs)` picks a well-conditioned set of
  complement pairs (S, −S); it needs at least m − 1 pairs.

**Decoder** (`GaussianAuditor.log_likelihood` / `predict`). The Gaussian log-likelihood is summed over examples for every legal
configuration. The variance is floored at 1e-10 and the decoder takes the argmax (first maximum on ties), as in the official
code. `predict` also returns the log-likelihood margin between the best and the runner-up configuration.

**Evaluation.**
* `overlap(true, pred, m)` counts matching coordinates (0..m); this is the official statistic v for r = m.
* `epsilon_lower_bound(v, m)` calls the official mean test (δ = 0, 95 %) on the overlaps of L independent evaluation trials.
* `epsilon_ceiling(m, L)` is the bound obtained by perfect predictions: 4.029 / 4.724 for m = 6 and L = 500 / 2000, and 5.320
  for m = 10, L = 500.

## Usage

```python
import numpy as np
import pooled_linear as pl

m = 6
combos, S_all = pl.sign_configs(m)              # combos[i]: indices of the forget batches of configuration i
# cal: configuration index -> calibration scores [Gamma_S, n] (one row per calibration model / noise reload)
# evz: scores of the audited models [L, n]; trues: their configuration indices (for evaluation only)
auditor = pl.fit(cal, m, method="ours_b")      # or "ours_a" / "b2"
pred, margin = auditor.predict(evz)
v = pl.overlap(trues, pred, m)
print("exact", np.mean(pred == trues), "eps_lb", pl.epsilon_lower_bound(v, m))

# sparse calibration for ours_b: calibrate only these configurations
design = pl.design_complement_pairs(m, n_pairs=m - 1)
```

Synthetic demo (scores drawn from the auditor's own model with an additive mean; it illustrates the API and is not evidence):

    python demo_synthetic.py            # m = 6
    python demo_synthetic.py --m 10

## Results from the full experiments (real models)

All ε are official mean-test lower bounds (δ = 0, 95 %, r = m) on fresh evaluation models; K is the total number of calibration
models or reloads. The research code and full reports are not part of this MVP. This implementation reproduces their predictions
exactly: identical predicted configurations and log-likelihood margins on the m = 6 and m = 10 R2D score files, and the same ε.

R2D numbers come from a **reconstructed R2D pipeline** (`RECONSTRUCTED_R2D_NOT_FULL_PAPER_REPRODUCTION`: the released
configuration trains near-chance base models, ≈ 1 % accuracy versus ≈ 4 % in the paper). The calibration models there are noise
reloads of one deterministic base model per configuration.

**m = 6, fresh L = 2000** (ceiling 4.724; B1 = the original batchwise auditor, Algorithm 2):

| setting | K | B1 | `b2` | `ours_a` | `ours_b` |
|---|---:|---:|---:|---:|---:|
| R2D, target ε = 2000 | 40 | 0.73 | 0.04 | 0.46 | **1.01** |
| R2D, target ε = 2000 | 200 | 1.27 | 0.84 | 1.11 | **1.83** |
| R2D, target ε = 2000 | 2000 | 1.51 | 1.98 | 2.10 | **2.26** |
| model clipping, step 20 | 40 | 0.42 | 0.01 | **0.82** | 0.42 |
| model clipping, step 20 | 160 | 0.68 | 1.23 | **1.62** | 0.65 |
| model clipping, step 20 | 1000 | 0.77 | **1.98** | 1.92 | 0.74 |

**m = 10, R2D target ε = 2000, uniform evaluation over all 252 configurations, L = 500** (ceiling 5.320):

| calibration | K | B1 | `b2` | `ours_a` | `ours_b` |
|---|---:|---:|---:|---:|---:|
| all 252 configurations, Γ = 2 | 504 | 1.04 | 0.02 | 0.49 | **1.86** (exact 0.67) |
| all 252 configurations, Γ = 4 | 1008 | 1.09 | 0.31 | 0.65 | **2.00** (exact 0.71) |
| 20 configurations (10 complement pairs), Γ = 8 | 160 | 0.61 | n/a | n/a | **1.25** (exact 0.44) |

What these runs showed:
* **Pooled variance.** It is accurate even when the true variances differ across configurations: at K = 40, its median absolute
  log-error against a high-budget reference is 0.24–0.34, versus 1.47 for the per-S variance. Its benefit is finite-sample
  stabilisation. Giving the decoder high-budget per-S variances instead changes ε by ≤ 0.02 (model clipping, step 20). Wherever every configuration can be
  calibrated, `ours_a` ≥ `b2` at small and moderate K.
* **Linear mean.** It wins when the score mean is close to additive in S, as in R2D. At m = 10 the per-S estimators collapse,
  because 2–4 reloads per configuration are too few for 252 means, while `ours_b` keeps working.
  * Fitted on 20 configurations, it decodes held-out configurations far above chance: 0.405 exact versus 1/252 at K = 160.
  * At matched K, calibrating 20, 40, 80 or 252 configurations gives similar accuracy, so the limit is K, not coverage.
* **It fails when the mean is not additive.** On model clipping, `ours_b` plateaus (≈ 0.6 exact) while `ours_a` and `b2` keep
  improving. Check additivity before relying on it, e.g. fit `ours_b` on half of the configurations and look at its decoding
  accuracy, or the R² of its predicted means, on the other half.
* **Abstention does not help.** Reporting only the most confident r < m coordinates did not improve ε; keep r = m.
* **Cost.** When noise reloads are cheap and base models are expensive (R2D), most of the cost is the base models. A uniform
  evaluation over the full space already needs the base models of most configurations: 221 of 252 at m = 10, L = 500. Sparse
  calibration then saves little (29 of 252 base models there) and costs ε (1.25 vs 1.86). Full coverage with Γ = 2 is the better
  choice in that regime.

## Limitations

* **The model is a diagonal Gaussian.** Examples are treated as independent given S, as in the official joint auditor.
* **`ours_a` and `b2` need every configuration calibrated.** `ours_b` needs a rank-m design. A minimal design (m − 1 complement
  pairs) extrapolates the means of the uncalibrated configurations with more error than it fits the calibrated ones, so use
  more pairs or more reloads per configuration.
* **The evidence is limited to the settings above:** reconstructed R2D (one training seed per configuration) and model clipping
  at m = 6, and R2D at m = 10. Larger m and other mechanisms are untested.
