# Spec: hard-point forget batches for unlearning audits

Implementation brief for a coding session. Read fully before writing code.

---

## 0. Target

**Question.** The audit in arXiv 2607.05898 deletes *random* forget batches. Certified unlearning (R2D, model clipping) promises ε for *any* forget set. Does building the forget batches from the **hardest** training points, and grouping them by **superclass** or **gradient direction**, raise the audited lower bound ε_LB?

**Algorithms (equal priority):** R2D and model clipping. They train very differently, so difficulty scores are computed **separately for each**:
- R2D: 35 full-batch steps, model near chance.
- Model clipping: 400 epochs of SGD, ~55% test accuracy.

**Auditor:** the paper's **Instantiation II**, i.e. the balanced sign-vector setup used for certified methods:
- **Setup:** m = 6 forget batches of B = 750 points, 3 of 6 deleted per run (|S| = 20), sign-vector conversion to pure ε at δ = 0, confidence ζ = 0.05.
- **Decoders:** report both, see §5.2.

**Deliverables**

1. Per-example difficulty scores for the 45,000 CIFAR-100 training points (excluding the validation split), one score file per algorithm.
2. Forget-pool split directories, in the exact format the existing audit code reads, for every configuration in §5.
3. Sanity checks (§2.3) and diagnostics (§4.4) for each split.
4. ε_LB for each configuration and algorithm, from the **unchanged** unlearning and conversion code.

**Hypotheses**

- ε_LB(hard pool) > ε_LB(random pool) > ε_LB(easy pool).
- Within the hard pool: gradient-clustered ≥ superclass-grouped > random assignment > gradient-spread.
- The effect is stronger for model clipping than for R2D, because the R2D model is barely trained (§2.3).

**Non-goals:** do not modify the unlearning algorithms or the ε_LB conversion. Only the forget-pool composition and the batch assignment change. The only auditor-side addition allowed is the joint decoder in §5.2, if it's missing from the repo.

---

## 1. Setup, codebase and data format

**Repo:** the user's fork of `Sahasrajit123/audit-unlearning-code` is already cloned locally. Work from the repository root; do not clone again.

```bash
git checkout master && git pull
git checkout -b forget-set-design           # do all work on this branch
mkdir forget_set_design                     # new code lives here (see §8)
```

- **Environment:** the README mentions a conda file `r2d.yml` in the repository root. If it exists, run `conda env create -f r2d.yml`. Otherwise install from `certified_unlearning_plus_grad_based_methods/requirements.txt` plus `torch>=2.0 numpy scipy matplotlib tqdm`.
- **Data:** generate the paper's baseline splits first, from `certified_unlearning_plus_grad_based_methods/`:
  ```bash
  python scripts/download_cifar100_batches.py
  python scripts/split_cifar100_train.py --batch_size 750 --split_mode uniform --seed 1
  ```
- **Do not edit** existing pipeline files. Any change needed (§5.2 joint decoder, §6 rule 3) goes in new files under `forget_set_design/`, or a separate, clearly explained commit.
- Commit after each step (§2–§5) with a message naming the step, and record the exact commands in `forget_set_design/README.md`.

**Relevant code**

- `certified_unlearning_plus_grad_based_methods/scripts/split_cifar100_train.py`: creates the splits. It holds out `--val_split` (default 0.1 → 5,000) of the 50,000 train images, then takes `--forget_fraction` (0.1 → 4,500) as the forget pool, written as batches of `--batch_size` (750 → 6 batches).
- **R2D:** `r2d_audit/`
  - `run_all_combinations_forget_unlearning.sh`: trains and unlearns every combination of 3 of 6 batches (20 runs).
  - `evaluation_scripts/evaluate_combination_models.py`: φ statistics with noise reloads.
  - `evaluation_scripts/evaluate_prediction_cumulative_model.py`: ε_LB.
  - Defaults: `tinynetcifar100`, full-batch GD (micro-batch 128 accumulation), 35 train + 5 unlearn epochs, lr 0.01, no shuffle. With full batch, T = 40 steps and K = 5 (`compute_T_and_K` in `forget_phi_noisy_loader.py`), so the rewind point is step 35.
- **Model clipping:** `certified_unlearning_plus_grad_based_methods/`. This pipeline is written in **JAX/Flax**, unlike `r2d_audit/` (PyTorch).
  - **Use the default config `configs/exp_cifar100_chg7.yaml`**, the one the README's commands use:
    - Training: `tiny_net_cifar100`, SGD with batch 128, 400 epochs, onecycle max_lr 0.06, momentum 0, weight decay 5e-4. The CIFAR-100 train transform uses random flip/crop augmentation.
    - Unlearning: `contractive_coefficients` (noisy projected SGD on the retain set), 50 epochs, constant lr 0.001, `init_model_clip: 30`, `model_clip: 5.0`, `sigma: 1e-4`, `epsilon_target: 1e5`, `delta: 1e-5`, `stop_epoch: 10`.
    - Post-unlearning: noiseless fine-tuning (onecycle, max_lr 0.1).
  - Other paper configs, for reference only: `exp_cifar100_chg.yaml` (ε_target 10, σ 0.02, clip 0.1) and `exp_cifar100_chg6.yaml` (ε_target 1e5, σ 1e-3, clip 0.225). `exp_cifar.yaml` is CIFAR-10.
  - **Note in the results:** chg7's σ = 1e-4 and C₂ = 5 match what the paper's text calls the *uncertified* model-clipping variant, so its claimed ε (1e5) is nominal.
  - `run_exhaustive_combinations_parallel.sh`: all C(6,3) combinations, one shared training run per combination plus `--num_unlearn_per_combo` unlearning trials (script default 35; its help text says 5, so pass it explicitly).
  - `compute_eps_bounds_sampled_combos.py` and `audit_utils.py`: audits. README examples audit `unlearn_style="epoch"`, `unlearn_itr=5`, `k=3`.

**Split directory format** (must be reproduced exactly; both pipelines read it):

```
<DATAROOT>/
├── train/  val/  test/  retain/
└── forget/
    ├── batch_00000.pkl      # (images, labels) tuple of NumPy arrays
    ├── ...
    └── batch_00005.pkl
```

The batch index in the filename is the batch identity. `retain/` = the 45,000 non-validation points minus the 4,500 forget points.

**First task:** read `split_cifar100_train.py` and write `make_custom_split.py`. It takes a fixed list of 4,500 global indices plus a batch assignment (index → batch 0–5), and writes this format using the **same** validation hold-out and seed as the paper's split. Check on a random assignment that its output matches the original script's file layout, dtypes and shapes, and that both pipelines load it.

---

## 2. Step 1: reference training runs (one per algorithm)

### 2.1 Common rules

- **Data:** all 45,000 non-validation points (same val hold-out and seed as above).
- **Why include every forget candidate:** the score measures how much a point influences training *when it is trained on* (Thm 4.3 compares training with vs. without x, along the trajectory ν_{t,D} with x ∈ D). Gradients of x on a model that never saw x would measure how unusual x is, not how hard it is to forget. Do **not** train the reference model without the forget pool.
- **Separation and validity:** each reference run happens once, before any audit run, and only decides which points go into which batch. Each audit run still draws the secret S uniformly at random, so the ε_LB conversion is unaffected (§6 rule 1). Log the seeds.
- **Same recipe as the audit's pre-unlearning training.** Training length and batch size shape the gradient norms, and so the ranking.

### 2.2 Recipes

| | R2D reference | Model-clipping reference |
|---|---|---|
| Framework | PyTorch | JAX/Flax |
| Model | `tinynetcifar100` (`r2d_audit/models.py`) | `tiny_net_cifar100` (`certified_unlearning_plus_grad_based_methods/src/models/tiny_net.py`) |
| Training | full-batch GD, lr 0.01, 35 epochs, no shuffle | `exp_cifar100_chg7.yaml` `training:` block (SGD, batch 128, 400 epochs, onecycle 0.06, weight decay 5e-4, flip/crop augmentation) |
| Steps T | 35 (one per epoch) | 400 × ⌈45,000 / 128⌉ ≈ 140,800 |
| Checkpoints | after **every** step (35), so the sum is exact | 35 evenly spaced in steps (≈ every 11.4 epochs); save every epoch if that's easier and select 35 |
| Sampling rate P(1) | 1 | 128 / 45,000 |

Use each pipeline's own training code, with only the checkpoint frequency changed (a copied config under `forget_set_design/configs/`). If the training loader uses data augmentation, keep it for training but compute gradients **without** augmentation (§3.3).

### 2.3 Sanity checks (log before scoring)

- Train and test accuracy of each reference model, and the train−test gap.
- **R2D warning:** the R2D model is expected to be near chance (~1% on 100 classes). There, "high score" means *not learned* rather than *memorized*, and hard pools may carry no extra audit signal. Report the accuracy next to every R2D result.

---

## 3. Step 2: difficulty score (Sepahvand et al., arXiv 2505.18786, Def. 4.2)

### 3.1 Definition

$$P(x,\alpha)=\sum_{t=1}^{T} C_{t,\alpha}\,\ln \mathbb{E}_{w\sim\nu_{t,D}}\, f_{t,\alpha}\big(\lVert g(x,w)\rVert_2\big),\qquad g(x,w)=\nabla_w \ell(w,x)\ \text{(all parameters)}$$

$$C_{t,\alpha}=\frac{1}{\alpha-1}\Big(\frac{p-1}{p}\Big)^{t+1},\qquad
\ln f_{t,\alpha}(g)=p\,\ln\!\Big(\sum_{k=0}^{o}\binom{o}{k}P(1)^k\big(1-P(1)\big)^{o-k}\exp\!\Big(\frac{g^2(k^2-k)}{2\sigma^2}\Big)\Big)$$

$$o=o_p^{\,t}(\alpha)\ \ (o_p \text{ applied } t \text{ times}),\qquad o_p(\alpha)=\frac{p}{p-1}\alpha-\frac1p,\qquad p=3T$$

- P(1) = sampling rate (batch size / dataset size).
- σ = training noise. For noiseless training the paper assumes an implicit σ ≤ 0.1; use σ = 0.1.
- The expectation over w uses one model per checkpoint.
- Compute o_t iteratively: `o = alpha; for t in 1..T: o = p/(p-1)*o - 1/p`. Check the indexing (whether step t uses t or t−1 applications) against the paper and report whether the ranking changes.
- Default α = 2. Also compute α ∈ {4, 8, 16} and report the Spearman correlation between rankings. If ρ > 0.95, use α = 2.

> Verify all formulas against the PDF of 2505.18786 (Def. 4.2, §5.1) before trusting them; they were transcribed from fetched page text. No official code was found, so implement from scratch.

### 3.2 R2D (full batch): exact closed form

P(1) = 1, so only the k = o term survives, and with one model per checkpoint:

$$\boxed{\hat P(x,\alpha)=\frac{p}{2\sigma^2}\sum_{t=1}^{35} C_{t,\alpha}\,(o_t^2-o_t)\,\lVert g(x,w_t)\rVert_2^2}$$

σ and p are global constants, so the **ranking depends only on the weighted sum of squared gradient norms**. The checkpoint weights are nearly flat: the last is about 1.2× the first at α = 2 and 1.34× at α = 8.

### 3.3 Model clipping (mini-batch): general form with a Riemann sum

- **Riemann sum over checkpoints** at steps s_1 < … < s_35 (s_0 = 0):

  $$\hat P = \sum_{i=1}^{35} \Big[C_{s_i,\alpha}\,\ln f_{s_i,\alpha}\big(\lVert g(x,w_{s_i})\rVert\big)\Big]\,(s_i-s_{i-1})$$

- **Non-integer o:** use the integer order ⌈o⌉ in the binomial sum. Rényi moments don't decrease with order, so this is conservative. Note the approximation in the README.
- **Overflow:** compute ln f entirely in log-space with `logsumexp` over k = 0..⌈o⌉, using log-binomial via `lgamma`. With σ = 0.1 the exponent g²(k²−k)/(2σ²) is huge, so never exponentiate directly.
- **Diagnostic:** also compute the full-batch-form ranking Σ_i w_{s_i} ‖g‖² (s_i − s_{i−1}) and report its Spearman correlation with the exact score. For large g/σ the two should nearly agree.

### 3.4 Per-example gradient norms (all parameters)

At each checkpoint, compute ‖∇_w ℓ(w_t, x)‖₂ for all 45,000 points.

- **R2D (PyTorch):** `torch.func.vmap(torch.func.grad(loss_fn))` in chunks of about 256, reducing to the norm right away. Do not store gradients. Alternative: Opacus `GradSampleModule`.
- **Model clipping (JAX/Flax):** `jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))` over chunks, wrapped in `jax.jit`, reducing to `optax.global_norm` per example. For penultimate features (§4.2), use Flax `capture_intermediates=True` or `sow`.
- **Settings:** eval mode, cross-entropy, **no augmentation**. GroupNorm has no batch statistics, so per-example gradients are well defined. If the model-clipping model uses BatchNorm, use eval-mode statistics and note it.
- **Output:** `grad_norms_<alg>.npy` with shape [35, 45000] in float32.

### 3.5 Cheap proxies, for comparison only

- `lastlayer_score`: ‖p_x − y_x‖₂ · ‖h_x‖₂ (the last-layer gradient norm, §4.2), aggregated like the score.
- `avg_gradnorm`: the plain mean of ‖g‖ over checkpoints.

Report the Spearman correlation of each proxy with P̂. Only P̂ is used for pool selection.

**Output:** `scores_<alg>.csv` with columns `global_idx, label, P_hat_a2, P_hat_a4, P_hat_a8, P_hat_a16, avg_gradnorm, lastlayer_score`.

---

## 4. Step 3: build forget pools and batch assignments (per algorithm)

### 4.1 Pools (4,500 points each, class-balanced: 45 per class)

| Pool | Rule |
|---|---|
| `hard` | Top 45 per class by P̂ |
| `easy` | Bottom 45 per class by P̂ |
| `random` | 45 per class, uniformly at random (fixed seed); can be shared across algorithms |

All pools in C1–C6 are class-balanced, so every comparison holds class composition fixed and isolates difficulty (C2/C3 vs. C1) or assignment (C4–C6 vs. C3). This keeps "hard points" separate from "hard classes", which the paper's adversarial split already covers. Report the Jaccard overlap between the R2D and model-clipping hard pools.

### 4.2 Gradient-direction representation (last layer)

For cross-entropy with a final linear layer, the last-layer gradient is

$$G_x=(p_x-y_x)\,h_x^\top\in\mathbb R^{C\times d}$$

with p = softmax output, y = one-hot label, h = penultimate features (input to the final `nn.Linear`). A forward pass is enough.

- At each checkpoint t: flatten G_x and normalize it to unit length.
- Direction vector: u_x = normalize(Σ_t ĝ_{x,t}). Alternative to report: the final checkpoint only.
- Memory: C·d floats per point (C = 100). If 4,500 × C·d is too large, use ⟨G_x, G_x'⟩ = (p_x−y_x)ᵀ(p_x'−y_x') · h_xᵀh_x' to build the 4,500 × 4,500 cosine matrix directly, or project with a fixed Gaussian matrix to 1,024 dimensions.
- Inspect each pipeline's TinyNet definition to locate the final linear layer and d.

### 4.3 Batch assignments (6 batches × 750, applied to a given pool)

| Assignment | Rule |
|---|---|
| `rand` | Random permutation, chunked into 6 × 750 (fixed seed) |
| `label` | **Superclass grouping.** CIFAR-100 has 20 superclasses × 5 classes, so the balanced pool has 225 points per superclass. Shuffle the superclass order with a fixed seed, sort the pool by (superclass order, class, score), and chunk into 6 × 750. Each batch holds 3–4 whole superclasses; at most one superclass is split across two batches. Coarse labels come from the original CIFAR-100 files (`coarse_labels`); check whether the split pickles keep them, and if not, map fine → coarse with the standard CIFAR-100 table and verify it on the original files. |
| `grad` | Balanced spherical k-means on u_x with k = 6, exactly 750 per cluster. Run spherical k-means (cosine), then equal-size assignment via `scipy.optimize.linear_sum_assignment` on the 4,500 × 4,500 cost matrix (each center repeated 750 times, cost = 1 − cosine). Iterate assignment → center update until stable (≤ 20 iterations). |
| `spread` (control) | Take the `grad` clusters and deal points round-robin so each batch gets 125 from each cluster. |

Fix and log all seeds. Save `assignment_<alg>_<pool>_<method>.csv` with columns `global_idx, batch_id`.

### 4.4 Diagnostics (per pool × assignment, before any audit)

Write `diagnostics_<alg>.json` with:

1. **Within-batch alignment:** ‖Σ_{x∈b} u_x‖ / |b| per batch. It is ≈ 1 for aligned batches and ≈ 0.037 (1/√750) for random directions.
2. **Across-batch orthogonality:** the 6 × 6 cosine matrix of batch-mean directions; report the mean off-diagonal absolute value.
3. **Retain masking:** for each batch, the mean cosine between the batch-mean direction and the k = 50 nearest retain directions (computed on a random 10,000-point retain subsample). High values mean the signal may be masked by similar retain data.
4. **Score summary:** mean, median and min of P̂ per batch, plus a class histogram per batch.
5. **Proxy agreement:** the Spearman correlations from §3.5.

---

## 5. Step 4: audit runs

### 5.1 Configuration matrix (run for **each** algorithm)

Pre-registered: report **every** row, never only the best.

| ID | Pool | Assignment | Purpose |
|---|---|---|---|
| C0 | paper uniform split (`--split_mode uniform --seed 1`) | paper's | reproduce the baseline |
| C1 | random | rand | our random baseline |
| C2 | easy | rand | lower reference |
| C3 | hard | rand | main hypothesis: does difficulty matter? |
| C4 | hard | label | grouping by superclass (semantic) |
| C5 | hard | grad | grouping by gradient direction |
| C6 | hard | spread | control: is direction meaningful? |

**Order:** follow the run order in §5.7.

### 5.2 Auditor: Instantiation II, two decoders

The paper uses Instantiation II for certified methods: 6 batches, balanced sign vector, conversion with Z = C(6,3) = 20, pure ε at δ = 0, ζ = 0.05.

- **Decoder B1, batchwise Gaussian (Alg. 2 style).** The repo's built-in audits fit per-point in/out Gaussians, rank batches by cumulative LLR, take the top 3 as deleted, and compute the overlap v:
  - R2D: `evaluate_prediction_cumulative_model.py`
  - Model clipping: `audit_utils.compute_*_bounds_for_all_runs_batch_pointwise` with k = 3, or `compute_eps_bounds_sampled_combos.py`

  Use these unchanged.
- **Decoder B2, joint sign-vector Gaussian (Alg. 3).** Fit one Gaussian per point **per S** (μ̂_x(S), v̂_x(S)) from calibration runs with known S. For a new run, compute Λ(S) = Σ_x log N(z_x; μ̂_x(S), v̂_x(S)) for all 20 S and guess Ŝ = argmax Λ(S). Feed the overlaps into the **same** `cum_runs_eps_lab` ε test.
  - The public repo's audits appear to implement only B1. The teammate working on pooled variance has a B2 implementation ("Original Joint"). **Ask the user for its path or the teammate's branch and reuse it.** Implement B2 under `forget_set_design/` only if it isn't available.
- Report both decoders. B2 is the primary result, because it matches the paper's certified-method audits.

### 5.3 R2D settings

Use the R2D defaults in §1, with the **same claimed ε grid for every configuration**: start with `--epsilon inf,1000,100` (inf = no noise, the upper reference).

- **Noise and calibration:** `--delta 1e-3`, `--num-shadow-reloads 100`.
- **Audit:** `--num-samples 500`, `--ci-delta 0.05`, `--metric phi`.
- **Report:** `epsilon_lb_avg_v` and `epsilon_lb_median_v`.

### 5.4 Model-clipping settings

Use the default `configs/exp_cifar100_chg7.yaml`, unchanged, for every configuration.

- **Runs:** `run_exhaustive_combinations_parallel.sh --config configs/exp_cifar100_chg7.yaml --data_dir <split> --forget_fraction 0.5 --num_unlearn_per_combo 35`. That's the script default; pass it explicitly. If the audit code doesn't already keep calibration and evaluation trials disjoint, split the 35 trials per combination into a calibration set and an evaluation set (§6 rule 3) and record the split.
- **Audit:** the README's default checkpoint, `unlearn_style="epoch"`, `unlearn_itr=5`, with k = 3.

### 5.5 Validating the score against real detectability (cheap, do early)

From the **C1** calibration runs of each algorithm, compute each forget point's empirical separation:

$$d_x = \frac{|\hat\mu_x^{+}-\hat\mu_x^{-}|}{\sqrt{(\hat v_x^{+}+\hat v_x^{-})/2}}$$

Report the Spearman correlation between d_x and P̂(x) over the 4,500 C1 points.
- **Weak or absent correlation:** the score doesn't track audit signal for that algorithm. Flag it before spending runs on C3–C6.
- **Use only as a check:** never select pool points from audit runs (§6 rule 1).

### 5.6 Outputs

Write `results.csv` with columns `algorithm, config_id, decoder, claimed_eps, eps_lb_avg, eps_lb_median, rho_lb, exact_match_rate, n_calib, n_eval, seed, ref_train_acc, ref_test_acc`. Produce one plot per algorithm and decoder: ε_LB vs. claimed ε (or vs. calibration budget), one line per configuration.

### 5.7 Run order

Do the whole sequence for **R2D first**. Its training is cheap (35 full-batch steps per combination), so it tests the full pipeline quickly. Then repeat it for **model clipping**, which is the expensive one (20 combinations × 400 epochs). Don't drop model clipping if R2D shows no effect, because its scores are likely more meaningful (§2.3).

0. **Setup:** write `make_custom_split.py` and check it against the paper's split format (§1).
1. **In parallel** (independent; the scores never use audit runs):
   - **Audits C0 and C1.** C0 checks the pipeline reproduces the paper. C1 is our baseline: run it with **3 seeds** to measure run-to-run noise in ε_LB.
   - **Reference training and difficulty scores** (§2–§3), plus the accuracy sanity check (§2.3).
2. **Score check (§5.5):** uses the C1 calibration runs and the scores. If the Spearman correlation is near zero for an algorithm, stop that algorithm here and report it.
3. **Audits C3, then C2.** C3 (hard pool, random assignment) is the main test; compare it with C1's seed-to-seed spread. C2 (easy pool) gives the lower reference; run it with C3 or right after.
4. **Only if C3 clearly beats C1** (difference larger than C1's run-to-run spread):
   - Build the `label`, `grad` and `spread` assignments and run the §4.4 diagnostics on them. This needs no audit runs.
   - Then run audits **C4, C5 and C6**.

Record in `forget_set_design/README.md` where each step stopped, and why, if a gate in step 2 or 4 fails.

---

## 6. Validity rules (do not violate)

1. Pools and assignments are fixed **before** any audit run, using only the reference runs in §2. Nothing about them may depend on audit outcomes or on the secret S.
2. The secret S (which 3 of 6 batches are deleted) must stay uniformly random over all 20 combinations. Do not alter the combination runners.
3. Calibration and evaluation must use disjoint randomness: separate noise reloads for R2D, separate unlearning trials for model clipping. Check that the evaluation scripts don't evaluate on draws used for fitting. If they do, flag it and add a disjoint split in `forget_set_design/`.
4. Report all configurations in §5.1. If a configuration is added after seeing results, either (a) rerun it with fresh seeds as a confirmatory run, or (b) apply a Bonferroni correction (`ci_delta = 0.05/k` for k configurations compared).
5. Within an algorithm, keep the claimed ε, δ, training recipe, number of runs and seeds the same across configurations. Only the split differs.
6. Claimed ε is nominal for R2D (G and L are estimated, not enforced). Interpret results as "the audit vs. the claimed ε".

---

## 7. Known risks and what to log

- **R2D under-training:** see §2.3. Expect a weaker or null effect on R2D, and always report the reference accuracy.
- **Interference:** hard points may interfere with each other (Dagréou et al. 2026). Watch whether `grad` beats `rand` on the hard pool.
- **Calibration stability:**
  - **Easy points:** they saturate φ, which can give near-zero variances and hit the variance floor. Log the per-point variance distribution per pool.
  - **Hard points:** they may make variances differ across S, which stresses the teammate's pooled-variance assumption. Log, per point, max_S v̂_x(S) / min_S v̂_x(S) for C1, C3 and C5 (B2 decoder).
- **Generalization masking:** see diagnostic 3 in §4.4.
- **Ceiling:** the perfect-prediction ε_LB ceiling at m = 6 is ≈ 4.0 (500 evaluation runs) to 4.7 (2,000). If a configuration reaches ~90% of it, note saturation; differences above that aren't meaningful.
- **Cost:**
  - Each configuration needs 20 training runs plus unlearning trials per algorithm.
  - Model-clipping training is 400 epochs per combination, the main cost.
  - Scores add two reference runs plus 35 per-example-gradient passes each.

---

## 8. Suggested file layout

```
forget_set_design/
├── configs/                    # copied training configs with checkpoint changes
├── make_custom_split.py        # §1
├── train_reference.py          # §2 (--alg r2d | modelclip)
├── compute_scores.py           # §3 -> grad_norms_<alg>.npy, scores_<alg>.csv
├── build_pools.py              # §4.1-4.3 -> assignment_*.csv, split dirs
├── diagnostics.py              # §4.4 -> diagnostics_<alg>.json
├── joint_decoder.py            # §5.2 B2, only if the teammate's version isn't available
├── validate_score.py           # §5.5
├── run_audits.sh               # §5 -> wraps the existing pipelines
├── collect_results.py          # §5.6 -> results.csv, plots
└── README.md                   # seeds, configs, commands actually run, open questions
```

Place `forget_set_design/` at the repository root of the fork, next to `r2d_audit/` and `certified_unlearning_plus_grad_based_methods/`. Import existing modules via `sys.path`; don't copy them.

**Open question to ask the user before running audits:** where is the teammate's joint (B2) decoder?

---

## 9. References

1. S. Sarmasarkar, A. Koloskova, S. Koyejo. *Auditing of Unlearning Algorithms.* arXiv:2607.05898, 2026. Instantiation II and joint sign-vector auditor (Alg. 3), batchwise auditor (Alg. 2), R2D and model clipping experiments (§7.1, App. D.1, G.2). Code: github.com/Sahasrajit123/audit-unlearning-code.
2. N. Sepahvand, A. Thudi, et al. *Leveraging Per-Instance Privacy for Machine Unlearning.* ICML 2025; arXiv:2505.18786. Score Def. 4.2, Thm. 4.3, Cor. 4.4; checkpoint estimation §5.1; proxies §5.3. No public code found.
3. S. Mu, D. Klabjan. *Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions.* arXiv:2409.09778. Thm. 3.1 (noise calibration).
4. A. Koloskova, Y. Allouah, et al. *Certified Unlearning for Neural Networks.* ICML 2025; arXiv:2506.06985. Model clipping, Eq. 4, Def. 2.1.
5. M. Dagréou, V. Dorseuil, M. Yaghini, N. Papernot, A. Bellet. *Detectability without Interference: Efficient Canary Design for Privacy Auditing in One Run.* arXiv:2605.27292. Self- vs. cross-influence canary selection; orthogonality reduces interference.
6. M. Boglioni, T. Liu, A. Ilyas, Z. S. Wu. *Optimizing Canaries for Privacy Auditing with Metagradient Descent.* arXiv:2507.15836. Optimized canaries beat random and mislabeled ones.
7. T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS 2023. Background on one-run auditing.
