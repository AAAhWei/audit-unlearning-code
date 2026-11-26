# Forget-set design for unlearning audits

Work log for `spec.md`: commands actually run, seeds, and open questions.
All commands run from the repository root unless a `cd` is shown.

## Environment

- Host: Lugia. GPU 0 is reserved for another user; use one of GPUs 1–6, and only when free.
- Conda env: `/tmp2/enzo/miniconda3/envs/unlearn-audit` (Python 3.11).
  `r2d.yml` is not in the repository, so the env was built from
  `certified_unlearning_plus_grad_based_methods/requirements.txt` plus `matplotlib` and
  `jax[cuda12]`.
- Data lives on local scratch: `certified_unlearning_plus_grad_based_methods/data` is a
  symlink to `/tmp2/enzo/audit-unlearning/data` (Lugia only).

## Step 0 (§1): data and `make_custom_split.py`

Baseline split of the paper (configuration C0):

```bash
cd certified_unlearning_plus_grad_based_methods
mkdir -p /tmp2/enzo/audit-unlearning/data/cifar100
ln -s /tmp2/enzo/audit-unlearning/data ./data
cp /tmp2/dataset/cifar100/cifar-100-python.tar.gz ./data/cifar100/
python scripts/download_cifar100_batches.py
python scripts/split_cifar100_train.py --batch_size 750 --split_mode uniform --seed 1
```

Result: `data/cifar100/data_split/cifar100_uniform_bs_750_seed1/` with 45,000 train,
5,000 val, 40,500 retain and 4,500 forget points (6 batches of 750).

### `make_custom_split.py`

Writes a split directory from 4,500 global indices and a batch id (0–5) per index.

- **Global index** = row in the original CIFAR-100 train file (0–49,999).
- **Shared with the paper split:** the validation hold-out, `train/` and `val/` replay the
  random draws of `split_cifar100_train.py` for `--seed 1 --val_split 0.1`, so they are
  identical in every custom split.
- **Depends on the assignment:** `forget/batch_0000k.pkl` holds the points assigned to
  batch k, in assignment order. `retain/` holds the other 40,500 points, shuffled with
  `--seed`.
- **Extra file:** `split_indices.npz` in the split directory records the global index of
  every row. The pipelines ignore it; the scoring and pool-building steps need it because
  the batch pickles carry no indices and no superclass labels.

```bash
python forget_set_design/utils/make_custom_split.py --assignment <a.csv> --out_dir <split dir>
```

`<a.csv>` has columns `global_idx,batch_id`. `--paper_uniform` and `--random
--assign_seed N` build the assignment internally.

### Check (run on CPU)

```bash
D=certified_unlearning_plus_grad_based_methods/data/cifar100/data_split
REF=$D/cifar100_uniform_bs_750_seed1

python forget_set_design/utils/make_custom_split.py --paper_uniform --out_dir $D/_check_paper_replay
CUDA_VISIBLE_DEVICES="" python forget_set_design/utils/check_custom_split.py \
    --split $D/_check_paper_replay --reference $REF --expect_identical_forget

python forget_set_design/utils/make_custom_split.py --random --assign_seed 0 --out_dir $D/_check_random
CUDA_VISIBLE_DEVICES="" python forget_set_design/utils/check_custom_split.py \
    --split $D/_check_random --reference $REF
```

Both runs passed every check:

- File layout, tuple type, dtypes and shapes match the paper split in all five folders.
- `train/`, `val/` and `test/` are identical to the paper split, row for row.
- With `--paper_uniform`, all six forget files are identical to the paper's, and `retain/`
  holds the same points.
- Forget and retain are disjoint, together equal the training set, and contain no
  validation point.
- Every row equals the original CIFAR-100 image at its recorded global index.
- Readers: the model-clipping loader (`load_cifar_splits_with_batch_subset`) and the R2D
  runner's `build_train_pickle` both load the split.

The two test splits were deleted afterwards.

## R2D pipeline: four modules missing from the release

`r2d_audit/datasets.py` imports `lacuna`, `eicu`, `cifar_pickle` and
`cifar_presplit_loader`. None is in the paper's repository or its git history, so
`main.py` could not start. They were added in a separate commit:

- `lacuna.py`, `eicu.py`: copied unmodified from the original R2D repository
  (`github.com/siqiaomu/r2d`, commit `8e2534c`, MIT; licence in
  `r2d_audit/LICENSE_siqiaomu_r2d.md`). Only needed for the imports. `eicu.py` needs
  `pandas`, which was added to the env (`pip install pandas`).
- `cifar_pickle.py`, `cifar_presplit_loader.py`: **reconstructed**, since they exist in
  neither repository. The paper (arXiv 2607.05898) states nothing about input
  normalisation, augmentation or loader order, so each choice follows the released code:

| Choice | Evidence |
|---|---|
| Inputs are raw pixels: float32, channels first, 0–255, no normalisation | All three audit scripts in `evaluation_scripts/` feed the model `torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)` from the uint8 pickles; training must match |
| No augmentation | `main.py` hard-codes `augment=False` |
| Files in sorted name order, rows in file order; shuffle only with `--shuffle` | The run script defaults `SHUFFLE=false`; training is full-batch, so order only affects rounding |
| Items are `(image, label, identity)`; identity = batch index of the file | `main.py`, `r2d.py`, `mia.py` unpack three values; `r2d_audit/README.md` |
| Returned keys: `train_loader`, `valid_loader`, `test_loader`, `retain_loader`, `forget_loader`, `num_forget_ids` | The keys `main.py` reads |
| `num_forget_ids` = points under `<root>/forget` | The loader is not passed the chosen batches. Gives unlearn run names `..._forget_4500_..._bs_40500_...`, matching the example in `forget_phi_noisy_loader.py`. Only used in file names |
| No `forget_ids` key | `main.py` would overwrite the runner's `forget_ids.json` with it |

Checks: on CPU, loader output equals the audit scripts' tensors and the expected row
order for both the training-phase and unlearning-phase layouts. On GPU, one combination
ran end to end through the unchanged runner (2 train epochs + 1 unlearn epoch,
`--forget_batch_indices 0,2,4`, full batch, micro-batch 128, lr 0.01).

**Still a reconstruction.** The C0 run is the test: the paper reports a base accuracy of
about 4% for R2D (Appendix G.2). If C0 does not land near it, revisit normalisation first.

## Open questions

1. **Ask the authors for `cifar_pickle.py` and `cifar_presplit_loader.py`** (issue on
   `Sahasrajit123/audit-unlearning-code`) to confirm the reconstruction.
2. **Joint (B2) decoder:** where is the teammate's implementation? The fork has a branch
   `origin/pooled-linear-auditor-mvp` with a `pooled_linear_auditor/` folder; not yet
   inspected.

## Step 1 (§5.7): R2D audits C0 and C1

`run_audits.sh` does one configuration with the `r2d_audit/` pipeline. It does not change
the pipeline. You select the GPU with `CUDA_VISIBLE_DEVICES`. The script does not examine
the GPU.

```bash
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C0    # split of the paper
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C1    # pool seeds 0, 1, 2
```

- C1 uses a class-balanced random pool (45 points for each class) and a random
  assignment. The script creates each split directory with
  `make_custom_split.py --random --balanced --assign_seed <pool seed>`.
- The three C1 seeds are pool seeds. The pipeline script gives no seed to the runner,
  thus each training run uses the runner default (seed 3).
- Settings: claimed epsilon `inf,100,1000,2000`, delta `1e-3`, 100 calibration noise
  draws for each configuration, 500 evaluation trials, `ci-delta` 0.05, calibration
  seed 0, evaluation seed 1.
- Results: `/tmp2/enzo/audit-unlearning/r2d/<config>/sweep/sweep_table.csv`.

### Decoders

The branch is rebased on `origin/pooled-linear-auditor-mvp`, which adds
`pooled_linear_auditor/` (the joint decoder `b2` and the variants `ours_a`, `ours_b`).

- **The R2D audit script of the paper is the joint decoder (B2), not B1.** spec.md
  section 5.2 says that `evaluate_prediction_cumulative_model.py` is the batchwise
  decoder. That is not correct. The script fits one Gaussian for each point and each
  run, adds the log-likelihoods of all points, and selects the run with the largest
  sum. This is Algorithm 3 of the paper.
- **`calibration_sweep.py` is the adapter** between the R2D runs and
  `pooled_linear_auditor/pooled_linear.py`. It calculates the phi scores with the
  functions of `r2d_audit/`, and it uses the epsilon test in
  `r2d_audit/cum_runs_eps_lab.py`. `run_audits.sh` starts it in step 2.
- **Why the adapter and not the script of the paper:** the script of the paper keeps
  the first noisy evaluation of each run in a cache and uses it again. With 500 samples
  and 20 runs, it evaluates only 20 different noise draws, but the epsilon test assumes
  500 independent trials. The adapter makes a new noise draw for each trial. It also
  uses different generators for calibration and evaluation (§6 rule 3).
- `OFFICIAL_AUDIT=1` starts the audit scripts of the paper also, for comparison.
- `METHODS=b2,ours_a,ours_b` adds the two variants. They use the same scores.
- Decoder B1 (batchwise) is not implemented for R2D on this branch.

## Step 1 (§2–§3): R2D reference model and difficulty scores

```bash
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_scores.sh
```

- `train_reference.py` trains on all 45,000 training points with the settings of the
  audit runs (35 full-batch steps, learning rate 0.01, seed 3). It calls the pipeline's
  own `main.run_epoch` and saves the weights after each step. On the data of a C0 run,
  its weights at steps 0, 1 and 6 are identical to the checkpoints of `main.py`.
- `compute_scores.py` calculates the per-example gradient norm of all parameters at each
  step, then the closed-form score of spec.md §3.2. The formula agrees with
  Definition 4.2 of arXiv 2505.18786 (HTML version).
- Reference model accuracy: train 1.13%, val 1.10%, test 1.12% (chance level).
- Rankings at α = 2, 4, 8, 16 agree (Spearman > 0.9999). The pools use α = 2.
- The index of the order (o_t or o_(t−1) at step t) does not change the ranking.
- 83% of the variance of the score is between classes. The pools are class-balanced,
  thus they select on the other 17%.

## Step 2 (§5.5): score check — the gate fails for R2D

```bash
python3 forget_set_design/r2d/validate_score.py --audit-dir <C1 sweep dir> \
    --split-dir <C1 split dir> --scores-csv <scores_r2d.csv> --out-dir <dir>
```

Spearman correlation between the score (α = 2) and the separation d_x of each point,
all points / mean of the per-class values (standard error ≈ 0.02):

| claimed ε | C1 seed 0 | C1 seed 1 | C1 seed 2 |
|---|---|---|---|
| inf | +0.02 / −0.01 | 0.00 / 0.00 | +0.01 / +0.02 |
| 2000 | −0.01 / −0.04 | −0.01 / −0.03 | +0.02 / −0.01 |
| 1000 | −0.01 / −0.04 | −0.01 / −0.03 | +0.02 / −0.01 |
| 100 | 0.00 / −0.01 | −0.06 / −0.02 | +0.03 / −0.03 |

The correlation is near zero on all three pools. The per-point signal itself is
repeatable (seed 0, noise-free: split-half Spearman 0.67 between two halves of the 20
configurations). Spec.md §5.7 says to stop the algorithm at this gate. We decided to
run C3 and C2 one time each as the direct test, and not to build C4–C6 for R2D unless
C3 is above the C1 range.

R2D results so far (ε lower bound, avg test, joint B2 decoder, base accuracy ≈ 1%):

| claimed ε | C0 | C1 seed 0 | C1 seed 1 | C1 seed 2 |
|---|---|---|---|---|
| 100 | 0.05 | 0.02 | 0.01 | 0.06 |
| 1000 | 1.04 | 0.82 | 0.82 | 1.16 |
| 2000 | 1.93 | 1.73 | 1.78 | 2.03 |

## Step 3 (§5.7): R2D audits C3 and C2

```bash
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C3    # hard pool
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C2    # easy pool
```

- `build_pools.py` selects the 45 points with the highest (hard) or lowest (easy) score
  in each class and puts them into 6 batches at random (seed 0). It reads only the
  scores file. Only the `rand` assignment is available.
- Files: `data_split/assignment_r2d_<pool>_rand.csv` and `.json`, split directories
  `c3_r2d_hard_rand` and `c2_r2d_easy_rand`. Both pass `check_custom_split.py`.
- The hard pool and the easy pool have no common point. Each has approximately 450
  points in common with each C1 pool, as chance gives.

### C2 and C3 results, and the configurations added after them

ε lower bound (avg test, joint B2 decoder, batch assignment seed 0):

| claimed ε | C2 easy | C1 (min – max of 3 seeds) | C3 hard |
|---|---|---|---|
| 100 | 0.02 | 0.01 – 0.06 | 0.05 |
| 1000 | 0.77 | 0.82 – 1.16 | 1.17 |
| 2000 | 1.57 | 1.73 – 2.03 | 2.15 |

The order is easy < random < hard, but C3 is not above C1 by more than the C1 range.
We added these configurations after we saw these results (spec.md §6, rule 4 applies):

```bash
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C3_seed1   # also C3_seed2, C2_seed1, C2_seed2
CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C3.1
```

- `C2_seed<N>` and `C3_seed<N>` use the same pool points as C2 and C3. Only the random
  batch assignment changes (seed N). They give the spread of C2 and C3.
- `C3.1` uses the `global_hard` pool: the 4,500 highest scores of all points, with no
  class balance. The pool has points from 32 classes, from 1 to 446 points for each
  class. The batch assignment is random, thus each batch is a mix of these classes.
  A difference from C1 can come from the class concentration and not from the score.
  It is an exploratory configuration and it is not in spec.md.

### Results with all seeds, and C3.1

ε lower bound (avg test, joint B2 decoder). C2 and C3: batch assignment seeds 0 / 1 / 2
of a fixed pool. C1: pool seeds 0 / 1 / 2.

| claimed ε | C2 easy | C1 random | C3 hard | C3.1 global hard |
|---|---|---|---|---|
| 100 | 0.02 / 0.01 / 0.03 | 0.02 / 0.02 / 0.07 | 0.05 / 0.04 / 0.06 | 0.03 |
| 1000 | 0.77 / 0.78 / 0.82 | 0.82 / 0.82 / 1.16 | 1.17 / 1.18 / 1.14 | 0.43 |
| 2000 | 1.58 / 1.63 / 1.65 | 1.73 / 1.78 / 2.03 | 2.15 / 2.18 / 2.03 | 0.77 |

- Each C3 seed is above each C2 seed. The difference is approximately 0.5 at ε = 2000.
- The C3 mean is above the C1 mean (2.12 and 1.85 at ε = 2000), but C1 seed 2 is equal
  to the lowest C3 seed. Thus the gate of spec.md §5.7, step 4 is not clearly open.
  We continued with C4 to C6 because the direction of the effect is the same for all seeds.
- C3.1 is the lowest configuration. Its calibrated noise is approximately 1.4 times the
  noise of the other configurations (σ = 0.00050 and 0.00036 at ε = 2000).
- The result directory of C3.1 is `C3_1`. `main.py` cuts the checkpoint path at the
  first period, thus a directory name with a period stops the unlearn step.

## Step 4 (§4.2–§4.4, §5.7): assignments label, grad and spread (C4, C5, C6)

```bash
# Gradient directions of all training points. The CPU is sufficient (about 30 minutes).
CUDA_VISIBLE_DEVICES="" python3 forget_set_design/r2d/compute_directions.py --device cpu \
    --ckpt-dir /tmp2/$USER/audit-unlearning/reference/r2d/checkpoints \
    --data-dir <data_split>/cifar100_uniform_bs_750_seed1 \
    --out-dir /tmp2/$USER/audit-unlearning/scores

CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh C4   # also C5, C6

# Direction statistics of the assignments (spec.md §4.4). No audit results are necessary.
python3 forget_set_design/utils/diagnostics.py --out-dir /tmp2/$USER/audit-unlearning/scores \
    --scores-csv /tmp2/$USER/audit-unlearning/scores/scores_r2d.csv \
    --directions /tmp2/$USER/audit-unlearning/scores/directions_r2d.npy \
    <data_split>/assignment_r2d_hard_{rand,label,grad,spread}.csv
```

- `compute_directions.py` writes `directions_r2d.npy`: the direction u_x of the
  last-layer gradient (100 × 128 = 12,800 dimensions, no projection), for all points.
- `build_pools.py --assignment label|grad|spread` makes the assignments. C4, C5 and C6
  use the same 4,500 points as C3. Only the batch of each point changes.
- `label`: 750 is not a multiple of 225 (the points of one superclass). Thus 4 of the 5
  batch borders cut a superclass, and not 1 as spec.md §4.3 says.
