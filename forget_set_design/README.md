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
python forget_set_design/make_custom_split.py --assignment <a.csv> --out_dir <split dir>
```

`<a.csv>` has columns `global_idx,batch_id`. `--paper_uniform` and `--random
--assign_seed N` build the assignment internally.

### Check (run on CPU)

```bash
D=certified_unlearning_plus_grad_based_methods/data/cifar100/data_split
REF=$D/cifar100_uniform_bs_750_seed1

python forget_set_design/make_custom_split.py --paper_uniform --out_dir $D/_check_paper_replay
CUDA_VISIBLE_DEVICES="" python forget_set_design/check_custom_split.py \
    --split $D/_check_paper_replay --reference $REF --expect_identical_forget

python forget_set_design/make_custom_split.py --random --assign_seed 0 --out_dir $D/_check_random
CUDA_VISIBLE_DEVICES="" python forget_set_design/check_custom_split.py \
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

## Open questions

1. **R2D pipeline does not import.** `r2d_audit/datasets.py` imports `lacuna`, `eicu`,
   `cifar_pickle` and `cifar_presplit_loader`, and none of these files is in the
   repository (or its git history). `main.py` imports `datasets`, so R2D training cannot
   start until they are obtained from the authors or rewritten. Only the runner's
   pickle-building function could be checked against the split format.
2. **Joint (B2) decoder:** where is the teammate's implementation? The fork has a branch
   `origin/pooled-linear-auditor-mvp` with a `pooled_linear_auditor/` folder; not yet
   inspected.
