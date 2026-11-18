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
