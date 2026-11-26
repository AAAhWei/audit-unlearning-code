#!/bin/bash
# This script calculates the difficulty scores for model clipping (spec.md sections 2 and 3).
#
# The script does two steps:
#   1. It trains the reference model on all 45,000 training points. The training is that
#      of certified_unlearning_plus_grad_based_methods/, with all 6 forget batches in the
#      training set and no unlearning trial. It saves the weights after each epoch.
#   2. compute_scores.py calculates the gradient norm of each point at 35 checkpoints,
#      then the difficulty scores.
#   3. validate_score.py compares the scores with the audit signal of each forget point
#      of a complete audit with a random pool. The script skips this step if that audit
#      does not exist.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/modelclip/run_scores.sh
#
#   <gpu>  The physical GPU id. You select the GPU. The script does not examine it.
#          The script uses one GPU.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_scores.sh
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_scores.sh
#
# DRY_RUN=1 prints the commands and does not start them.
# The script skips each step that is complete. Use the same command to continue
# after an interruption. Set FORCE=1 to calculate the scores again.
# If the training stopped before its end, the script stops and gives the directory
# that you must delete first.
#
# Output files:
#   $REF_DIR/config_used.yaml               The config file of the reference model.
#   $REF_DIR/runs/<id>/ckpt/checkpoint_<e>  The weights after the epoch <e>.
#   $REF_DIR/runs/<id>/run_train.log        The output of the training.
#   $REF_DIR/scores.log                     The output of step 2.
#   $SCORE_DIR/grad_norms_modelclip.npy     The gradient norms [35, 45000].
#   $SCORE_DIR/scores_modelclip.csv         The scores of each point.
#   $SCORE_DIR/scores_modelclip.json        The checks of the scores.
#   $SCORE_DIR/validate_score_modelclip_<split>.json   The result of step 3.
#
# Time and disk space: the training is approximately 25 minutes. The checkpoints of the
# 400 epochs are approximately 0.8 GB.
#
# Settings are environment variables. The default values are in brackets.
#   ENV_BIN      Directory of the python environment.
#                [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   SPLITS       Directory of the split directories.
#                [/tmp2/enzo/audit-unlearning/data/cifar100/data_split]
#   SPLIT        Name of the split directory. All split directories have the same
#                training points. [cifar100_uniform_bs_750_seed1]
#   REF_DIR      Directory for the reference model.
#                [/tmp2/enzo/audit-unlearning/reference/modelclip]
#   SCORE_DIR    Directory for the scores. [/tmp2/enzo/audit-unlearning/scores]
#   BASE_CONFIG  Config file of the pipeline. This script changes only the checkpoints
#                to keep and the log directory. Use the config file of the audit runs.
#                [certified_unlearning_plus_grad_based_methods/configs/exp_cifar100_chg7.yaml]
#   NUM_CKPT     Number of checkpoints for the scores. [35]
#   ALPHAS       List of Renyi orders. [2,4,8,16]
#   SIGMA        Noise level of the score. [0.1]
#   CHUNK        Number of points for one gradient calculation. [500]
#   CHECK_AUDIT  Audit directory for step 3. [/tmp2/enzo/audit-unlearning/modelclip/MC1/audit]
#   CHECK_SPLIT  Name of the split directory of that audit. [c1_random_rand_seed0]
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PIPELINE="$REPO/certified_unlearning_plus_grad_based_methods"

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
SPLITS=${SPLITS:-/tmp2/enzo/audit-unlearning/data/cifar100/data_split}
SPLIT=${SPLIT:-cifar100_uniform_bs_750_seed1}
REF_DIR=${REF_DIR:-/tmp2/enzo/audit-unlearning/reference/modelclip}
SCORE_DIR=${SCORE_DIR:-/tmp2/enzo/audit-unlearning/scores}
BASE_CONFIG=${BASE_CONFIG:-$PIPELINE/configs/exp_cifar100_chg7.yaml}
NUM_CKPT=${NUM_CKPT:-35}
ALPHAS=${ALPHAS:-2,4,8,16}
SIGMA=${SIGMA:-0.1}
CHUNK=${CHUNK:-500}
CHECK_AUDIT=${CHECK_AUDIT:-/tmp2/enzo/audit-unlearning/modelclip/MC1/audit}
CHECK_SPLIT=${CHECK_SPLIT:-c1_random_rand_seed0}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

DATA="$SPLITS/$SPLIT"
RUNS="$REF_DIR/runs"
USED="$REF_DIR/config_used.yaml"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
GPUS=${CUDA_VISIBLE_DEVICES:-}
[ -n "$GPUS" ] || die "CUDA_VISIBLE_DEVICES is not set. Set it to the GPU id that you want to use."
[[ "$GPUS" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must be one GPU id. The value is '$GPUS'."
[ -x "$ENV_BIN/python3" ] || die "$ENV_BIN does not contain python3. Set ENV_BIN to the correct directory."
[ -f "$BASE_CONFIG" ] || die "The config file $BASE_CONFIG does not exist."
for PART in train retain forget test; do
    [ -d "$DATA/$PART" ] || die "The directory $DATA/$PART does not exist. Create the data split first."
done
[ -d "$SPLITS/cifar100_no_forget/train" ] || die "The directory $SPLITS/cifar100_no_forget/train does not exist."

export PATH="$ENV_BIN:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
# JAX then allocates GPU memory when it needs the memory, not all at the start.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# These flags are the same as in run_audits.sh.
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_strict_conv_algorithm_picker=false"
export JAX_DISABLE_MOST_FASTER_PATHS=1

echo "=== difficulty scores for model clipping  split=$DATA  gpu=$GPUS  host=$(hostname -s)  $(date '+%F %T')"

# ---------------------------------------------------------------- config file
# The pipeline config, with a checkpoint after each epoch and the log directory.
make_config() {
    python3 - "$BASE_CONFIG" "$1" "$REF_DIR/wandb" <<'EOF'
import sys
import yaml

source, target, log_dir = sys.argv[1:4]
with open(source) as f:
    config = yaml.safe_load(f)
config["training"]["checkpoint_save_every_n_epochs"] = 1
config["wandb"]["log_dir"] = log_dir
with open(target, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
EOF
}

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$RUNS" "$REF_DIR/wandb"
    make_config "$USED.new"
    if [ -f "$USED" ] && ! cmp -s "$USED" "$USED.new"; then
        rm -f "$USED.new"
        die "$USED has different settings. Use the same settings, or use a different REF_DIR."
    fi
    mv "$USED.new" "$USED"
fi
EPOCHS=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["epochs"])' "$BASE_CONFIG")

# ---------------------------------------------------------------- step 1
# Print the run directories of the reference model.
run_dirs() {
    [ -d "$RUNS" ] || return 0
    find "$RUNS" -mindepth 2 -maxdepth 2 -name run_vars.json -printf '%h\n' | sort
}

echo "=== [1/3] Reference training: $EPOCHS epochs on the retain set and all forget batches of $DATA"
RUN_DIR=$(run_dirs)
if [ -n "$RUN_DIR" ] && [ "$(echo "$RUN_DIR" | wc -l)" = "1" ] && [ ! -f "$RUNS/_run_0.dir" ] &&
   [ -d "$RUN_DIR/ckpt/checkpoint_$EPOCHS" ]; then
    echo "The reference model is complete. The script skips this step."
elif [ -n "$RUN_DIR" ] || [ -f "$RUNS/_run_0.dir" ]; then
    die "$RUNS has a run that is not complete. Delete $RUNS and start the script again."
else
    # --forget_fraction 1.0 puts all forget batches into the training set. Thus the
    # model trains on all training points. --num_unlearn_per_combo 0 does no unlearning.
    CMD=(python3 experiment_unlearning_exhaustive_combinations.py
         --config "$USED" --results_dir "$RUNS" --combo_idx 0
         --forget_fraction 1.0 --data_dir "$DATA" --num_unlearn_per_combo 0
         --shuffle_train_samples)
    echo "+ (cd $PIPELINE && ${CMD[*]})"
    echo "  log: $REF_DIR/train.out and run_train.log in the run directory"
    if [ "$DRY_RUN" != "1" ]; then
        START=$SECONDS
        (cd "$PIPELINE" && "${CMD[@]}" > "$REF_DIR/train.out" 2>&1) ||
            die "The training failed. Read $REF_DIR/train.out and run_train.log in the run directory."
        RUN_DIR=$(run_dirs)
        [ -d "$RUN_DIR/ckpt/checkpoint_$EPOCHS" ] || die "The training wrote no checkpoint_$EPOCHS in $RUN_DIR/ckpt."
        echo "    the training is complete: $(((SECONDS - START) / 60)) min"
        grep -h "Training took\|Test_accuracy" "$RUN_DIR/run_train.log" || true
    fi
fi

# ---------------------------------------------------------------- step 2
echo "=== [2/3] Difficulty scores: $NUM_CKPT checkpoints, alpha = $ALPHAS, sigma = $SIGMA"
if [ "$DRY_RUN" = "1" ] && [ -z "$RUN_DIR" ]; then RUN_DIR="$RUNS/<id>"; fi
if [ "$FORCE" != "1" ] && [ -f "$SCORE_DIR/scores_modelclip.csv" ] && [ -f "$SCORE_DIR/scores_modelclip.json" ] &&
   [ "$SCORE_DIR/scores_modelclip.json" -nt "$RUN_DIR/ckpt/checkpoint_$EPOCHS" ]; then
    echo "The scores are complete. The script skips this step."
else
    CMD=(python3 "$REPO/forget_set_design/modelclip/compute_scores.py"
         --run-dir "$RUN_DIR" --data-dir "$DATA"
         --raw-train-dir "$SPLITS/cifar100_no_forget/train" --out-dir "$SCORE_DIR"
         --num-checkpoints "$NUM_CKPT" --alphas "$ALPHAS" --sigma "$SIGMA" --chunk "$CHUNK")
    echo "+ ${CMD[*]}"
    echo "  log: $REF_DIR/scores.log"
    if [ "$DRY_RUN" != "1" ]; then
        "${CMD[@]}" 2>&1 | tee "$REF_DIR/scores.log" | grep --line-buffered -E "^\[scores\]|ERROR|Error|Traceback" || true
        [ "${PIPESTATUS[0]}" = "0" ] || die "The command failed. Read $REF_DIR/scores.log."
    fi
fi

# ---------------------------------------------------------------- step 3
echo "=== [3/3] Check: the scores and the audit signal of $CHECK_AUDIT"
if ! ls "$CHECK_AUDIT"/scores_step*.npz > /dev/null 2>&1; then
    echo "$CHECK_AUDIT contains no scores_step<s>.npz file. The script skips this step."
else
    CMD=(python3 "$REPO/forget_set_design/modelclip/validate_score.py"
         --audit-dir "$CHECK_AUDIT" --split-dir "$SPLITS/$CHECK_SPLIT" --scores-csv "$SCORE_DIR/scores_modelclip.csv"
         --raw-train-dir "$SPLITS/cifar100_no_forget/train" --out-dir "$SCORE_DIR")
    echo "+ ${CMD[*]}"
    if [ "$DRY_RUN" != "1" ]; then
        # This command uses the CPU only.
        CUDA_VISIBLE_DEVICES="" "${CMD[@]}" 2>&1 | grep -v "P_hat_a\(4\|8\|16\)" || true
    fi
fi

echo "=== Complete."
echo "Reference model: $RUN_DIR"
echo "Scores:          $SCORE_DIR/scores_modelclip.csv"
echo "Checks:          $SCORE_DIR/scores_modelclip.json"
