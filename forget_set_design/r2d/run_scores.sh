#!/bin/bash
# This script calculates the difficulty scores for R2D (spec.md sections 2 and 3).
#
# The script does two steps:
#   1. train_reference.py trains the reference model on all 45,000 training points.
#      It saves the weights after each of the 35 steps.
#   2. compute_scores.py calculates the gradient norm of each point at each step,
#      then the difficulty scores.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_scores.sh
#
#   <gpu>  The physical GPU id. You select the GPU. The script does not examine it.
#          The script uses one GPU.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_scores.sh
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_scores.sh
#
# DRY_RUN=1 prints the commands and does not start them.
# The script skips each step that is complete. Use the same command to continue
# after an interruption. Set FORCE=1 to do all steps again.
#
# Output files:
#   $REF_DIR/checkpoints/step_000.pt to step_035.pt   The weights at each step.
#   $REF_DIR/reference_meta.json                      The settings and the accuracy values.
#   $REF_DIR/train.log, $REF_DIR/scores.log           The full output of each step.
#   $SCORE_DIR/grad_norms_r2d.npy                     The gradient norms [35, 45000].
#   $SCORE_DIR/scores_r2d.csv                         The scores of each point.
#   $SCORE_DIR/scores_r2d.json                        The checks of the scores.
#
# Settings are environment variables. The default values are in brackets.
#   ENV_BIN    Directory of the python environment.
#              [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   SPLIT      Name of the split directory. All split directories have the same
#              train/ files. [cifar100_uniform_bs_750_seed1]
#   REF_DIR    Directory for the reference model. [/tmp2/enzo/audit-unlearning/reference/r2d]
#   SCORE_DIR  Directory for the scores. [/tmp2/enzo/audit-unlearning/scores]
#   ALPHAS     List of Renyi orders. [2,4,8,16]
#   SIGMA      Noise level of the score. [0.1]
#   CHUNK      Number of points for one gradient calculation. [500]
#   THREADS    Number of CPU threads. [4]
set -euo pipefail

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
SPLIT=${SPLIT:-cifar100_uniform_bs_750_seed1}
REF_DIR=${REF_DIR:-/tmp2/enzo/audit-unlearning/reference/r2d}
SCORE_DIR=${SCORE_DIR:-/tmp2/enzo/audit-unlearning/scores}
ALPHAS=${ALPHAS:-2,4,8,16}
SIGMA=${SIGMA:-0.1}
CHUNK=${CHUNK:-500}
THREADS=${THREADS:-4}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

# These values are the same as the training phase of the audit runs (run_audits.sh).
TRAIN_EPOCHS=35
LR=0.01
MICRO_BS=128
SEED=3

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FSD="$REPO/forget_set_design/r2d"
SPLITS=$(realpath "$REPO/certified_unlearning_plus_grad_based_methods/data/cifar100/data_split")
DATA="$SPLITS/$SPLIT"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
GPUS=${CUDA_VISIBLE_DEVICES:-}
[ -n "$GPUS" ] || die "CUDA_VISIBLE_DEVICES is not set. Set it to the GPU id that you want to use."
[[ "$GPUS" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must be one GPU id. The value is '$GPUS'."
[ -x "$ENV_BIN/python3" ] || die "$ENV_BIN does not contain python3. Set ENV_BIN to the correct directory."
[ -d "$DATA/train" ] || die "The directory $DATA/train does not exist. Create the data split first."
[ -d "$SPLITS/cifar100_no_forget/train" ] || die "The directory $SPLITS/cifar100_no_forget/train does not exist."

export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
# Python then writes each line to the log file immediately.
export PYTHONUNBUFFERED=1
PY="$ENV_BIN/python3"

# Print a command and its log file. Start the command only if DRY_RUN is not 1.
# The full output goes to the log file. The terminal shows the lines with a [tag] only.
run_logged() {
    local log=$1; shift
    echo "+ $*"
    echo "  log: $log"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    mkdir -p "$(dirname "$log")"
    "$@" 2>&1 | tee "$log" | grep --line-buffered -E "^\[(reference|scores)\]|ERROR|Error|Traceback" || true
    [ "${PIPESTATUS[0]}" = "0" ] || die "The command failed. Read $log."
}

# ---------------------------------------------------------------- step 1
LAST_CKPT=$(printf "%s/checkpoints/step_%03d.pt" "$REF_DIR" "$TRAIN_EPOCHS")
echo "=== [1/2] Reference training: $TRAIN_EPOCHS steps on $DATA/train"
if [ "$FORCE" != "1" ] && [ -f "$LAST_CKPT" ] && [ -f "$REF_DIR/reference_meta.json" ]; then
    echo "The reference model is complete. The script skips this step."
else
    run_logged "$REF_DIR/train.log" "$PY" "$FSD/train_reference.py" \
        --alg r2d --data-dir "$DATA" --out-dir "$REF_DIR" \
        --epochs "$TRAIN_EPOCHS" --lr "$LR" --micro-batch-size "$MICRO_BS" --seed "$SEED" \
        --threads "$THREADS" --device cuda:0
fi

# ---------------------------------------------------------------- step 2
echo "=== [2/2] Difficulty scores: alpha = $ALPHAS, sigma = $SIGMA"
if [ "$FORCE" != "1" ] && [ -f "$SCORE_DIR/scores_r2d.csv" ] && [ -f "$SCORE_DIR/scores_r2d.json" ] \
        && [ "$SCORE_DIR/scores_r2d.json" -nt "$LAST_CKPT" ]; then
    echo "The scores are complete. The script skips this step."
else
    run_logged "$REF_DIR/scores.log" "$PY" "$FSD/compute_scores.py" \
        --alg r2d --ckpt-dir "$REF_DIR/checkpoints" --data-dir "$DATA" \
        --raw-train-dir "$SPLITS/cifar100_no_forget/train" --out-dir "$SCORE_DIR" \
        --alphas "$ALPHAS" --sigma "$SIGMA" --chunk "$CHUNK" \
        --threads "$THREADS" --device cuda:0
fi

echo "=== Complete."
echo "Reference model: $REF_DIR/reference_meta.json"
echo "Scores:          $SCORE_DIR/scores_r2d.csv"
echo "Checks:          $SCORE_DIR/scores_r2d.json"
