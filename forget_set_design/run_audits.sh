#!/bin/bash
# This script does one audit configuration of spec.md section 5.
# It uses the r2d_audit/ pipeline and does not change that pipeline.
#
# The script does two steps for each configuration:
#   1. It trains and unlearns all 20 combinations of 3 forget batches from 6.
#   2. It audits the unlearned models with the joint sign-vector decoder (decoder B2).
#      joint_decoder_r2d.py does this step. It calculates the phi scores with noise
#      draws, predicts the deleted batches, and calculates the epsilon lower bounds.
#
# OFFICIAL_AUDIT=1 adds a third step. That step starts the two audit scripts of the
# pipeline (evaluate_combination_models.py, evaluate_prediction_cumulative_model.py).
# These scripts use the same joint decoder rule. But they use only one noise draw for
# each run in the evaluation, thus their evaluation samples are not independent.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/run_audits.sh <config>
#
#   <gpu>     The physical GPU id. You select the GPU. The script does not examine it.
#   <config>  C0          Use the uniform split directory of the paper.
#             C1          Use a class-balanced random pool for each pool seed 0, 1, 2.
#             C1_seed<N>  Use a class-balanced random pool for pool seed N only.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/run_audits.sh C0
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/run_audits.sh C1
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/run_audits.sh C1
#
# DRY_RUN=1 prints the commands and does not start them.
# The script skips each step that is complete. Use the same command to continue
# after an interruption.
# The script writes the results to $OUT_ROOT/<config>/joint/joint_summary.csv.
# With OFFICIAL_AUDIT=1, it also writes $OUT_ROOT/<config>/audit/rho_audit_summary.csv.
#
# Settings are environment variables. The default values are in brackets.
#   ENV_BIN      Directory of the python environment.
#                [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   OUT_ROOT     Root directory for the results. [/tmp2/enzo/audit-unlearning/r2d]
#   NUM_JOBS     Number of parallel runs on each GPU. [2]
#   EPSILONS     List of claimed epsilon values. [inf,100,1000,2000]
#   DELTA        Delta for the noise calibration. [1e-3]
#                The paper does not give its delta value.
#   RELOADS      Number of calibration noise draws for each model. [100]
#   NUM_SAMPLES  Number of evaluation trials in the audit. [500]
#   CI_DELTA     Confidence parameter of the lower bounds. [0.05]
#   METHODS      List of decoder methods: b2, ours_a, ours_b. [b2]
#                b2 is the decoder of the paper.
#   CALIB_SEED   Seed of the calibration noise. [0]
#   EVAL_SEED    Seed of the evaluation noise. [1]
#   OFFICIAL_AUDIT  Set to 1 to start the audit scripts of the pipeline also. [0]
#   THREADS      Number of CPU threads for each process. [4]
set -euo pipefail

CONFIG=${1:-}
if [ -z "$CONFIG" ]; then
    sed -n '2,51p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
OUT_ROOT=${OUT_ROOT:-/tmp2/enzo/audit-unlearning/r2d}
NUM_JOBS=${NUM_JOBS:-2}
EPSILONS=${EPSILONS:-inf,100,1000,2000}
DELTA=${DELTA:-1e-3}
RELOADS=${RELOADS:-100}
NUM_SAMPLES=${NUM_SAMPLES:-500}
CI_DELTA=${CI_DELTA:-0.05}
METHODS=${METHODS:-b2}
CALIB_SEED=${CALIB_SEED:-0}
EVAL_SEED=${EVAL_SEED:-1}
OFFICIAL_AUDIT=${OFFICIAL_AUDIT:-0}
THREADS=${THREADS:-4}
DRY_RUN=${DRY_RUN:-0}

# spec.md section 5.3 sets these values. All configurations use the same values.
# The paper (arXiv 2607.05898, Appendix G.2) uses T = 40 and K = 5 with full-batch
# updates and a learning rate of 0.01. T = 40 is 35 training epochs plus 5 unlearning
# epochs. The paper does not give the micro-batch size. 128 is the default of the
# pipeline script.
TRAIN_EPOCHS=35
UNLEARN_EPOCHS=5
LR=0.01
MICRO_BS=128
N_COMBOS=20
RUN_SUBDIR="bs_full_train_${TRAIN_EPOCHS}_unlearn_${UNLEARN_EPOCHS}_lr_${LR//./_}_comb3"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R2D_DIR="$REPO/r2d_audit"
SPLITS=$(realpath "$REPO/certified_unlearning_plus_grad_based_methods/data/cifar100/data_split")

N_STEPS=2
if [ "$OFFICIAL_AUDIT" = "1" ]; then N_STEPS=3; fi

die() { echo "ERROR: $*" >&2; exit 1; }

# Print a command. Start the command only if DRY_RUN is not 1.
run() {
    echo "+ $*"
    if [ "$DRY_RUN" != "1" ]; then "$@"; fi
}

# ---------------------------------------------------------------- checks
# The pipeline script needs the physical GPU ids as an argument.
# The script gets these ids from CUDA_VISIBLE_DEVICES.
GPUS=${CUDA_VISIBLE_DEVICES:-}
[ -n "$GPUS" ] || die "CUDA_VISIBLE_DEVICES is not set. Set it to the GPU id that you want to use."
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "CUDA_VISIBLE_DEVICES must be a list of GPU ids. The value is '$GPUS'."
[ -x "$ENV_BIN/python3" ] || die "$ENV_BIN does not contain python3. Set ENV_BIN to the correct directory."
[ -d "$SPLITS" ] || die "The directory $SPLITS does not exist. Create the data splits first."

export PATH="$ENV_BIN:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

# ---------------------------------------------------------------- one configuration
# Audit one configuration.
# Arguments: <name of the configuration> <split directory>
audit_one() {
    local NAME=$1 DATA=$2
    local BASE="$OUT_ROOT/$NAME"
    local RES="$BASE/$RUN_SUBDIR"
    [ -d "$DATA/forget" ] || [ "$DRY_RUN" = "1" ] || die "$DATA does not contain a forget directory."

    echo
    echo "=== $NAME  split=$DATA  gpus=$GPUS  host=$(hostname -s)  $(date '+%F %T')"
    [ "$DRY_RUN" = "1" ] || mkdir -p "$BASE"

    # Step 1: train and unlearn.
    # A run is complete when its directory contains run_metadata.json.
    local DONE=0
    if [ -d "$RES" ]; then
        DONE=$(find "$RES" -maxdepth 2 -name run_metadata.json | wc -l)
    fi
    if [ "$DONE" -ge "$N_COMBOS" ]; then
        echo "[1/$N_STEPS] All $N_COMBOS runs are complete. The script skips the training."
    else
        echo "[1/$N_STEPS] Train and unlearn. $DONE of $N_COMBOS runs are complete."
        if [ "$DONE" -gt 0 ]; then
            # Do only the runs that are not complete.
            # Argument 13 of the pipeline script selects one combination. The index starts at 1.
            for I in $(seq 1 "$N_COMBOS"); do
                if [ ! -f "$RES/run_$(printf '%03d' "$I")/run_metadata.json" ]; then
                    run ./run_all_combinations_forget_unlearning.sh "$DATA" "$BASE" \
                        cifar100 tinynetcifar100 "$GPUS" 1 full "$MICRO_BS" \
                        "$TRAIN_EPOCHS" "$UNLEARN_EPOCHS" false "$LR" "$I"
                fi
            done
        else
            run ./run_all_combinations_forget_unlearning.sh "$DATA" "$BASE" \
                cifar100 tinynetcifar100 "$GPUS" "$NUM_JOBS" full "$MICRO_BS" \
                "$TRAIN_EPOCHS" "$UNLEARN_EPOCHS" false "$LR"
        fi
        if [ "$DRY_RUN" != "1" ]; then
            DONE=$(find "$RES" -maxdepth 2 -name run_metadata.json | wc -l)
            [ "$DONE" -ge "$N_COMBOS" ] || die "Only $DONE of $N_COMBOS runs are complete. Read the logs in $RES/logs/."
            echo "Test accuracy of the trained models. The paper gives approximately 4% for R2D."
            grep -h "\[train\] final test accuracy" "$RES"/logs/run_*.log | sort | uniq -c
        fi
    fi

    # Step 2: joint sign-vector decoder (decoder B2).
    # An empty marker file records that this step is complete for these settings.
    local SETTINGS="eps${EPSILONS}_delta${DELTA}_calib${RELOADS}_eval${NUM_SAMPLES}"
    local JOINT_TAG="$BASE/joint/.done_${SETTINGS}_${METHODS}_seeds${CALIB_SEED}_${EVAL_SEED}"
    if [ -f "$JOINT_TAG" ]; then
        echo "[2/$N_STEPS] The joint decoder audit is complete. The script skips this step."
    else
        echo "[2/$N_STEPS] Audit with the joint decoder."
        run python3 "$REPO/forget_set_design/joint_decoder_r2d.py" \
            --results-dir "$RES" --data-dir "$DATA" --output-dir "$BASE/joint" \
            --epsilon "$EPSILONS" --delta "$DELTA" --num-calib "$RELOADS" \
            --num-eval "$NUM_SAMPLES" --methods "$METHODS" --ci-delta "$CI_DELTA" \
            --calib-seed "$CALIB_SEED" --eval-seed "$EVAL_SEED" \
            --config-id "$NAME" --device cuda:0
        if [ "$DRY_RUN" != "1" ]; then touch "$JOINT_TAG"; fi
    fi
    echo "=== $NAME: joint decoder results: $BASE/joint/joint_summary.csv"

    # Step 3 (only with OFFICIAL_AUDIT=1): the audit scripts of the pipeline.
    if [ "$OFFICIAL_AUDIT" = "1" ]; then
        echo "[3/$N_STEPS] Start the audit scripts of the pipeline."
        run python3 evaluation_scripts/evaluate_combination_models.py \
            --results-dir "$RES" --data-dir "$DATA" --model-type unlearnt \
            --epsilon "$EPSILONS" --delta "$DELTA" --num-shadow-reloads "$RELOADS" \
            --device cuda:0 --output-dir "$BASE/stats"
        run python3 evaluation_scripts/evaluate_prediction_cumulative_model.py \
            --results-dir "$RES" --data-dir "$DATA" --stats-dir "$BASE/stats" \
            --output-dir "$BASE/audit" --epsilon "$EPSILONS" --delta "$DELTA" \
            --metric phi --num-samples "$NUM_SAMPLES" --ci-delta "$CI_DELTA" \
            --num-shadow-reloads "$RELOADS" --no-generate-if-missing --device cuda:0
        run python3 collect_rho_audit.py --results-dir "$BASE/audit"
        echo "=== $NAME: pipeline audit results: $BASE/audit/rho_audit_summary.csv"
    fi
}

# Create the C1 split directory for one pool seed if it does not exist. Then audit it.
# Argument: <pool seed>
c1_one() {
    local SEED=$1
    local DATA="$SPLITS/c1_random_rand_seed$SEED"
    if [ -d "$DATA/forget" ]; then
        echo "The split directory $DATA exists. The script uses it."
    else
        run python3 "$REPO/forget_set_design/make_custom_split.py" \
            --random --balanced --assign_seed "$SEED" --out_dir "$DATA" \
            --save_assignment "$SPLITS/assignment_random_rand_seed$SEED.csv"
    fi
    audit_one "C1_seed$SEED" "$DATA"
}

cd "$R2D_DIR"
[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT_ROOT"
case "$CONFIG" in
    C0)        audit_one C0 "$SPLITS/cifar100_uniform_bs_750_seed1" ;;
    C1)        for S in 0 1 2; do c1_one "$S"; done ;;
    C1_seed*)  S=${CONFIG#C1_seed}
               [[ "$S" =~ ^[0-9]+$ ]] || die "The configuration '$CONFIG' is not correct. Use C1_seed<N> with a number N."
               c1_one "$S" ;;
    *)         die "The configuration '$CONFIG' is not known. Use C0, C1 or C1_seed<N>." ;;
esac
