#!/bin/bash
# This script does one audit configuration of spec.md section 5.
# It uses the r2d_audit/ pipeline and does not change that pipeline.
#
# The script does two steps for each configuration:
#   1. It trains and unlearns all 20 combinations of 3 forget batches from 6.
#   2. It audits the unlearned models with the joint sign-vector decoder (decoder B2).
#      joint_decoder.py does this step. It calculates the phi scores with noise
#      draws, predicts the deleted batches, and calculates the epsilon lower bounds.
#
# OFFICIAL_AUDIT=1 adds a third step. That step starts the two audit scripts of the
# pipeline (evaluate_combination_models.py, evaluate_prediction_cumulative_model.py).
# These scripts use the same joint decoder rule. But they use only one noise draw for
# each run in the evaluation, thus their evaluation samples are not independent.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_audits.sh <config>
#
#   <gpu>     The physical GPU id. You select the GPU. The script does not examine it.
#   <config>  C0          Use the uniform split directory of the paper.
#             C1          Use a class-balanced random pool for each pool seed 0 to 5.
#             C1_seed<N>  Use a class-balanced random pool for pool seed N only.
#             C2          Use the easy pool: the lowest difficulty scores in each class.
#             C3          Use the hard pool: the highest difficulty scores in each class.
#                         C2 and C3 do the batch assignment seeds 0, 1 and 2.
#             C3.1        Use the global hard pool: the highest difficulty scores of
#                         all points. The classes do not have the same number of
#                         points. This configuration is not in spec.md.
#                         C3.1 does the batch assignment seed 0 only.
#             C4          Use the hard pool. Put the points of the same superclass
#                         into the same batch (assignment label).
#             C5          Use the hard pool. Put the points with a similar gradient
#                         direction into the same batch (assignment grad).
#             C6          Use the hard pool. Give each batch the same number of points
#                         from each gradient cluster (assignment spread).
#                         C4, C5 and C6 do the seed 0 only. C5 and C6 need the
#                         directions file from compute_directions.py.
#             C4.1, C5.1  Use the random pool of C1_seed<POOL_SEED> with the
#                         assignment of C4 (label) or C5 (grad).
#             C4.2, C5.2  Use the easy pool of C2 with the assignment of C4 or C5.
#                         These four configurations are not in spec.md. They show if
#                         the result of C4 and C5 needs the hard pool. They do the
#                         seed 0 only.
#             <config>_seed<N>
#                         For C2 to C6 and their variants: use the same pool with the
#                         batch assignment seed N only.
#                         The results of the seed 0 have the names C2, C3 and C3_1.
#                         The directory names use C3_1 because a period in the path
#                         stops main.py.
#                         These configurations need the scores file of run_scores.sh.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_audits.sh C0
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_audits.sh C1
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_audits.sh C1
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
#   SCORES_CSV   Scores file for C2 to C6.
#                [/tmp2/enzo/audit-unlearning/scores/scores_r2d.csv]
#   DIRECTIONS   Directions file for C5, C5.1, C5.2 and C6.
#                [/tmp2/enzo/audit-unlearning/scores/directions_r2d.npy]
#   POOL_SEED    Pool seed of the C1 pool that C4.1 and C5.1 use. [0]
set -euo pipefail

CONFIG=${1:-}
if [ -z "$CONFIG" ]; then
    sed -n '2,84p' "$0" | sed 's/^# \{0,1\}//'
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
SCORES_CSV=${SCORES_CSV:-/tmp2/enzo/audit-unlearning/scores/scores_r2d.csv}
DIRECTIONS=${DIRECTIONS:-/tmp2/enzo/audit-unlearning/scores/directions_r2d.npy}
POOL_SEED=${POOL_SEED:-0}
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

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
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
    [[ "$BASE" != *.* ]] || die "The output directory $BASE contains a period. main.py cannot use this path."
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
        run python3 "$REPO/forget_set_design/r2d/joint_decoder.py" \
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
        run python3 "$REPO/forget_set_design/utils/make_custom_split.py" \
            --random --balanced --assign_seed "$SEED" --out_dir "$DATA" \
            --save_assignment "$SPLITS/assignment_random_rand_seed$SEED.csv"
    fi
    audit_one "C1_seed$SEED" "$DATA"
}

# Create the split directory for a pool from the difficulty scores if it does not
# exist. Then audit it.
# Arguments: <configuration without the seed, for example C3> <assignment seed>
# The seed 0 uses the names without a seed. Thus the first C2 and C3 runs keep their names.
pool_one() {
    local CONF=$1 SEED=$2
    local POOL ASSIGN
    POOL=$(pool_of "$CONF")
    ASSIGN=$(assignment_of "$CONF")
    local NAME=$CONF SUFFIX=""
    if [ "$SEED" != "0" ]; then NAME="${CONF}_seed$SEED"; SUFFIX="_seed$SEED"; fi
    local TAG="r2d_${POOL}_${ASSIGN}$SUFFIX"
    local EXTRA=()
    if [ "$ASSIGN" = "grad" ] || [ "$ASSIGN" = "spread" ]; then EXTRA=(--directions "$DIRECTIONS"); fi
    if [ "$POOL" = "random" ]; then
        # The pool is the C1 pool. Thus only the batch of each point is different from C1.
        local POOL_FILE="$SPLITS/assignment_random_rand_seed$POOL_SEED.csv"
        [ -f "$POOL_FILE" ] || [ "$DRY_RUN" = "1" ] \
            || die "The pool file $POOL_FILE does not exist. Start C1_seed$POOL_SEED first."
        EXTRA+=(--pool-file "$POOL_FILE")
        if [ "$POOL_SEED" != "0" ]; then
            NAME="${NAME}_pool$POOL_SEED"; TAG="r2d_${POOL}${POOL_SEED}_${ASSIGN}$SUFFIX"
        fi
    fi
    # The directory names must not contain a period. main.py cuts the checkpoint path
    # at the first period to make the name of the unlearn run. Thus C3.1 uses C3_1.
    NAME=${NAME//./_}
    local DATA="$SPLITS/$(echo "${CONF//./_}" | tr '[:upper:]' '[:lower:]')_$TAG"
    if [ -d "$DATA/forget" ]; then
        echo "The split directory $DATA exists. The script uses it."
    else
        [ -f "$SCORES_CSV" ] || die "The scores file $SCORES_CSV does not exist. Start run_scores.sh first."
        [ "$ASSIGN" != "grad" ] && [ "$ASSIGN" != "spread" ] || [ -f "$DIRECTIONS" ] \
            || die "The directions file $DIRECTIONS does not exist. Start compute_directions.py first."
        run python3 "$REPO/forget_set_design/utils/build_pools.py" --alg r2d --pool "$POOL" \
            --assignment "$ASSIGN" --assign-seed "$SEED" --out-stem "assignment_$TAG" \
            --scores-csv "$SCORES_CSV" --out-dir "$SPLITS" "${EXTRA[@]}"
        run python3 "$REPO/forget_set_design/utils/make_custom_split.py" \
            --assignment "$SPLITS/assignment_$TAG.csv" --out_dir "$DATA"
    fi
    audit_one "$NAME" "$DATA"
}

# Give the pool of a configuration: C2 to C6.
pool_of() {
    case "$1" in
        C2|C4.2|C5.2) echo easy ;;
        C3|C4|C5|C6)  echo hard ;;
        C4.1|C5.1)    echo random ;;
        C3.1)        echo global_hard ;;
    esac
}

# Give the batch assignment of a configuration: C2 to C6.
assignment_of() {
    case "$1" in
        C4|C4.1|C4.2) echo label ;;
        C5|C5.1|C5.2) echo grad ;;
        C6) echo spread ;;
        *)  echo rand ;;
    esac
}

cd "$R2D_DIR"
[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT_ROOT"
case "$CONFIG" in
    C0)        audit_one C0 "$SPLITS/cifar100_uniform_bs_750_seed1" ;;
    C1)        for S in 0 1 2 3 4 5; do c1_one "$S"; done ;;
    C1_seed*)  S=${CONFIG#C1_seed}
               [[ "$S" =~ ^[0-9]+$ ]] || die "The configuration '$CONFIG' is not correct. Use C1_seed<N> with a number N."
               c1_one "$S" ;;
    C2|C3)     for S in 0 1 2; do pool_one "$CONFIG" "$S"; done ;;
    C3.1|C4|C5|C6|C4.1|C4.2|C5.1|C5.2)
               pool_one "$CONFIG" 0 ;;
    C[2-6]_seed*|C3.1_seed*|C[45].[12]_seed*)
               S=${CONFIG#*_seed}
               [[ "$S" =~ ^[0-9]+$ ]] || die "The configuration '$CONFIG' is not correct. Use <config>_seed<N> with a number N."
               pool_one "${CONFIG%_seed*}" "$S" ;;
    *)         die "The configuration '$CONFIG' is not known. Use C0 to C6, C3.1, C4.1, C4.2, C5.1, C5.2 or <config>_seed<N>." ;;
esac
