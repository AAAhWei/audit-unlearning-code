#!/bin/bash
# Run one audit configuration of spec.md section 5 end to end, with the unchanged
# r2d_audit/ pipeline:
#   1. train + unlearn all 20 combinations of 3 of 6 forget batches
#   2. phi statistics with noise reloads
#   3. epsilon lower bounds (decoder B1) and a summary CSV
#
# Usage:
#   forget_set_design/run_audits.sh <config> <gpu>
#
#   <config>  C0          the paper's uniform split
#             C1          class-balanced random pool, pool seeds 0, 1 and 2 in turn
#             C1_seed<N>  one of them
#   <gpu>     ONE physical GPU id. Check nvidia-smi first; the script refuses a
#             reserved GPU and a GPU running another user's process.
#
# Examples:
#   forget_set_design/run_audits.sh C0 4
#   forget_set_design/run_audits.sh C1 4
#   DRY_RUN=1 forget_set_design/run_audits.sh C1 4      # print the commands only
#
# Finished steps are skipped, so the same command resumes an interrupted run.
# Results: $OUT_ROOT/<config>/audit/rho_audit_summary.csv
#
# Settings (environment variables, defaults in brackets):
#   ENV_BIN      python environment [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   OUT_ROOT     results root on scratch [/tmp2/enzo/audit-unlearning/r2d]
#   NUM_JOBS     parallel runs on the GPU [2]
#   EPSILONS     claimed epsilon grid [inf,1000,100]
#   DELTA        noise delta [1e-3]
#   RELOADS      noise reloads per model [100]
#   NUM_SAMPLES  sampled runs in the audit [500]
#   CI_DELTA     confidence slack [0.05]
#   THREADS      CPU threads per process [4]
set -euo pipefail

CONFIG=${1:-}
GPU=${2:-}
if [ -z "$CONFIG" ] || [ -z "$GPU" ]; then
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
OUT_ROOT=${OUT_ROOT:-/tmp2/enzo/audit-unlearning/r2d}
NUM_JOBS=${NUM_JOBS:-2}
EPSILONS=${EPSILONS:-inf,1000,100}
DELTA=${DELTA:-1e-3}
RELOADS=${RELOADS:-100}
NUM_SAMPLES=${NUM_SAMPLES:-500}
CI_DELTA=${CI_DELTA:-0.05}
THREADS=${THREADS:-4}
DRY_RUN=${DRY_RUN:-0}

# Fixed by the spec (section 5.3): R2D defaults, same for every configuration.
TRAIN_EPOCHS=35
UNLEARN_EPOCHS=5
LR=0.01
MICRO_BS=128
N_COMBOS=20
RUN_SUBDIR="bs_full_train_${TRAIN_EPOCHS}_unlearn_${UNLEARN_EPOCHS}_lr_${LR//./_}_comb3"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R2D_DIR="$REPO/r2d_audit"
SPLITS=$(realpath "$REPO/certified_unlearning_plus_grad_based_methods/data/cifar100/data_split")

die() { echo "ERROR: $*" >&2; exit 1; }
run() {
    echo "+ $*"
    if [ "$DRY_RUN" != "1" ]; then "$@"; fi
}

# ---------------------------------------------------------------- checks
[[ "$GPU" =~ ^[0-9]+$ ]] || die "<gpu> must be one GPU id, got '$GPU'"
[ -x "$ENV_BIN/python3" ] || die "no python3 under $ENV_BIN (set ENV_BIN)"
[ -d "$SPLITS" ] || die "split directory not found: $SPLITS"

# Shared-server GPU policy.
HOST=$(hostname -s)
case "$HOST" in
    Celebi) die "all GPUs on Celebi are reserved; run on another host" ;;
    Mew)    [ "$GPU" != "3" ] || die "GPU 3 on Mew is reserved" ;;
    Lugia)  [ "$GPU" != "0" ] || die "GPU 0 on Lugia is reserved" ;;
esac

nvidia-smi -i "$GPU" > /dev/null 2>&1 || die "GPU $GPU not found on $HOST"
OTHERS=""
for PID in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    OWNER=$(ps -o user= -p "$PID" 2>/dev/null | tr -d ' ' || true)
    if [ "$OWNER" != "$(id -un)" ]; then
        OTHERS="$OTHERS ${OWNER:-unknown}(pid $PID)"
    fi
done
[ -z "$OTHERS" ] || die "GPU $GPU is in use by:$OTHERS. Pick a free GPU."

export PATH="$ENV_BIN:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

# ---------------------------------------------------------------- one configuration
audit_one() {    # audit_one <name> <split dir>
    local NAME=$1 DATA=$2
    local BASE="$OUT_ROOT/$NAME"
    local RES="$BASE/$RUN_SUBDIR"
    [ -d "$DATA/forget" ] || [ "$DRY_RUN" = "1" ] || die "no forget/ under $DATA"

    echo
    echo "=== $NAME  split=$DATA  gpu=$GPU  host=$HOST  $(date '+%F %T')"
    [ "$DRY_RUN" = "1" ] || mkdir -p "$BASE"

    # 1. train + unlearn
    local DONE=0
    if [ -d "$RES" ]; then
        DONE=$(find "$RES" -maxdepth 2 -name run_metadata.json | wc -l)
    fi
    if [ "$DONE" -ge "$N_COMBOS" ]; then
        echo "[1/3] $DONE/$N_COMBOS runs already complete, skipping training"
    else
        echo "[1/3] train + unlearn ($DONE/$N_COMBOS runs complete so far)"
        if [ "$DONE" -gt 0 ]; then
            # Rerun only the missing combinations (argument 13 = 1-based index).
            for I in $(seq 1 "$N_COMBOS"); do
                if [ ! -f "$RES/run_$(printf '%03d' "$I")/run_metadata.json" ]; then
                    run ./run_all_combinations_forget_unlearning.sh "$DATA" "$BASE" \
                        cifar100 tinynetcifar100 "$GPU" 1 full "$MICRO_BS" \
                        "$TRAIN_EPOCHS" "$UNLEARN_EPOCHS" false "$LR" "$I"
                fi
            done
        else
            run ./run_all_combinations_forget_unlearning.sh "$DATA" "$BASE" \
                cifar100 tinynetcifar100 "$GPU" "$NUM_JOBS" full "$MICRO_BS" \
                "$TRAIN_EPOCHS" "$UNLEARN_EPOCHS" false "$LR"
        fi
        if [ "$DRY_RUN" != "1" ]; then
            DONE=$(find "$RES" -maxdepth 2 -name run_metadata.json | wc -l)
            [ "$DONE" -ge "$N_COMBOS" ] || die "only $DONE/$N_COMBOS runs finished; see $RES/logs/"
            echo "Accuracy of the trained models (paper reports about 4% for R2D):"
            grep -h "\[train\] final test accuracy" "$RES"/logs/run_*.log | sort | uniq -c
        fi
    fi

    # 2. phi statistics
    local STATS_TAG="$BASE/stats/.done_eps${EPSILONS}_delta${DELTA}_reloads${RELOADS}"
    if [ -f "$STATS_TAG" ]; then
        echo "[2/3] phi statistics already computed, skipping"
    else
        echo "[2/3] phi statistics"
        CUDA_VISIBLE_DEVICES=$GPU run python3 evaluation_scripts/evaluate_combination_models.py \
            --results-dir "$RES" --data-dir "$DATA" --model-type unlearnt \
            --epsilon "$EPSILONS" --delta "$DELTA" --num-shadow-reloads "$RELOADS" \
            --device cuda:0 --output-dir "$BASE/stats"
        if [ "$DRY_RUN" != "1" ]; then touch "$STATS_TAG"; fi
    fi

    # 3. lower bounds
    echo "[3/3] epsilon lower bounds"
    CUDA_VISIBLE_DEVICES=$GPU run python3 evaluation_scripts/evaluate_prediction_cumulative_model.py \
        --results-dir "$RES" --data-dir "$DATA" --stats-dir "$BASE/stats" \
        --output-dir "$BASE/audit" --epsilon "$EPSILONS" --delta "$DELTA" \
        --metric phi --num-samples "$NUM_SAMPLES" --ci-delta "$CI_DELTA" \
        --num-shadow-reloads "$RELOADS" --no-generate-if-missing --device cuda:0

    run python3 collect_rho_audit.py --results-dir "$BASE/audit"
    echo "=== $NAME done: $BASE/audit/rho_audit_summary.csv"
}

c1_one() {       # c1_one <pool seed>
    local SEED=$1
    local DATA="$SPLITS/c1_random_rand_seed$SEED"
    if [ -d "$DATA/forget" ]; then
        echo "Split exists: $DATA"
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
               [[ "$S" =~ ^[0-9]+$ ]] || die "bad config '$CONFIG'"
               c1_one "$S" ;;
    *)         die "unknown config '$CONFIG' (use C0, C1 or C1_seed<N>)" ;;
esac
