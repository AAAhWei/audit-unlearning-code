#!/bin/bash
# This script does the model-clipping audit of one forget-set configuration.
#
# For each of the 20 combinations of 3 of 6 forget batches, the script trains one model
# and then does TRIALS unlearning trials. Each trial has its own noise seed. Then the
# script audits the trials with the joint decoders b2, ours_a and ours_b, for different
# calibration budgets. A calibration budget is the number of calibration trials for
# each sign vector.
#
# The training and the unlearning are those of
# certified_unlearning_plus_grad_based_methods/. This script does not change that code.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/modelclip/run_audits.sh <config> [<first> [<count>]]
#
#   <gpu>     The physical GPU id. You select the GPU. The script does not examine it.
#             The script uses one GPU.
#   <config>  MC1    Random pool, random batches. The split is that of C1_seed0.
#             MC2    Easy pool, random batches.
#             MC3    Hard pool, random batches.
#             MC3.1  Global hard pool, random batches. The pool is the highest
#                    difficulty scores of all points. The classes do not have equal
#                    numbers of points.
#             MC4    Hard pool, batches of the assignment "label".
#             MC4.1  Random pool, batches of the assignment "label". The split is that
#                    of C4.1.
#             MC4.2  Easy pool, batches of the assignment "label".
#             MC5    Hard pool, batches of the assignment "grad".
#             MC5.1  Random pool (that of MC1), batches of the assignment "grad".
#             MC5.2  Easy pool, batches of the assignment "grad".
#             MC6    Hard pool, batches of the assignment "spread".
#             MC1_seed<n>  A different random pool, n = 1 to 5. The split is that of
#                          C1_seed<n>. MC1 is the seed 0.
#             MC2_seed<n>, MC3_seed<n>
#                          The pool of MC2 or MC3 with a different random selection
#                          of the batches, n = 1 to 9.
#             The hard pool and the easy pool use the difficulty scores of model
#             clipping. Start run_scores.sh first. The assignments "grad" and "spread"
#             use the gradient directions of compute_directions.py.
#             The script creates the split directory if it does not exist.
#             Set DATA to use a different split directory with a different name.
#   <first>   The first combination, 0 to 19. [0]
#   <count>   The number of combinations. [all combinations from <first>]
#
# Examples:
#   TEST=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC1
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC1
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC1 0 10
#   CUDA_VISIBLE_DEVICES=5 forget_set_design/modelclip/run_audits.sh MC1 10 10
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC4.1
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC1_seed1
#   STEPS=20 OUT_NAME=audit_step20 CUDA_VISIBLE_DEVICES=4 forget_set_design/modelclip/run_audits.sh MC1
#
# TEST=1 measures the time and the disk space. It does combination 0 only, with 2
# trials, in the directory <config>_test. It does not start the audit.
# DRY_RUN=1 prints the commands and does not start them.
#
# The script skips each combination that is complete. Use the same command to continue
# after an interruption. If a combination stopped before its end, the script stops and
# gives the directory that you must delete first.
# The audit starts only when all 20 combinations are complete. Thus you can divide the
# combinations between two GPUs, as in the examples. Then start the command with no
# <first> again to do the audit.
# To audit the trials again with different settings, start the script again with
# those settings. The script skips the combinations and does only the audit. It keeps
# the scores of each step in a file. If that file exists, the script reads it and does
# only the fits. Set FORCE=1 to calculate the scores again. Use a different OUT_NAME
# for each group of settings that you want to keep.
#
# Output files in $OUT_ROOT/<config>/ (C4.1 becomes MC4_1):
#   config_used.yaml          The config file of the runs.
#   runs/<id>/                One directory for each combination: the trained model
#                             (ckpt/), the trials (ckpt_trial_<n>/) and the log files.
#   logs/combo_<n>.out        The output of the start of each combination.
#   audit/sweep_table.csv     The mean, the standard deviation, the minimum and the
#                             maximum of the lower bound for each cell.
#   audit/sweep_step<s>.csv   One row for each fit of a decoder at step <s>.
#   audit/sweep_step<s>.json  The settings of that file.
#   audit/scores_step<s>.npz  The scores of all trials at step <s>.
#   audit/audit.log           The full output of the audit.
# OUT_NAME changes the name of the directory audit/.
#
# Settings are environment variables. The default values are in brackets.
#   ENV_BIN       Directory of the python environment.
#                 [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   OUT_ROOT      Directory of the results. [/tmp2/enzo/audit-unlearning/modelclip]
#   SPLITS        Directory of the split directories.
#                 [/tmp2/enzo/audit-unlearning/data/cifar100/data_split]
#   DATA          Split directory. The default is the split of <config>.
#   SCORES_CSV    Scores file of model clipping, for the hard pool and the easy pool.
#                 [/tmp2/enzo/audit-unlearning/scores/scores_modelclip.csv]
#   DIRECTIONS    Directions file of model clipping, for MC5, MC5.1, MC5.2 and MC6.
#                 [/tmp2/enzo/audit-unlearning/scores/directions_modelclip.npy]
#   BASE_CONFIG   Config file of the pipeline. This script changes only the stop step,
#                 the checkpoints to keep and the log directory.
#                 [certified_unlearning_plus_grad_based_methods/configs/exp_cifar100_chg7.yaml]
#   MAX_STEP      Each trial stops after this number of unlearning steps. [30]
#   KEEP_EVERY    Each trial keeps the checkpoints of the steps that are multiples of
#                 this number. One checkpoint is approximately 2 MB. [10]
#   STEPS         List of the steps to audit. [10,20,30]
#   TRIALS        Number of unlearning trials for each combination. [150]
#   NUM_CALIB     Number of calibration trials for each combination. The audit uses
#                 the first trials for the calibration. [50]
#   NUM_EVAL      Number of evaluation trials for each combination. The audit uses
#                 the trials after the calibration trials. [100]
#   PARALLEL      Number of combinations that run at the same time on the GPU. [1]
#   OUT_NAME      Name of the audit directory in $OUT_ROOT/<config>/. [audit]
#   BUDGETS       List of calibration budgets. The maximum is NUM_CALIB. [2,5,10,20,50]
#   MAX_GROUPS    Maximum number of fits for one budget. [10]
#   METHODS       List of decoder methods. [b2,ours_a,ours_b]
#   DESIGN_PAIRS  List of calibration designs. 0 uses all sign vectors. N uses N
#                 complement pairs (2 * N sign vectors), for ours_b only. [0]
#   CI_DELTA      Confidence parameter of the lower bounds. [0.05]
set -euo pipefail

CONFIG=${1:-}
if [ -z "$CONFIG" ]; then
    sed -n '2,113p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PIPELINE="$REPO/certified_unlearning_plus_grad_based_methods"

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
OUT_ROOT=${OUT_ROOT:-/tmp2/enzo/audit-unlearning/modelclip}
SPLITS=${SPLITS:-/tmp2/enzo/audit-unlearning/data/cifar100/data_split}
DATA=${DATA:-}
SCORES_CSV=${SCORES_CSV:-/tmp2/enzo/audit-unlearning/scores/scores_modelclip.csv}
DIRECTIONS=${DIRECTIONS:-/tmp2/enzo/audit-unlearning/scores/directions_modelclip.npy}
BASE_CONFIG=${BASE_CONFIG:-$PIPELINE/configs/exp_cifar100_chg7.yaml}
MAX_STEP=${MAX_STEP:-30}
KEEP_EVERY=${KEEP_EVERY:-10}
STEPS=${STEPS:-10,20,30}
NUM_CALIB=${NUM_CALIB:-50}
NUM_EVAL=${NUM_EVAL:-100}
PARALLEL=${PARALLEL:-1}
OUT_NAME=${OUT_NAME:-audit}
BUDGETS=${BUDGETS:-2,5,10,20,50}
MAX_GROUPS=${MAX_GROUPS:-10}
METHODS=${METHODS:-b2,ours_a,ours_b}
DESIGN_PAIRS=${DESIGN_PAIRS:-0}
CI_DELTA=${CI_DELTA:-0.05}
FORCE=${FORCE:-0}
TEST=${TEST:-0}
DRY_RUN=${DRY_RUN:-0}

# These values are the same as in r2d/run_audits.sh: 20 combinations of 3 of 6 batches.
N_COMBOS=20
FORGET_FRACTION=0.5
# This is the default value of run_exhaustive_combinations_parallel.sh.
SEED_OFFSET=10000

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
GPUS=${CUDA_VISIBLE_DEVICES:-}
[ -n "$GPUS" ] || die "CUDA_VISIBLE_DEVICES is not set. Set it to the GPU id that you want to use."
[[ "$GPUS" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must be one GPU id. The value is '$GPUS'."
[ -x "$ENV_BIN/python3" ] || die "$ENV_BIN does not contain python3. Set ENV_BIN to the correct directory."
[ -f "$BASE_CONFIG" ] || die "The config file $BASE_CONFIG does not exist."
[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "PARALLEL must be a positive integer. The value is '$PARALLEL'."

export PATH="$ENV_BIN:$PATH"

# Create the split directory for a pool from the difficulty scores of model clipping
# if it does not exist. Set DATA to that directory.
# Arguments: <pool: hard, easy, global_hard or random>
#            <assignment: rand, label, grad or spread> [<seed of the batch selection>]
pool_split() {
    local POOL=$1 ASSIGN=$2 SEED=${3:-0}
    local TAG="modelclip_${POOL}_${ASSIGN}"
    if [ "$SEED" != "0" ]; then TAG="${TAG}_seed$SEED"; fi
    # The name of the split directory does not include the seed part of <config>.
    local PREFIX=${CONFIG%_seed*}
    DATA="$SPLITS/$(echo "${PREFIX//./_}" | tr '[:upper:]' '[:lower:]')_$TAG"
    if [ -d "$DATA/forget" ]; then return 0; fi
    [ -f "$SCORES_CSV" ] || die "The scores file $SCORES_CSV does not exist. Start modelclip/run_scores.sh first."
    local EXTRA=()
    if [ "$ASSIGN" = "grad" ] || [ "$ASSIGN" = "spread" ]; then
        [ -f "$DIRECTIONS" ] || die "The directions file $DIRECTIONS does not exist. Start modelclip/compute_directions.py first."
        EXTRA+=(--directions "$DIRECTIONS")
    fi
    if [ "$POOL" = "random" ]; then
        # The pool is that of MC1. Thus only the batch of each point is different from MC1.
        local POOL_FILE="$SPLITS/assignment_random_rand_seed0.csv"
        [ -f "$POOL_FILE" ] || die "The pool file $POOL_FILE does not exist."
        EXTRA+=(--pool-file "$POOL_FILE")
    fi
    local BUILD=(python3 "$REPO/forget_set_design/utils/build_pools.py" --alg modelclip --pool "$POOL"
                 --assignment "$ASSIGN" --assign-seed "$SEED" --out-stem "assignment_$TAG"
                 --scores-csv "$SCORES_CSV" --out-dir "$SPLITS" "${EXTRA[@]}")
    local SPLIT=(python3 "$REPO/forget_set_design/utils/make_custom_split.py"
                 --assignment "$SPLITS/assignment_$TAG.csv" --out_dir "$DATA")
    echo "+ ${BUILD[*]}"
    echo "+ ${SPLIT[*]}"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    # These two commands use the CPU only.
    CUDA_VISIBLE_DEVICES="" "${BUILD[@]}"
    CUDA_VISIBLE_DEVICES="" "${SPLIT[@]}"
}

if [ -z "$DATA" ]; then
    case "$CONFIG" in
        MC1)   DATA="$SPLITS/c1_random_rand_seed0" ;;
        MC4.1) DATA="$SPLITS/c4_1_r2d_random_label" ;;
        MC2)   pool_split easy rand ;;
        MC3)   pool_split hard rand ;;
        MC4)   pool_split hard label ;;
        MC4.2) pool_split easy label ;;
        MC5)   pool_split hard grad ;;
        MC5.1) pool_split random grad ;;
        MC5.2) pool_split easy grad ;;
        MC6)   pool_split hard spread ;;
        MC3.1) pool_split global_hard rand ;;
        MC1_seed[1-5])      DATA="$SPLITS/c1_random_rand_seed${CONFIG#MC1_seed}" ;;
        MC2_seed[1-9])      pool_split easy rand "${CONFIG#MC2_seed}" ;;
        MC3_seed[1-9])      pool_split hard rand "${CONFIG#MC3_seed}" ;;
        *) die "The configuration '$CONFIG' is not known. Use MC1 to MC6, MC3.1, MC4.1, MC4.2, MC5.1, MC5.2, MC1_seed<n>, MC2_seed<n> or MC3_seed<n>, or set DATA." ;;
    esac
fi
[ -d "$DATA/forget" ] || [ "$DRY_RUN" = "1" ] || die "$DATA does not contain a forget directory."
[ -d "$DATA/retain" ] || [ "$DRY_RUN" = "1" ] || die "$DATA does not contain a retain directory."

FIRST=${2:-0}
COUNT=${3:-$((N_COMBOS - FIRST))}
if [ "$TEST" = "1" ]; then
    TRIALS=${TRIALS:-2}
    FIRST=0
    COUNT=1
else
    TRIALS=${TRIALS:-$((NUM_CALIB + NUM_EVAL))}
fi
[[ "$FIRST" =~ ^[0-9]+$ && "$COUNT" =~ ^[1-9][0-9]*$ ]] || die "<first> and <count> must be integers."
[ $((FIRST + COUNT)) -le "$N_COMBOS" ] || die "<first> + <count> must be a maximum of $N_COMBOS."
[ $((MAX_STEP % KEEP_EVERY)) -eq 0 ] || die "MAX_STEP ($MAX_STEP) must be a multiple of KEEP_EVERY ($KEEP_EVERY)."
for S in ${STEPS//,/ }; do
    [ "$S" -le "$MAX_STEP" ] && [ $((S % KEEP_EVERY)) -eq 0 ] ||
        die "The step $S of STEPS must be a multiple of KEEP_EVERY ($KEEP_EVERY) and a maximum of MAX_STEP ($MAX_STEP)."
done

# The result directories have no period in their names, as for r2d/run_audits.sh.
NAME=${CONFIG//./_}
if [ "$TEST" = "1" ]; then NAME="${NAME}_test"; fi
BASE="$OUT_ROOT/$NAME"
RUNS="$BASE/runs"
USED="$BASE/config_used.yaml"

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
# JAX then allocates GPU memory when it needs the memory, not all at the start.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# These flags are the default of run_exhaustive_combinations_parallel.sh.
export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops --xla_gpu_strict_conv_algorithm_picker=false"
export JAX_DISABLE_MOST_FASTER_PATHS=1

echo "=== $NAME  split=$DATA  gpu=$GPUS  host=$(hostname -s)  $(date '+%F %T')"
echo "    combinations $FIRST to $((FIRST + COUNT - 1)), $TRIALS trials each, stop at step $MAX_STEP"
echo "    results: $BASE"

# ---------------------------------------------------------------- config file
# The pipeline config, with the stop step, the checkpoints to keep and the log directory.
make_config() {
    python3 - "$BASE_CONFIG" "$1" "$MAX_STEP" "$KEEP_EVERY" "$BASE/wandb" <<'EOF'
import sys
import yaml

source, target, max_step, keep_every, log_dir = sys.argv[1:6]
with open(source) as f:
    config = yaml.safe_load(f)
unlearning = config["unlearning"]
unlearning["max_unlearn_step"] = int(max_step)
# Keep only the step checkpoints that are multiples of keep_every. Keep no epoch checkpoint.
unlearning["save_only_cleanup_retained"] = True
unlearning["checkpoint_keep_first_n_steps"] = 0
unlearning["checkpoint_keep_multiple_phases"] = [[0, int(keep_every)]]
unlearning["checkpoint_keep_first_n_epochs"] = 0
unlearning["checkpoint_keep_last_n_epochs"] = 0
unlearning["checkpoint_keep_every_n_epochs"] = 0
config["wandb"]["log_dir"] = log_dir
with open(target, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
EOF
}

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$RUNS" "$BASE/logs" "$BASE/wandb"
    make_config "$USED.new"
    if [ -f "$USED" ] && ! cmp -s "$USED" "$USED.new"; then
        rm -f "$USED.new"
        die "$USED has different settings. Use the same settings, or use a different OUT_ROOT."
    fi
    mv "$USED.new" "$USED"
fi

# ---------------------------------------------------------------- combinations
# Print the run directory of the combination $1, or nothing.
run_dir_of() {
    [ -d "$RUNS" ] || return 0
    python3 - "$RUNS" "$1" <<'EOF'
import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*/run_vars.json")):
    with open(path) as f:
        if int(json.load(f)["combo_idx"]) == int(sys.argv[2]):
            print(path.parent)
EOF
}

# Return 0 if the combination $1 is complete. Stop if it has a run that is not complete.
is_complete() {
    local index=$1 dir
    dir=$(run_dir_of "$index")
    if [ -z "$dir" ] && [ ! -f "$RUNS/_run_$index.dir" ]; then return 1; fi
    if [ "$(echo "$dir" | wc -l)" = "1" ] && [ ! -f "$RUNS/_run_$index.dir" ] &&
       [ -d "$dir/ckpt_trial_$((TRIALS - 1))/unlearn_step_$MAX_STEP" ]; then
        return 0
    fi
    die "Combination $index has a run that is not complete: ${dir:-$(cat "$RUNS/_run_$index.dir")}. Delete that directory and $RUNS/_run_$index.dir. Then start the script again."
}

run_combo() {
    local index=$1 start=$SECONDS
    local cmd=(python3 experiment_unlearning_exhaustive_combinations.py
               --config "$USED" --results_dir "$RUNS" --combo_idx "$index"
               --forget_fraction "$FORGET_FRACTION" --data_dir "$DATA"
               --num_unlearn_per_combo "$TRIALS" --unlearn_seed_offset "$SEED_OFFSET"
               --shuffle_train_samples)
    echo "+ (cd $PIPELINE && ${cmd[*]})"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    if (cd "$PIPELINE" && "${cmd[@]}" > "$BASE/logs/combo_$index.out" 2>&1); then
        echo "    combination $index is complete: $(((SECONDS - start) / 60)) min  $(date '+%F %T')"
    else
        echo "ERROR: Combination $index failed. Read $BASE/logs/combo_$index.out and the log files in its run directory." >&2
        return 1
    fi
}

FAILED=0
for INDEX in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
    if is_complete "$INDEX"; then
        echo "    combination $INDEX is complete. The script skips it."
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
        wait -n || FAILED=1
    done
    run_combo "$INDEX" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n || FAILED=1
done
[ "$FAILED" = "0" ] || die "One or more combinations failed."

if [ "$TEST" = "1" ]; then
    if [ "$DRY_RUN" != "1" ]; then
        echo "=== $NAME: the test is complete."
        grep -h "Training took\|Unlearning took" "$RUNS"/*/run_train.log "$RUNS"/*/run_unlearn_trial_*.log || true
        du -sh "$RUNS"/*/ckpt "$RUNS"/*/ckpt_trial_0
    fi
    exit 0
fi

# ---------------------------------------------------------------- audit
if [ "$DRY_RUN" != "1" ]; then
    for INDEX in $(seq 0 $((N_COMBOS - 1))); do
        if ! is_complete "$INDEX"; then
            echo "=== $NAME: combination $INDEX is not complete. The audit starts when all $N_COMBOS combinations are complete."
            exit 0
        fi
    done
fi

# The audit does not train a model and it does no unlearning trial. It does not change
# the files of the trials.
OUT="$BASE/$OUT_NAME"
CMD=(python3 "$REPO/forget_set_design/modelclip/calibration_sweep.py"
     --results-dir "$RUNS" --output-dir "$OUT" --steps "$STEPS"
     --num-calib "$NUM_CALIB" --num-eval "$NUM_EVAL" --budgets "$BUDGETS"
     --max-groups "$MAX_GROUPS" --methods "$METHODS" --design-pairs "$DESIGN_PAIRS"
     --ci-delta "$CI_DELTA" --config-id "$NAME")
if [ "$FORCE" = "1" ]; then CMD+=(--force); fi

echo "=== $NAME: audit  $(date '+%F %T')"
echo "+ ${CMD[*]}"
echo "  log: $OUT/audit.log"
if [ "$DRY_RUN" = "1" ]; then exit 0; fi

mkdir -p "$OUT"
if ! "${CMD[@]}" 2>&1 | tee -a "$OUT/audit.log"; then
    die "The audit failed. Read $OUT/audit.log."
fi
echo "=== $NAME: audit results: $OUT/sweep_table.csv"
