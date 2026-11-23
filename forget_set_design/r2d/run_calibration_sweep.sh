#!/bin/bash
# This script compares the decoder methods b2, ours_a and ours_b for different
# calibration budgets, for one audit configuration.
#
# A calibration budget is the number of calibration noise draws for each sign vector.
# The script uses the unlearned models of a complete audit. It does not train a model.
# Start run_audits.sh for the configuration first.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> forget_set_design/r2d/run_calibration_sweep.sh <config>
#
#   <gpu>     The physical GPU id. You select the GPU. The script does not examine it.
#             The script uses one GPU.
#   <config>  The name of a result directory of run_audits.sh, for example C1_seed0,
#             C3, C4 or C4_1. C4.1 is the same as C4_1.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_calibration_sweep.sh C3
#   EPSILONS=10,30,100 CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_calibration_sweep.sh C4
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 forget_set_design/r2d/run_calibration_sweep.sh C3
#
# DRY_RUN=1 prints the command and does not start it.
# The script skips each claimed epsilon that is complete with the same settings.
# Use the same command to continue after an interruption. Set FORCE=1 to do all
# values again.
#
# Output files in $OUT_ROOT/<config>/sweep/:
#   sweep_eps<value>.csv    One row for each fit of a decoder.
#   sweep_eps<value>.json   The settings of that file.
#   sweep_table.csv         The mean, the standard deviation, the minimum and the
#                           maximum of the lower bound for each cell.
#   sweep.log               The full output.
# The script does not change the files of run_audits.sh.
#
# Time: one calibration seed for one claimed epsilon is approximately 3 minutes.
# The default settings are 1 seed and 3 epsilon values, thus approximately 10 minutes.
#
# Settings are environment variables. The default values are in brackets.
#   ENV_BIN       Directory of the python environment.
#                 [/tmp2/enzo/miniconda3/envs/unlearn-audit/bin]
#   OUT_ROOT      Directory of the audit results. [/tmp2/enzo/audit-unlearning/r2d]
#   DATA          Split directory of the configuration. The default is the split
#                 directory that the audit of run_audits.sh recorded.
#   EPSILONS      List of claimed epsilon values. inf is not permitted. [100,1000,2000]
#                 Use values for which the lower bound is not at its maximum. For
#                 C4, C5 and their variants, use lower values, for example 10,30,100.
#   BUDGETS       List of calibration budgets. [2,10,100]
#   NUM_CALIB     Number of calibration noise draws for each sign vector and each
#                 calibration seed. [100]
#   MAX_GROUPS    Maximum number of fits for one budget and one calibration seed. [10]
#   CALIB_SEEDS   List of calibration seeds. [0]
#   EVAL_SEED     Seed of the evaluation noise. [1]
#   NUM_SAMPLES   Number of evaluation trials. [500]
#   METHODS       List of decoder methods. [b2,ours_a,ours_b]
#   DESIGN_PAIRS  List of calibration designs. 0 uses all sign vectors. N uses N
#                 complement pairs (2 * N sign vectors), for ours_b only. [0,5]
#   DELTA         Delta for the noise calibration. [1e-3]
#   CI_DELTA      Confidence parameter of the lower bounds. [0.05]
#   THREADS       Number of CPU threads. [4]
set -euo pipefail

CONFIG=${1:-}
if [ -z "$CONFIG" ]; then
    sed -n '2,58p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ENV_BIN=${ENV_BIN:-/tmp2/enzo/miniconda3/envs/unlearn-audit/bin}
OUT_ROOT=${OUT_ROOT:-/tmp2/enzo/audit-unlearning/r2d}
DATA=${DATA:-}
EPSILONS=${EPSILONS:-100,1000,2000}
BUDGETS=${BUDGETS:-2,10,100}
NUM_CALIB=${NUM_CALIB:-100}
MAX_GROUPS=${MAX_GROUPS:-10}
CALIB_SEEDS=${CALIB_SEEDS:-0}
EVAL_SEED=${EVAL_SEED:-1}
NUM_SAMPLES=${NUM_SAMPLES:-500}
METHODS=${METHODS:-b2,ours_a,ours_b}
DESIGN_PAIRS=${DESIGN_PAIRS:-0,5}
DELTA=${DELTA:-1e-3}
CI_DELTA=${CI_DELTA:-0.05}
THREADS=${THREADS:-4}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}

# These values are the same as in run_audits.sh.
N_COMBOS=20
RUN_SUBDIR="bs_full_train_35_unlearn_5_lr_0_01_comb3"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
GPUS=${CUDA_VISIBLE_DEVICES:-}
[ -n "$GPUS" ] || die "CUDA_VISIBLE_DEVICES is not set. Set it to the GPU id that you want to use."
[[ "$GPUS" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must be one GPU id. The value is '$GPUS'."
[ -x "$ENV_BIN/python3" ] || die "$ENV_BIN does not contain python3. Set ENV_BIN to the correct directory."

# The result directories have no period in their names.
NAME=${CONFIG//./_}
BASE="$OUT_ROOT/$NAME"
RES="$BASE/$RUN_SUBDIR"
[ -d "$RES" ] || die "The directory $RES does not exist. Start run_audits.sh for $CONFIG first."
DONE=$(find "$RES" -maxdepth 2 -name run_metadata.json | wc -l)
[ "$DONE" -ge "$N_COMBOS" ] || die "Only $DONE of $N_COMBOS runs of $NAME are complete. Start run_audits.sh for $CONFIG first."

if [ -z "$DATA" ]; then
    # The audit of run_audits.sh recorded its split directory.
    RECORD=$(find "$BASE/joint" -maxdepth 1 -name 'joint_eps*.json' 2>/dev/null | sort | head -n 1)
    [ -n "$RECORD" ] || die "$BASE/joint contains no joint_eps*.json file. Set DATA to the split directory."
    DATA=$("$ENV_BIN/python3" -c 'import json, sys; print(json.load(open(sys.argv[1]))["data_dir"])' "$RECORD")
fi
[ -d "$DATA/forget" ] || die "$DATA does not contain a forget directory."

export PATH="$ENV_BIN:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
# Python then writes each line to the log file immediately.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- sweep
OUT="$BASE/sweep"
CMD=(python3 "$REPO/forget_set_design/r2d/calib_sweep.py"
     --results-dir "$RES" --data-dir "$DATA" --output-dir "$OUT"
     --epsilon "$EPSILONS" --delta "$DELTA" --budgets "$BUDGETS"
     --num-calib "$NUM_CALIB" --max-groups "$MAX_GROUPS" --calib-seeds "$CALIB_SEEDS"
     --eval-seed "$EVAL_SEED" --num-eval "$NUM_SAMPLES" --methods "$METHODS"
     --design-pairs "$DESIGN_PAIRS" --ci-delta "$CI_DELTA" --config-id "$NAME"
     --device cuda:0)
if [ "$FORCE" = "1" ]; then CMD+=(--force); fi

echo "=== $NAME  split=$DATA  gpu=$GPUS  host=$(hostname -s)  $(date '+%F %T')"
echo "+ ${CMD[*]}"
echo "  log: $OUT/sweep.log"
if [ "$DRY_RUN" = "1" ]; then exit 0; fi

mkdir -p "$OUT"
"${CMD[@]}" 2>&1 | tee -a "$OUT/sweep.log"
[ "${PIPESTATUS[0]}" = "0" ] || die "The command failed. Read $OUT/sweep.log."
echo "=== $NAME: sweep results: $OUT/sweep_table.csv"
