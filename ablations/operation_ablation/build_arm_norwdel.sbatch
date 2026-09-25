#!/bin/bash
#SBATCH --job-name=ablation_norwdel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ablation_norwdel_%A_%a.log
# norwdel arm ("w/o <rewrite>") build worker. NT is passed explicitly (never $SLURM_ARRAY_TASK_COUNT, which is the
# number of RESUBMITTED tasks on a partial re-run). Each task builds the shards with index % NT == task id.
# usage: sbatch --array=0-<NT-1> build_arm_norwdel.sbatch <NT> <procs>
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
export PYTHONUNBUFFERED=1
NT=${1:?NT}; P=${2:-16}; K=${SLURM_ARRAY_TASK_ID:?}
${REFINER_PY:-python} $ROOT/ablations/operation_ablation/build_arm_norwdel.py build "$NT" "$K" "$P"
rc=$?; echo "[$(date -u)] task $K rc=$rc"; exit $rc
