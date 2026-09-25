#!/bin/bash
#SBATCH --job-name=ablation_noext
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/ablation_noext_%A_%a.log
# noext arm ("w/o <extract>"): one array task = shards[task::NT], rebuilt from the raw sidecar with the numbered input
# re-rendered from Dripper step 1 (only pages whose noext text depends on it). Resumable per shard.
# usage: sbatch --array=0-<NT-1> build_arm_noext.sbatch <NT> [procs]
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
export PYTHONUNBUFFERED=1
unset OPS_TOLERANT OPS_TOLERANT_MIN RENDER_PROCS LIMIT_FILES
ABL=${ABL:-$WORK_DIR/dclm_pipeline/operation_ablation}
NT=${1:?ntasks}; P=${2:-${SLURM_CPUS_PER_TASK:-32}}; TASK=${SLURM_ARRAY_TASK_ID:-0}
${REFINER_PY:-python} $ROOT/ablations/operation_ablation/build_arm_noext.py "$ABL" "$ABL/noext" $TASK $NT $P
rc=$?; echo "[$(date -u)] task $TASK done rc=$rc"; exit $rc
