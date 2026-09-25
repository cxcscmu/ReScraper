#!/bin/bash
#SBATCH --job-name=ux_leg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# UltraX stage 4: post parquet -> corpus text shards ($UX_WORK/text). usage: sbatch --array=0-7 --export=ALL,NT=8 ux_leg.sbatch
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
export UX_WORK=${UX_WORK:-$WORK_DIR/dclm_pipeline/resiliparse_raw_ultrax}; W=$UX_WORK
NT=${NT:?}
export PYTHONUNBUFFERED=1
find $W/text -maxdepth 1 -name "*.jsonl.gz.tmp*" -delete 2>/dev/null
${REFINER_PY:-python} $ROOT/baselines/ultrax/ux_build_leg.py "${SLURM_ARRAY_TASK_ID:-0}" "$NT"
rc=$?
echo "[$(date -u)] task ${SLURM_ARRAY_TASK_ID:-0} rc=$rc shards=$(ls $W/text 2>/dev/null | grep -c jsonl.gz$)"
exit $rc
