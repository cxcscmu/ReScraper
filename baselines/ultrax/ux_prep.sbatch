#!/bin/bash
#SBATCH --job-name=ux_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# UltraX stage 1 ($UX_SRC/*.jsonl.gz -> $UX_WORK/in/*.parquet), 16-way split. Resumable (existing outputs skipped).
# usage: sbatch --array=0-15 ux_prep.sbatch
# Check afterwards: the number of rows in in/ must equal the number of input documents with non-empty text
# (for the raw resiliparse extraction of the pool: 10,319 shards, 18,001,982 rows).
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
export PYTHONUNBUFFERED=1
${REFINER_PY:-python} $ROOT/baselines/ultrax/ux_pool_prep.py "${SLURM_ARRAY_TASK_ID:?}" 16
