#!/bin/bash
#SBATCH --job-name=rescue_build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=800G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=logs/rescue_build_%A_%a.log
# Stage-1 rescue decisions + FineWeb-Edu sidecar over the pool, node-packed (1 array task = 1 node = 8 ranks,
# WORLD = NODES*8), resumable per shard.
# usage: sbatch --array=0-13 --export=ALL,WORLD=112[,EDU_KEEP=1.5,EDU_RW=1.0,PROSE_MIN=150,OUT_DIR=...] rescue_build.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?} ${RW_MODEL:?RePro 1B rephraser checkpoint}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORLD=${WORLD:?}
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$((SLURM_ARRAY_TASK_ID*8+SLURM_PROCID)); ${REFINER_PY:-python} $ROOT/data_construction/rescue/rescue_build.py \$RANK $WORLD"
echo "[$(date -u)] task $SLURM_ARRAY_TASK_ID rc=$?"
