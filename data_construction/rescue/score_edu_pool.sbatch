#!/bin/bash
#SBATCH --job-name=score_edu_pool
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-31%32
#SBATCH --output=logs/score_edu_pool_%A_%a.log
# FineWeb-Edu (bf16, batch 128) over the pages the two-stage refiner deleted. Single-GPU tasks; task k owns
# shards [k::32] and skips shards already written.
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false EDU_BS=128
${REFINER_PY:-python} $ROOT/data_construction/rescue/score_edu_pool.py $SLURM_ARRAY_TASK_ID 32
