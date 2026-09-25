#!/bin/bash
#SBATCH --job-name=rewrite_pool
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/rewrite_pool_%A_%a.log
# RePro 1B rewrites (T=1.0, top-p 0.9) of the deleted pages with int_edu >= 1. Node-packed with NT ranks per node
# (7 in our run), resumable per shard, atomic writes.
# usage: sbatch --array=0-3 --export=ALL,WORLD=28 rewrite_pool.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?} ${RW_MODEL:?RePro 1B rephraser checkpoint}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORLD=${WORLD:?}; NT=${NT:-7}
A=${SLURM_ARRAY_TASK_ID:-0}
export VLLM_CACHE_ROOT=/tmp/vllm_rewrite_${SLURM_ARRAY_JOB_ID:-0}_${A}
srun --ntasks=$NT --gpus-per-task=1 bash -c "RANK=\$((${A}*$NT+SLURM_PROCID)); ${REFINER_PY:-python} $ROOT/data_construction/rescue/rewrite_pool.py \$RANK $WORLD"
rc=$?; echo "[$(date -u)] task $A rc=$rc"; exit $rc
