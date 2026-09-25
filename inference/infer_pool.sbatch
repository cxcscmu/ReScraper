#!/bin/bash
#SBATCH --job-name=rescraper_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/rescraper_infer_%A_%a.log
# Pool inference of the ReScraper model over the raw-HTML universe (Dripper step 1 records). Node-packed:
# 1 array task = 1 node = 8 ranks (one GPU each), WORLD = NODES*8, resumable per shard.
# The released corpus: 28 nodes x 8 H200 (WORLD 224), ~1.0-1.7 h per node task.
# usage: sbatch --array=0-27 --export=ALL,WORLD=224,INFER_MODEL_PATH=$STAGE2_CKPT,INFER_OUT_DIR=$WORK_DIR/dclm_pipeline/rescraper_corpus/text infer_pool.sbatch
#        backfill: add INFER_STEMS_FILE=<file with the missing shard stems> and WORLD=<number of stems or 8*(K+1)>
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export INFER_SYSTEM_PROMPT_FILE=${INFER_SYSTEM_PROMPT_FILE:-$ROOT/prompts/student_system_stage2.txt}
export INFER_OUT_DIR=${INFER_OUT_DIR:-$WORK_DIR/dclm_pipeline/rescraper_corpus/text}
WORLD=${WORLD:?}; : ${INFER_MODEL_PATH:?}
echo "[$(date -u)] node-task $SLURM_ARRAY_TASK_ID model=$INFER_MODEL_PATH world=$WORLD out=$INFER_OUT_DIR"
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$((SLURM_ARRAY_TASK_ID*8+SLURM_PROCID)); echo rank=\$RANK gpu=\$CUDA_VISIBLE_DEVICES; ${REFINER_PY:-python} $ROOT/inference/infer_pool.py \$RANK $WORLD"
rc=$?; echo "[$(date -u)] node-task $SLURM_ARRAY_TASK_ID done rc=$rc"; exit $rc
