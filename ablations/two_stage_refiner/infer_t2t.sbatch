#!/bin/bash
#SBATCH --job-name=t2t_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/t2t_infer_%A_%a.log
# Full-pool text-to-text inference of the two-stage refiner (input = Dripper step-2 text, 10,318 shards),
# node-packed: each array task = 1 node = 8 ranks; WORLD = NODES*8. Output: <two_stage_pool>/<INFER_MODEL_TAG>/<stem>.jsonl.gz
# rows in step-2 order, <delete> -> empty text. Resumable per shard, atomic writes. Greedy decoding, 100k-char chunks.
# Usage: sbatch --array=0-<N-1> --export=ALL,INFER_MODEL_PATH=$TWO_STAGE_CKPT,INFER_MODEL_TAG=best,WORLD=<N*8> infer_t2t.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; R=${WORK_DIR:?set WORK_DIR}
export PYTHONUNBUFFERED=1
export INFER_TEMPERATURE=${INFER_TEMPERATURE:-0.0} INFER_TOP_P=${INFER_TOP_P:-1.0} INFER_CHUNK_CHARS=${INFER_CHUNK_CHARS:-100000}
export VLLM_CACHE_ROOT=/tmp/vllm_cache_${SLURM_JOB_ID}
export INFER_SYSTEM_PROMPT_FILE=${INFER_SYSTEM_PROMPT_FILE:-$ROOT/prompts/two_stage_refiner_system.txt}
INPUT_DIR=${INPUT_DIR:-$R/dclm_pipeline/two_stage_pool/input}
WORLD=${WORLD:?}
: ${INFER_MODEL_PATH:?} ${INFER_MODEL_TAG:?}
export INFER_MODEL_PATH INFER_MODEL_TAG
echo "[$(date -u)] t2t node-task $SLURM_ARRAY_TASK_ID model=$INFER_MODEL_PATH tag=$INFER_MODEL_TAG world=$WORLD T=$INFER_TEMPERATURE chunk=$INFER_CHUNK_CHARS"
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$((SLURM_ARRAY_TASK_ID*8+SLURM_PROCID)); echo rank=\$RANK gpu=\$CUDA_VISIBLE_DEVICES; ${REFINER_PY:-python} $ROOT/ablations/two_stage_refiner/infer_t2t.py \$RANK $INPUT_DIR $WORLD"
rc=$?
echo "[$(date -u)] node-task $SLURM_ARRAY_TASK_ID done rc=$rc"
exit $rc
