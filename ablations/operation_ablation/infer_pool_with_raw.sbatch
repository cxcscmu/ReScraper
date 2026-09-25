#!/bin/bash
#SBATCH --job-name=ablation_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=800G
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=logs/ablation_infer_%A_%a.log
# One new full-pool pass of the released model with the release settings (T=1.0, top-p 1.0, 3,072 new tokens),
# writing the release text rows to <ABL>/full/text AND the per-page raw sidecar to <ABL>/raw, from which every
# ablation arm is built. Clears the executor's non-default switches so the release defaults apply.
# Array mode: RANK = task*8 + procid, WORLD = 8 * tasks (our run: --array=0-27, WORLD=224, 28 x 8 H200, 1.0-1.7 h/task).
# usage: sbatch --array=0-27 --export=ALL,INFER_MODEL_PATH=$STAGE2_CKPT,ABL=$WORK_DIR/dclm_pipeline/operation_ablation,WORLD=224 infer_pool_with_raw.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset OPS_TOLERANT OPS_TOLERANT_MIN RENDER_PROCS LIMIT_FILES
ABL=${ABL:-$WORK_DIR/dclm_pipeline/operation_ablation}
export INFER_SYSTEM_PROMPT_FILE=${INFER_SYSTEM_PROMPT_FILE:-$ROOT/prompts/student_system_stage2.txt}
export INFER_OUT_DIR=${INFER_OUT_DIR:-$ABL/full/text} ABLATION_RAW_DIR=${ABLATION_RAW_DIR:-$ABL/raw}
: ${INFER_MODEL_PATH:?}; export INFER_MODEL_PATH
WORLD=${WORLD:?}; TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "[$(date -u)] node-task $TASK model=$INFER_MODEL_PATH world=$WORLD out=$INFER_OUT_DIR raw=$ABLATION_RAW_DIR stems_file=${INFER_STEMS_FILE:-}"
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$(($TASK*8+SLURM_PROCID)); echo rank=\$RANK gpu=\$CUDA_VISIBLE_DEVICES; ${REFINER_PY:-python} $ROOT/ablations/operation_ablation/infer_pool_with_raw.py \$RANK $WORLD"
rc=$?; echo "[$(date -u)] node-task $TASK done rc=$rc"; exit $rc
