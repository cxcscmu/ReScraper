#!/bin/bash
#SBATCH --job-name=teacher_refine
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0-15
#SBATCH --mem=1400G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/teacher_refine_%A_%a.log
# Qwen3.8-27B refinement labels for the ~1.6M SFT seed pages: 16 node tasks x 8 ranks (one GPU each) = WORLD 128,
# resumable per rank part (a finished part is skipped on requeue). Our run used 8x H200 per node.
# Requires an environment whose vLLM/transformers can serve Qwen3.8-27B ($TEACHER_PY, see requirements/teacher.txt).
# If the node driver is older than the CUDA runtime of that environment, prepend the CUDA forward-compat
# libraries to LD_LIBRARY_PATH.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_CACHE_ROOT=/tmp/vllm_teacher_${SLURM_JOB_ID}
WORLD=${WORLD:-128}
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$((SLURM_ARRAY_TASK_ID*8+SLURM_PROCID)); echo rank=\$RANK gpu=\$CUDA_VISIBLE_DEVICES; ${TEACHER_PY:-python} $ROOT/data_construction/teacher_refine/label_qwen27b.py \$RANK $WORLD"
echo "[$(date -u +%FT%TZ)] rc=$?"
