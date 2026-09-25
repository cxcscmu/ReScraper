#!/bin/bash
#SBATCH --job-name=proxc_inf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%A_%a.log
# resiliparse + ProX-C full-pool run. 1 array task = 1 node = 8 ranks (one GPU each), WORLD = NODES*8,
# resumable per shard (proxc_pool.py skips shards whose output exists).
# usage: sbatch --array=0-27 --export=ALL,INFER_OUT_DIR=$WORK_DIR/dclm_pipeline/prox_c/text,WORLD=224 infer_pool_proxc.sbatch
# Do NOT set VLLM_WORKER_MULTIPROC_METHOD=spawn: with spawn, vLLM re-imports the main module in the engine-core
# child and proxc_pool.py builds a multiprocessing.Pool at module scope, which then fails at start-up.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# If pyarrow/sqlite fail to load with a missing CXXABI_1.3.15 symbol on GPU nodes, prepend the Python
# environment's own lib/ directory to LD_LIBRARY_PATH (the system libstdc++ is older than the env's).
# Give every array task its OWN torch.compile cache: vLLM names its compile-cache dirs by LOCAL rank, so with a
# shared default cache dir all nodes compile into the same paths at once and corrupt a cold cache.
export VLLM_CACHE_ROOT=/tmp/vllmcache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$VLLM_CACHE_ROOT"
WORLD=${WORLD:?}; : ${INFER_OUT_DIR:?}
PY=${REFINER_PY:-python}          # vLLM env with resiliparse installed
echo "[$(date -u)] node-task $SLURM_ARRAY_TASK_ID host=$(hostname) world=$WORLD out=$INFER_OUT_DIR model=${PROXC_MODEL:-default}"
srun --ntasks=8 --gpus-per-task=1 bash -c "RANK=\$((SLURM_ARRAY_TASK_ID*8+SLURM_PROCID)); echo rank=\$RANK gpu=\$CUDA_VISIBLE_DEVICES; $PY $ROOT/baselines/prox_c/proxc_pool.py \$RANK $WORLD"
rc=$?
# capture rc before any other command ($(date) inside an echo would reset $?)
echo "[$(date -u)] node-task $SLURM_ARRAY_TASK_ID done rc=$rc"
exit $rc
