#!/bin/bash
#SBATCH --job-name=eval_chunk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/eval_chunk_%A_%a.log
# One shard of the DCLM Core eval suite (mmlu_and_lowvar split into eval/shards/chunk0-6.yaml) on the latest
# checkpoint of a pretraining run, 1 GPU per chunk. Then run core_sheet.sh <name>.
# Usage: sbatch --array=0-6 [--export=ALL,EV_MODEL=open_lm_1b_swiglutorch.json] eval_chunk.sh <run_dir> <name>
#   run_dir = <PRETRAIN_DIR>/<scale>/<READ>/pretrain_logs/<READ>-<open_lm run name>
#   EV_MODEL: d=1024_l=24_h=8.json for the 400M setting (default), open_lm_1b_swiglutorch.json for the 1B setting
set -uo pipefail
RUN_DIR=${1:?run_dir}; NAME=${2:?name}
MODEL_CFG=${EV_MODEL:-d=1024_l=24_h=8.json}
CODE=${DCLM_DIR:?set DCLM_DIR}
CHUNK=chunk${SLURM_ARRAY_TASK_ID:-0}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 WANDB_MODE=offline
cd "$CODE"; export PYTHONPATH=$CODE
[[ -d $RUN_DIR/checkpoints ]] || { echo "FATAL: no checkpoints in $RUN_DIR"; exit 1; }
EPOCH=$(ls -1v "$RUN_DIR/checkpoints/"epoch_*.pt 2>/dev/null | tail -1 | xargs -n1 basename | sed 's/\.pt$//')
[[ -n $EPOCH ]] || { echo "FATAL: no ckpt"; exit 1; }
RESULT=$CODE/eval_results/chunks/$NAME/$EPOCH/metrics_${CHUNK}.json
mkdir -p "$(dirname "$RESULT")"
PORT=$((20000 + (SLURM_JOB_ID + ${SLURM_ARRAY_TASK_ID:-0} * 137) % 20000))
CUDA_VISIBLE_DEVICES=0 ${DCLM_TORCHRUN:-torchrun} --master_port "$PORT" --nproc_per_node 1 eval/eval_openlm_ckpt.py \
  --donot-compute-perplexity \
  --checkpoint "$RUN_DIR/checkpoints/${EPOCH}.pt" \
  --model "$CODE/training/open_lm_configs/$MODEL_CFG" \
  --config "$RUN_DIR/params.txt" \
  --eval-yaml "$CODE/eval/shards/${CHUNK}.yaml" \
  --output-file "$RESULT" \
  --use-temp-working-dir
RC=$?
[[ -f $RESULT ]] && echo "CHUNK_JSON_WRITTEN $RESULT"
exit $RC
