#!/bin/bash
#SBATCH --job-name=eval_packed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# The same 7 chunk commands as eval_chunk.sh, packed one chunk per GPU on a single 8-GPU node (~1.25 h for a 1B
# model), then core_sheet.sh (merge + eval_sheet.py). Resumable: a chunk whose metrics json exists is skipped.
# Guard: training.train can exit 0 after an open_lm crash, so only a checkpoint whose stats say
# is_final_checkpoint=True is evaluated.
# usage: sbatch -J eval_<corpus> eval_packed.sbatch <tokenized_dataset_name (READ)> [scale: 1b (default) | 400m]
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
READ=${1:?name}; SCALE=${2:-1b}
R=${WORK_DIR:?set WORK_DIR}; CODE=${DCLM_DIR:?set DCLM_DIR}; PY=${DCLM_PY:-python}
if [[ $SCALE == 1b ]]; then MODEL_CFG=open_lm_1b_swiglutorch.json; else MODEL_CFG=d=1024_l=24_h=8.json; fi
NAME=${SCALE}_$READ
RUN_DIR=$(ls -d ${PRETRAIN_DIR:-$R/pretrain}/$SCALE/$READ/pretrain_logs/$READ-* 2>/dev/null | head -1)
[[ -n $RUN_DIR ]] || { echo "FATAL no run dir under $SCALE/$READ"; exit 1; }
ls $RUN_DIR/checkpoints/epoch_*.pt >/dev/null 2>&1 || { echo "FATAL no checkpoints in $RUN_DIR"; exit 1; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 WANDB_MODE=offline
cd "$CODE"; export PYTHONPATH=$CODE
EPOCH=$(ls -1v "$RUN_DIR/checkpoints/"epoch_*.pt 2>/dev/null | tail -1 | xargs -n1 basename | sed "s/\.pt$//")
[[ -n $EPOCH ]] || { echo "FATAL: no ckpt"; exit 1; }
echo "[$(date -u)] eval $NAME run_dir=$RUN_DIR epoch=$EPOCH"
$PY - "$RUN_DIR/checkpoints/stats_${EPOCH#epoch_}.pt" <<'PYEOF' || { echo "FATAL: $EPOCH is not the final checkpoint - training did not finish"; exit 1; }
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("stats", sys.argv[1], "is_final_checkpoint", s.get("is_final_checkpoint"), "step", s.get("step"), "samples_seen", s.get("samples_seen"))
sys.exit(0 if s.get("is_final_checkpoint") is True else 1)
PYEOF
mkdir -p logs
pids=()
for K in 0 1 2 3 4 5 6; do
  CHUNK=chunk$K
  RESULT=$CODE/eval_results/chunks/$NAME/$EPOCH/metrics_${CHUNK}.json
  if [[ -s $RESULT ]]; then echo "$CHUNK already done: $RESULT"; continue; fi
  mkdir -p "$(dirname "$RESULT")"
  PORT=$((20000 + (SLURM_JOB_ID + K * 137) % 20000))
  ( CUDA_VISIBLE_DEVICES=$K ${DCLM_TORCHRUN:-torchrun} --master_port "$PORT" --nproc_per_node 1 eval/eval_openlm_ckpt.py \
      --donot-compute-perplexity \
      --checkpoint "$RUN_DIR/checkpoints/${EPOCH}.pt" \
      --model "$CODE/training/open_lm_configs/$MODEL_CFG" \
      --config "$RUN_DIR/params.txt" \
      --eval-yaml "$CODE/eval/shards/${CHUNK}.yaml" \
      --output-file "$RESULT" \
      --use-temp-working-dir > logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${CHUNK}.log 2>&1
    echo "[$(date -u)] $CHUNK rc=$? json=$([[ -s $RESULT ]] && echo yes || echo NO)" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
n=$(ls -1 $CODE/eval_results/chunks/$NAME/$EPOCH/metrics_chunk*.json 2>/dev/null | wc -l)
echo "[$(date -u)] chunks with json: $n / 7"
[[ $n -eq 7 ]] || { echo "FATAL: missing chunk json(s); resubmit this job - finished chunks are skipped"; exit 1; }
bash $ROOT/pretraining/core_sheet.sh "$NAME"
