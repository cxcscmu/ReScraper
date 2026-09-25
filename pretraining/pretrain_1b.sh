#!/bin/bash
#SBATCH --job-name=pt1b
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# 1B setting: DCLM scale 1b_1x_fast (1.4B params, 28.8B training tokens: 54,923 steps x 256 x 2048) on an
# already-tokenized corpus. Our runs: 8 nodes x 8 H200, ~4.3 h of training plus requeues.
# Optional checkpoint pruner (PRUNE=1): open_lm writes epoch_N, optimizer_N, then stats_N; once stats_N exists,
# epoch_k/optimizer_k for k < N are deleted (stats_*.pt are kept, so --resume latest still works). DCLM's
# training.train has no pass-through for open_lm's --delete-previous-checkpoint, hence a pruner instead of a flag.
# Usage: sbatch -J pt1b_<corpus> pretrain_1b.sh <tokenized_dataset_name (READ)> <master_port>
set -uo pipefail
READ=${1:?tokenized dataset name (without .json)}
PORT=${2:?master port}
R=${WORK_DIR:?set WORK_DIR}; CODE=${DCLM_DIR:?set DCLM_DIR}
L=${PRETRAIN_DIR:-$R/pretrain}/1b/$READ
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline WANDB_DIR=$L/wandb
mkdir -p "$L/pretrain_logs" "$WANDB_DIR"
cd "$CODE"
DC=$CODE/exp_data/datasets/tokenized/$READ.json
[[ -s $DC ]] || { echo "FATAL: no data config $DC"; exit 1; }
for d in "$L/pretrain_logs/$READ"-*; do
  [[ -d $d ]] || continue
  ls "$d"/checkpoints/stats_*.pt >/dev/null 2>&1 || { echo "removing never-checkpointed $d"; rm -rf "$d"; }
done
prune() {
  local d last f n
  for d in "$L/pretrain_logs/$READ"-*/checkpoints; do
    [[ -d $d ]] || continue
    last=$(ls "$d" | sed -n -E "s/^stats_([0-9]+)\.pt$/\1/p" | sort -n | tail -1)
    [[ -n $last ]] || continue
    for f in "$d"/epoch_*.pt "$d"/optimizer_*.pt; do
      [[ -e $f ]] || continue
      n=$(basename "$f" .pt); n=${n##*_}
      if [[ $n =~ ^[0-9]+$ ]] && (( n < last )); then rm -f "$f" && echo "[prune $(date +%T)] rm $f (newest complete checkpoint = $last)"; fi
    done
  done
}
PRUNER=
if [[ ${PRUNE:-0} == 1 ]]; then ( while true; do sleep 300; prune; done ) & PRUNER=$!; fi
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$PORT
echo "[$(date +%T)] 1b_1x_fast $READ nodes=$SLURM_JOB_NUM_NODES world=$((SLURM_JOB_NUM_NODES*8))"
srun --kill-on-bad-exit=1 bash -c "
  export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT PYTHONPATH=$CODE
  ${DCLM_TORCHRUN:-torchrun} \
    --nnodes=\$SLURM_JOB_NUM_NODES \
    --nproc-per-node=8 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m training.train -- \
      --scale 1b_1x_fast \
      --data-config $DC \
      --logs $L/pretrain_logs \
      --num-checkpoints 20 \
      --multiple-data-passes \
      --ignore-parse-errors \
      --data-tolerate-num-ckpts 20 \
      --data-tolerate-error-p 0.5"
rc=$?
if [[ -n $PRUNER ]]; then kill $PRUNER 2>/dev/null; prune; fi
exit $rc
