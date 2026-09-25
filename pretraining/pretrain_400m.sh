#!/bin/bash
#SBATCH --job-name=pt400m
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# 400M setting: DCLM scale 411m_1x (412M params, 8.2B training tokens, global batch 512 x 2048) on an
# already-tokenized corpus (corpus/tokenize.sh).
# global_bs = 512 divides 8N for N in 1/2/4/8/16, so the node count can follow availability.
# Our runs: 4 nodes x 8 H200. Resumes from the newest checkpoint on requeue (--resume latest is DCLM's default).
# Usage: sbatch -J pt400m_<corpus> pretrain_400m.sh <tokenized_dataset_name (READ)> <master_port>
set -uo pipefail
READ=${1:?tokenized dataset name}; PORT=${2:?port}
R=${WORK_DIR:?set WORK_DIR}; CODE=${DCLM_DIR:?set DCLM_DIR}
L=${PRETRAIN_DIR:-$R/pretrain}/400m/$READ
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
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$PORT
echo "[$(date -u +%FT%TZ)] 411m_1x $READ nodes=$SLURM_JOB_NUM_NODES world=$((SLURM_JOB_NUM_NODES*8)) port=$PORT"
srun --kill-on-bad-exit=1 bash -c "
  export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT PYTHONPATH=$CODE
  ${DCLM_TORCHRUN:-torchrun} \
    --nnodes=\$SLURM_JOB_NUM_NODES \
    --nproc-per-node=8 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m training.train -- \
      --scale 411m_1x \
      --data-config $DC \
      --logs $L/pretrain_logs \
      --num-checkpoints 20 \
      --multiple-data-passes \
      --ignore-parse-errors \
      --data-tolerate-num-ckpts 20 \
      --data-tolerate-error-p 0.5"
