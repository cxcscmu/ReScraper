#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Multi-node SFT launcher (srun + torchrun; the HF Trainer picks up the DeepSpeed ZeRO-3 config).
# Global batch must be 192 = micro x accum x nodes x 8 GPUs (Stage 1: 8 nodes x accum 3; Stage 2: 4 nodes x accum 6),
# with --loss_scale 24 (the default) the accumulated gradient is then identical across these topologies.
# Resumes from the newest checkpoint on requeue.
# Usage: sbatch -J <name> train_multinode.sbatch <data.jsonl> <outdir> <port> [model] [epochs] [micro] [accum] [extra flags]
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
DATA=${1:?data}; OUT=${2:?outdir}; PORT=${3:-29511}
MODEL=${4:-Qwen/Qwen3-0.6B}; EPOCHS=${5:-3}; MICRO=${6:-2}; ACCUM=${7:-3}
EXTRA="${8:-}"   # extra flags for sft_train.py, e.g. --tag_loss_weight 5
PREPACK=${PREPACK:?prepacked dir}; [[ -d $PREPACK/train ]] || { echo FATAL: $PREPACK missing; exit 1; }
GB=$(( MICRO * ACCUM * SLURM_JOB_NUM_NODES * 8 ))
[[ $GB -eq 192 ]] || { echo "FATAL: global batch $GB != 192"; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_PROJECT=${WANDB_PROJECT:-rescraper-sft} WANDB_DIR=${WANDB_DIR:-logs/wandb}
export WANDB_NAME=$SLURM_JOB_NAME
export GENEVAL_LOG_DIR=${GENEVAL_LOG_DIR:-logs}
export HF_DATASETS_CACHE=/tmp/hfds_${SLURM_JOB_ID}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 NCCL_DEBUG=WARN
mkdir -p "$WANDB_DIR" "$HF_DATASETS_CACHE"
cd "$ROOT/training"
[[ -s $DATA ]] || { echo "FATAL: $DATA missing"; exit 1; }
# RESUME is a plain string on purpose: an array expanded inside the double-quoted srun command below would
# split into two argv words. Checkpoint steps are sorted numerically on their own.
RESUME=
LATEST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed "s#.*/checkpoint-##" | sort -n | tail -1 || true)
[[ -n ${LATEST:-} && -f $OUT/checkpoint-$LATEST/trainer_state.json ]] && RESUME="--resume_from_checkpoint $OUT/checkpoint-$LATEST"
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "[$(date -u +%FT%TZ)] $SLURM_JOB_NAME nodes=$SLURM_JOB_NUM_NODES master=$MASTER data=$DATA"
srun --kill-on-bad-exit=1 bash -c "
  ${TORCHRUN:-torchrun} --nnodes=\$SLURM_JOB_NUM_NODES --nproc-per-node=8 --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER --master_port=$PORT \
    sft_train.py \
      --data_path $DATA --model_path $MODEL --output_dir $OUT \
      --max_seq_len 32768 --epochs $EPOCHS --lr 8e-5 \
      --per_device_batch_size $MICRO --gradient_accumulation_steps $ACCUM \
      --val_size 1000 --gen_eval_every 50 --gen_eval_samples 64 \
      --save_steps 30 --save_total_limit 20 --use_packing \
      --deepspeed deepspeed_zero3.json --prepacked_dir $PREPACK $EXTRA $RESUME"
rc=$?
echo "[$(date -u +%FT%TZ)] rc=$rc"
exit $rc
