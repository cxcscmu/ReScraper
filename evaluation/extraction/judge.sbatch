#!/bin/bash
#SBATCH --job-name=extjudge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# gpt-oss-120b extraction judge (+ sanity controls), one model load per array task; array task i takes gid % SHARD_N == i.
# usage: sbatch --array=0-<K-1> --export=ALL,SHARD_N=<K> judge.sbatch <ext_texts.jsonl> <out_dir>
# Hardware: 1x 96 GB GPU (RTX PRO 6000 Blackwell), TP1, GPU_UTIL 0.94, VLLM_MXFP4_USE_MARLIN=1 on that card; ~10 requests/s.
# Do not run it on 48 GB cards at TP2: with this vLLM/MXFP4 build gpt-oss-120b emits "!!!!..." until max_tokens (the
# sanity gate catches it and the job exits 3).
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed "s/^NVIDIA //; s/[^A-Za-z0-9]/_/g")
NG=$(nvidia-smi -L | wc -l); MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
export TP=$NG GPU_NAME JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0} JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b} MAXCH=${MAXCH:-24000} GPU_UTIL=${GPU_UTIL:-0.94}
export SHARD_N=${SHARD_N:-1} SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
[[ $NG -gt 1 ]] && export VLLM_ENABLE_V1_MULTIPROCESSING=1
[[ $GPU_NAME == *PRO_6000* ]] && export VLLM_MXFP4_USE_MARLIN=1
echo "restarts=${SLURM_RESTART_COUNT:-0} gpus=${NG}x$GPU_NAME mem=${MEM}MiB model=$JUDGE_MODEL util=$GPU_UTIL shard=$SHARD_ID/$SHARD_N"
$PYJUDGE $EV/extraction/judge_run.py "$1" "$2"
rc=$?; echo "[$(date)] rc=$rc"; exit $rc
