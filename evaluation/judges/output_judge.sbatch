#!/bin/bash
#SBATCH --job-name=outJ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# gpt-oss-120b output judge (Figure 1(b), appendix page-quality table): 1x 96 GB GPU, TP1, MXFP4 Marlin; about 0.6 min per 1,000 texts.
# usage: python output_judge.py build <heldout_pipeline work dir> output_judge_in.jsonl; sbatch output_judge.sbatch output_judge_in.jsonl <out output_judge_5000.jsonl>
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed "s/^NVIDIA //; s/[^A-Za-z0-9]/_/g") TP=1
[[ $GPU_NAME == *PRO_6000* ]] && export VLLM_MXFP4_USE_MARLIN=1
echo "gpu=$GPU_NAME restarts=${SLURM_RESTART_COUNT:-0}"
$PYJUDGE $EV/judges/output_judge.py "$1" "$2"
rc=$?; echo "[$(date)] rc=$rc"; exit $rc
