#!/bin/bash
#SBATCH --job-name=prepack_shard
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=3:00:00
#SBATCH --output=logs/prepack_shard_%A_%a.log
# Sharded prepack: sbatch --array=0-7 --export=ALL,DATA=..,OUT=..,SP=..,NT=8 prepack_shard.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 NPROC=40
export HF_DATASETS_CACHE=/tmp/hfds_shard_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p $HF_DATASETS_CACHE
${REFINER_PY:-python} $ROOT/training/prepack_shard.py $DATA $OUT $SP $SLURM_ARRAY_TASK_ID $NT; rc=$?
rm -rf $HF_DATASETS_CACHE; echo "shard $SLURM_ARRAY_TASK_ID rc=$rc"; exit $rc
