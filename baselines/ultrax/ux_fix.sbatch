#!/bin/bash
#SBATCH --job-name=ux_fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=3:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Sweep for whatever the ux_inf array left behind (a dead worker does not fail its task): inference for every in/
# shard without an inf/ output, then post-processing for every inf/ shard without a post/ output. Submitted by
# ux_gate.sbatch after ux_validate.py --clean (never while ux_inf tasks run). Recomputes the missing sets itself,
# so it can be re-run until both are empty.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
export UX_WORK=${UX_WORK:-$WORK_DIR/dclm_pipeline/resiliparse_raw_ultrax}; W=$UX_WORK
UXD=${ULTRAX_DIR:?set ULTRAX_DIR to a checkout of https://github.com/OpenBMB/UltraX}
UXM=${ULTRAX_MODEL:-$(ls -d "${HF_HOME:?set HF_HOME}"/hub/models--openbmb--UltraX-0.6B-Preview/snapshots/*/models/UltraX | head -1)}
PY=${REFINER_PY:-python}
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/tmp/vllm_fix_${SLURM_JOB_ID}
M=$W/in_missing_${SLURM_JOB_ID}; P=$W/inf_missing_${SLURM_JOB_ID}; mkdir -p "$M" "$P"
comm -23 <(ls $W/in | sort) <(ls $W/inf | sort) | while read -r f; do ln -sfn "$W/in/$f" "$M/$f"; done
n=$(ls "$M" | wc -l)
echo "[$(date -u)] fix host=$(hostname) restart=${SLURM_RESTART_COUNT:-0}: $n shards without inference output"
cd $UXD || exit 1
if [ "$n" -gt 0 ]; then
  $PY stage2_large_scale_execution/inference/inference.py \
    --input_dir "$M" --model_path "$UXM" --output_dir $W/inf --num_gpus 8 --max_chars 48000
  echo "[$(date -u)] fix inference rc=$?  inf=$(ls $W/inf | wc -l)"
fi
comm -23 <(ls $W/inf | sort) <(ls $W/post | sort) | while read -r f; do ln -sfn "$W/inf/$f" "$P/$f"; done
m=$(ls "$P" | wc -l)
echo "[$(date -u)] fix: $m inf shards without post output"
if [ "$m" -gt 0 ]; then
  $PY stage2_large_scale_execution/post_processing/post_process_and_execute.py \
    --input_dir "$P" --output_dir $W/post --num_workers 90
  echo "[$(date -u)] fix post rc=$?  post=$(ls $W/post | wc -l)"
fi
exit 0
