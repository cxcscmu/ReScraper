#!/bin/bash
#SBATCH --job-name=ux_inf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=3:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# UltraX stages 2+3 per array task (one node, 8 GPUs): the officially released openbmb/UltraX-0.6B-Preview over this
# task's slice of $UX_WORK/in, with the UltraX repository's own large-scale scripts:
#   stage2_large_scale_execution/inference/inference.py --num_gpus 8 --max_chars 48000   (greedy decoding is
#       hard-coded in inference.py)
#   stage2_large_scale_execution/post_processing/post_process_and_execute.py --num_workers 90   (executes the
#       predicted cleaning functions -> post/<stem>.parquet with columns original, cleaned, processed_functions)
# Post-processing runs inside the same task on the task's own shards; this only changes which allocation runs it.
# Resumable: both UltraX scripts skip any shard whose output parquet exists, so a requeued task redoes
# only what is missing. Partial writes from an interrupted task are caught by ux_validate.py in ux_gate.sbatch.
# usage: sbatch --array=0-27 --export=ALL,NT=28 ux_inf.sbatch
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
export UX_WORK=${UX_WORK:-$WORK_DIR/dclm_pipeline/resiliparse_raw_ultrax}; W=$UX_WORK
NT=${NT:?}
UXD=${ULTRAX_DIR:?set ULTRAX_DIR to a checkout of https://github.com/OpenBMB/UltraX}
UXM=${ULTRAX_MODEL:-$(ls -d "${HF_HOME:?set HF_HOME}"/hub/models--openbmb--UltraX-0.6B-Preview/snapshots/*/models/UltraX | head -1)}
PY=${REFINER_PY:-python}          # vLLM env able to run the UltraX repository scripts
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/tmp/vllm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
# If pyarrow fails with a missing CXXABI_1.3.15 symbol on GPU nodes, prepend the Python environment's lib/ dir to
# LD_LIBRARY_PATH (the system libstdc++ is older than the env's).
A=${SLURM_ARRAY_TASK_ID:-0}
PART=$(printf "%s/in_parts/%03d" "$W" "$A")
echo "[$(date -u)] task $A/$NT host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} model=$UXM out=$W/inf"
$PY $ROOT/baselines/ultrax/ux_slice_in.py "$A" "$NT" || exit 1
cd $UXD || exit 1
$PY stage2_large_scale_execution/inference/inference.py \
  --input_dir "$PART" --model_path "$UXM" --output_dir $W/inf --num_gpus 8 --max_chars 48000
rc=$?
echo "[$(date -u)] task $A inference rc=$rc  outputs=$(ls $W/inf 2>/dev/null | wc -l)"
[ $rc -eq 0 ] || exit $rc
Q=$(printf "%s/inf_parts/%03d" "$W" "$A"); mkdir -p "$Q"
n=0; for f in $(ls "$PART"); do if [ -f "$W/inf/$f" ]; then ln -sfn "$W/inf/$f" "$Q/$f"; n=$((n+1)); fi; done
echo "[$(date -u)] task $A post over $n/$(ls $PART | wc -l) inf shards"
$PY stage2_large_scale_execution/post_processing/post_process_and_execute.py \
  --input_dir "$Q" --output_dir $W/post --num_workers 90
rc=$?
echo "[$(date -u)] task $A post rc=$rc  post=$(ls $W/post 2>/dev/null | wc -l)"
exit $rc
