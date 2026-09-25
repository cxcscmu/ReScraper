#!/bin/bash
#SBATCH --job-name=dripper_ultrax_inf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Dripper -> UltraX, stages 2+3: the official UltraX-0.6B refinement of Dripper step2 text, one input part per node
# job. Resume-safe via dripper_ultrax_todo.py (validates/deletes corrupt outputs, symlinks only missing work).
# Post-processing (execute the predicted edits) runs on the node's CPUs after inference, unless POSTPROC=0
# (then run dripper_ultrax_post.sbatch for the part).
# usage: sbatch dripper_ultrax_infer.sbatch <part 0..5>
set -uo pipefail
K=${1:?usage: sbatch dripper_ultrax_infer.sbatch <part>}
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
W=${DRIPPER_UX_WORK:-$WORK_DIR/dclm_pipeline/dripper_ultrax/work}
UXD=${ULTRAX_DIR:?set ULTRAX_DIR to a checkout of https://github.com/OpenBMB/UltraX}
UXM=${ULTRAX_MODEL:-$(ls -d "${HF_HOME:?set HF_HOME}"/hub/models--openbmb--UltraX-0.6B-Preview/snapshots/*/models/UltraX 2>/dev/null | head -1)}
[[ -f $UXM/config.json ]] || { echo "FATAL: UltraX model dir missing"; exit 1; }
PY=${REFINER_PY:-python}
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/dripper_ultrax_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"

echo "[$(date -u)] part=$K todo-build (limit=${TODO_LIMIT:-0})"
$PY "$ROOT/baselines/ultrax/dripper_ultrax/dripper_ultrax_todo.py" --inp "$W/in/part$K" --out "$W/inf/part$K" \
    --todo "$W/todo_p${K}_r${TODO_REM:-0}" --limit "${TODO_LIMIT:-0}" --mod "${TODO_MOD:-1}" --rem "${TODO_REM:-0}" || exit 1

if [[ -n $(ls -A "$W/todo_p${K}_r${TODO_REM:-0}" 2>/dev/null) ]]; then
  cd "$UXD"
  $PY stage2_large_scale_execution/inference/inference.py \
      --input_dir "$W/todo_p${K}_r${TODO_REM:-0}" \
      --model_path "$UXM" \
      --output_dir "$W/inf/part$K" \
      --num_gpus "${GPUS:-8}" \
      --max_chars 48000 || { echo "inference FAILED"; exit 1; }
else
  echo "no todo - inference already complete for part$K"
fi

if [[ ${POSTPROC:-1} -eq 0 ]]; then echo "postproc skipped (split mode)"; exit 0; fi
echo "[$(date -u)] post-process part=$K"
cd "$UXD"
$PY stage2_large_scale_execution/post_processing/post_process_and_execute.py \
    --input_dir "$W/inf/part$K" \
    --output_dir "$W/post/part$K" \
    --num_workers 64 || { echo "postprocess FAILED"; exit 1; }

nin=$(ls "$W/in/part$K" | wc -l); npost=$(ls "$W/post/part$K" 2>/dev/null | wc -l)
echo "[$(date -u)] part$K in=$nin post=$npost"
