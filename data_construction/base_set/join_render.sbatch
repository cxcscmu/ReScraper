#!/bin/bash
#SBATCH --job-name=join_render
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-31%8
#SBATCH --output=logs/join_render_%A_%a.log
# Raw-HTML join + canonical rendering for one half of the seed set (CPU only).
# NT is fixed to 32 on purpose: never slice by $SLURM_ARRAY_TASK_COUNT, which equals the number of tasks
# actually submitted and is wrong when only failed tasks are re-submitted.
# usage: sbatch join_render.sbatch <dir with wanted.jsonl>
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
DM=${1:?dir with wanted.jsonl}
TID=$SLURM_ARRAY_TASK_ID
[ -s "$DM/wanted.jsonl" ] || { echo "no $DM/wanted.jsonl - abort"; exit 1; }
if [ -s "$DM/join_${TID}.jsonl" ]; then echo "task $TID already done"; exit 0; fi
export PYTHONUNBUFFERED=1
${REFINER_PY:-python} $ROOT/data_construction/base_set/join_render.py "$DM" $TID 32
