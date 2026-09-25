#!/bin/bash
#SBATCH --job-name=dripper_step2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# Step 2 of the Dripper teacher run: main_html -> text (webkit_txt) for every step-1 shard.
# CPU-only, so it can run WHILE step-1 inference is still going: each task skips shards whose .jsonl.gz already
# exists, so re-running picks up new ones. Task i handles shards[i::NT] of the sorted step-1 file list.
# usage: sbatch --array=0-15 run_convert_text.sbatch
#   When re-running only some array indices, pass the ORIGINAL task count explicitly (--export=ALL,NT=16):
#   SLURM_ARRAY_TASK_COUNT is then the number of re-run tasks and would change the slicing.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
IN=${DRIPPER_STEP1_DIR:-$WORK_DIR/dclm_pipeline/dripper/step1_inference}
OUT=${DRIPPER_STEP2_DIR:-$WORK_DIR/dclm_pipeline/dripper/step2_text}
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
PY=${REFINER_PY:-python}          # env with the `dripper` package (MinerU-HTML) and bs4

TID=${SLURM_ARRAY_TASK_ID:-0}
NT=${NT:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "[$(date +%T)] convert task $TID/$NT on $(hostname)"
$PY "$ROOT/data_construction/dripper/convert_main_html_to_text.py" \
    --in_dir "$IN" --out_dir "$OUT" \
    --task_id "$TID" --num_tasks "$NT" --workers "${SLURM_CPUS_PER_TASK:-32}"
echo "[$(date +%T)] done; converted so far: $(ls "$OUT"/*.jsonl.gz 2>/dev/null | wc -l)"
