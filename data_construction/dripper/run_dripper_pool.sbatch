#!/bin/bash
#SBATCH --job-name=dripper_step1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
# Step 1 of the Dripper teacher run: Dripper (MinerU-HTML) main-content inference over the whole source pool.
#
# One array task = one whole node (8 GPUs). Each task owns SHARDS_PER_TASK consecutive pool shards (sorted file
# list) and runs 8 workers, one per GPU, each striding through the task's slice. Resumable: a shard with a .done
# marker is skipped, so --requeue after an interruption is safe.
# Output: $DRIPPER_STEP1_DIR/<stem>.jsonl ({"input": raw html, "main_html": ...}), per-shard logs in logs/ under it.
# usage: sbatch --array=0-161 run_dripper_pool.sbatch       (162 tasks x 64 shards = 10,368 >= the number of pool shards)
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
POOL=${HTML_POOL_DIR:-$WORK_DIR/pools/dclm-pool-400m-1x-html-jsonl-step3a-10pct}
OUT=${DRIPPER_STEP1_DIR:-$WORK_DIR/dclm_pipeline/dripper/step1_inference}
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY=${REFINER_PY:-python}          # vLLM env with the `dripper` package (MinerU-HTML) installed
MODEL=${DRIPPER_MODEL:-$(ls -d "${HF_HOME:?set HF_HOME}"/hub/models--opendatalab--MinerU-HTML/snapshots/*/ | head -1)}
SHARDS_PER_TASK=${SHARDS_PER_TASK:-64}
GPUS=8

mapfile -t ALL < <(ls "$POOL"/*.jsonl.gz | sort)
TOTAL=${#ALL[@]}
TID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( TID * SHARDS_PER_TASK ))
echo "[$(date +%T)] node=$(hostname) task=$TID shards[$START..$((START+SHARDS_PER_TASK-1))] of $TOTAL"

run_gpu() {                       # $1 = local gpu index; strides through this task's slice
  local g=$1 n=0 s=0
  for (( k=g; k<SHARDS_PER_TASK; k+=GPUS )); do
    local idx=$(( START + k ))
    (( idx >= TOTAL )) && break
    local shard="${ALL[$idx]}"
    local base; base=$(basename "$shard" .jsonl.gz)
    [[ -f "$OUT/${base}.done" ]] && { s=$((s+1)); continue; }
    if CUDA_VISIBLE_DEVICES=$g $PY \
          "$ROOT/data_construction/dripper/step1_dripper_inference.py" run_shard \
          --shard_file "$shard" --output_dir "$OUT" --model_path "$MODEL" \
          --shard_id "$idx" --batch_size 500 \
          > "$LOGD/${base}.out" 2>&1; then
      mv -f "$OUT/shard_$(printf '%03d' "$idx").jsonl" "$OUT/${base}.jsonl" 2>/dev/null
      touch "$OUT/${base}.done"; n=$((n+1))
    else
      echo "$(date +%T) FAILED $base (task $TID gpu $g)" >> "$LOGD/failures.txt"
    fi
  done
  echo "[$(date +%T)] gpu$g: done=$n skipped=$s"
}

for (( g=0; g<GPUS; g++ )); do run_gpu "$g" & done
wait
echo "[$(date +%T)] TASK $TID END; pool progress: $(ls "$OUT"/*.done 2>/dev/null | wc -l)/$TOTAL"
