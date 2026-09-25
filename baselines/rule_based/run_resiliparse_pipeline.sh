#!/bin/bash
# run_resiliparse_pipeline.sh
#
# Resiliparse HTML-to-text extraction of the source pool, followed by the RefinedWeb rule stack (DCLM "step3b",
# dclm_baseline_refinedweb_post_lang.yaml), BFF dedup and tokenization. Produces:
#   - $OUT_ROOT/resiliparse_extract/...        the resiliparse extraction of every pool page (input of UltraX)
#   - $OUT_ROOT/step3b_post_lang/...           RefinedWeb-rule output before dedup
#   - corpus `refinedweb_rule`                 RefinedWeb-rule after BFF dedup (main table: RefinedWeb-rule, 7.04B;
#                                              Figure 3: resiliparse bar)
# The input pool is the DCLM 400M-1x pool sample that has already been URL- and language-filtered upstream (DCLM
# step3a), so step3a is not run again here.
#
# Runs on a submit host: submits sbatch jobs (DCLM process_no_ray.py from pretraining/dclm_patches) and polls squeue
# between phases.
# env: WORK_DIR, DCLM_DIR (DCLM checkout with pretraining/dclm_patches applied), RESCRAPER_ROOT, PIPE_PY (DCLM data-processing
#      Python env with resiliparse), HTML_POOL_DIR, optional RESILIPARSE_OUT, CPU_PARTITION
# usage: bash run_resiliparse_pipeline.sh

set -euo pipefail

POLL=120
N_JOBS=40
CPUS=32
MEM=200G
WALLTIME=2-00:00:00

: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
: ${DCLM_DIR:?set DCLM_DIR to the DCLM checkout}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository}
PY=${PIPE_PY:-python3}
PART=${CPU_PARTITION:-cpu}
RAW_POOL=${HTML_POOL_DIR:-$WORK_DIR/pools/dclm-pool-400m-1x-html-jsonl-step3a-10pct}
OUT_ROOT=${RESILIPARSE_OUT:-$WORK_DIR/dclm_pipeline/resiliparse}

LOG=$PWD/logs
RUNLISTS=$OUT_ROOT/runlists
mkdir -p "$LOG" "$RUNLISTS" "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_single_job() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; do
        log "  waiting for job $jid ..."
        sleep 60
    done
    log "  job $jid finished"
}

wait_jobs() {
    local pattern=$1
    while squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "^${pattern}"; do
        local n
        n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^${pattern}" || true)
        log "  waiting ... $n jobs matching '$pattern' still running/pending"
        sleep "$POLL"
    done
    log "  all '$pattern' jobs finished"
}

rm -f "$RUNLISTS/resiliparse_pool_rel.txt" "$RUNLISTS/resiliparse_extract_rel.txt"

# ============================================================
# 1. Prep: input shard list
# ============================================================
log "=== PHASE 1: Prep ==="
$PY - "$RAW_POOL" "$RUNLISTS/resiliparse_pool_rel.txt" <<'PY'
import sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(root.glob("*.jsonl.gz"))
out.write_text("".join(f"{p.name}\n" for p in files))
print(f"shard count = {len(files)}")
PY
TOTAL=$(wc -l < "$RUNLISTS/resiliparse_pool_rel.txt")
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))
log "  prep done: TOTAL=$TOTAL shards, CHUNK=$CHUNK"

# ============================================================
# 2. Resiliparse extract (HTML -> plain text)
# ============================================================
log "=== PHASE 2: Resiliparse extract ==="

for START in $(seq 0 $CHUNK $((TOTAL-1))); do
    sbatch -p "$PART" --cpus-per-task=$CPUS --mem=$MEM --time=$WALLTIME \
        --job-name=rp_${START} --output="$LOG/rp_%j.out" \
        --wrap "cd $DCLM_DIR && $PY ray_processing/process_no_ray.py \
        --raw_data_dirpath $RAW_POOL \
        --shard_list_file $RUNLISTS/resiliparse_pool_rel.txt \
        --shard_start $START --shard_count $CHUNK \
        --readable_name resiliparse_extract \
        --output_dir $OUT_ROOT/resiliparse_extract \
        --config_path baselines/baselines_configs/resiliparse_extract.yaml \
        --source_name cc --workers $CPUS \
        --disable_filter_io_dump --skip_dataset_metadata --overwrite"
done
log "  submitted resiliparse jobs"
wait_jobs "rp_"

# ============================================================
# 3. Build resiliparse-extract shard list for step3b
# ============================================================
log "=== PHASE 3: Build resiliparse-extract shard list ==="

RP_PD=$OUT_ROOT/resiliparse_extract/resiliparse_extract/processed_data
$PY - "$RP_PD" "$RUNLISTS/resiliparse_extract_rel.txt" <<'PY'
import sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(f for f in root.iterdir() if f.name.endswith(('.jsonl.zst', '.jsonl.gz', '.jsonl')))
out.write_text(''.join(f'{f.name}\n' for f in files))
print(f'resiliparse_extract shard count = {len(files)}')
PY
RP_COUNT=$(wc -l < "$RUNLISTS/resiliparse_extract_rel.txt")
RP_CHUNK=$(( (RP_COUNT + N_JOBS - 1) / N_JOBS ))
log "  resiliparse extract produced $RP_COUNT shards"

# ============================================================
# 4. Step 3b: RefinedWeb heuristic filters
# ============================================================
log "=== PHASE 4: Step3b (RefinedWeb rules) ==="

for START in $(seq 0 $RP_CHUNK $((RP_COUNT-1))); do
    sbatch -p "$PART" --cpus-per-task=$CPUS --mem=$MEM --time=$WALLTIME \
        --job-name=rs3b_${START} --output="$LOG/rs3b_%j.out" \
        --wrap "cd $DCLM_DIR && $PY ray_processing/process_no_ray.py \
        --raw_data_dirpath $RP_PD \
        --shard_list_file $RUNLISTS/resiliparse_extract_rel.txt \
        --shard_start $START --shard_count $RP_CHUNK \
        --readable_name step3b_post_lang \
        --output_dir $OUT_ROOT/step3b_post_lang \
        --config_path baselines/baselines_configs/dclm_baseline_refinedweb_post_lang.yaml \
        --source_name cc --workers $CPUS \
        --disable_filter_io_dump --skip_dataset_metadata --overwrite"
done
log "  submitted step3b jobs"
wait_jobs "rs3b_"

# ============================================================
# 5. Step 4: BFF dedup (13-grams, 0.8, old-both, 65B expected n-grams) + 6. tokenization, shared corpus steps.
#    Dedup input: $OUT_ROOT/step3b_post_lang/dclm_baseline_refinedweb_post_lang/processed_data
#    Dedup output: $OUT_ROOT/step4_dedup_oldboth_65b
# ============================================================
log "=== PHASE 5: Step4 BFF dedup + tokenize (corpus refinedweb_rule) ==="
S4_JOB=$(sbatch --parsable -p "$PART" "$ROOT/corpus/dedup_bff.sh" refinedweb_rule)
TOK_JOB=$(sbatch --parsable -p "$PART" --dependency=afterok:$S4_JOB "$ROOT/corpus/tokenize.sh" refinedweb_rule)
log "  submitted dedup job $S4_JOB, tokenize job $TOK_JOB"
wait_single_job "$TOK_JOB"

log "=========================================="
log "  RESILIPARSE PIPELINE COMPLETE"
log "  OUT_ROOT: $OUT_ROOT"
log "=========================================="
