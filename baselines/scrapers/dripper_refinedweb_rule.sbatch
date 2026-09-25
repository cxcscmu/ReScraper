#!/bin/bash
#SBATCH --job-name=dripper_rwrule
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Dripper + RefinedWeb-rule (Figure 3, Dripper bar at 1B): Dripper step-2 text (data_construction/dripper/)
#   -> the same RefinedWeb rule stack as the resiliparse baseline (dclm_baseline_refinedweb_post_lang.yaml, DCLM
#      process_no_ray.py) -> BFF dedup -> tokenization (shared corpus steps, corpus name dripper_refinedweb_rule).
# Mirrors run_resiliparse_pipeline.sh step for step (step3a is skipped: the pool is already language-filtered
# upstream), so the only difference from the resiliparse baseline is the extractor.
# Resumable: the rule phase writes a .marks/step3b.done marker and is skipped when present.
# env: WORK_DIR, DCLM_DIR (with pretraining/dclm_patches applied), RESCRAPER_ROOT, PIPE_PY
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
: ${DCLM_DIR:?set DCLM_DIR to the DCLM checkout}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
LEG=$WORK_DIR/dclm_pipeline/dripper
TXT=${DRIPPER_STEP2_DIR:-$LEG/step2_text}
MARK=$LEG/.marks
LOGD=$LEG/logs_pipeline
mkdir -p "$MARK" "$LOGD"
cd "$DCLM_DIR"

PY=${PIPE_PY:-python}
export PYTHONPATH=$DCLM_DIR
log(){ echo "[$(date +%F\ %T)] $*"; }

# ---------- Phase 1: step3b heuristic filters (same yaml as the resiliparse baseline) ----------
if [[ ! -f $MARK/step3b.done ]]; then
  log "=== Phase 1: step3b_post_lang ==="
  $PY ray_processing/process_no_ray.py \
      --source_name cc --workers "${SLURM_CPUS_PER_TASK:-32}" \
      --raw_data_dirpath "$TXT" \
      --readable_name step3b_post_lang \
      --output_dir "$LEG/step3b_post_lang" \
      --config_path baselines/baselines_configs/dclm_baseline_refinedweb_post_lang.yaml \
      --overwrite > "$LOGD/step3b.log" 2>&1 \
    && touch "$MARK/step3b.done" && log "step3b OK" || { log "step3b FAILED (see $LOGD/step3b.log)"; exit 1; }
fi
S3B=$LEG/step3b_post_lang/dclm_baseline_refinedweb_post_lang/processed_data
log "rule output: $S3B"

# ---------- Phase 2: BFF dedup (65B old-both) + tokenization, shared corpus steps ----------
S4=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} $ROOT/corpus/dedup_bff.sh dripper_refinedweb_rule)
TK=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterok:$S4 $ROOT/corpus/tokenize.sh dripper_refinedweb_rule)
log "submitted dedup $S4 and tokenize $TK for corpus dripper_refinedweb_rule"
