#!/bin/bash
#SBATCH --job-name=ablation_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/ablation_build_%j.log
# Shard gate + build_arms.py (nodel / noedit / norw arms from the raw sidecar; malformed F3 pages dropped from every
# arm; the full arm verified byte for byte). The corpus is accepted when 10,315..10,320 shards are present (10,318
# = 10,320 pool shards minus the 2 shards whose rendering never terminates, exactly as the released corpus).
# Writes $ABL/EXPECT_SHARDS for the post-filter jobs.
# usage: sbatch [--export=ALL,RELEASE_TEXT=$WORK_DIR/dclm_pipeline/rescraper_corpus/text] build_arms.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${WORK_DIR:?}
ABL=${ABL:-$WORK_DIR/dclm_pipeline/operation_ablation}
find "$ABL/raw" "$ABL/full/text" "$ABL"/{nodel,noedit,norw}/text -maxdepth 1 -name "*.jsonl.gz.tmp*" -print -delete 2>/dev/null | sed "s/^/rm orphan /"
NR=$(ls "$ABL/raw" | grep -c "_raw.jsonl.gz$"); NT=$(ls "$ABL/full/text" | grep -c "_processed.jsonl.gz$")
echo "[$(date -u)] shard gate: raw=$NR full_text=$NT"
if [ "$NR" != "$NT" ]; then echo "FATAL raw/text shard counts differ"; exit 1; fi
if [ "$NT" -ge 10315 ] && [ "$NT" -le 10320 ]; then EXPECT=$NT; else echo "FATAL only $NT shards (need 10315..10320, expected 10318)"; exit 1; fi
CMP=""; [ -n "${RELEASE_TEXT:-}" ] && CMP="--compare-release $RELEASE_TEXT"
${REFINER_PY:-python} $ROOT/ablations/operation_ablation/build_arms.py "$ABL" 32 $CMP
rc=$?
if [ "$rc" = "0" ]; then echo "$EXPECT" > "$ABL/EXPECT_SHARDS"; echo "[$(date -u)] BUILD gate OK expect=$EXPECT"; else echo "[$(date -u)] BUILD FAILED rc=$rc"; fi
exit $rc
