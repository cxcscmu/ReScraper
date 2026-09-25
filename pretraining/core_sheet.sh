#!/bin/bash
# Merge the 7 eval chunks of a run and compute the Core table with eval_sheet.py (centered accuracy averaged over
# the 22 low-variance tasks; commonsense_qa is dropped). Always report Core from this TSV, never from a field of
# the metrics json. Usage: core_sheet.sh <name>
set -uo pipefail
NAME=${1:?name}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; CODE=${DCLM_DIR:?set DCLM_DIR}; PY=${DCLM_PY:-python}
OUTD=${CORE_DIR:-${WORK_DIR:?set WORK_DIR}/core}; mkdir -p "$OUTD"
nch=$(ls -1 "$CODE/eval_results/chunks/$NAME/"epoch_*/metrics_chunk*.json 2>/dev/null | wc -l); echo "chunks found: $nch (expected 7)"
[[ $nch -eq 7 ]] || { echo "FATAL: expected 7 chunks"; exit 1; }
TMP=$(mktemp)
$PY "$ROOT/pretraining/eval_merge.py" "$NAME" | tee $TMP
grep -q MERGED_OK $TMP || { echo "FATAL: merge not OK"; exit 1; }
MERGED=$(ls -1 "$CODE/eval_results/$NAME/"epoch_*/metrics_mmlu_and_lowvar.json 2>/dev/null | head -1); [[ -s $MERGED ]] || { echo "FATAL: no merged json"; exit 1; }
OUT=$OUTD/core_${NAME}_$(date -u +%Y%m%d_%H%M%S).tsv
$PY "$ROOT/pretraining/eval_sheet.py" --eval_meta_data "$CODE/eval/eval_meta_data.csv" --eval_results "$MERGED" --output "$OUT" || { echo "eval_sheet FAILED"; exit 1; }
echo "================ CORE $NAME ================"; cat "$OUT"; cp "$OUT" "$OUTD/core_${NAME}_LATEST.tsv"
rm -f $TMP
