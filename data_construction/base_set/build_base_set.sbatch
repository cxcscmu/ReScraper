#!/bin/bash
#SBATCH --job-name=build_base_set
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/build_base_set_%j.log
# The no-rewrite base SFT set (Stage-1 parent, 1,383,115 rows in our run) from one or more halves of the
# seed set. For every half <dir> (with wanted.jsonl and join_*.jsonl):
#   1. build_staged.py   -> <dir>/sft_staged.jsonl          (Dripper removals + teacher decision on the same ids)
#   2. build_dfirst.py   -> <dir>/sft_dfirst.jsonl           + verify_dfirst.py round-trip gate
#   3. build_norw.py     -> <dir>/sft_dfirst_norw.jsonl      (every <rewrite> -> <delete>)
#   4. check_stem_disjoint.py: the half's shards must be disjoint from the held-out shards
# then 5. merge_gate.py union: concatenate all halves, seeded shuffle, structural gate + tag mix.
# In our run the first half additionally went through leak_filter.py between steps 1 and 2 (see README).
# usage: sbatch build_base_set.sbatch <out.jsonl> <half_dir> [<half_dir> ...]
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; : ${HELDOUT_SHARDS:?stem list of the held-out shards}
PY=${REFINER_PY:-python}; S=$ROOT/data_construction/base_set
export PYTHONUNBUFFERED=1
OUT=${1:?out.jsonl}; shift
PARTS=()
for DM in "$@"; do
  echo "=== half $DM"
  [ -s "$DM/sft_staged.jsonl" ] || $PY $S/build_staged.py "$DM" "$DM/sft_staged.jsonl" || exit 1
  STAGED=$DM/sft_staged.jsonl
  if [ "${LEAK_FILTER_DIR:-}" != "" ] && [ "${LEAK_FILTER_HALF:-}" = "$DM" ]; then
    $PY $S/leak_filter.py "$LEAK_FILTER_DIR" "$STAGED" "$DM/sft_staged_filtered.jsonl" || exit 1
    STAGED=$DM/sft_staged_filtered.jsonl
  fi
  $PY $S/build_dfirst.py "$STAGED" "$DM/sft_dfirst.jsonl" || exit 1
  $PY $S/verify_dfirst.py "$STAGED" "$DM/sft_dfirst.jsonl" 4000 || exit 1
  $PY $S/build_norw.py "$DM/sft_dfirst.jsonl" "$DM/sft_dfirst_norw.jsonl" || exit 1
  $PY $S/check_stem_disjoint.py "$DM/wanted.jsonl" "$HELDOUT_SHARDS" || { echo "FATAL: shard overlap with held-out"; exit 1; }
  PARTS+=("$DM/sft_dfirst_norw.jsonl")
done
echo "=== union + seeded shuffle + structural gate"
$PY $S/merge_gate.py union "$OUT" "${PARTS[@]}" || exit 1
echo "BASE_SET_DONE $OUT"
