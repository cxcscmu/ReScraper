#!/bin/bash
#SBATCH --job-name=tokenize_par
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=112G
#SBATCH --time=6:00:00
#SBATCH --output=logs/tokenize_par_%A_%a.log
# Optional parallel variant of tokenize.sh (used for the operation-ablation corpora): array task k tokenizes every
# N-th dedup shard into $TOK_OUT/p<k>/ with its own manifest and exp_data json; tokenize_merge.py then stitches the
# parts into the single layout tokenize.sh produces (within-part shuffle only; open_lm shuffles shard order per
# epoch anyway). Retries a part up to 3 times (a transient Ray actor failure is the known mode).
# usage: sbatch --array=0-7 tokenize_parallel.sh <corpus> <N>     (N must equal the array size)
#        then: sbatch --dependency=afterok:<array> --wrap "$REFINER_PY corpus/tokenize_merge.py <corpus> <N>"
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
source $ROOT/corpus/corpora.sh
LEG=${1:?corpus}; N=${2:?N}; K=${SLURM_ARRAY_TASK_ID:?}
corpus_vars "$LEG" || exit 2
export PYTHONUNBUFFERED=1
PART=$TOK_IN/.parts_$N/p$K
rm -rf "$PART"; mkdir -p "$PART" "$TOK_OUT"
ls "$TOK_IN" | grep -E "\.jsonl(\.gz|\.zst|\.zstd)?$" | sort | awk -v k=$K -v n=$N "NR % n == k" | while read f; do ln -s "$TOK_IN/$f" "$PART/$f"; done
NSH=$(ls "$PART" | wc -l)
echo "[$(date -u)] $LEG part $K/$N: $NSH shards -> $TOK_OUT/p$K"
cd "$CODE"; export PYTHONPATH=$CODE
rc=1
for att in 1 2 3; do
  rm -rf "$TOK_OUT/p$K" "/tmp/rs_${SLURM_JOB_ID}_${K}_${att}"
  echo "[$(date -u)] part $K attempt $att/3 starting ($NSH shards)"
  $PY ray_processing/tokenize_shuffle.py --input "$PART" --output "$TOK_OUT/p$K" \
    --readable_name "${READ}_p$K" --content_key text --tokenizer EleutherAI/gpt-neox-20b \
    --ray_spill_location "/tmp/rs_${SLURM_JOB_ID}_${K}_${att}" --overwrite
  rc=$?
  TARS=$(ls "$TOK_OUT/p$K"/*.tar 2>/dev/null | wc -l)
  echo "[$(date -u)] part $K attempt $att/3 rc=$rc tars=$TARS"
  if [ "$rc" = "0" ]; then break; fi
  sleep 60
done
echo "[$(date -u)] part $K rc=$rc tars=$(ls $TOK_OUT/p$K/*.tar 2>/dev/null | wc -l)"; exit $rc
