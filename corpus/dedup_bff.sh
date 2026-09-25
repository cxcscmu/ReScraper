#!/bin/bash
#SBATCH --job-name=dedup_bff
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dedup_bff_%j.log
# BFF Bloom-filter deduplication with DCLM's settings (identical for every corpus in the paper):
# 13-gram, threshold 0.8, remove-type old-both, 65e9 expected n-grams, fp-rate 0.01.
# Requires the bff binary built in $DCLM_DIR/dedup/bff (cargo build --release).
# NOT resumable: partial output is cleared before each fresh run; a .marks/dedup.done marker skips reruns.
# usage: sbatch dedup_bff.sh <corpus name from corpora.sh>
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
source $ROOT/corpus/corpora.sh
corpus_vars "${1:?corpus}" || exit 2
mkdir -p "$MARKD" "$LOGD"
[[ -f $MARKD/dedup.done ]] && { echo "dedup already done"; exit 0; }
rm -rf "$DEDUP_OUT"; mkdir -p "$DEDUP_OUT"
echo "[$(date +%F' '%T)] $CORPUS dedup start in=$DEDUP_IN"
( cd "$CODE/dedup/bff" && target/release/bff bff \
    --inputs "$DEDUP_IN" --output-directory "$DEDUP_OUT" \
    --expected-ngram-count 65000000000 --fp-rate 0.01 \
    --min-ngram-size 13 --max-ngram-size 13 --filtering-threshold 0.8 \
    --remove-type old-both ) \
  && touch "$MARKD/dedup.done" && echo "[$(date +%F' '%T)] dedup OK ($(ls "$DEDUP_OUT" | wc -l) shards)" \
  || { echo "dedup FAILED"; exit 1; }
