#!/bin/bash
#SBATCH --job-name=postfilter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/postfilter_%j.log
# Post-filter one corpus: <CDIR>/text -> <CDIR>/text_clean (see postfilter.py). Reads text/ and never writes into it.
# This job is the gate of the unattended chain: it is submitted with --dependency=afterany:<inference array>,
# which fires even when array tasks FAILED or TIMED OUT, so it refuses to run on an incomplete corpus.
# Dedup/tokenize hang off afterok, so failing here stops the chain cleanly.
# usage: sbatch [--export=ALL,EXPECT_SHARDS=N] postfilter.sbatch <corpus name from corpus/corpora.sh>
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
source $ROOT/corpus/corpora.sh
corpus_vars "${1:?corpus}" || exit 2
export PYTHONUNBUFFERED=1
EXPECT=${EXPECT_SHARDS:-10320}
# Orphan .tmp files from preempted/requeued ranks. Safe here: this job only runs once every
# inference task of the array has reached a terminal state, so nothing is still writing.
find "$CDIR/text" -maxdepth 1 -name "*.jsonl.gz.tmp*" -delete 2>/dev/null
n=$(ls "$CDIR/text"/*.jsonl.gz 2>/dev/null | wc -l)
echo "[$(date -u)] postfilter $CORPUS in=$CDIR/text shards=$n expect=$EXPECT"
if [ "$n" -lt "$EXPECT" ]; then
  echo "FATAL incomplete corpus: $n of $EXPECT shards present in $CDIR/text."
  echo "Backfill the missing shards and resubmit; text_clean/ is resumable so finished shards are skipped."
  exit 1
fi
exec ${REFINER_PY:-python} $ROOT/inference/postfilter.py "$CDIR/text" "$CDIR/text_clean" 32
