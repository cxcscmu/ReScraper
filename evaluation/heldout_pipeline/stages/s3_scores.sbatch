#!/bin/bash
#SBATCH --job-name=hp_scores
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# Stage 3b (1 GPU, about 20 min for ~24k unique texts): DataMan + FineWeb-Edu on every unique text (evaluation/scorers,
# one call per shard; finished shards whose INPUT_MD5 matches are skipped). Used by both the greedy and the
# release-decoding chain (the latter scores only the texts s3_reuse.py found no cached score for).
# usage: sbatch s3_scores.sbatch <page_table>      (HP_WORK selects a non-default work dir)
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
PT=$1; TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}; WK=${HP_WORK:-$HELDOUT_PIPELINE_DIR/work/$TAG}
DM=$EV/scorers
mkdir -p $WK/score_out
for f in $WK/score_in/uniq_*.jsonl; do
  [ -s $f ] || continue
  b=$(basename $f .jsonl); H=$(md5sum < $f | cut -c1-32)
  for d in dataman edu; do   # a shard's scores are reused only if they were computed on this exact input
    [ -d $WK/score_out/${d}_$b ] && [ "$(cat $WK/score_out/${d}_$b/INPUT_MD5 2>/dev/null)" != "$H" ] && rm -rf $WK/score_out/${d}_$b
  done
  if [ ! -s $WK/score_out/dataman_$b/$b.scores.jsonl ]; then
    $PY $DM/score_dataman.py --inputs $b=$f --output-dir $WK/score_out/dataman_$b || exit 1
    echo $H > $WK/score_out/dataman_$b/INPUT_MD5
  fi
  if [ ! -s $WK/score_out/edu_$b/$b.edu.jsonl ]; then
    $PY $DM/score_edu.py --inputs $b=$f --output-dir $WK/score_out/edu_$b || exit 1
    echo $H > $WK/score_out/edu_$b/INPUT_MD5
  fi
done
echo "S3_SCORES_OK"
