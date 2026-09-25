#!/bin/bash
#SBATCH --job-name=div_count
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.log
# Distinct n-gram counting over the 376 shards, one process per corpus, in parallel (7-22 min per corpus; memory grows
# with the exact n-gram unions). usage: sbatch count.sbatch <corpus> [<corpus> ...]
#   corpora: rescraper ultrax proxc refinedweb_rule fineweb_rule resiliparse_raw ; then: python build_json.py <out.json>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1
mkdir -p logs
for c in "$@"; do
  $PY $EV/diversity/ngram_scale.py $c > logs/count_${c}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
grep -h DONE logs/count_*_${SLURM_JOB_ID}.log
