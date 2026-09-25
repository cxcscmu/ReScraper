#!/bin/bash
#SBATCH --job-name=prepack_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/prepack_merge_%j.log
# merge shards (if not merged yet) + light check.  sbatch --export=ALL,OUT=..,NT=8 prepack_merge.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; PY=${REFINER_PY:-python}
export PYTHONUNBUFFERED=1
[ -s $OUT/stats.json ] && grep -q shards $OUT/stats.json || { $PY $ROOT/training/prepack_merge.py $OUT $NT || exit 1; }
$PY $ROOT/training/prepack_light_check.py $OUT || exit 1
echo PREPACK_MERGE_OK $OUT
