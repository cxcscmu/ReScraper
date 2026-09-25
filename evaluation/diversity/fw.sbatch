#!/bin/bash
#SBATCH --job-name=div_fw
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
# FineWeb-rule corpus of the 376 shards (not plotted; part of diversity_ngrams_pool376.json): raw resiliparse text out of
# the UltraX parquets (REFINER_PY, needs pyarrow), then the FineWeb-rule chain (datatrove 0.2.0 env), 22 processes.
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
export OMP_NUM_THREADS=1
$PY $EV/diversity/extract_raw.py || exit 1
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HELDOUT_PIPELINE_DIR/cache} $PYDT $EV/diversity/fw_scale.py 22
