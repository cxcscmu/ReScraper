#!/bin/bash
#SBATCH --job-name=hp_rel_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Release-decoding variant: figure data files *_5k_rel.json in $HELDOUT_PIPELINE_DIR/data. usage: sbatch rel_build.sbatch <page_table> <W>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
export TOKENIZERS_PARALLELISM=true HP_WORK=$2 HP_SUF=${HP_SUF:-5k_rel} HP_OUT=${HP_OUT:-$HELDOUT_PIPELINE_DIR/data} HP_DECODING=release
$PY $EV/heldout_pipeline/stages/s4_build.py "$1" || exit 1
echo "REL_BUILD_OK"
