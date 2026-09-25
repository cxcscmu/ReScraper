#!/bin/bash
#SBATCH --job-name=hp_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Stage 4 (CPU, 3-5 min): figure data files. usage: sbatch s4_build.sbatch <page_table>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
export TOKENIZERS_PARALLELISM=true
$PY $EV/heldout_pipeline/stages/s4_build.py "$1" || exit 1
echo "S4_BUILD_OK"
