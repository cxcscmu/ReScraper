#!/bin/bash
#SBATCH --job-name=hp_gather
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Stage 1b + 3a (CPU, 2-3 min): execute the greedy student programs (+ teacher fidelity), then collect the texts to score.
# usage: sbatch s3_gather.sbatch <page_table>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
set -o pipefail
PT=$1; TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}; WK=${HP_WORK:-$HELDOUT_PIPELINE_DIR/work/$TAG}
$PY $EV/heldout_pipeline/stages/s1_ours_exec.py "$PT" 2>&1 | tail -60 || exit 1
[ -s $WK/ours.jsonl ] || exit 1
$PY $EV/heldout_pipeline/stages/s3_gather.py "$PT" || exit 1
echo "S3_GATHER_OK"
