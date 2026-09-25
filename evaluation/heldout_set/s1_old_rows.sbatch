#!/bin/bash
#SBATCH --job-name=heldout_s1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.log
# Held-out set step 1 (CPU, ~20 min): provenance / re-render / pool join / FineWeb-Edu of the 3,161 existing staged
# rows, WARC ids of the two-stage refiner's SFT source, shard sets of the other pool SFT files.
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
W=$HELDOUT5K_DIR/work; mkdir -p $W
export PROCS=${PROCS:-6} EDU_THREADS=${EDU_THREADS:-8}
$PY $EV/heldout_set/s1_old_rows.py $W || { echo HELDOUT_S1_FAIL; exit 1; }
echo "HELDOUT_S1_OK"
