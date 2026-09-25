#!/bin/bash
#SBATCH --job-name=heldout_s2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Held-out set step 2 (CPU, ~15 min): exclusion + seeded pick of 2 new held-out shards + candidate pages (render, seed
# filter, LID, stage-1 feasibility, dedup) + pool join + FineWeb-Edu. The fastText LID runs in the DCLM data env.
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
W=$HELDOUT5K_DIR/work; SEED=${SEED:-5000}
export PROCS=${PROCS:-6} EDU_THREADS=${EDU_THREADS:-8}
echo "seed=$SEED"
$PY $EV/heldout_set/s2a_new_shards.py $W $W/two_stage_refiner_stems.txt $W/two_stage_src_warc.tsv $SEED || { echo HELDOUT_S2_FAIL_2a; exit 1; }
$PYPIPE $EV/heldout_set/s2b_lid.py $W || { echo HELDOUT_S2_FAIL_2b; exit 1; }
$PY $EV/heldout_set/s2c_new_candidates.py $W || { echo HELDOUT_S2_FAIL_2c; exit 1; }
echo "HELDOUT_S2_OK"
